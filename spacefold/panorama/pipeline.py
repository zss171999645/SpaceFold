"""End-to-end single-panorama inference."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from spacefold._atomic import result_directory
from spacefold.dependencies import panorama_paths
from spacefold.geometry import TreeBatch, load_geometry_checkpoint, register_volfill_source
from spacefold.geometry.surface import (
    audit_edt_crops,
    audit_tree,
    capped_squared_edt,
    multiscale_tree,
    predicted_surface_points,
    surface_indices,
)

from . import _validate_camera_height, _validate_sampling
from .backbones import load_depth_backbone, panorama_features
from .frame import CAMERA_TO_WORLD, RoomFrame, room_from_distance
from .model import (
    PanoramaProjector,
    load_component,
    load_generator,
    load_latent_stats,
    sample_latent,
)


def _image(path: Path) -> tuple[torch.Tensor, str]:
    """Decode the upright panorama and identify the exact bytes used for inference."""
    encoded = path.read_bytes()
    with Image.open(BytesIO(encoded)) as source:
        orientation = source.getexif().get(274, 1)
        image = source if orientation == 1 else ImageOps.exif_transpose(source)
        if image.width != image.height * 2:
            raise ValueError(
                f"Expected a level 2:1 panorama, received {image.width} × {image.height}"
            )
        pixels = np.array(
            image.convert("RGB").resize((2048, 1024), Image.Resampling.BICUBIC), copy=True
        )
    rgb = torch.from_numpy(pixels).permute(2, 0, 1)[None].float() / 255.0
    return rgb, hashlib.sha256(encoded).hexdigest()


def _image_conditions(image, da2_root, da2_checkpoint, dit_checkpoint, device):
    projector = PanoramaProjector()
    metadata = load_component(dit_checkpoint, "conditioner_adapter", projector)
    projector = projector.requires_grad_(False).eval().to(device)
    backbone = load_depth_backbone(da2_root, da2_checkpoint, device)
    distance, pyramid = panorama_features(backbone, image)
    axes = torch.from_numpy(CAMERA_TO_WORLD).to(device)[None]
    confidence = torch.ones_like(image[:, :1])
    with torch.autocast(device.type, dtype=torch.bfloat16):
        tokens = projector(pyramid, axes, confidence)
    if tokens.shape != (1, 2048, 768) or not torch.isfinite(tokens).all():
        raise ValueError("Invalid panorama conditioning tokens")
    return distance.numpy(), tokens, metadata


def _visible_tree(points: np.ndarray, frame: RoomFrame, sample_id: str):
    indices, _ = surface_indices(points, frame.lower, frame.upper)
    if len(indices) == 0:
        raise ValueError("The estimated room bounds contain no visible surface samples")
    occupied = np.zeros((1024,) * 3, dtype=bool)
    occupied.reshape(-1)[indices] = True
    squared = capped_squared_edt(occupied)
    del occupied
    audit_edt_crops(squared, indices)
    tree = multiscale_tree(squared, frame.lower, frame.upper)
    del squared
    audit_tree(tree, indices)
    tree.provenance.update(sample_id=sample_id, visible_source="panorama_depth")
    return tree


def _encode_visible(vae, tree, device: torch.device) -> torch.Tensor:
    batch = TreeBatch.from_trees([tree]).to(device)
    with torch.autocast(device.type, dtype=torch.float16):
        mean, _ = vae.encoder(batch)
    # The training cache stored the posterior mean in FP16 before normalization.
    visible = mean[0].float().cpu().half()
    if visible.shape != (16, 32, 32, 32) or not torch.isfinite(visible).all():
        raise ValueError("Invalid visible-scene latent from the geometry encoder")
    return visible


def _write_ply(path: Path, points: np.ndarray) -> None:
    """Write point positions in metres, with a uniform grey display colour."""
    xyzrgb = np.dtype(
        [(name, "<f4") for name in ("x", "y", "z")]
        + [(name, "u1") for name in ("red", "green", "blue")]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for start in range(0, len(points), 1_000_000):
            chunk = points[start : start + 1_000_000]
            data = np.empty(len(chunk), dtype=xyzrgb)
            data["x"], data["y"], data["z"] = chunk.T
            data["red"] = data["green"] = data["blue"] = 180
            data.tofile(handle)


@torch.inference_mode()
def generate(
    *,
    panorama: Path,
    dit_checkpoint: Path,
    vae_checkpoint: Path,
    da2_checkpoint: Path | None = None,
    latent_stats: Path,
    da2_root: Path | None = None,
    volfill_root: Path | None = None,
    camera_height: float,
    output: Path,
    steps: int = 50,
    guidance: float = 3.0,
    seed: int = 0,
    device: str = "cuda:0",
) -> None:
    """Generate and publish one complete result directory on the local filesystem."""
    camera_height = _validate_camera_height(camera_height)
    steps, guidance, seed = _validate_sampling(steps, guidance, seed)
    target = torch.device(device)
    if target.type != "cuda" or torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError(
            "SpaceFold panorama inference requires a CUDA-enabled PyTorch installation"
        )
    if torch.cuda.get_device_capability(target)[0] < 8:
        raise RuntimeError(
            "SpaceFold panorama inference requires native BF16 (CUDA capability 8.0+)"
        )
    torch.cuda.set_device(target.index if target.index is not None else torch.cuda.current_device())
    volfill_root, da2_root, da2_checkpoint = panorama_paths(
        volfill_root=volfill_root, da2_root=da2_root, da2_checkpoint=da2_checkpoint
    )
    register_volfill_source(volfill_root)
    torch.set_num_threads(2)
    with result_directory(output) as work:
        mean, std = load_latent_stats(latent_stats)
        image, rgb_digest = _image(panorama)
        image = image.to(target)
        print("Estimating radial distance and panorama image tokens", flush=True)
        distance, tokens, metadata = _image_conditions(
            image, da2_root, da2_checkpoint, dit_checkpoint, target
        )
        np.save(work / "da2_distance_fp32.npy", distance)
        del image
        torch.cuda.empty_cache()

        print("Building the visible-scene octree", flush=True)
        frame, visible_points = room_from_distance(distance, camera_height)
        tree = _visible_tree(visible_points, frame, f"panorama_{rgb_digest[:16]}")
        tree.save(work / "visible_native_tree.npz")
        del visible_points

        print("Encoding visible geometry", flush=True)
        vae, _ = load_geometry_checkpoint(vae_checkpoint, target)
        visible = _encode_visible(vae, tree, target)
        del tree
        np.save(work / "visible_latent_fp16.npy", visible.numpy())
        torch.cuda.empty_cache()

        print(f"Sampling the scene latent in {steps} Euler steps", flush=True)
        generator = load_generator(dit_checkpoint, volfill_root, target)
        normalized_visible = ((visible.float() - mean) / std)[None].to(target)
        normalized = sample_latent(
            generator, normalized_visible, tokens, steps=steps, guidance=guidance, seed=seed
        )
        raw = normalized * std + mean
        if not torch.isfinite(raw).all():
            raise ValueError("The sampler produced non-finite latent values")
        np.savez_compressed(
            work / "generated_latent.npz",
            normalized=normalized.numpy(),
            raw=raw.numpy(),
            bbox_min=frame.lower,
            bbox_max=frame.upper,
        )
        del generator, tokens
        torch.cuda.empty_cache()

        print("Decoding and writing the scene point cloud", flush=True)
        with torch.autocast(target.type, dtype=torch.float16):
            decoded = vae.decoder(raw[None].to(target), teacher=None)
        if not all(torch.isfinite(level.values).all() for level in decoded):
            raise ValueError("The geometry decoder produced non-finite distances")
        points, counts = predicted_surface_points(
            decoded, bbox_min=frame.lower, bbox_max=frame.upper
        )
        if len(points) == 0 or not np.isfinite(points).all():
            raise ValueError("The geometry decoder produced an empty or non-finite surface")
        np.savez_compressed(
            work / "prediction_points.npz",
            points=points,
            bbox_min=frame.lower,
            bbox_max=frame.upper,
        )
        _write_ply(work / "generated_geometry.ply", points)
        record = {
            "panorama_sha256": rgb_digest,
            "weights": metadata.get("weights"),
            "checkpoint_step": metadata.get("step"),
            "coordinate_unit": "metre",
            "camera_height_m": camera_height,
            "camera_position": frame.camera.tolist(),
            "camera_to_world_axes": CAMERA_TO_WORLD.tolist(),
            "radial_distance_scale": frame.scale,
            "bbox_min": frame.lower.tolist(),
            "bbox_max": frame.upper.tolist(),
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "generated_points": len(points),
            "points_per_resolution": counts,
        }
        (work / "scene.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(points):,} scene points to {output / 'generated_geometry.ply'}", flush=True)
