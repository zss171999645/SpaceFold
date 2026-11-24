"""Command-line interface for the native octree and geometry codec."""

from __future__ import annotations

import argparse
from pathlib import Path


def _prepare(args):
    from ._grid import AdaptiveTUDF
    from .octree import bridge_legacy_r256, uniform_band_tree

    envelope = AdaptiveTUDF.load_compact(args.input)
    reference = bridge_legacy_r256(envelope, provenance={"sample_id": args.input.stem})
    del envelope
    tree = uniform_band_tree(reference, preserve_terminal_surface=True)
    tree.save(args.output)


def _build(args):
    import numpy as np

    from .surface import (
        audit_edt_crops,
        audit_tree,
        capped_squared_edt,
        multiscale_tree,
        surface_indices,
    )

    with np.load(args.input, allow_pickle=False) as source:
        lo, hi = source["bbox_min"], source["bbox_max"]
        ids, _ = surface_indices(source["points"], lo, hi)
    if not len(ids):
        raise ValueError("the input has no points inside its bounding box")
    occupied = np.zeros((1024,) * 3, dtype=bool)
    occupied.flat[ids] = True
    squared = capped_squared_edt(occupied)
    del occupied
    audit_edt_crops(squared, ids)
    tree = multiscale_tree(squared, lo, hi)
    del squared
    audit_tree(tree, ids)
    tree.provenance["sample_id"] = args.input.stem
    tree.save(args.output)


def _cuda_codec(args):
    import torch

    from .checkpoint import load_geometry_checkpoint

    if args.volfill_root is not None:
        from ._source import register_volfill_source

        register_volfill_source(args.volfill_root)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("geometry encoding and decoding require a CUDA device with spconv")
    model, _ = load_geometry_checkpoint(args.weights, device)
    return model, device


def _encode(args):
    import torch

    from ._io import save_npz
    from .model import LATENT_SHAPE, TreeBatch
    from .octree import UnifiedTUDF

    tree = UnifiedTUDF.load(args.input)
    model, device = _cuda_codec(args)
    batch = TreeBatch.from_trees([tree]).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        mean, _ = model.encoder(batch)
    latent = mean[0].detach().to(device="cpu", dtype=torch.float16)
    if tuple(latent.shape) != LATENT_SHAPE or not torch.isfinite(latent).all():
        raise ValueError("the encoder returned an invalid geometry latent")
    save_npz(args.output, raw=latent.numpy(), bbox_min=tree.bbox_min, bbox_max=tree.bbox_max)


def _decode(args):
    import numpy as np
    import torch

    from ._io import save_npz
    from .model import LATENT_SHAPE
    from .surface import _bbox, predicted_surface_points

    with np.load(args.input, allow_pickle=False) as source:
        raw = np.asarray(source["raw"], dtype=np.float32)
        lo, hi = _bbox(source["bbox_min"], source["bbox_max"])
    if raw.shape != LATENT_SHAPE or not np.isfinite(raw).all():
        raise ValueError("expected a finite raw latent of shape [16, 32, 32, 32]")
    model, device = _cuda_codec(args)
    z = torch.from_numpy(raw).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        levels = model.decoder(z, teacher=None)
    if any(not torch.isfinite(level.values).all() for level in levels):
        raise ValueError("the decoder returned nonfinite distances")
    points, _ = predicted_surface_points(levels, bbox_min=lo, bbox_max=hi)
    save_npz(args.output, points=points, bbox_min=lo, bbox_max=hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler, description in (
        ("build", _build, "Build a native octree from bounded XYZ surface points."),
        (
            "prepare",
            _prepare,
            "Convert a compact R256/R512/R1024 envelope into the five-level octree.",
        ),
        ("encode", _encode, "Encode a native octree into its posterior-mean latent."),
        ("decode", _decode, "Decode a raw geometry latent into world-space points."),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--input", type=Path, required=True, help="Input NPZ path")
        command.add_argument("--output", type=Path, required=True, help="New output NPZ path")
        if name in ("encode", "decode"):
            command.add_argument(
                "--weights", type=Path, required=True, help="Geometry safetensors checkpoint"
            )
            command.add_argument(
                "--volfill-root",
                type=Path,
                help="Path to the VolFill checkout if its modules are outside the environment",
            )
            command.add_argument("--device", default="cuda", help="CUDA device, for example cuda:0")
        command.set_defaults(handler=handler)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.handler(args)
    print(args.output)


if __name__ == "__main__":
    main()
