"""Project panorama features onto target-frame rays and sample scene latents."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from . import _validate_sampling
from .backbones import build_dit


def load_latent_stats(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read per-channel training moments and return broadcastable FP32 tensors."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"mean", "std"}:
            raise ValueError("The latent statistics NPZ must contain mean and std")
        mean, std = archive["mean"], archive["std"]
    for name, value in (("mean", mean), ("std", std)):
        if value.dtype != np.float32 or value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError(f"Expected 16 finite float32 values for latent {name}")
    if (std <= 1.0e-6).any():
        raise ValueError("Latent standard deviations must exceed 1e-6")
    return torch.from_numpy(mean)[:, None, None, None], torch.from_numpy(std)[:, None, None, None]


def _fibonacci_rays(count: int) -> torch.Tensor:
    index = torch.arange(count, dtype=torch.float32) + 0.5
    y = 1.0 - 2.0 * index / count
    radius = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
    azimuth = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return torch.stack((radius * azimuth.sin(), y, radius * azimuth.cos()), dim=-1)


def _direction_embedding(rays: torch.Tensor, channels: int) -> torch.Tensor:
    unit = F.normalize(rays.float(), dim=-1)
    latitude = torch.asin(unit[..., 1].clamp(-1.0, 1.0))
    longitude = torch.atan2(unit[..., 0], unit[..., 2])
    frequencies = torch.arange(1, channels // 4 + 1, dtype=unit.dtype, device=unit.device)
    latitude = latitude[..., None] * frequencies
    longitude = longitude[..., None] * frequencies
    return torch.cat((latitude.sin(), latitude.cos(), longitude.sin(), longitude.cos()), dim=-1)


def _sample_erp(feature: torch.Tensor, camera_rays: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample pixel-centre ERP features across the longitude seam."""
    height, width = feature.shape[-2:]
    unit = F.normalize(camera_rays.float(), dim=-1)
    longitude = torch.atan2(unit[..., 0], unit[..., 2])
    latitude = torch.asin(unit[..., 1].clamp(-1.0, 1.0))
    pixel_x = (longitude / math.tau + 0.5) * width - 0.5
    pixel_y = (0.5 - latitude / math.pi) * height - 0.5
    grid_x = 2.0 * (pixel_x + 1.0 + 0.5) / (width + 2) - 1.0
    grid_y = 2.0 * (pixel_y + 0.5) / height - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2).to(feature.dtype)
    wrapped = torch.cat((feature[..., -1:], feature, feature[..., :1]), dim=-1)
    samples = F.grid_sample(
        wrapped, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return samples.squeeze(-1).transpose(1, 2).contiguous()


class PanoramaProjector(nn.Module):
    """Project three DA² feature levels into 2,048 target-frame image tokens."""

    def __init__(self):
        super().__init__()
        rays = _fibonacci_rays(2048)
        self.register_buffer("rays", rays, persistent=False)
        self.register_buffer("positions", _direction_embedding(rays, 768), persistent=False)
        self.input_projection = nn.Sequential(nn.LayerNorm(896), nn.Linear(896, 768))

    def forward(
        self,
        pyramid: list[torch.Tensor],
        camera_to_world: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        channels = tuple(level.shape[1] for level in pyramid)
        if channels != (512, 256, 128):
            raise ValueError(f"DA² pyramid channels must be (512, 256, 128), got {channels}")
        camera_rays = torch.einsum("bji,nj->bni", camera_to_world.float(), self.rays)
        features = torch.cat([_sample_erp(level, camera_rays) for level in pyramid], dim=-1)
        tokens = self.input_projection(features)
        tokens = tokens + self.positions.to(tokens.dtype)[None]
        support = _sample_erp(confidence, camera_rays).clamp(0.0, 1.0)
        return tokens * support.to(dtype=tokens.dtype)


def load_component(checkpoint: Path, name: str, module: nn.Module) -> dict[str, str]:
    """Strictly restore one component from the flat, tensor-only generator file."""
    prefix = name + "."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
        metadata = archive.metadata() or {}
        if metadata.get("format") != "spacefold.panorama.v1":
            raise ValueError("Expected a SpaceFold panorama generator safetensors file")
        keys = archive.keys()
        if {key.partition(".")[0] for key in keys} != {"dit", "conditioner_adapter"}:
            raise ValueError("The generator must contain DiT and panorama projector parameters")
        state = {
            key[len(prefix) :]: archive.get_tensor(key) for key in keys if key.startswith(prefix)
        }
        for key, expected in module.state_dict().items():
            if key in state and state[key].dtype != expected.dtype:
                raise ValueError(
                    f"{prefix}{key} has dtype {state[key].dtype}, expected {expected.dtype}"
                )
        module.load_state_dict(state, strict=True)
    return metadata


class PanoramaGenerator(nn.Module):
    """SpaceFold flow model over a 16 × 32³ latent with visible-scene conditioning."""

    def __init__(self, volfill_root: Path):
        super().__init__()
        self.dit = build_dit(volfill_root)

    def forward(
        self,
        noisy: torch.Tensor,
        time: torch.Tensor,
        image_tokens: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        return self.dit(torch.cat((noisy, visible), dim=1), time * 1000.0, image_tokens)


def load_generator(
    checkpoint: Path,
    volfill_root: Path,
    device: torch.device,
) -> PanoramaGenerator:
    """Load the generator for use with precomputed image tokens and visible geometry."""
    model = PanoramaGenerator(volfill_root)
    load_component(checkpoint, "dit", model.dit)
    return model.requires_grad_(False).eval().to(device)


@torch.inference_mode()
def sample_latent(
    generator: PanoramaGenerator,
    visible: torch.Tensor,
    image_tokens: torch.Tensor,
    *,
    steps: int,
    guidance: float,
    seed: int,
) -> torch.Tensor:
    """Integrate the conditional flow from Gaussian noise at t=1 to t=0."""
    steps, guidance, seed = _validate_sampling(steps, guidance, seed)
    if visible.device.type != "cuda" or visible.device != image_tokens.device:
        raise ValueError("Visible geometry and image tokens must be on the same CUDA device")
    if visible.shape != (1, 16, 32, 32, 32) or image_tokens.shape != (1, 2048, 768):
        raise ValueError("Expected one visible 16 × 32³ latent and 2,048 × 768 image tokens")
    device = visible.device
    rng = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn(visible.shape, device=device, generator=rng)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    empty_image = torch.zeros_like(image_tokens)
    empty_geometry = torch.zeros_like(visible)
    for index in range(steps):
        time = times[index].expand(1)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            velocity = generator(sample, time, image_tokens, visible)
            if guidance != 1.0:
                unconditional = generator(sample, time, empty_image, empty_geometry)
                velocity = unconditional + guidance * (velocity - unconditional)
        sample = sample + (times[index + 1] - times[index]) * velocity.float()
    return sample[0].float().cpu()
