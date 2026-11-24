"""Load the public DA² and VolFill networks from user-supplied source trees."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


def _source_module(source: Path):
    """Load a real upstream module, preserving package-relative imports."""
    source = source.resolve(strict=True)
    identity = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    name = f"_spacefold_upstream_{identity}"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import upstream source at {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # A failed import must not leave a partially initialized package cached.
        del sys.modules[name]
        raise
    return module


def load_depth_backbone(root: Path, checkpoint: Path, device: torch.device) -> nn.Module:
    """Restore the DA² model package without importing its dataset or demo CLI."""
    source = root / "src/da2/model/__init__.py"
    configuration = root / "configs/infer.json"
    if not source.is_file() or not configuration.is_file():
        raise FileNotFoundError(f"Expected a DA-2 checkout containing {source} and {configuration}")
    with configuration.open(encoding="utf-8") as handle:
        config = json.load(handle)
    height, width = 546, 1092
    config["inference"].update(min_pixels=height * width, max_pixels=height * width)
    config["spherevit"]["sphere"].update(width=width, height=height, hfov=math.tau, vfov=math.pi)
    network = _source_module(source).SphereViT(config=config)
    network.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    return network.requires_grad_(False).eval().to(device)


def build_dit(root: Path) -> nn.Module:
    """Construct the checkpoint's 32-channel input, 16-channel output transformer."""
    from spacefold.geometry import register_volfill_source

    register_volfill_source(root)
    from volfill.amodal.model.dit.dit3d import CoarseTUDFDiT

    return CoarseTUDFDiT(
        resolution=32,
        in_channels=32,
        model_channels=960,
        cond_channels=768,
        out_channels=16,
        num_blocks=16,
        num_heads=15,
        mlp_ratio=4.0,
        patch_size=1,
        pe_mode="ape",
        use_fp16=False,
        use_checkpoint=False,
        qk_rms_norm=True,
        qk_rms_norm_cross=False,
    )


@torch.inference_mode()
def panorama_features(
    model: nn.Module,
    panorama: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Get FP32 relative radial distance and the BF16 image-conditioning pyramid.

    Both passes call DA²'s own forward. Hooks capture the three published
    decoder upsampling stages for SpaceFold's learned spherical projection.
    """
    image = F.interpolate(panorama, size=(546, 1092), mode="bilinear", align_corners=False)
    with torch.autocast(image.device.type, enabled=False):
        distance = model(image.float())
    if (
        distance.shape != (1, 546, 1092)
        or not torch.isfinite(distance).all()
        or (distance <= 0).any()
    ):
        raise ValueError("DA² produced an invalid radial distance map")

    pyramid: list[torch.Tensor] = []

    def collect_stage(_module, _inputs, output):
        pyramid.append(output)

    stages = model.get_submodule("vit_w_esphere._vit_w_esphere.up_sampler")
    with ExitStack() as hooks, torch.autocast(image.device.type, dtype=torch.bfloat16):
        for stage in stages:
            hooks.enter_context(stage.register_forward_hook(collect_stage))
        model(image)
    if len(pyramid) != 3:
        raise ValueError(f"Expected three DA² pyramid levels, received {len(pyramid)}")
    return distance[0].float().cpu(), pyramid
