"""Generate a metric 3D scene from one level, 2:1 equirectangular RGB panorama."""

import argparse
from pathlib import Path

from spacefold.dependencies import DependencyError, panorama_paths

from . import _validate_camera_height, _validate_sampling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panorama", type=Path, required=True, help="Input equirectangular image.")
    parser.add_argument(
        "--dit-checkpoint", type=Path, required=True, help="SpaceFold generator safetensors."
    )
    parser.add_argument(
        "--vae-checkpoint", type=Path, required=True, help="SpaceFold geometry VAE safetensors."
    )
    parser.add_argument(
        "--da2-checkpoint", type=Path, help="Override the managed DA² model.safetensors."
    )
    parser.add_argument(
        "--latent-stats",
        type=Path,
        required=True,
        help="SpaceFold mean and std in numeric NPZ format.",
    )
    parser.add_argument("--da2-root", type=Path, help="Override the managed DA-2 source checkout.")
    parser.add_argument(
        "--volfill-root", type=Path, help="Override the managed VolFill source checkout."
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        required=True,
        help="Measured or assumed camera height in metres.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for geometry and intermediate results.",
    )
    parser.add_argument("--steps", type=int, default=50, help="Euler steps (default 50).")
    parser.add_argument(
        "--guidance", type=float, default=3.0, help="Classifier-free guidance scale (default 3)."
    )
    parser.add_argument("--seed", type=int, default=0, help="CUDA noise seed (default 0).")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default cuda:0).")
    args = parser.parse_args()
    try:
        _validate_camera_height(args.camera_height)
        _validate_sampling(args.steps, args.guidance, args.seed)
    except ValueError as error:
        parser.error(str(error))
    for path in (
        args.panorama,
        args.dit_checkpoint,
        args.vae_checkpoint,
        args.latent_stats,
    ):
        if not path.is_file():
            parser.error(f"Missing input file: {path}")
    try:
        args.volfill_root, args.da2_root, args.da2_checkpoint = panorama_paths(
            volfill_root=args.volfill_root,
            da2_root=args.da2_root,
            da2_checkpoint=args.da2_checkpoint,
        )
    except (DependencyError, OSError) as error:
        parser.error(str(error))

    from .pipeline import generate

    generate(**vars(args))


if __name__ == "__main__":
    main()
