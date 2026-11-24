"""Single-panorama scene generation with SpaceFold."""

from numbers import Real


def _validate_camera_height(height: float) -> None:
    if not isinstance(height, Real) or isinstance(height, bool) or not 0.5 <= height <= 3.0:
        raise ValueError("Camera height must lie in [0.5, 3.0] metres")


def _validate_sampling(steps: int, guidance: float, seed: int) -> None:
    if type(steps) is not int or not 1 <= steps <= 200:
        raise ValueError("Euler steps must be an integer in [1, 200]")
    if not isinstance(guidance, Real) or isinstance(guidance, bool) or not 0.0 <= guidance <= 10.0:
        raise ValueError("Guidance must lie in [0, 10]")
    if type(seed) is not int or not -(2**63) <= seed < 2**64:
        raise ValueError("Seed must be an integer in the torch.Generator seed range")
