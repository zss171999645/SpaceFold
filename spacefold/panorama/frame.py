"""Recover a level room frame and a metric scale from relative radial distance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _validate_camera_height

# The panorama camera looks along +z, with +x to the right and +y up.
# The output room frame has +x to the right, +y forward and +z up.
CAMERA_TO_WORLD = np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)


@dataclass(frozen=True)
class RoomFrame:
    camera: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scale: float


def _scene_bounds(points: np.ndarray, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(points[:, :2] - camera[None, :2], axis=1)
    nearby = points[radius <= 12.0]
    if len(nearby) < 4096:
        nearby = points
    if len(nearby) < 4096:
        raise ValueError("At least 4,096 depth samples are required to fit room bounds")

    left = np.quantile(nearby[:, :2], 0.02, axis=0).astype(np.float32) - 0.5
    right = np.quantile(nearby[:, :2], 0.98, axis=0).astype(np.float32) + 0.5
    above_floor = nearby[nearby[:, 2] >= -0.25]
    if len(above_floor) < 1024:
        above_floor = nearby
    ceiling = max(float(np.quantile(above_floor[:, 2], 0.98)), 2.4)
    side = min(max(float((right - left).max()), ceiling, 4.0), 14.0)
    xy = (left + right) * 0.5
    lower = np.asarray([xy[0] - side / 2, xy[1] - side / 2, (ceiling - side) / 2], np.float32)
    upper = np.asarray([xy[0] + side / 2, xy[1] + side / 2, (ceiling + side) / 2], np.float32)
    return lower, upper


def room_from_distance(distance: np.ndarray, camera_height: float) -> tuple[RoomFrame, np.ndarray]:
    """Back-project every native DA² pixel and fit the bounding cube in metres."""
    _validate_camera_height(camera_height)
    if distance.shape != (546, 1092) or not np.isfinite(distance).all() or (distance <= 0).any():
        raise ValueError("Expected a finite positive 546 × 1092 radial distance map")
    height, width = distance.shape
    x = np.arange(width, dtype=np.float64) + 0.5
    y = np.arange(height, dtype=np.float64) + 0.5
    longitude = (x - width / 2) / width * math.tau
    latitude = (height / 2 - y) / height * math.pi
    latitude, longitude = np.meshgrid(latitude, longitude, indexing="ij")
    rays = np.stack(
        (
            np.cos(latitude) * np.sin(longitude),
            np.cos(latitude) * np.cos(longitude),
            np.sin(latitude),
        ),
        axis=-1,
    ).astype(np.float32)
    distance = distance.astype(np.float64)
    downward = -rays[..., 2].astype(np.float64)
    floor_region = downward >= 0.25
    floor_drop = float(np.quantile(distance[floor_region] * downward[floor_region], 0.90))
    if not np.isfinite(floor_drop) or floor_drop <= 1.0e-8:
        raise ValueError("Could not estimate metric scale from downward-looking panorama rays")
    scale = camera_height / floor_drop
    camera = np.asarray([0, 0, camera_height], dtype=np.float32)
    points = (camera + distance[..., None] * scale * rays).astype(np.float32).reshape(-1, 3)
    lower, upper = _scene_bounds(points, camera)
    return RoomFrame(camera, lower, upper, scale), points
