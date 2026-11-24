"""Build the native surface octree and extract terminal points in world coordinates."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ._grid import AdaptiveTUDF, _bbox, child_coords, coords_to_points
from .octree import (
    SURFACE_SAFE_POLICY,
    _pool_any,
    bridge_legacy_r256,
    uniform_band_tree,
)


def surface_indices(points, lo, hi):
    """Map world XYZ points to unique ZYX voxel IDs inside a half-open box."""
    lo, hi = _bbox(lo, hi)
    points = np.asarray(points)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.dtype.kind not in "iuf"
        or not np.isfinite(points).all()
    ):
        raise ValueError("expected finite world XYZ points of shape [N, 3]")
    inside = ((points >= lo) & (points < hi)).all(axis=1)
    grid = np.floor((points[inside] - lo) / ((hi - lo) / 1024))
    # Subtraction can round a valid point just below the upper face to index 1024.
    xyz = np.clip(grid, 0, 1023).astype(np.uint32)
    ids = (xyz[:, 2] * 1024 + xyz[:, 1]) * 1024 + xyz[:, 0]
    return np.unique(ids), inside


def capped_squared_edt(occupied):
    """Compute squared Euclidean distance capped at nine using separable erosions."""
    if occupied.ndim != 3 or occupied.dtype != np.bool_ or not occupied.any():
        raise ValueError("expected a nonempty boolean occupancy volume")
    squared = np.full(occupied.shape, 9, dtype=np.uint8)
    squared[occupied] = 0
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = 5
        structure = -(np.arange(-2, 3, dtype=np.int16) ** 2).reshape(shape)
        squared = ndimage.grey_erosion(squared, structure=structure, mode="constant", cval=9)
    return squared


def audit_edt_crops(squared, seed_ids):
    if squared.shape != (1024,) * 3 or not len(seed_ids):
        raise ValueError("expected a 1024-cubed distance field and occupied voxel IDs")
    audits = []
    for index in np.linspace(0, len(seed_ids) - 1, 4, dtype=int):
        seed = int(seed_ids[index])
        center = (seed // 1024**2, seed // 1024 % 1024, seed % 1024)
        lower = tuple(min(max(c - 24, 0), 976) for c in center)
        crop = squared[tuple(slice(v, v + 48) for v in lower)]
        oracle = np.minimum(np.rint(ndimage.distance_transform_edt(crop != 0) ** 2), 9).astype(
            np.uint8
        )
        inner = (slice(3, -3),) * 3
        if not np.array_equal(crop[inner], oracle[inner]):
            raise ValueError("native distance field differs from the crop distance transform")
        audits.append({"lower_zyx": list(lower), "inner_voxels": 42**3, "exact": True})
    return audits


def multiscale_tree(squared, lo, hi):
    """Build band-three distance fields at each level and preserve the native surface."""
    if squared.shape != (1024,) * 3 or squared.dtype != np.uint8:
        raise ValueError("expected an R1024 uint8 squared distance field")
    lo, hi = _bbox(lo, hi)
    occ512 = _pool_any(squared == 0)
    mask512 = ndimage.maximum_filter(occ512, size=5, mode="constant", cval=0)
    ref512 = np.argwhere(mask512).astype(np.int32)
    ref256 = np.argwhere(
        ndimage.maximum_filter(_pool_any(mask512), size=3, mode="constant", cval=0)
    ).astype(np.int32)
    del mask512
    q512 = capped_squared_edt(occ512)
    base = np.sqrt(capped_squared_edt(_pool_any(occ512)).astype(np.float32)).astype(np.float16)
    midcoords = child_coords(ref256)
    middle = np.sqrt(q512[tuple(midcoords.T)].astype(np.float32)).astype(np.float16).reshape(-1, 8)
    finecoords = child_coords(ref512)
    fine = (
        np.sqrt(squared[tuple(finecoords.T)].astype(np.float32)).astype(np.float16).reshape(-1, 8)
    )
    del q512, occ512, midcoords, finecoords
    adaptive = AdaptiveTUDF(
        base_tudf=base,
        bbox_min=lo,
        bbox_max=hi,
        truncation_voxels=3.0,
        refine_base_parent_zyx=ref256,
        middle_children_tudf=middle,
        refine_middle_parent_zyx=ref512,
        fine_children_tudf=fine,
    )
    return uniform_band_tree(bridge_legacy_r256(adaptive), preserve_terminal_surface=True)


def audit_tree(tree, expected_ids):
    tree.validate()
    if tuple(level.resolution for level in tree.levels) != (64, 128, 256, 512, 1024):
        raise ValueError("expected tree resolutions 64, 128, 256, 512, 1024")
    if tree.provenance["policy"] != SURFACE_SAFE_POLICY:
        raise ValueError("tree uses a different refinement policy")
    terminal = tree.surface_coords_by_level(0.5)
    if any(len(terminal[r]) for r in (64, 128, 256, 512)):
        raise ValueError("unexpected coarse terminal surface")
    coords = terminal[1024].astype(np.uint32)
    actual = np.sort((coords[:, 0] * 1024 + coords[:, 1]) * 1024 + coords[:, 2])
    if not np.array_equal(actual, expected_ids):
        raise ValueError("terminal surface differs from the input voxel set")
    return {
        "all_native_seeds_exact": True,
        "native_surface_count": len(actual),
        "coarse_terminal_surface_count": 0,
    }


def predicted_surface_points(
    outputs,
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    batch_index: int = 0,
) -> tuple[np.ndarray, dict[int, int]]:
    """Extract world-space voxel centers for the selected decoded scene."""
    bbox_min, bbox_max = _bbox(bbox_min, bbox_max)
    coordinate_sets = {}
    for level in outputs:
        surface = (
            (~level.split) & (level.values.float() <= -2 / 3) & (level.coords[:, 0] == batch_index)
        )
        coordinate_sets[int(level.resolution)] = (
            level.coords[surface, 1:].detach().cpu().numpy().astype(np.int32, copy=False)
        )
    parts = [
        coords_to_points(coords, bbox_min, bbox_max, resolution)
        for resolution, coords in coordinate_sets.items()
        if len(coords)
    ]
    points = (
        np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
        if parts
        else np.empty((0, 3), dtype=np.float32)
    )
    return points, {resolution: len(coords) for resolution, coords in coordinate_sets.items()}
