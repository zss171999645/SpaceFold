"""Multilevel truncated unsigned distance fields and their lossless NPZ encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from ._grid import (
    CHILD_OFFSETS_ZYX,
    AdaptiveTUDF,
    coords_to_linear,
    linear_to_coords,
    pack_tudf_palette4,
    unpack_tudf_palette4,
)
from ._io import save_npz

FORMAT = "level_generic_tudf_palette4_v1"


SURFACE_SAFE_POLICY = "uniform_child_band_or_descendant_surface_safe_v2"


@dataclass(frozen=True)
class TUDFLevel:
    """The root uses ZYX raster order. Children follow each split parent in order."""

    resolution: int
    values: np.ndarray
    split: np.ndarray


@dataclass(frozen=True)
class UnifiedTUDF:
    levels: tuple[TUDFLevel, ...]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    truncation_voxels: float
    provenance: dict

    def validate(self) -> None:
        if not self.levels:
            raise ValueError("an octree needs at least one level")
        base = self.levels[0].resolution
        if base < 1 or base & (base - 1):
            raise ValueError("base resolution must be a power of two")
        if not np.isfinite(self.truncation_voxels) or self.truncation_voxels <= 0:
            raise ValueError("invalid truncation")
        lower, upper = np.asarray(self.bbox_min), np.asarray(self.bbox_max)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not (np.isfinite(lower).all() and np.isfinite(upper).all() and (upper > lower).all())
        ):
            raise ValueError("invalid bbox")
        expected = base**3
        for depth, level in enumerate(self.levels):
            if level.resolution != base * 2**depth:
                raise ValueError("levels must double consecutively")
            if level.values.shape != (expected,) or level.values.dtype != np.float16:
                raise ValueError("values must be aligned float16 nodes")
            if level.split.shape != (expected,) or level.split.dtype != np.bool_:
                raise ValueError("split must be an aligned boolean mask")
            if (
                not np.isfinite(level.values).all()
                or np.any(level.values < 0)
                or np.any(level.values > self.truncation_voxels)
            ):
                raise ValueError("values outside the truncated distance range")
            expected = 8 * int(np.count_nonzero(level.split))
        if expected:
            raise ValueError("the finest level cannot split")

    def iter_coords(self):
        """Generate deterministic coordinates without storing redundant indices."""
        coords = linear_to_coords(
            np.arange(self.levels[0].resolution ** 3), self.levels[0].resolution
        )
        for index, level in enumerate(self.levels):
            yield level, coords
            if index + 1 < len(self.levels):
                parents = coords[level.split]
                coords = (2 * parents[:, None, :] + CHILD_OFFSETS_ZYX[None]).reshape(-1, 3)

    def surface_coords_by_level(self, threshold_voxels: float = 0.5) -> dict[int, np.ndarray]:
        self.validate()
        if not 0 <= threshold_voxels < self.truncation_voxels:
            raise ValueError("surface threshold must lie inside the band")
        return {
            level.resolution: np.ascontiguousarray(
                coords[(~level.split) & (level.values <= threshold_voxels)], dtype=np.int32
            )
            for level, coords in self.iter_coords()
        }

    def save(self, path: str | Path) -> None:
        """Save the exact palette values and little-endian split bits to a new NPZ file."""
        self.validate()
        if self.truncation_voxels != 3.0:
            raise ValueError("palette4 storage is qualified only for band3")
        arrays = {
            "storage_format": np.array(FORMAT),
            "resolutions": np.array([level.resolution for level in self.levels], dtype=np.int32),
            "bbox_min": self.bbox_min,
            "bbox_max": self.bbox_max,
            "truncation_voxels": np.array(self.truncation_voxels, dtype=np.float32),
            "provenance_json": np.array(json.dumps(self.provenance, sort_keys=True)),
        }
        for level in self.levels:
            key = str(level.resolution)
            arrays["values_" + key] = pack_tudf_palette4(level.values.reshape(-1, 8))
            arrays["split_" + key] = np.packbits(level.split, bitorder="little")
        save_npz(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> UnifiedTUDF:
        with np.load(path, allow_pickle=False) as arrays:
            if str(arrays["storage_format"].item()) != FORMAT:
                raise ValueError("unrecognized unified octree format")
            levels = []
            for resolution in arrays["resolutions"].tolist():
                key = str(resolution)
                values = unpack_tudf_palette4(arrays["values_" + key]).reshape(-1)
                bits = arrays["split_" + key]
                if bits.size != (values.size + 7) // 8:
                    raise ValueError("incorrect split bitset size")
                split = np.unpackbits(bits, bitorder="little", count=values.size).astype(bool)
                levels.append(TUDFLevel(resolution, values, split))
            tree = cls(
                tuple(levels),
                arrays["bbox_min"],
                arrays["bbox_max"],
                float(arrays["truncation_voxels"].item()),
                json.loads(str(arrays["provenance_json"].item())),
            )
        tree.validate()
        return tree


def _pool_any(mask: np.ndarray) -> np.ndarray:
    resolution = mask.shape[0]
    if mask.shape != (resolution,) * 3 or resolution % 2:
        raise ValueError("expected an even cubic mask")
    half = resolution // 2
    return mask.reshape(half, 2, half, 2, half, 2).any(axis=(1, 3, 5))


def _coarse_tudf(surface: np.ndarray, truncation: float) -> np.ndarray:
    if not surface.any():
        raise ValueError("source has no surface seeds")
    # Pool the surface mask before measuring distance at the coarser level.
    return np.minimum(distance_transform_edt(~surface), truncation).astype(np.float16)


def _rows(query: np.ndarray, reference: np.ndarray, resolution: int) -> np.ndarray:
    """Strict bijective payload reorder, with explicit duplicate/missing checks."""
    codes = coords_to_linear(reference, resolution)
    order = np.argsort(codes)
    sorted_codes = codes[order]
    if len(sorted_codes) > 1 and np.any(sorted_codes[1:] == sorted_codes[:-1]):
        raise ValueError("duplicate reference parents")
    qcodes = coords_to_linear(query, resolution)
    if len(qcodes) != len(codes):
        raise ValueError("refined parent count changed")
    positions = np.searchsorted(sorted_codes, qcodes)
    if np.any(positions >= len(codes)) or not np.array_equal(sorted_codes[positions], qcodes):
        raise ValueError("missing reference parents")
    if len(np.unique(qcodes)) != len(codes):
        raise ValueError("duplicate requested parents")
    return order[positions]


def bridge_legacy_r256(tree: AdaptiveTUDF, *, provenance: dict | None = None) -> UnifiedTUDF:
    """Add R64/R128 levels to a band-three R256/R512/R1024 envelope.

    Coarse levels use distance transforms of pooled surface seeds. The finer
    payloads are reordered into parent-major, child-minor order.
    """
    tree.validate()
    if tree.base_resolution != 256 or tree.truncation_voxels != 3.0:
        raise ValueError("legacy bridge requires an R256 band3 tree")
    surface128 = _pool_any(tree.base_tudf == 0)
    value128 = _coarse_tudf(surface128, tree.truncation_voxels)
    value64 = _coarse_tudf(_pool_any(surface128), tree.truncation_voxels)
    required256 = tree.base_tudf < tree.truncation_voxels
    required256[tuple(tree.refine_base_parent_zyx.T)] = True
    split128 = _pool_any(required256)
    split64 = _pool_any((value128 < tree.truncation_voxels) | split128)

    levels = [TUDFLevel(64, value64.reshape(-1), split64.reshape(-1))]
    root_coords = linear_to_coords(np.flatnonzero(split64), 64)
    coords128 = (root_coords[:, None] * 2 + CHILD_OFFSETS_ZYX).reshape(-1, 3)
    levels.append(TUDFLevel(128, value128[tuple(coords128.T)], split128[tuple(coords128.T)]))
    parents128 = coords128[levels[-1].split]
    coords256 = (parents128[:, None] * 2 + CHILD_OFFSETS_ZYX).reshape(-1, 3)
    ref256 = coords_to_linear(tree.refine_base_parent_zyx, 256)
    split256 = np.isin(coords_to_linear(coords256, 256), ref256)
    levels.append(TUDFLevel(256, tree.base_tudf[tuple(coords256.T)], split256))
    rows256 = _rows(coords256[split256], tree.refine_base_parent_zyx, 256)
    coords512 = (coords256[split256, None] * 2 + CHILD_OFFSETS_ZYX).reshape(-1, 3)
    split512 = np.isin(
        coords_to_linear(coords512, 512), coords_to_linear(tree.refine_middle_parent_zyx, 512)
    )
    levels.append(TUDFLevel(512, tree.middle_children_tudf[rows256].reshape(-1), split512))
    rows512 = _rows(coords512[split512], tree.refine_middle_parent_zyx, 512)
    fine_values = tree.fine_children_tudf[rows512].reshape(-1)
    levels.append(TUDFLevel(1024, fine_values, np.zeros(fine_values.size, dtype=bool)))
    result = UnifiedTUDF(
        tuple(levels),
        tree.bbox_min.copy(),
        tree.bbox_max.copy(),
        tree.truncation_voxels,
        {
            **(provenance or {}),
            "policy": "legacy_geometry_preserving_r64_bridge_v1",
            "coarse_target": "R256_zero_mask_revoxelization_then_EDT_band3",
            "fine_target": "unchanged_legacy_payload",
            "encoder_dilation_in_new_split_labels": False,
        },
    )
    result.validate()
    return result


def uniform_band_tree(tree: UnifiedTUDF, *, preserve_terminal_surface=False) -> UnifiedTUDF:
    """Keep a split when a child lies in the distance band or has finer descendants.

    The surface safeguard preserves a split whose removal would turn a coarse
    zero-valued voxel into a new surface point.
    """
    tree.validate()
    guard_counts = {}
    decisions = [np.zeros_like(level.split) for level in tree.levels]
    for depth in range(len(tree.levels) - 2, -1, -1):
        level, children = tree.levels[depth : depth + 2]
        required_children = (children.values < tree.truncation_voxels) | decisions[depth + 1]
        needed = required_children.reshape(-1, 8).any(axis=1)
        if preserve_terminal_surface:
            # A zero-valued parent cannot replace an empty child subtree without
            # introducing a new coarse surface. Preserve that existing path.
            protect = (level.values[level.split] <= 0.5) & ~needed
            guard_counts[str(level.resolution)] = int(protect.sum())
            needed = needed | protect
        decisions[depth][level.split] = needed

    retained_rows = np.arange(tree.levels[0].values.size, dtype=np.int64)
    levels = []
    for depth, original in enumerate(tree.levels):
        mask = decisions[depth][retained_rows]
        levels.append(TUDFLevel(original.resolution, original.values[retained_rows], mask))
        if depth + 1 < len(tree.levels):
            # Original children are in old-selected-parent / ordered-child
            # order. Maintain that exact order when taking a subset of parents.
            child_groups = np.cumsum(original.split, dtype=np.int64) - 1
            selected_groups = child_groups[retained_rows[mask]]
            retained_rows = (selected_groups[:, None] * 8 + np.arange(8)).reshape(-1)
    result = UnifiedTUDF(
        tuple(levels),
        tree.bbox_min.copy(),
        tree.bbox_max.copy(),
        tree.truncation_voxels,
        {
            **tree.provenance,
            "envelope_policy": tree.provenance.get("policy"),
            "policy": SURFACE_SAFE_POLICY
            if preserve_terminal_surface
            else "uniform_child_band_or_descendant_v1",
            "split_definition": (
                "any_child(band or descendants) or prevent_new_coarse_surface"
                if preserve_terminal_surface
                else "any_child(d_child < truncation_voxels or split_child)"
            ),
            **(
                {"surface_guard_counts": guard_counts, "surface_threshold_voxels": 0.5}
                if preserve_terminal_surface
                else {}
            ),
            "encoder_dilation_in_split_labels": False,
        },
    )
    result.validate()
    return result
