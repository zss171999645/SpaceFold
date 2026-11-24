"""Voxel coordinates, band-three palette packing and the three-level TUDF envelope."""

from dataclasses import dataclass

import numpy as np

CHILD_OFFSETS_ZYX = np.asarray(
    [(z, y, x) for z in range(2) for y in range(2) for x in range(2)],
    dtype=np.int32,
)


TUDF_PALETTE_FLOAT16 = np.asarray(
    [0.0, 1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0, np.sqrt(5.0), np.sqrt(6.0), np.sqrt(8.0), 3.0],
    dtype=np.float16,
)


def pack_tudf_palette4(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float16)
    if array.ndim != 2 or array.shape[1] != 8:
        raise ValueError(f"expected TUDF child values [N,8], got {array.shape}")
    indices = np.searchsorted(TUDF_PALETTE_FLOAT16, array)
    if np.any(indices >= len(TUDF_PALETTE_FLOAT16)):
        raise ValueError("TUDF values fall outside the compact palette")
    if not np.array_equal(TUDF_PALETTE_FLOAT16[indices], array):
        unknown = np.unique(array[TUDF_PALETTE_FLOAT16[indices] != array])
        raise ValueError(f"TUDF values are not exactly palette representable: {unknown[:8]}")
    codes = np.asarray(indices, dtype=np.uint8)
    return np.ascontiguousarray(codes[:, 0::2] | (codes[:, 1::2] << 4))


def unpack_tudf_palette4(
    packed: np.ndarray, *, palette: np.ndarray = TUDF_PALETTE_FLOAT16
) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"expected packed TUDF values [N,4], got {values.shape}")
    codes = np.empty((len(values), 8), dtype=np.uint8)
    codes[:, 0::2] = values & 0x0F
    codes[:, 1::2] = values >> 4
    palette = np.asarray(palette, dtype=np.float16)
    if codes.size and int(codes.max()) >= len(palette):
        raise ValueError("packed TUDF code falls outside its palette")
    return np.ascontiguousarray(palette[codes], dtype=np.float16)


def coords_to_linear(coords_zyx: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.asarray(coords_zyx, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"expected coordinates [N,3], got {coords.shape}")
    if coords.size and (coords.min() < 0 or coords.max() >= int(resolution)):
        raise ValueError("coordinates fall outside their resolution")
    return np.asarray(
        (coords[:, 0] * int(resolution) + coords[:, 1]) * int(resolution) + coords[:, 2],
        dtype=np.int64,
    )


def linear_to_coords(indices: np.ndarray, resolution: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size and (indices.min() < 0 or indices.max() >= int(resolution) ** 3):
        raise ValueError("linear indices fall outside their resolution")
    z, remainder = np.divmod(indices, int(resolution) ** 2)
    y, x = np.divmod(remainder, int(resolution))
    return np.ascontiguousarray(np.stack((z, y, x), axis=1), dtype=np.int32)


def child_coords(parents_zyx: np.ndarray) -> np.ndarray:
    parents = np.asarray(parents_zyx, dtype=np.int32)
    if parents.ndim != 2 or parents.shape[1] != 3:
        raise ValueError(f"expected parents [N,3], got {parents.shape}")
    children = parents[:, None, :] * 2 + CHILD_OFFSETS_ZYX[None, :, :]
    return np.ascontiguousarray(children.reshape(-1, 3), dtype=np.int32)


def coords_to_points(
    coords_zyx: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int,
) -> np.ndarray:
    coords = np.asarray(coords_zyx, dtype=np.float32)
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    voxel_size = (bbox_max - bbox_min) / float(resolution)
    return np.ascontiguousarray(
        bbox_min[None, :] + (coords[:, ::-1] + 0.5) * voxel_size[None, :],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class AdaptiveTUDF:
    base_tudf: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    truncation_voxels: float
    refine_base_parent_zyx: np.ndarray
    middle_children_tudf: np.ndarray
    refine_middle_parent_zyx: np.ndarray
    fine_children_tudf: np.ndarray

    @property
    def base_resolution(self) -> int:
        return int(self.base_tudf.shape[0])

    def validate(self) -> None:
        resolution = self.base_resolution
        if self.base_tudf.shape != (resolution, resolution, resolution):
            raise ValueError("base TUDF must be cubic")
        if self.base_tudf.dtype != np.float16:
            raise ValueError("base TUDF must be float16")
        if self.middle_children_tudf.shape != (
            len(self.refine_base_parent_zyx),
            8,
        ):
            raise ValueError("middle child payload does not match base parents")
        if self.fine_children_tudf.shape != (
            len(self.refine_middle_parent_zyx),
            8,
        ):
            raise ValueError("fine child payload does not match middle parents")
        middle_parent = coords_to_linear(self.refine_middle_parent_zyx // 2, resolution)
        base_parent = coords_to_linear(self.refine_base_parent_zyx, resolution)
        if len(middle_parent) and not np.all(np.isin(middle_parent, base_parent)):
            raise ValueError("fine nodes are missing refined base ancestors")

    @classmethod
    def load_compact(cls, path):
        """Read a three-level band-three envelope in the compact palette format."""
        with np.load(path, allow_pickle=False) as data:
            if str(data["storage_format"].item()) != "octree_bitmask_palette4_v2":
                raise ValueError("expected the compact three-level octree format")
            resolution = int(data["base_resolution"].item())
            if resolution != 256:
                raise ValueError("expected a base resolution of 256")
            if not np.array_equal(data["tudf_palette"], TUDF_PALETTE_FLOAT16):
                raise ValueError("the input uses a different TUDF palette")
            packed = data["refine_base_mask_bits"]
            if packed.shape != (resolution**3 // 8,) or packed.dtype != np.uint8:
                raise ValueError("invalid root refinement bitset")
            base = linear_to_coords(
                np.flatnonzero(np.unpackbits(packed, bitorder="little")), resolution
            )
            masks = data["refine_middle_child_mask"]
            if masks.shape != (len(base),) or masks.dtype != np.uint8:
                raise ValueError("refined parent count differs from the child mask count")
            parent, child = np.nonzero(
                masks[:, None] & np.left_shift(1, np.arange(8, dtype=np.uint8))
            )
            middle = np.ascontiguousarray(
                base[parent] * 2 + CHILD_OFFSETS_ZYX[child], dtype=np.int32
            )
            result = cls(
                data["base_tudf"],
                data["bbox_min"],
                data["bbox_max"],
                float(data["truncation_voxels"].item()),
                base,
                unpack_tudf_palette4(data["middle_children_codes4"]),
                middle,
                unpack_tudf_palette4(data["fine_children_codes4"]),
            )
        if result.base_resolution != resolution:
            raise ValueError("base resolution disagrees with the distance field")
        result.validate()
        return result
