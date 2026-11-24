"""Regression tests for the CPU geometry tools."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spacefold.geometry._grid import AdaptiveTUDF, coords_to_points
from spacefold.geometry.octree import TUDFLevel, UnifiedTUDF, bridge_legacy_r256
from spacefold.geometry.surface import surface_indices


class GeometryCoordinatesTests(unittest.TestCase):
    def test_half_open_bounds_include_the_last_representable_point(self):
        lower = np.full(3, -1, dtype=np.float32)
        upper = np.ones(3, dtype=np.float32)
        points = np.stack([lower, np.nextafter(upper, lower), upper, np.nextafter(lower, -np.inf)])

        ids, inside = surface_indices(points, lower, upper)

        np.testing.assert_array_equal(inside, [True, True, False, False])
        np.testing.assert_array_equal(ids, [0, 1024**3 - 1])

    def test_outside_points_are_filtered_before_grid_arithmetic(self):
        points = np.array([[0, 0, 0], [np.finfo(np.float32).max, 0, 0]], dtype=np.float32)
        lower = np.zeros(3, dtype=np.float32)
        upper = np.ones(3, dtype=np.float32)

        with np.errstate(over="raise", invalid="raise"):
            ids, inside = surface_indices(points, lower, upper)

        np.testing.assert_array_equal(inside, [True, False])
        np.testing.assert_array_equal(ids, [0])

    def test_float64_bounds_keep_the_voxel_origin_in_both_directions(self):
        lower = np.array([10_000_000.125, 2.125, -4.0])
        upper = lower + [1.0, 3.0, 5.0]
        xyz = np.array([[450, 18, 2], [71, 19, 44]])
        points = lower + (xyz + 0.5) * ((upper - lower) / 1024)

        ids, inside = surface_indices(points, lower, upper)
        centers = coords_to_points(xyz[:, ::-1], lower, upper, 1024)

        np.testing.assert_array_equal(inside, [True, True])
        expected = [2 * 1024**2 + 18 * 1024 + 450, 44 * 1024**2 + 19 * 1024 + 71]
        np.testing.assert_array_equal(ids, expected)
        np.testing.assert_array_equal(centers, points.astype(np.float32))


class GeometryArchiveTests(unittest.TestCase):
    def test_packed_distances_must_contain_bytes(self):
        root = TUDFLevel(2, np.ones(8, dtype=np.float16), np.zeros(8, dtype=bool))
        tree = UnifiedTUDF((root,), np.zeros(3), np.ones(3), 3, {})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.npz"
            damaged = Path(directory) / "damaged.npz"
            tree.save(source)
            with np.load(source, allow_pickle=False) as archive:
                fields = dict(archive)
            fields["values_2"] = fields["values_2"].astype(np.uint16) + 256
            np.savez(damaged, **fields)

            with self.assertRaisesRegex(ValueError, "uint8"):
                UnifiedTUDF.load(damaged)

    def test_invalid_base_distances_are_detected_before_pruning(self):
        distances = np.full((256, 256, 256), 3, dtype=np.float16)
        distances[0, 0, 0] = 0
        distances[-1, -1, -1] = np.nan
        parents = np.empty((0, 3), dtype=np.int32)
        children = np.empty((0, 8), dtype=np.float16)
        tree = AdaptiveTUDF(
            distances, np.zeros(3), np.ones(3), 3, parents, children, parents, children
        )

        with self.assertRaisesRegex(ValueError, "base TUDF"):
            bridge_legacy_r256(tree)

    def test_cli_rejects_a_dangling_output_before_loading_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "points.npz"
            output = Path(directory) / "result.npz"
            target = Path(directory) / "missing.npz"
            np.savez(source, points=np.empty((0, 3)), bbox_min=np.zeros(3), bbox_max=np.ones(3))
            output.symlink_to(target)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spacefold.geometry",
                    "build",
                    "--input",
                    source,
                    "--output",
                    output,
                ],
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("output already exists", result.stderr)
            self.assertTrue(output.is_symlink())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
