"""Regression tests for the CPU geometry tools."""

import unittest

import numpy as np

from spacefold.geometry._grid import coords_to_points
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


if __name__ == "__main__":
    unittest.main()
