from mcrppy.gravitational_allocation import GravitationalAllocation
from mcrppy.point_pattern import PointPattern
from mcrppy.spatial_windows import BallWindow
import math
import numpy as np

import pytest

def test_estimated_intensity():
    nb_points = 12
    window = BallWindow([0,0], 1)
    window_volume = math.pi
    points = window.rand(n=nb_points)
    expected = nb_points/window_volume
    point_pattern = PointPattern(points, window, 14)
    point_pattern_ga = GravitationalAllocation(point_pattern)
    result = point_pattern_ga.estimated_intensity
    np.testing.assert_equal(result, expected)

def test_starting_points_2d():
    window = BallWindow([0,0], 1)
    points = np.array([[1, 0], [0,0]])
    expected = [np.array([[2,0], [0, 0],[2,0] ]), np.array([[1,0], [-1, 0],[1,0] ])]
    scale = 2/math.pi
    point_pattern = PointPattern(points, window, 5)
    point_pattern_ga = GravitationalAllocation(point_pattern)
    result = point_pattern_ga._starting_points_2d(nb_points=3, scale=scale)
    np.testing.assert_array_almost_equal(result, expected)
