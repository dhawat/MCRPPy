from mcrppy.gravitational_allocation import GravitationalAllocation
from mcrppy.point_pattern import PointPattern
from mcrppy.spatial_windows import BallWindow, BoxWindow
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

def test__trajectory():
    window = BoxWindow([[0,1], [0,1]])
    points = np.array([[0, 1], [0, -1], [-1, 1]])
    point_pattern = PointPattern(points, window, 15)
    point_pattern_ga = GravitationalAllocation(point_pattern)
    x = np.array([[0,0]])
    result = point_pattern_ga._trajectory(x, stoptime=1, stepsize=1/2)
    expected = np.array([[0, 0], [1/4, -1/4]])
    np.testing.assert_array_almost_equal(expected, result)

def test_trajectory():
    window = BoxWindow([[0,1], [0,1]])
    points = np.array([[0, 1], [0, -1], [-1, 1]])
    point_pattern = PointPattern(points, window, 15)
    point_pattern_ga = GravitationalAllocation(point_pattern)
    x = np.array([[0,0], [0,2]])
    result = point_pattern_ga.trajectory(x, stoptime=1, stepsize=1/2)
    expected = [np.array([[0, 0], [1/4, -1/4]]), np.array([[0, 2], [1/4, 35/12 - 3*math.pi]])]
    np.testing.assert_array_almost_equal(expected, result)
