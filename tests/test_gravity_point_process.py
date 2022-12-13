import pytest
from GPPY.gravity_point_process import GravityPointProcess
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BallWindow
import numpy as np


@pytest.mark.parametrize(
    "k, epsilon, expected",
    [
        (0, 1, np.array([[0, 1]])),
        (1, 1, np.array([[0, -1]])),
        (2, 1, np.array([[0.7, 0.9]])),
        (2, [1, 1], np.array([[0.7, 0.9], [0.7, 0.9]]))
    ],
)
def test_pushed_point(k, epsilon, expected):
    "test pushed point with point_pattern of 3 simple points and one step in time"
    points = np.array([[0, 0], [0, -1], [1, 1]])
    window = BallWindow(center=[0, 0], radius=1.1)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp._pushed_point(k=k, epsilon=epsilon, stop_time=1)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "points, correction, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, np.array([[0, 1], [0, -1], [0.7, 0.9]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, np.array([[0, 1], [0, -1], [0.7, 0.9]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False, np.array([[0, 1], [0, -2], [1.7, 1.9]]))
    ],
)
def test_pushed_point_process(points, correction, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    window = BallWindow(center=[0, 0], radius=1.1)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.pushed_point_process(epsilon=1, stop_time=1, correction=correction)[0]
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, correction, p, q, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, 5, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 1.1, np.array([[-1/2, -1/2], [-1/5, -7/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False,5, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5], [1.7, 1.9]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), False, 3, 1.1, np.array([[-1/2, -1/2], [-1/5, -7/5], [1.7, 1.9]]))
    ],
)
def test_pushed_point_process_with_p_q(points, correction, p, q, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    window = BallWindow(center=[0, 0], radius=10)
    point_pattern = PointPattern(points, window, intensity=1 )
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.pushed_point_process(epsilon=1, stop_time=1, correction=correction, p=p, q=q)[0]
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, correction, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, np.array([[0, 1], [0, -1/2], [17/13, 6/13]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, np.array([[0, 1], [0, -1/2], [17/13, 6/13]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False, np.array([[0, 1], [0, -3/2], [2, 1]]))
    ],
)
def test_equilibrium_point_process(points, correction, expected):
    "test with point_pattern of 3 simple points and one step in time"
    window = BallWindow(center=[0, 0], radius=1.1)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.equilibrium_point_process(epsilon=1, stop_time=1, correction=correction)
    np.testing.assert_array_almost_equal(result, expected, decimal=7)
