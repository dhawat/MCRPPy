import pytest
from GPPY.gravity_point_process import GravityPointProcess
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BallWindow
import numpy as np


@pytest.mark.parametrize(
    "k, expected",
    [
        (0, np.array([[-0.5, 0.5]])),
        (1, np.array([[-0.2, -1.4]])),
        (2, np.array([[0.7, 0.9]])),
    ],
)
def test_pushed_point(k, expected):
    "test pushed point with point_pattern of 3 simple points and one step in time"
    points = np.array([[0, 0], [0, -1], [1, 1]])
    window = BallWindow(center=[0, 0], radius=1.1)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp._pushed_point(k=k, epsilon=1, stop_time=1)
    np.testing.assert_array_almost_equal(result, expected)


def test_pushed_point_process():
    "test with point_pattern of 3 simple points and one step in time"
    points = np.array([[0, 0], [0, -1], [1, 1]])
    window = BallWindow(center=[0, 0], radius=1.1)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    expected = np.array([[-0.5, 0.5], [-0.2, -1.4], [0.7, 0.9]])
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.pushed_point_process(epsilon=1, stop_time=1)
    np.testing.assert_array_almost_equal(result, expected)


def test_equilibrium_point_process():
    "test with point_pattern of 3 simple points and one step in time"
    points = np.array([[0, 0], [0, -1.0], [1, 1]])
    window = BallWindow(center=[0, 0], radius=1.1)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    expected = np.array(
        [
            [-5e-01, 5e-01],
            [-2.77555756e-17, -1.0],
            [8e-01, 6e-01],
        ]
    )
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.equilibrium_point_process(epsilon=1, stop_time=1)
    np.testing.assert_array_almost_equal(result, expected, decimal=7)
