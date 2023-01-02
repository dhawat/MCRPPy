import pytest
from GPPY.gravity_point_process import GravityPointProcess
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BallWindow
import numpy as np


@pytest.mark.parametrize(
    "k, epsilon, expected",
    [
        (0, 1, np.array([[-0.5, 0.5]])),
        (1, 1, np.array([[-1/5, -7/5]])),
        (2, 1, np.array([[0.7, 0.9]])),
        (2, [1, 1], np.array([[0.7, 0.9], [0.7, 0.9]]))
    ],
)
def test_pushed_point_without_p(k, epsilon, expected):
    "test pushed point with point_pattern of 3 simple points and one step in time with corrected force"
    points = np.array([[0, 0], [0, -1], [1, 1]])
    window = BallWindow(center=[0, 0], radius=2)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp._pushed_point(k=k, epsilon=epsilon, stop_time=1)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "points, correction, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, np.array([[-0.5, 0.5], [-1/5, -7/5], [0.7, 0.9]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, np.array([[-0.5, 0.5], [-1/5, -7/5], [0.7, 0.9]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False, np.array([ [-0.5, 0.5], [-1/5, -12/5], [1.7, 1.9]])),
        (np.array([[0,0,0], [1, 0, -0.5]]), True, np.array([[-4*np.sqrt(4)/(5*np.sqrt(5)), 0, 2*np.sqrt(4)/(5*np.sqrt(5))], [-1/3 + 4*np.sqrt(4)/(5*np.sqrt(5)), 0, 1/6 - 2*np.sqrt(4)/(5*np.sqrt(5))]])),
        (np.array([[0,0,0], [1, 0, -0.5]]), False, np.array([[-4*np.sqrt(4)/(5*np.sqrt(5)), 0, 2*np.sqrt(4)/(5*np.sqrt(5))], [1 +4*np.sqrt(4)/(5*np.sqrt(5)), 0, -1/2- 2*np.sqrt(4)/(5*np.sqrt(5))]]))
    ],
)
def test_pushed_point_process(points, correction, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    d= points.shape[1]
    window = BallWindow(center=[0]*d, radius=5)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.pushed_point_process(epsilon=1, stop_time=1, correction=correction)[0]
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, correction, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, np.array([[-0.5, 0.5], [-1/5, -7/5], [0.7, 0.9]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, np.array([[-0.5, 0.5], [-1/5, -7/5], [0.7, 0.9]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False, np.array([ [-0.5, 0.5], [-1/5, -12/5], [1.7, 1.9]])),
        (np.array([[0,0,0], [1, 0, -0.5]]), True, np.array([[-4*np.sqrt(4)/(5*np.sqrt(5)), 0, 2*np.sqrt(4)/(5*np.sqrt(5))], [-1/3 + 4*np.sqrt(4)/(5*np.sqrt(5)), 0, 1/6 - 2*np.sqrt(4)/(5*np.sqrt(5))]])),
        (np.array([[0,0,0], [1, 0, -0.5]]), False, np.array([[-4*np.sqrt(4)/(5*np.sqrt(5)), 0, 2*np.sqrt(4)/(5*np.sqrt(5))], [1 +4*np.sqrt(4)/(5*np.sqrt(5)), 0, -1/2- 2*np.sqrt(4)/(5*np.sqrt(5))]]))
    ],
)
def test_pushed_point_process_without_pool(points, correction, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    d= points.shape[1]
    window = BallWindow(center=[0]*d, radius=5)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.pushed_point_process(epsilon=1, correction=correction, multiprocess=False)[0]
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, correction, p, q, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, 5, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 1.1, np.array([[-1/2, -1/2], [-1/5, -7/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False,5, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5], [1.7, 1.9]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), False, 3, 1.1, np.array([[-1/2, -1/2], [-1/5, -7/5], [1.7, 1.9]])),
        (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 1]]), True, 4, 0.1, np.array([[0, -1/(2*np.sqrt(2)), 1-1/(2*np.sqrt(2))], [0, -1/(5*np.sqrt(5)), -2 + -2/(5*np.sqrt(5)) + 4/3*np.pi], [0, 1 + 1/(2*np.sqrt(2)) + 1/(5*np.sqrt(5)) - 4/3*np.pi, 1 + 1/(2*np.sqrt(2)) + 2/(5*np.sqrt(5)) - 4/3*np.pi]])),
        (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 1]]), False, 4, 0.1, np.array([[0, -1/(2*np.sqrt(2)), 1-1/(2*np.sqrt(2))], [0, -1/(5*np.sqrt(5)), -2 + -2/(5*np.sqrt(5))], [0, 1 + 1/(2*np.sqrt(2)) + 1/(5*np.sqrt(5)), 1 + 1/(2*np.sqrt(2)) + 2/(5*np.sqrt(5))]])),
    ],
)
def test_pushed_point_process_with_p_q(points, correction, p, q, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    d = points.shape[1]
    window = BallWindow(center=[0]*d, radius=10)
    point_pattern = PointPattern(points, window, intensity=1 )
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.pushed_point_process(epsilon=1, stop_time=1, correction=correction, p=p, q=q)[0]
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, correction, p, q, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, 5, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 1.1, np.array([[-1/2, -1/2], [-1/5, -7/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False,5, 0.1, np.array([[-0.5, 0.5], [-1/5, -12/5], [1.7, 1.9]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), False, 3, 1.1, np.array([[-1/2, -1/2], [-1/5, -7/5], [1.7, 1.9]])),
        (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 1]]), True, 4, 0.1, np.array([[0, -1/(2*np.sqrt(2)), 1-1/(2*np.sqrt(2))], [0, -1/(5*np.sqrt(5)), -2 + -2/(5*np.sqrt(5)) + 4/3*np.pi], [0, 1 + 1/(2*np.sqrt(2)) + 1/(5*np.sqrt(5)) - 4/3*np.pi, 1 + 1/(2*np.sqrt(2)) + 2/(5*np.sqrt(5)) - 4/3*np.pi]])),
        (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 1]]), False, 4, 0.1, np.array([[0, -1/(2*np.sqrt(2)), 1-1/(2*np.sqrt(2))], [0, -1/(5*np.sqrt(5)), -2 + -2/(5*np.sqrt(5))], [0, 1 + 1/(2*np.sqrt(2)) + 1/(5*np.sqrt(5)), 1 + 1/(2*np.sqrt(2)) + 2/(5*np.sqrt(5))]])),
    ],
)
def test_pushed_point_process_with_p_q_without_pool(points, correction, p, q, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    d = points.shape[1]
    window = BallWindow(center=[0]*d, radius=10)
    point_pattern = PointPattern(points, window, intensity=1 )
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.pushed_point_process(epsilon=1, stop_time=1, correction=correction, p=p, q=q, multiprocess=False)[0]
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, correction, p, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 2, np.array([[-0.5, 0.5], [0, -2 + np.pi], [3/2 - np.pi, 3/2 - np.pi]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), False, 2, np.array([[-0.5, 0.5], [0, -2], [3/2 , 3/2]])),
        (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 1]]), True, 2, np.array([[0, -1/(2*np.sqrt(2)), 1-1/(2*np.sqrt(2))], [0, 0, -2 + 4/3*np.pi], [0, 1 + 1/(2*np.sqrt(2)) - 4/3*np.pi, 1 + 1/(2*np.sqrt(2)) - 4/3*np.pi]])),
        (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 1]]), False, 2, np.array([[0, -1/(2*np.sqrt(2)), 1-1/(2*np.sqrt(2))], [0, 0, -2 ], [0, 1 + 1/(2*np.sqrt(2)), 1 + 1/(2*np.sqrt(2))]])),
    ],
)
def test_pushed_point_process_with_kd_tree(points, correction, p, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    d = points.shape[1]
    window = BallWindow(center=[0]*d, radius=10)
    point_pattern = PointPattern(points, window, intensity=1 )
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.pushed_point_process(epsilon=1, stop_time=1, correction=correction, p=p)[0]
    np.testing.assert_array_almost_equal(result, expected)
@pytest.mark.parametrize(
    "points, correction, expected",
    [
        (np.array([[0, 0], [0, -1]]), True, np.array([[0, 1], [0, -1/2]])),
        #(np.array([[0, -1], [0, 0], [1, 1]]), True, np.array([[-0.5, 0.5], [3/35, -29/35], [17/13, 6/13]])),
        #(np.array([[0, -1], [0, 0], [1, 1]]), False, np.array([[-0.5, 0.5], [3/35, -29/35 -1], [2, 1]]))
    ],
)
def test_equilibrium_point_process(points, correction, expected):
    "test with point_pattern of 3 simple points and one step in time"
    window = BallWindow(center=[0, 0], radius=2)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = GravityPointProcess(point_pattern)
    result = gravity_pp.equilibrium_point_process(epsilon=1, stop_time=1, correction=correction)
    np.testing.assert_array_almost_equal(result, expected, decimal=7)
