import pytest
from rpppy.repelled_point_process import RepelledPointProcess, epsilon_critical
from rpppy.point_pattern import PointPattern
from rpppy.spatial_windows import BallWindow
import numpy as np

# test _repelled_point without p
@pytest.mark.parametrize(
    "k, epsilon, expected",
    [
        (0, 1, np.array([[-0.5, 0.5]])),
        (1, 1, np.array([[-1/5, -7/5]])),
        (2, 1, np.array([[0.7, 0.9]])),
        (2, [1, 1], np.array([[0.7, 0.9], [0.7, 0.9]])),
        (0, 2.0, np.array([[-1, 1]])),
        (0, [2.0, 1], np.array([[-1, 1],[-0.5, 0.5] ]))
    ],
)
def test_repelled_point_without_p(k, epsilon, expected):
    "test pushed point with point_pattern of 3 simple points and one step in time with corrected force"
    points = np.array([[0, 0], [0, -1], [1, 1]])
    window = BallWindow(center=[0, 0], radius=2)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = RepelledPointProcess(point_pattern)
    result = gravity_pp._repelled_point(k=k, epsilon=epsilon, stop_time=1)
    np.testing.assert_array_almost_equal(result, expected)

# test repelled_point_process without p
@pytest.mark.parametrize(
    "points, correction, core_number, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 2, np.array([[-0.5, 0.5], [-1/5, -7/5], [0.7, 0.9]])),
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 1, np.array([[-0.5, 0.5], [-1/5, -7/5], [0.7, 0.9]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, 2, np.array([[-0.5, 0.5], [-1/5, -7/5], [0.7, 0.9]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False, 1, np.array([ [-0.5, 0.5], [-1/5, -12/5], [1.7, 1.9]])),
        (np.array([[0,0,0], [1, 0, -0.5]]), True, 3,  np.array([[-4*np.sqrt(4)/(5*np.sqrt(5)), 0, 2*np.sqrt(4)/(5*np.sqrt(5))], [-1/3 + 4*np.sqrt(4)/(5*np.sqrt(5)), 0, 1/6 - 2*np.sqrt(4)/(5*np.sqrt(5))]])),
        (np.array([[0,0,0], [1, 0, -0.5]]), False, 1, np.array([[-4*np.sqrt(4)/(5*np.sqrt(5)), 0, 2*np.sqrt(4)/(5*np.sqrt(5))], [1 +4*np.sqrt(4)/(5*np.sqrt(5)), 0, -1/2- 2*np.sqrt(4)/(5*np.sqrt(5))]]))
    ],
)
def test_repelled_point_process(points, correction, core_number, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    d= points.shape[1]
    window = BallWindow(center=[0]*d, radius=5)
    point_pattern = PointPattern(points, window, intensity=1 / np.pi)
    gravity_pp = RepelledPointProcess(point_pattern)
    result = gravity_pp.repelled_point_process(epsilon=1, stop_time=1, correction=correction, core_number=core_number)
    np.testing.assert_array_almost_equal(result, expected)

# test repelled_point_process with p
@pytest.mark.parametrize(
    "points, correction, p, core_number, expected",
    [
        (np.array([[0, 0], [0, -1], [1, 1]]), True, 3, 1, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), True, 5, 3, np.array([[-0.5, 0.5], [-1/5, -12/5 + np.pi], [1.7 - np.pi, 1.9 - np.pi]])),
        (np.array([[0, -1], [0, 0], [1, 1]]), False, 5, 2, np.array([[-0.5, 0.5], [-1/5, -12/5], [1.7, 1.9]])),
        (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 1]]), True, 4, 2, np.array([[0, -1/(2*np.sqrt(2)), 1-1/(2*np.sqrt(2))], [0, -1/(5*np.sqrt(5)), -2 + -2/(5*np.sqrt(5)) + 4/3*np.pi], [0, 1 + 1/(2*np.sqrt(2)) + 1/(5*np.sqrt(5)) - 4/3*np.pi, 1 + 1/(2*np.sqrt(2)) + 2/(5*np.sqrt(5)) - 4/3*np.pi]])),
        (np.array([[0, 0, 0], [0, 0, -1], [0, 1, 1]]), False, 4, 1, np.array([[0, -1/(2*np.sqrt(2)), 1-1/(2*np.sqrt(2))], [0, -1/(5*np.sqrt(5)), -2 + -2/(5*np.sqrt(5))], [0, 1 + 1/(2*np.sqrt(2)) + 1/(5*np.sqrt(5)), 1 + 1/(2*np.sqrt(2)) + 2/(5*np.sqrt(5))]])),
    ],
)
def test_repelled_point_process_with_p(points, correction, p, core_number, expected):
    "test with point_pattern of 3 simple points and one step in time, with one epsilon"
    d = points.shape[1]
    window = BallWindow(center=[0]*d, radius=10)
    point_pattern = PointPattern(points, window, intensity=1 )
    gravity_pp = RepelledPointProcess(point_pattern)
    result = gravity_pp.repelled_point_process(epsilon=1, stop_time=1, correction=correction, p=p, core_number=core_number)
    np.testing.assert_array_almost_equal(result, expected)

def test_epsilon_critical():
    np.testing.assert_equal(1/4, epsilon_critical(2, 1/np.pi))
