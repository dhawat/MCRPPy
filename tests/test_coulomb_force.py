import numpy as np
import pytest
from rpppy.coulomb_force import force_base, force_k, _select_point_in_ball
from scipy.spatial import KDTree
from rpppy.utils import volume_unit_ball

@pytest.mark.parametrize(
    "n, d, intensity, x, test",
    [
        (4, 2, 1 / np.pi, np.zeros((1, 2)), 1),
        (2, 3, 1, np.full(shape=(1, 3), fill_value=2), 2),
    ],
)
def test_force_base_from_point_with_mass(n, d, intensity, x, test):
    """Test the force function of one point repeated at zero"""
    points = np.full(shape=(n, d), fill_value=1)
    kappa = volume_unit_ball(d)
    result = [force_base(x, points, intensity), force_base(x, points, intensity, correction=False)]
    if test == 1:
        expected = [np.atleast_2d(-points[1] * n / d)]*2
    if test == 2:
        f_2 = points[1] * n / (np.sqrt(d) ** d)
        f_1 =  f_2 - intensity * kappa * x
        expected = [np.atleast_2d(f_1), np.atleast_2d(f_2)]
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "k, points, intensity, expected",
    [
        (2, np.array([[1, 1], [-1, 1], [0, 0], [0, 1]]), 1/np.pi, np.array([[0, -2]])),
        (1, np.array([[1]*4, [0]*4, [1]*4, [1]*4]), 1, np.full((1, 4),-3 / 16)),
        (
            2,
            np.array([[0, 1], [2, 0], [1, 0], [-1, 4]]),
            1,
            np.array([[-2/5 - np.pi, -7 / 10]]),
        ),
    ],
)
def test_force_k(k, points, intensity, expected):
    "test on simple data"
    result = force_k(k, points, intensity)
    np.testing.assert_array_almost_equal(result, expected)

# test force_k with p not None
@pytest.mark.parametrize(
    "k, points, intensity, p, expected",
    [
        (2, np.array([[1, 1], [-1, 1], [0, 0], [0, 1]]), 1/np.pi, 10, np.array([[0, -2]])),
        (1, np.array([[1, 2, -1], [0]*3, [6, -3, 17]]), 7, np.sqrt(7), np.array([[-1/(6*np.sqrt(6)), -1/(3*np.sqrt(6)), 1/(6*np.sqrt(6))]])),
        (
            2,
            np.array([[0, 1], [2, 0], [1, 0], [-1, 4]]),
            1, 2,
            np.array([[-1/2, -1/2]]),
        ),
    ],
)
def test_force_k_p(k, points, intensity, p, expected):
    "test on simple data"
    kd_tree = KDTree(points)
    result = force_k(k, points, intensity, p=p, kd_tree=kd_tree)
    np.testing.assert_array_almost_equal(result, expected)

# test _select_point_in_ball
@pytest.mark.parametrize(
    "idx, p",
    [(1, 1),
     (2, 1),
     (5, 7),
     (0, 3),
     (4, 3)]
)

def test_select_point_in_ball(idx, p):
    points = np.array([[0,1], [1,0],
                       [1/2, 1/2], [2,1],
                       [1.01, 0], [-1/2, 1],
                       [0,0], [-15, 6], [2, 2] ])
    x = points[idx]
    kd_tree = KDTree(points)
    result = _select_point_in_ball(idx, points, kd_tree, p)
    expected = []
    for y in points.tolist():
        if np.linalg.norm(y-x)>0 and  np.linalg.norm(y-x)<=p:
            expected.append(y)
    np.testing.assert_array_equal(result, np.array(expected))
