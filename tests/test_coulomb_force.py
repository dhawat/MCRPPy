import numpy as np
import pytest
from rpppy.coulomb_force import force_base, force_k
from rpppy.utils import volume_unit_ball
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BallWindow

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
        expected = [np.atleast_2d(points[1] * n / d)]*2
    if test == 2:
        f_2 = -points[1] * n / (np.sqrt(d) ** d)
        f_1 =  f_2 + intensity * kappa * x
        expected = [np.atleast_2d(f_1), np.atleast_2d(f_2)]
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "k, x, points, intensity, expected",
    [
        (2, np.full((1, 2), 0), np.full((4, 2), 1), 1, np.full((1, 2), 3 / 2)),
        (1, np.full((1, 4), 0), np.full((4, 4), 1), 1, np.full((1, 4), 3 / 16)),
        (
            2,
            np.array([[1, 0]]),
            np.array([[0, 1], [2, 0], [-1, 4], [-5, 6]]),
            1,
            np.array([[5 / 12 + np.pi, 7 / 12]]),
        ),
    ],
)
def test_force_k(k, x, points, intensity, expected):
    "test on simple data"
    d = x.shape[1]
    window = BallWindow(center=[0]*d, radius=12)
    point_pattern = PointPattern(points, window, intensity)
    result = force_k(k, x, point_pattern)
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "p, q, x, points, correction, intensity, expected",
    [
        (10, 0.1, np.full((1, 2), 0), np.full((4, 2), 1), True, 1, np.full((1, 2), 2)),
        (10, 0.1, np.full((1, 2), 1), np.full((4, 2), 2), True, 1/np.pi, np.full((1, 2), 3)),
        (10, 0.1, np.full((1, 3), 1), np.full((3, 3), 2), True, 1, np.full((1, 3), 1/np.sqrt(3) + 4/3*np.pi)),
        (10, 4.5, np.array([[1, 0]]),
            np.array([[0, 1], [2, 0], [-1, 4], [-5, 6], [-3, 1]]),
            False, 5,
            np.array([[-1 / 12, 1 / 12]]),
        ),
    ],
)
def test_force_homogeneous(p, q, x, points, correction, intensity, expected):
    "test on simple data"
    d = x.shape[1]
    window = BallWindow(center=[0]*d, radius=12)
    point_pattern = PointPattern(points, window, intensity)
    result = force_homogeneous(x, point_pattern, p=p, q=q, correction=correction)
    np.testing.assert_array_almost_equal(result, expected)
