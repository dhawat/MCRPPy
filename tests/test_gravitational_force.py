import numpy as np
import pytest
from GPPY.gravitational_force import force, force_k
from GPPY.utils import volume_unit_ball


@pytest.mark.parametrize(
    "n, d, intensity, z, test",
    [
        (4, 2, 1 / np.pi, np.zeros((1, 2)), 1),
        (2, 3, 1, np.full(shape=(1, 3), fill_value=2), 2),
    ],
)
def test_force_from_point_with_mass(n, d, intensity, z, test):
    """Test the force function on 0 of one point repeated"""
    points = np.full(shape=(n, d), fill_value=1)
    kappa = volume_unit_ball(d)
    result = force(z, points, intensity)
    if test == 1:
        expected = np.atleast_2d(points[1] * n / d)
    if test == 2:
        expected = np.atleast_2d(
            -points[1] * n / (np.sqrt(d) ** d) + intensity * kappa * z
        )
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
    result = force_k(k, x, points, intensity)
    np.testing.assert_array_almost_equal(result, expected)
