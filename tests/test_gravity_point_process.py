import numpy as np
import pytest
from gravitypointprocess.gravity_point_process import _force, _volume_unit_ball


@pytest.mark.parametrize(
    "n, d, intensity, z, test",
    [
        (4, 2, 1 / np.pi, np.zeros((1, 2)), 1),
        (2, 3, 1, np.full(shape=(1, 3), fill_value=2), 2),
    ],
)
def test_foce_from_point_with_mass(n, d, intensity, z, test):
    """Test the force function of the same repeated point on 0"""
    points = np.full(shape=(n, d), fill_value=1)
    kappa = _volume_unit_ball(d)
    result = _force(z, points, intensity)
    if test == 1:
        expected = np.atleast_2d(points[1] * n / d)
    if test == 2:
        expected = np.atleast_2d(
            -points[1] * n / (np.sqrt(d) ** d) + intensity * kappa * z
        )
    np.testing.assert_array_almost_equal(result, expected)
