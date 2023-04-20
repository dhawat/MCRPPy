from inspect import Parameter
import numpy as np
import pytest

from mcrppy.spatial_windows import AnnulusWindow
from structure_factor.utils import get_random_number_generator

##### BallWindow

#Unit annulus: annulus of large radius 2 and small radius 1
VOLUME_UNIT_ANNULUS = {
    1: 2,
    2: 3*np.pi,
    3: 28 * np.pi / 3,
    4: np.pi ** 2 / 2 * 15,
    5: 8 * np.pi ** 2 / 15 * 31
}


@pytest.mark.parametrize(
    "dimension, seed",
    (
        (1, None),
        (2, None),
        (3, None),
        (4, None),
        (5, None)
    ),
)
def test_volume_unit_annulus(dimension, seed):
    rng = get_random_number_generator(seed)
    center = rng.normal(size=dimension)
    ball = AnnulusWindow(center, small_radius=1, large_radius=2)
    np.testing.assert_almost_equal(ball.volume, VOLUME_UNIT_ANNULUS[dimension])


@pytest.mark.parametrize(
    "dimension, seed",
    (
        [2, None],
        [3, None],
        [4, None],
        [10, None],
        [100, None],
    ),
)
def test_center_not_belongs_to_annulus(dimension, seed):
    rng = get_random_number_generator(seed)
    center = rng.normal(size=dimension)
    annulus = AnnulusWindow(center)
    assert center not in annulus


@pytest.mark.parametrize(
    "point, test",
    [
        [[0, 1], "in"],
        [[0, -1], "in"],
        [[1, 0], "in"],
        [[-2, 0],"in"],
        [[1.5, 1],"in"],
        [[1, -1.7], "in"],
        [[-2.5, 0],"out"],
        [[1.5, 1.5],"out"],
        [[0, 0], "out"],
        [[1.5, 7],"out"],
        [[0, 0.7], "out"],
    ],
)
def test_unit_annulus_contains_points(point, test):
    d = 2
    center = np.zeros(d)
    annulus = AnnulusWindow(center, small_radius=1, large_radius=2)
    if test == "in":
        assert point in annulus
    else:
        assert point not in annulus


# @pytest.mark.parametrize(
#     "center, nb_points, seed",
#     (
#         [[0.4], 1, 1],
#         [[0.4, 4], 1, None],
#         [[0.4, 4, 40], 1, 4],
#         [[0.4, 4, 40], 3, 4],
#         [[0.4], 100, None],
#         [[0.4, 4], 100, 2],
#         [[0.4, 4, 40], 100, None],
#     ),
# )
# def test_random_points_fall_inside_annulus(center, nb_points, seed):
#     rng = get_random_number_generator(seed)
#     annulus = AnnulusWindow(center, small_radius=2, large_radius=10)
#     random_points = annulus.rand(nb_points, seed=rng)
#     indicator_vector = annulus.indicator_function(random_points)
#     assert np.all(indicator_vector)
