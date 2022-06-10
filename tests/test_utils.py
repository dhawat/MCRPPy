import pytest
import numpy as np
from GPPY.utils import sort_points_by_increasing_distance


def test_sort_points_by_inc_dist():
    points = np.array([[1, 2, 3], [0, 0, 0], [-1, 4, 5], [0, 1, 0], [1, 4, 5]])
    expected = np.array([[0, 0, 0], [0, 1, 0], [1, 2, 3], [-1, 4, 5], [1, 4, 5]])
    result = sort_points_by_increasing_distance(points)
    np.testing.assert_array_almost_equal(result, expected)
