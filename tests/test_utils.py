import pytest
import numpy as np
from rpppy.utils import sort_points_by_increasing_distance, sort_output_push_point


def test_sort_points_by_inc_dist():
    points = np.array([[1, 2, 3], [0, 0, 0], [-1, 4, 5], [0, 1, 0], [1, 4, 5]])
    expected = np.array([[0, 0, 0], [0, 1, 0], [1, 2, 3], [-1, 4, 5], [1, 4, 5]])
    result = sort_points_by_increasing_distance(points)
    np.testing.assert_array_almost_equal(result, expected)


def test_sort_output_push_point():
    x = [np.array([[ 1.18, -0.39,  0.871],
        [ 3.09, -0.2,  1]]),
         np.array([[ 2.18, 3,  0],
        [ 9, 2,  1]])]
    epsilon = [1, 2]
    expected = [np.array([[ 1.18, -0.39,  0.871],
        [ 2.18, 3,  0]]), np.array([[ 3.09, -0.2,  1], [ 9, 2,  1]])]
    result = sort_output_push_point(x, epsilon)
    np.testing.assert_array_almost_equal(result, expected)
