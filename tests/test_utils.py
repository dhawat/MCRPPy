import pytest
import numpy as np
from mcrppy.utils import sort_points_by_increasing_distance, reshape_output_repelled_point, jaccobi_measure, volume_unit_ball


def test_sort_points_by_inc_dist():
    points = np.array([[1, 2, 3], [0, 0, 0], [-1, 4, 5], [0, 1, 0], [1, 4, 5]])
    expected = np.array([[0, 0, 0], [0, 1, 0], [1, 2, 3], [-1, 4, 5], [1, 4, 5]])
    result = sort_points_by_increasing_distance(points)
    np.testing.assert_array_almost_equal(result, expected)


def test_reshape_output_repelled_point():
    x = [np.array([[ 1.18, -0.39,  0.871],
        [ 3.09, -0.2,  1]]),
         np.array([[ 2.18, 3,  0],
        [ 9, 2,  1]])]
    epsilon = [1, 2]
    expected = [np.array([[ 1.18, -0.39,  0.871],
        [ 2.18, 3,  0]]), np.array([[ 3.09, -0.2,  1], [ 9, 2,  1]])]
    result = reshape_output_repelled_point(x, epsilon)
    np.testing.assert_array_almost_equal(result, expected)

def test_jaccobi_measure():
    x = np.array([[1, 1/2, 0], [1/2, 0, 0], [0, 1.1, 0]])
    #x= np.array([ [1/2, 0, 0]])
    jac_params = np.array([[1, 1, 0], [2, 0, 1]]).T
    expected = np.array([0, 9/8, 0])
    result = jaccobi_measure(x, jac_params)
    np.testing.assert_array_almost_equal(result, expected)

def test_volume_unit_ball_window():
    d=2
    expected = np.pi
    result = volume_unit_ball(d)
    np.testing.assert_equal(expected, result)
