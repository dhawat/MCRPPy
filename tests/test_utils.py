import pytest
import numpy as np
from mcrppy.utils import sort_points_by_increasing_distance, reshape_output_repelled_point, jaccobi_measure, volume_unit_ball, _find_sum_of_coef_of_cubic_term, error, mse, regression_line


def test_sort_points_by_inc_dist():
    points = np.array([[1, 2, 3], [0, 0, 0], [-1, 4, 5], [0, 1, 0], [1, 4, 5]])
    expected = np.array([[0, 0, 0], [0, 1, 0], [1, 2, 3], [-1, 4, 5], [1, 4, 5]])
    result = sort_points_by_increasing_distance(points)
    np.testing.assert_array_almost_equal(result, expected)

def test_volume_unit_ball_window():
    d=2
    expected = np.pi
    result = volume_unit_ball(d)
    np.testing.assert_equal(expected, result)

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

def test_find_sum_of_coefficient_of_cubic_term():
    poly = lambda x: 2*x[:,0]**2 - 4*x[:,1]**2 + 1.23*x[:,2]**2 + 3*x[:,0] -2
    d=3
    expected = np.array([2 -4 + 1.23])
    result = _find_sum_of_coef_of_cubic_term(poly,d)
    np.testing.assert_almost_equal(expected, result)

@pytest.mark.parametrize(
    "approx, exact, expected",
    ( [[1, 2, 3, 0], 0, [1, 2, 3, 0]],
      [ [1, 2, 3, 0], 1, [0, 1, 2, -1]],
       [1, None, "NAN"],)
)
def test_error(approx, exact, expected):
    result = error(approx,exact)
    np.testing.assert_equal(result, expected)

@pytest.mark.parametrize(
    "exact, expected",
    [
    (None, "NAN"),
    (0, np.array([2, 4, 10, 0])),
    (1, np.array([1, 1, 17, 1])),
    ]
)
def test_mse(exact, expected):
    mean = [1, 2, -3, 0]
    std = [1, 0, 1, 0]
    result = mse(mean, std, exact)
    np.testing.assert_equal(result, expected)

def test_regression_line():
    x = np.random.rand(100)*5
    y = -2.5*x + 1.5
    _, slope, _ = regression_line(x,y, log=False)
    expected = -2.5
    np.testing.assert_almost_equal(expected, slope)
