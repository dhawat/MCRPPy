import numpy as np
import pytest
from GPPY.numerical_integration import delyon_portier_integration, kernel, leave_one_out_kernel_estimator, bandwidth_delyon_portier, function_test_1_delyon_portier

@pytest.mark.parametrize(
    "x, expected",
    [(np.array([[0, 0]]), 12/(2*np.pi)),
     (np.array([[0, 0, 0]]), 15/(2*np.pi)),
     (np.array([[0, 0], [0.5, 0], [-1,2]]), np.array([12/(2*np.pi), 9/(4*np.pi), 0]))]
)
def test_kernel(x, expected):
    result = kernel(x)
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "idx_out, x, points, bandwidth, expected",
    [(1, np.array([[1, 0]]), np.array([[0, 0], [1, 0]]), 1, 0),
     (0, np.array([[0.5, 0]]), np.array([[0, 0], [1, 0]]), 1, 9/(4*np.pi)),
     (0, np.array([[0.5, 0]]), np.array([[0, 0], [1, 0], [0.5, 0.5]]), 1, 9/(4*np.pi))]
)
def test_leave_one_out_kernel_estimator(idx_out, x, points, bandwidth, expected):
    result = leave_one_out_kernel_estimator(idx_out, x, points, bandwidth)
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, expected",
    [(np.array([[0, 0], [0.5, 0], [0, 0.5]]), (8*np.pi)/(9*(11/2 - 5 /np.sqrt(2))))]
)
def test_delyon_portier_integration(points, expected):
    f = lambda x: 2*np.linalg.norm(x, axis=1)
    bandwidth=1
    result = delyon_portier_integration(f, points, bandwidth)
    np.testing.assert_array_almost_equal(result, expected)

def test_bandwidth_delyon_portier():
    d = 2
    nb_points = 2**7
    sigma = 1
    expected =(12/5)**(1/6)
    result = bandwidth_delyon_portier(d, sigma, nb_points)
    np.testing.assert_equal(result, expected)

@pytest.mark.parametrize(
    "x, expected",
    [(np.array([[1, 0.3323]]), 0),
     (np.array([[1/2, 1/2]]), 4)]
)
def test_function_test_1(x, expected):
    result = function_test_1_delyon_portier(x)
    np.testing.assert_almost_equal(result, expected)
