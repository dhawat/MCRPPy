import numpy as np
import pytest
from GPPY.numerical_integration import delyon_portier_integration, kernel, leave_one_out_kernel_estimator, bandwidth_0_delyon_portier, function_test_1_delyon_portier, variance_kernel, monte_carlo_integration
from structure_factor.spatial_windows import UnitBallWindow
from GPPY.spatial_windows import AnnulusWindow


def f_1(x):
    #indicator ball unit window
    d = x.shape[1]
    window = UnitBallWindow(center=[0]*d)
    return window.indicator_function(x)

@pytest.mark.parametrize(
    "points, expected",
    ([np.array([[0, 0], [1, 2], [0.5, 0.2], [1, 0], [2, -1]]), 3/5 ],
     [np.array([[0, 0, 0], [1, 2, 0], [0.5, 0.2, -0.1], [1, 0, 0], [3, 2, -1]]), 3/5 ]
    )
)
def test_monte_carlo_integration(points, expected):
    result = monte_carlo_integration(f=f_1,points= points)
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "x, choice, expected",
    [(np.array([[0, 0]]), "DelPor", 12/(2*np.pi)),
     (np.array([[0, 0, 0]]), "DelPor", 15/(2*np.pi)),
     (np.array([[0, 0], [0.5, 0], [-1,2]]), "DelPor", np.array([12/(2*np.pi), 9/(4*np.pi), 0])),
     (np.array([[0,0], [1, 5], [0.5,0]]), "Epanechnikov", np.array([2/np.pi, 0, 3/(2*np.pi)]))]
)
def test_kernel(x, choice, expected):
    result = kernel(x, choice)
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

def test_bandwidth_0_delyon_portier():
    points= np.array([[0, 1], [1, 1], [-1,-1]])
    expected =(2**9/5)**(1/6)
    result = bandwidth_0_delyon_portier(points)
    np.testing.assert_equal(result, expected)

@pytest.mark.parametrize(
    "x, expected",
    [(np.array([[1, 0.3323]]), 0),
     (np.array([[1/2, 1/2]]), 4)]
)
def test_function_test_1(x, expected):
    result = function_test_1_delyon_portier(x)
    np.testing.assert_almost_equal(result, expected)


def test_variance_kernel():
    points = np.array([[0, 0], [0.25, 0], [0.5, 0.5]])
    x = np.array([[0, 0.5]])
    idx_out=1
    bandwidth = 2
    result = variance_kernel(idx_out, x, points, bandwidth)
    expected = 0
    np.testing.assert_almost_equal(result, expected)
