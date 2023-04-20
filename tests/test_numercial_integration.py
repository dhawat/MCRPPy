import numpy as np
import pytest
from structure_factor.spatial_windows import UnitBallWindow, BoxWindow
from structure_factor.point_pattern import PointPattern
import mcrppy.monte_carlo_methods as ni
import math


def f_1(x):
    #indicator ball unit window
    d = x.shape[1]
    window = UnitBallWindow(center=[0]*d)
    return window.indicator_function(x)

def f(x):
    #indicator ball unit window
    return f_1(x)*1

@pytest.mark.parametrize(
    "points, expected",
    ([np.array([[0, 0], [1, 2], [0.5, 0.2], [1, 0], [2, -1]]), 3/5 ],
     [np.array([[0, 0, 0], [1, 2, 0], [0.5, 0.2, -0.1], [1, 0, 0], [3, 2, -1]]), 3/5 ]
    )
)
def test_monte_carlo_integration(points, expected):
    result = ni.monte_carlo_method(f=f_1,points= points)
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "x, choice, expected",
    [(np.array([[0, 0]]), "DelPor", 12/(2*np.pi)),
     (np.array([[0, 0, 0]]), "DelPor", 15/(2*np.pi)),
     (np.array([[0, 0], [0.5, 0], [-1,2]]), "DelPor", np.array([12/(2*np.pi), 9/(4*np.pi), 0])),
     (np.array([[0,0], [1, 5], [0.5,0]]), "Epanechnikov", np.array([2/np.pi, 0, 3/(2*np.pi)]))]
)
def test_kernel(x, choice, expected):
    result = ni.kernel(x, choice)
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "idx_out, x, points, bandwidth, expected",
    [(1, np.array([[1, 0]]), np.array([[0, 0], [1, 0]]), 1, 0),
     (0, np.array([[0.5, 0]]), np.array([[0, 0], [1, 0]]), 1, 9/(4*np.pi)),
     (0, np.array([[0.5, 0]]), np.array([[0, 0], [1, 0], [0.5, 0.5]]), 1, 9/(4*np.pi))]
)
def test_leave_one_out_kernel_estimator(idx_out, x, points, bandwidth, expected):
    result = ni.leave_one_out_kernel_estimator(idx_out, x, points, bandwidth)
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, expected",
    [(np.array([[0, 0], [0.5, 0], [0, 0.5]]), (8*np.pi)/(9*(11/2 - 5 /np.sqrt(2))))]
)
def test_delyon_portier_integration(points, expected):
    window = BoxWindow([[-1,1]]*2)
    point_pattern = PointPattern(points, window)
    f = lambda x: 2*np.linalg.norm(x, axis=1)
    bandwidth=1
    result = ni.delyon_portier_integration(f, point_pattern, bandwidth)
    np.testing.assert_array_almost_equal(result, expected)

def test_bandwidth_0_delyon_portier():
    points= np.array([[0, 1], [1, 1], [-1,-1]])
    expected =(2**9/5)**(1/6)
    result = ni.bandwidth_0_delyon_portier(points)
    np.testing.assert_equal(result, expected)

@pytest.mark.parametrize(
    "x, expected",
    [(np.array([[1, 0.3323]]), 0),
     (np.array([[1/2, 1/2]]), 4)]
)
def test_function_test_1(x, expected):
    result = ni.function_test_1_delyon_portier(x)
    np.testing.assert_almost_equal(result, expected)


def test_variance_kernel():
    points = np.array([[0, 0], [0.25, 0], [0.5, 0.5]])
    x = np.array([[0, 0.5]])
    idx_out=1
    bandwidth = 2
    result = ni.variance_kernel(idx_out, x, points, bandwidth)
    expected = 0
    np.testing.assert_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, bandwidth, h_0, expected",
    [(np.array([[0, 0], [0, 1/4], [0, 1/2], [-1, 0]]),
      1/2,
      1/5,
      (np.array([[0, 0], [0, 1/4], [0, 1/2]]),
       np.array([1,1,1]),
       np.array([
           9/(2*math.pi), 9/math.pi, 9/(2*math.pi)]))
      )]
)
def test_integrand_estimate_core( points, bandwidth, h_0, expected):
    window = BoxWindow([[-1,1]]*2)
    point_pattern = PointPattern(points, window)
    result = ni._integrand_estimate_core(f, point_pattern, np.array([bandwidth]), h_0)
    for i in range(3):
        np.testing.assert_almost_equal(result[i], expected[i])


@pytest.mark.parametrize(
    "x, points, bandwidth, h_0, expected",
    [(np.array([[0, 1/4]]), np.array([[0, 0], [0, 1/4], [0, 1/2], [-1, 0]]), 1/2, 1/5, 5**2*2/(3**3))]
)
def test_integrand_estimate(x, points, bandwidth, h_0, expected):
    window = BoxWindow([[-1,1]]*2)
    point_pattern = PointPattern(points, window)
    result = ni.integrand_estimate(x, f, point_pattern, np.array([bandwidth]), h_0)
    np.testing.assert_almost_equal(result, expected)

@pytest.mark.parametrize(
    " points, bandwidth, h_0, expected",
    [(np.array([[0, 0], [0, 1/4], [0, 1/2], [-1, 0]]), 1/2, 1/5, 5*math.pi/(3**3))]
)
def test_integrand_estimate( points, bandwidth, h_0, expected):
    window = BoxWindow([[-1,1]]*2)
    point_pattern = PointPattern(points, window)
    result = ni.integral_integrand_estimate( f, point_pattern, np.array([bandwidth]), h_0)
    np.testing.assert_almost_equal(result, expected)
