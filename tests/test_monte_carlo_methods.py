import numpy as np
import pytest
from mcrppy.spatial_windows import UnitBallWindow, BoxWindow
from mcrppy.point_pattern import PointPattern
import mcrppy.monte_carlo_base as ni
from mcrppy.monte_carlo_base import monte_carlo_method, importance_sampling_mc, control_variate_mc, estimate_control_variate_parameter, estimate_control_variate_proposal
import math


def f_1(x):
    #indicator ball unit window
    d = x.shape[1]
    window = UnitBallWindow(center=[0]*d)
    return window.indicator_function(x)

def f(x):
    #indicator ball unit window
    return f_1(x)*1

# test monte_carlo_method
@pytest.mark.parametrize(
    "points, expected",
    ([np.array([[0, 0], [1, 2], [0.5, 0.2], [1, 0], [2, -1]]), 3/5 ],
     [np.array([[0, 0, 0], [1, 2, 0], [0.5, 0.2, -0.1], [1, 0, 0], [3, 2, -1]]), 3/5 ]
    )
)
def test_monte_carlo_method(points, expected):
    result = monte_carlo_method(f=f_1,points= points)
    np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.parametrize(
    "points, expected",
    ([np.array([[0, 0], [1, 2], [-1, 0]]), (math.sqrt(5) + 3)/18 ],
    )
)
def test_importance_sampling_mc(points, expected):
    f = lambda x: np.linalg.norm(x, axis=1)
    proposal = lambda x: f(x)**2 +1
    result = importance_sampling_mc(points, f, proposal)
    np.testing.assert_array_almost_equal(result, expected)

# test control_variate_mc
def  test_control_variate_mc():
    points = np.array([[0, 0], [1, 2], [-1, 0]])
    f = lambda x: np.linalg.norm(x, axis=1)
    proposal = lambda x: f(x)**2 +1
    mean_proposal = 1
    c = 2
    result = control_variate_mc(points, f, proposal, mean_proposal, c)
    expected = (math.sqrt(5) - 11)/3
    np.testing.assert_array_almost_equal(result, expected)

# test estimate_control_variate_parameter
def  test_estimate_control_variate_parameter():
    points = np.array([[0, 0], [1, 2], [-1, 0]])
    f = lambda x: np.linalg.norm(x, axis=1)
    proposal = lambda x: f(x)**2 +1
    result = estimate_control_variate_parameter(points, f, proposal)
    expected = (3*math.sqrt(5) - 1)/14
    np.testing.assert_array_almost_equal(result, expected)

# @pytest.mark.parametrize(
#     "polydegree, expected",
#     ([1, ],
#      [2, ]
#     )
# )
# # test estimate_control_variate_proposal
# def  test_estimate_control_variate_proposal(polydegree, expected):
#     points = np.array([[0, 0], [1, 2], [-1, 0]])
#     if polydegree==1:
#         f = lambda x: np.sum(x, axis=1) + 1
#     elif polydegree==2:
#         f = lambda x: np.sum(x, axis=1)**2 + 2*np.sum(x, axis=1) -1
#     proposal = estimate_control_variate_proposal(points, f, polydegree)
#     result = proposal(points)
#     np.testing.assert_array_almost_equal(result, expected)

# test kernel
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

# test leave_one_out_kernel_estimator
@pytest.mark.parametrize(
    "idx_out, x, points, bandwidth, expected",
    [(1, np.array([[1, 0]]), np.array([[0, 0], [1, 0]]), 1, 0),
     (0, np.array([[0.5, 0]]), np.array([[0, 0], [1, 0]]), 1, 9/(4*np.pi)),
     (0, np.array([[0.5, 0]]), np.array([[0, 0], [1, 0], [0.5, 0.5]]), 1, 9/(4*np.pi))]
)
def test_leave_one_out_kernel_estimator(idx_out, x, points, bandwidth, expected):
    result = ni.leave_one_out_kernel_estimator(idx_out, x, points, bandwidth)
    np.testing.assert_array_almost_equal(result, expected)

# test delyon_portier_mc
@pytest.mark.parametrize(
    "points, expected",
    [(np.array([[0, 0], [0.5, 0], [0, 0.5]]), (8*np.pi)/(9*(11/2 - 5 /np.sqrt(2))))]
)
def test_delyon_portier_mc(points, expected):
    window = BoxWindow([[-1,1]]*2)
    point_pattern = PointPattern(points, window)
    f = lambda x: 2*np.linalg.norm(x, axis=1)
    bandwidth=1
    result = ni.delyon_portier_mc(f, point_pattern, bandwidth)
    np.testing.assert_array_almost_equal(result, expected)

# test bandwidth_0_delyon_portier
def test_bandwidth_0_delyon_portier():
    points= np.array([[0, 1], [1, 1], [-1,-1]])
    expected =(2**9/5)**(1/6)
    result = ni.bandwidth_0_delyon_portier(points)
    np.testing.assert_equal(result, expected)

# test variance_kernel
def test_variance_kernel():
    points = np.array([[0, 0], [0.25, 0], [0.5, 0.5]])
    x = np.array([[0, 0.5]])
    idx_out=1
    bandwidth = 2
    result = ni.variance_kernel(idx_out, x, points, bandwidth)
    expected = 0
    np.testing.assert_almost_equal(result, expected)

# test integrand_estimatore_core
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

# test integrand_estimate
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
