import statistics as stat
import warnings
import scipy as sp

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from mcrppy.plot_functions import _plot_proposal
from mcrppy.utils import _find_sum_of_coef_of_cubic_term, volume_unit_ball
from mcrppy.spatial_windows import BoxWindow


def monte_carlo_method(points, f, weights=None):
    if weights is None:
        points_nb = points.shape[0]
        weights = 1/points_nb
    return np.sum(f(points)*weights)

def importance_sampling_mc(points, f, proposal, weights=None):
    h = lambda x: f(x)/proposal(x)
    return monte_carlo_method(points, h, weights)

def control_variate_mc(points, f, proposal, mean_proposal, c=None, weights=None):
    if c is None:
        c= estimate_control_variate_parameter(points, f, proposal)
    h = lambda x: f(x) - c*(proposal(x) - mean_proposal)
    return monte_carlo_method(points, h, weights)

def estimate_control_variate_parameter(points, f, proposal):
    r"""
    :math:`\frac{\sum_{\mathbf{x} \in \mathcal{B}^{\prime} } f(\mathbf{x}) (h(\mathbf{x}) - \bar{h})}{\sum_{\mathbf{x} \in \mathcal{B}^{\prime}} (h(\mathbf{x}) - \bar{h})^2}`
    """
    a  = proposal(points) - stat.mean(proposal(points))
    numerator = sum(f(points)*a)
    denominator = sum(a**2)
    return numerator/denominator
#! add test
def estimate_control_variate_proposal(points, f, poly_degree=2, plot=False):
    d = points.shape[1]
    y = f(points)
    # create a polynomial features object to create 'poly_degree' degree polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    # transform the input data to include the 'poly_degree' degree polynomial features
    points_poly = poly.fit_transform(points)
    # create a linear regression model
    model = LinearRegression()
    # fit the model to the data
    model.fit(points_poly, y)
    # the regressed model
    proposal = lambda x: model.predict(poly.fit_transform(x))
    print("coef", model.coef_)
    if plot:
        _plot_proposal(f, proposal, dim=points.shape[1])
    # mean of proposal for centered uniform law over the unit cube
    if poly_degree==2:
        d = points.shape[1]
        mean_proposal = model.intercept_ + _find_sum_of_coef_of_cubic_term(proposal, d)/12
        print("Mean proposal theoretical:", mean_proposal, "Estiamted:", np.mean(proposal(points)) )
    elif poly_degree==1:
        mean_proposal= model.intercept_
    else:
        mean_proposal=np.mean(proposal(points))
    return proposal, mean_proposal


#! The following is not used for the moment
def delyon_portier_mc(f, point_pattern, bandwidth=None, correction=False):
    points = point_pattern.points
    nb_points = points.shape[0]
    numerator = f(points)
    if bandwidth is None:
        #start_time = time.time()
        bandwidth=find_bandwidth(point_pattern, f, correction).x.item()
        #end_time=time.time()-start_time
        #print("Time bandwidth search=", int(end_time/60), "min", (end_time%60), "s")
    denominator = np.array([leave_one_out_kernel_estimator(i, points[i], points, bandwidth) for i in range(nb_points)])
    if correction:
        v = np.array([variance_kernel(i, points[i], points, bandwidth) for i in range(nb_points)])
        correction = 1 - v/denominator**2
        result = np.sum(numerator/denominator*correction)/nb_points
    else:
        result = np.sum(numerator/denominator)/nb_points
    return result

def leave_one_out_kernel_estimator(idx_out, x, points, bandwidth):
    nb_points = points.shape[0]
    d = points.shape[1]
    points = np.delete(points, idx_out, axis=0)
    estimator = np.sum(kernel((x-points)/bandwidth , choice="DelPor"))
    estimator /= (nb_points - 1)*bandwidth**d
    if estimator==0:
        warnings.warn(message="Leave-one-out estimator is 0. hint: increase bandwidth value.")
    return estimator

def bandwidth_0_delyon_portier(points):
    d = points.shape[1]
    nb_points = points.shape[0]
    sigma_2 = stat.mean([stat.stdev(points[:,i])**2 for i in range(d)])
    numerator = d*2**(d+5)*sp.special.gamma(d/2 + 3)
    denominator = (2*d +1)*nb_points
    return np.sqrt(sigma_2)*(numerator/denominator)**(1/(4+d))

def kernel(x, choice="DelPor"):
    d = x.shape[1]
    norm_x = np.linalg.norm(x, axis=1)
    support = (norm_x < 1)*1
    if choice=="DelPor":
        k = 1/(2*volume_unit_ball(d))*(d+1)*(d+2 -(d+3) * norm_x)* support
    elif choice=="Epanechnikov":
        k = 1/(2*volume_unit_ball(d))*(d+2)*(1 - norm_x**2)* support
    return k

def variance_kernel(idx_out, x, points, bandwidth):
    nb_points = points.shape[0]
    d = points.shape[1]
    leave_one_out = leave_one_out_kernel_estimator(idx_out, x, points, bandwidth)
    points = np.delete(points, idx_out, axis=0)
    ker = kernel((x-points)/bandwidth, choice="DelPor")/bandwidth**d
    result = np.sum((ker - leave_one_out)**2)/((nb_points-1)*(nb_points-2))
    return result

def integrand_estimate(x, f, point_pattern, bandwidth, h_0=None):
    #eq (4) DelPor2016
    points = point_pattern.points
    if h_0 is None:
        h_0 = bandwidth_0_delyon_portier(points)
    points, numerator, denominator=_integrand_estimate_core( f, point_pattern, bandwidth, h_0)
    nb_points, d = points.shape
    kernel_factor = kernel((x-points)/h_0, choice="Epanechnikov")/(h_0**d)
    estimate = np.sum(numerator/denominator*kernel_factor)/nb_points
    return estimate

def integral_integrand_estimate(f, point_pattern, bandwidth, h_0=None):
    points = point_pattern.points
    if h_0 is None:
        h_0 = bandwidth_0_delyon_portier(points)
    #eq after eq (5) DelPor2016
    points, numerator, denominator=_integrand_estimate_core( f, point_pattern, bandwidth, h_0)
    nb_points = points.shape[0]
    return np.sum(numerator/denominator)/nb_points

def _integrand_estimate_core( f, point_pattern, bandwidth, h_0):
    #! support window should be centered boxwindow
    if not isinstance(point_pattern.window, BoxWindow):
        raise TypeError(message="Actually, the observation window should be a centered BoxWindow.")
    points = point_pattern.points
    d = points.shape[1]
    l = np.diff(point_pattern.window.bounds, axis=1)[0].item()
    support = BoxWindow([[-l/2 + h_0, l/2-h_0]]*d) #eq (9) DelPor2016
    points = points[support.indicator_function(points)]
    nb_points = points.shape[0]
    numerator = f(points)
    bandwidth= bandwidth.item()
    denominator = np.array([leave_one_out_kernel_estimator(i, points[i], points, bandwidth) for i in range(nb_points)])
    return points, numerator, denominator

def find_bandwidth(point_pattern, f, correction, **kwargs):
    points = point_pattern.points
    x_0=bandwidth_0_delyon_portier(points)
    res = sp.optimize.minimize(_func_to_optimize, x_0, (point_pattern, f, correction), **kwargs)
    return res

def _func_to_optimize(bandwidth, point_pattern, f, correction):
    #function to optimize to find the bandwidth, Sec 5.2 DelPor2016
    points = point_pattern.points
    nb_points = points.shape[0]
    f_tilde = lambda x:np.array([integrand_estimate(x[i], f, point_pattern, bandwidth) for i in range(nb_points)])
    estimate_integral_f_tilde= delyon_portier_mc(f_tilde, point_pattern, bandwidth, correction)
    integral_f_tilde = integral_integrand_estimate(f, point_pattern, bandwidth)
    return abs(estimate_integral_f_tilde - integral_f_tilde)
