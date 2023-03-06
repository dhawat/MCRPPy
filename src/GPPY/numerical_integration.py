import numpy as np
from GPPY.spatial_windows import UnitBallWindow, BoxWindow, BallWindow
from GPPY.point_pattern import PointPattern
import scipy as sp
import statistics as stat
import warnings
from sklearn.linear_model import LinearRegression

def monte_carlo_integration(points, f, weights=None):
    if weights is None:
        points_nb = points.shape[0]
        weights = 1/points_nb
    #print("f_x", f(points), "w", weights, "pod", f(points)*weights)
    return np.sum(f(points)*weights)

def importance_sampling_integration(points, f, proposal):
    points_nb = points.shape[0]
    return np.sum(f(points)/proposal(points))/points_nb

def control_variate_integration(points, f, proposal, mean_proposal, c=None):
    points_nb = points.shape[0]
    if c is None:
        c= estimate_control_variate_parameter(points, f, proposal)
    return np.sum(f(points) - c*(proposal(points) - mean_proposal))/points_nb

#todo add test
def estimate_control_variate_parameter(points, f, proposal):
    mean_f = stat.mean(f(points))
    mean_proposal = stat.mean(proposal(points))
    a  = proposal(points) - mean_proposal
    numerator = sum((f(points)-mean_f)*a)
    denominator = sum(a**2)
    return numerator/denominator
#todo add test
def estimate_control_variate_proposal(points, f):
    y = f(points)
    reg = LinearRegression().fit(points,y)
    coef = reg.coef_
    intercept = reg.intercept_
    proposal = lambda x: np.sum(coef*x, axis=1) + intercept
    mean_proposal = intercept
    return proposal, mean_proposal

def sobol_sequence(window, nb_points, discrepancy=False, **kwargs):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
    #! add warning window should be centered box window
    #! add test
    sobol_pp = sobol_point_pattern(window, nb_points, discrepancy, **kwargs)
    return sobol_pp.points

def sobol_point_pattern(window, nb_points, discrepancy=False, **kwargs):
    d = window.dimension
    if isinstance(window, BoxWindow):
        l = np.max(np.diff(window.bounds))
    elif isinstance(window, BallWindow):
        l = 2*window.radius
        nb_points = int(nb_points/window.volume*(l**d))
    sobol = sp.stats.qmc.Sobol(d=d, **kwargs)
    #m = int(np.log(N)/np.log(2))
    #points_unit_box = sobol.random_base2(m=m)
    points_unit_box = sobol.random(n=nb_points)
    points = (points_unit_box - 0.5)*l
    sobol_pp = PointPattern(points, window).restrict_to_window(window)
    if discrepancy:
        return sobol_pp, sp.stats.qmc.discrepancy(points_unit_box)
    else:
        return sobol_pp


def delyon_portier_integration(f, point_pattern, bandwidth=None, correction=False):
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

def function_test_1_delyon_portier(x):
    d = x.shape[1]
    unit_box = BoxWindow([[0,1]]*d)
    support = np.array([int(unit_box.indicator_function(p) == True) for p in x])
    return np.prod(2*np.sin(np.pi*x)**2)*support

def kernel(x, choice="DelPor"):
    d = x.shape[1]
    unit_ball = UnitBallWindow(center=[0]*d)
    norm_x = np.linalg.norm(x, axis=1)
    support = (norm_x < 1)*1
    if choice=="DelPor":
        k = 1/(2*unit_ball.volume)*(d+1)*(d+2 -(d+3) * norm_x)* support
    elif choice=="Epanechnikov":
        k = 1/(2*unit_ball.volume)*(d+2)*(1 - norm_x**2)* support
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
    estimate_integral_f_tilde= delyon_portier_integration(f_tilde, point_pattern, bandwidth, correction)
    integral_f_tilde = integral_integrand_estimate(f, point_pattern, bandwidth)
    return abs(estimate_integral_f_tilde - integral_f_tilde)
