import numpy as np
from structure_factor.spatial_windows import UnitBallWindow, BoxWindow
import scipy as sp
import statistics as stat

def monte_carlo_integration(points, f, weights=None):
    if weights is None:
        points_nb = points.shape[0]
        weights = 1/points_nb
    return np.sum(f(points)*weights)

def importance_sampling_integration(points, f, proposal):
    points_nb = points.shape[0]
    return np.sum(f(points)/proposal(points))/points_nb

def control_variate_integration(points, f, proposal, mean_proposal, c):
    points_nb = points.shape[0]
    return np.sum(f(points) + c*(proposal(points) - mean_proposal))/points_nb


def sobol_sequence(window, nb_points, discrepancy=False, **kwargs):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html
    #! add warning window should be centered box window
    #! add test
    d = window.dimension
    l = np.max(np.diff(window.bounds))
    sobol = sp.stats.qmc.Sobol(d=d, **kwargs)
    #m = int(np.log(N)/np.log(2))
    #points_unit_box = sobol.random_base2(m=m)
    points_unit_box = sobol.random(n=nb_points)
    points = (points_unit_box - 0.5)*l
    if discrepancy:
        return points, sp.stats.qmc.discrepancy(points_unit_box)
    else:
        return points

def delyon_portier_integration(f, points, bandwidth, correction=False):
    nb_points = points.shape[0]
    numerator = f(points)
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
    estimator = np.sum(kernel((x-points)/bandwidth))
    estimator /= (nb_points - 1)*bandwidth**d
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
    support = norm_x < 1
    #todo utiliser np.as_array avec type int au lieu
    support = np.array([int(s ==  True) for s in support])
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
    ker = kernel((x-points)/bandwidth)/bandwidth**d
    result = np.sum((ker - leave_one_out)**2)/((nb_points-1)*(nb_points-2))
    return result

def integrand_estimate(x, f, points, bandwidth):
    nb_points = points.shape[0]
    d = points.shape[1]
    numerator = f(points)
    denominator = np.array([leave_one_out_kernel_estimator(i, points[i], points, bandwidth) for i in range(nb_points)])
    kernel_factor = kernel((x-points)/bandwidth, choice="Epanechnikov")/(bandwidth**d)
    result = np.sum(numerator/denominator*kernel_factor)/nb_points
    result/=nb_points
    return result
