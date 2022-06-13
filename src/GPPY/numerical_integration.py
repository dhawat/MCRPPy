import numpy as np
from structure_factor.spatial_windows import UnitBallWindow, BoxWindow
import scipy as sp

def monte_carlo_integration(f, points, weights):
    points_nb = points.shape[0]
    return np.sum(f(points)*weights)/points_nb

def delyon_portier_integration(f, points, bandwidth):
    nb_points = points.shape[0]
    numerator = f(points)
    denominator = np.array([leave_one_out_kernel_estimator(i, points[i], points, bandwidth) for i in range(nb_points)])
    result = np.sum(numerator/denominator)/nb_points
    return result

def kernel(x):
    d = x.shape[1]
    unit_ball = UnitBallWindow(center=[0]*d)
    norm_x = np.linalg.norm(x, axis=1)
    support = norm_x < 1
    support = np.array([int(s == True) for s in support])
    k = 1/(2*unit_ball.volume)*(d+1)*(d+2 -(d+3) * np.linalg.norm(x, axis=1))* support
    return k

def leave_one_out_kernel_estimator(idx_out, x, points, bandwidth):
    nb_points = points.shape[0]
    d = points.shape[1]
    points = np.delete(points, idx_out, axis=0)
    estimator = np.sum(kernel((x-points)/bandwidth))
    estimator /= (nb_points - 1)*bandwidth**d
    return estimator

def bandwidth_delyon_portier(d, sigma, nb_points):
    return sigma*((d*2**(d+5)*sp.special.gamma(d/2 + 3))/((2*d +1)*nb_points))**(1/(4+d))

def function_test_1_delyon_portier(x):
    d = x.shape[1]
    unit_box = BoxWindow([[0,1]]*d)
    support = np.array([int(unit_box.indicator_function(p) == True) for p in x])
    return np.prod(2*np.sin(np.pi*x)**2)*support
