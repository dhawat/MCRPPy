import numpy as np
from GPPY.utils import volume_unit_ball
from numba import jit
import copy
import matplotlib as plt
from structure_factor.spatial_windows import BallWindow
from GPPY.spatial_windows import AnnulusWindow

#@jit(nopython=True)
#todo try to compute the pairwise distance of all points once each time this matrix will contains all the needed distances for this step in time. In the code below we compute each distance twice. consider function which decide between paralellizing and vectorizing (matrix of force) depending on the complexity for the number of points and the steps in time
def force_base(x, points, intensity=None, correction=True):
    r"""
    .. math::
            F(x) = \sum_{z \in \mathcal{Z}, \|z\|_2 \uparrow} \limits \frac{z-x}{\|z-x\|_2^d} + \rho \kappa_d x

    Args:
        x (_type_): 1 times d array
        points (_type_): N times d array arrange by increasing Euclidean distance from the origine.
        intensity (_type_): _description_
    """
    d = points.shape[1]
    x = np.atleast_2d(x)
    numerator = (points - x).astype("float")
    denominator = np.linalg.norm(numerator, axis=1) ** d
    np.divide(numerator, np.atleast_2d(denominator).T, out=numerator)
    kappa_d = volume_unit_ball(d)
    if correction:
        # force with correcting error due to infinite sum and arranging from zero
        #! todo error if intensity is None
        force_x = np.sum(numerator, axis=0) + (intensity) * kappa_d * x
    else:
        # force while arranging the sum from x
        force_x = np.atleast_2d(np.sum(numerator, axis=0))
    return force_x

def force_inhomogeneous(x, point_pattern, betta, correction=True):
    r"""
    .. math::
            F(x) = \frac{\batta}{\rho(x)}(\sum_{z \in \mathcal{Z}, \|z\|_2 \uparrow} \limits \frac{z-x}{ \|z-x\|_2^d} + \kappa_d x)

    Args:
        x (_type_): 1 times d array
        points (_type_): N times d array arrange by increasing Euclidean distance from the origine.
        intensity (_type_): _description_
    """
    points = point_pattern.points
    intensity = point_pattern.intensity
    d = points.shape[1]
    x = np.atleast_2d(x)
    numerator = (points - x).astype("float")
    denominator = intensity(points)*(np.linalg.norm(numerator, axis=1) ** d)
    np.divide(numerator, np.atleast_2d(denominator).T, out=numerator)
    kappa_d = volume_unit_ball(d)
    if correction:
        force_x = np.sum(numerator, axis=0) + kappa_d * x
    else:
        force_x = np.atleast_2d(np.sum(numerator, axis=0))
    return force_x*betta/intensity(x) #! remove intensity(x)

def force_truncated_k(p, q, k, x, points, intensity, correction=True, ax=None):
    assert k <= points.shape[0] - 1
    points = np.delete(points, k, axis=0)
    #todo creat annulus window class
    window = BallWindow(center=x.ravel(), radius=p)
    if q==0:
        support=window.indicator_function(points)
    else:
        subwindow = BallWindow(center=x.ravel(), radius=q)
        support = np.logical_and(window.indicator_function(points), np.logical_not(subwindow.indicator_function(points)))
    points_support = points[support]
    if ax is not None:
        ax.scatter(points_support[:,0], points_support[:,1], c="r", s=1, label="points used")
        ax.plot(x[0], x[1], 'b.', label="x")
        window.plot(axis=ax, color='g', label="big window")
        subwindow.plot(axis=ax, color="y", label="small window")
        ax.legend()
    #print("total", points.shape, "actual", points_support.shape)
    force_x = force_base(x, points_support, intensity, correction=correction)
    return force_x


def force_homogeneous(x, point_pattern, p=None, q=0, correction=True):
    points = point_pattern.points
    intensity = point_pattern.intensity
    if p is None:
        window = point_pattern.window
    elif q==0:
        #todo add test
        window = BallWindow(center=x.ravel(), radius=p)
    else:
        window = AnnulusWindow(center=x.ravel(), large_radius=p, small_radius=q)
    points_in_window = points[window.indicator_function(points)]
    #print("total", points.shape, "actual", points_support.shape)
    force_x = force_base(x, points_in_window, intensity=intensity, correction=correction)
    return force_x

def force_k(k, x, point_pattern, inhomogeneous=False, **kwargs):
    point_pattern_ = copy.deepcopy(point_pattern)
    points = point_pattern_.points
    assert k <= points.shape[0] - 1
    point_pattern_.points = np.delete(points, k, axis=0)
    if inhomogeneous:
        force_x = force_inhomogeneous(x, point_pattern_, **kwargs)
    else:
        force_x = force_homogeneous(x, point_pattern_, **kwargs)
    return force_x



# @jit(nopython=True)
# def force_fast(x, numerator, denominator, intensity, kappa_d):
#     numerator = np.divide(numerator, np.atleast_2d(denominator).T)
#     force_x = np.sum(numerator, axis=0) + (intensity) * kappa_d * x
#     return force_x

# def force_ffast(x, points, intensity):
#     d = points.shape[1]
#     x = np.atleast_2d(x)
#     numerator = points - x
#     denominator = np.linalg.norm(numerator, axis=1) ** d
#     d = points.shape[1]
#     kappa_d = volume_unit_ball(d)
#     return force_fast(x, numerator, denominator, intensity, kappa_d)
