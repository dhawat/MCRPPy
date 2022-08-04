import numpy as np
from GPPY.utils import volume_unit_ball
from numba import jit
from structure_factor.spatial_windows import BallWindow

#@jit(nopython=True)
#todo try to compute the pairwise distance of all points once each time this matrix will contains all the needed distances for this step in time. In the code below we compute each distance twice. consider function which decide between paralellizing and vectorizing (matrix of force) depending on the complexity for the number of points and the steps in time
def force(x, points, intensity):
    r"""
    .. math::
            F(x) = \sum_{z \in \mathcal{Z}, \|z\|_2 \uparrow} \limits \frac{z-x}{\|z-x\|_2^d} - \rho \kappa_d x

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
    force_x = np.sum(numerator, axis=0) + (intensity) * kappa_d * x
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


def force_k(k, x, points, intensity):
    r""".. math::
            F_k(x) = \sum_{j, \|z_j\|_2 \uparrow}^{j \neq k} \limits \frac{z_j-x}{\|z_j-x\|_2^d} - \rho \kappa_d x

    _extended_summary_

    Args:
        k (_type_): index of point excluded from force
        x (_type_): _description_
        points (_type_): _description_
        intensity (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert k <= points.shape[0] - 1
    points = np.delete(points, k, axis=0)
    force_x = force(x, points, intensity)
    return force_x


def force_truncated_k(p, q, k, x, points, intensity):
    assert k <= points.shape[0] - 1
    points = np.delete(points, k, axis=0)
    window = BallWindow(center=x.ravel(), radius=p)
    subwindow = BallWindow(center=x.ravel(), radius=q)
    support = np.logical_and(window.indicator_function(points), np.logical_not(subwindow.indicator_function(points)))
    points_support = points[support]
    force_x = force(x, points_support, intensity)
    return force_x
