import numpy as np
from mcrppy.utils import volume_unit_ball

def force_base(x, points, intensity=None, correction=True):
    r"""
    The Coulomb force exerted by the charged particles `points` on the particule `x` of the same charge as the particles `points`. Formally described by the following equation

    .. math::

            F(x) = \sum_{z \in \mathcal{Z}, \|z\|_2 \uparrow} \frac{x-z}{\|x-z\|_2^d} - \rho \kappa_d x,

    where :math:`\mathcal{Z}` is the configuration of points, :math:`\kappa_d` is the volume of the unit ball in :math:`\mathbb{R}^d`, and :math:`\rho` is the expected number of points of :math:`\mathcal{Z}` per unit volume (called intensity).
    if `correction` is set to False, the expression of :math:`F` does not include the term `\rho \kappa_d x`.

    Args:
        x (np.ndarray): d-dimensional point on which the force is evaluated (1 times d array).
        points (np.ndarray): d-dimensional points exerting the force on `x` (N times d array). Default to None.
        intensity (float, optional): Expected number of points per unit volume.
        correction (bool, optional): If True the above expression of :math:`F` is used else, the term `\rho \kappa_d x` is removed from the expression of :math:`F`. Default to True.

    Returns:
        (np.ndarray):
        Force on `x`(1 times d array).

    """
    d = points.shape[1]
    x = np.atleast_2d(x)
    numerator = (x - points).astype("float")
    denominator = np.linalg.norm(numerator, axis=1) ** d
    np.divide(numerator, np.atleast_2d(denominator).T, out=numerator)
    kappa_d = volume_unit_ball(d)
    if correction:
        #! todo error if intensity is None
        force_x = np.sum(numerator, axis=0) - intensity * kappa_d * x
    else:
        force_x = np.atleast_2d(np.sum(numerator, axis=0))
    return force_x


def force_k(k, points, intensity, x=None, p=None, kd_tree=None):
    r""" Coulombic force exerted by `points` on `x` if not None, else, on the k-th point of `points`.
    If `p` is not None, it corresponds to the force exerted by the points in a ball centered at `x` of radius `p`.
    Formally, if `p` is None the force expression is

    .. math::

            F(x) = \sum_{z \in \mathcal{Z} \setminus \{x\}, \|z\|_2 \uparrow} \frac{x-z}{\|x-z\|_2^d} - \rho \kappa_d x,

    else,

    .. math::

            F(x) = \sum_{z \in \mathcal{Z} \setminus \{x\} \cap B(x,p), \|z-x\|_2 \uparrow} \frac{x-z}{\|x-z\|_2^d},

    where :math:`\mathcal{Z}` is the configuration of points, :math:`\kappa_d` is the volume of the unit ball in :math:`\mathbb{R}^d`, and :math:`\rho` is the expected number of points of :math:`\mathcal{Z}` per unit volume (called intensity).

    Args:
        k (int): Index of the point of `points`on which the force is computed.
        points (np.ndarray): d-dimensional points exerting the force on `x` (N times d array).
        intensity (float, optional): Expected number of points per unit volume.
        x (np.ndarray, optional): d-dimensional point on which the force is evaluated (1 times d array). Defaults to None.
        p (float, optional): Radius of the ball centered at `x` containing the points exerting the force on `x` if not None, else, on the k-th point of `points`. Defaults to None.
        kd_tree (scipy.spatial.KDTree, optional): kd-tree of `points`. Defaults to None.

    Returns:
        (np.ndarray):
        Force on `x` if not None, else, on the k-th point of `points` (1 times d array).
    """
    if x is None:
        x = points[k]
    assert k <= points.shape[0] - 1
    if p is not None:
        points = _select_point_in_ball(idx_x=k, points=points, kd_tree=kd_tree, p=p)
        force_x = force_base(x, points, intensity=intensity, correction=False)
    else:
        # removing the point x from point_pattern
        points = np.delete(points, k, axis=0)
        force_x = force_base(x, points, intensity=intensity, correction=True)
    return force_x


def _select_point_in_ball(idx_x, points, kd_tree, p):
    """Select the points in the ball of radius p centered at x.
    """
    x = points[idx_x]
    idx_points_in_window = kd_tree.query_ball_point(x=x.ravel(), r=p)
    idx_points_in_window.remove(idx_x)
    return points[idx_points_in_window]
