import numpy as np
from mcrppy.utils import volume_unit_ball

def force_base(x, points, intensity=None, correction=True):
    r"""
    .. math::
            F(x) = \sum_{z \in \mathcal{Z}, \|z\|_2 \uparrow} \limits \frac{x-z}{\|x-z\|_2^d} - \rho \kappa_d x

    Args:
        x (np.1darray): d dimensional points on which the force is evaluated (1 times d array).
        points (np.ndarray): d dimensional points exerting the force (N times d array).
        intensity (float): Expected number of points exerting the force per unit volume.
    """
    d = points.shape[1]
    x = np.atleast_2d(x)
    numerator = (x - points).astype("float")
    denominator = np.linalg.norm(numerator, axis=1) ** d
    np.divide(numerator, np.atleast_2d(denominator).T, out=numerator)
    kappa_d = volume_unit_ball(d)
    if correction:
        # force with correcting error due to infinite sum and arranging from zero
        #! todo error if intensity is None
        force_x = np.sum(numerator, axis=0) - intensity * kappa_d * x
    else:
        # force while arranging the sum from x
        force_x = np.atleast_2d(np.sum(numerator, axis=0))
    return force_x


def force_k(k, points, intensity, x=None, p=None, kd_tree=None):
    """ Coulombic force exerted by the points in the point_pattern (deprived of x) on x.
    If p is not None, it corresponds to the force exerted by the points in an ball centered at x of radius p.
    Args:
        x (_type_): _description_
        point_pattern (_type_): _description_
        correction (bool, optional): _description_. Defaults to True.
        p (_type_, optional): _description_. Defaults to None.
        kd_tree (_type_, optional): _description_. Defaults to None.
        k (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
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

    Args:
        x (_type_): _description_
        idx_x (_type_): index of x in the kd_tree
        kd_tree (_type_): kd_tree of points
        p (_type_): annulus large radius.
    """
    x = points[idx_x]
    idx_points_in_window = kd_tree.query_ball_point(x=x.ravel(), r=p)
    idx_points_in_window.remove(idx_x)
    #idx_points_in_annulus = [i if i<idx_x else i-1 for i in idx_points_in_window] #kd tree is built on all points (without removing k)
    return points[idx_points_in_window]
