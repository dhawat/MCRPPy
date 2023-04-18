import numpy as np
from rpppy.utils import volume_unit_ball
import copy
from rpppy.spatial_windows import AnnulusWindow

def force_base(x, points, intensity=None, correction=True):
    r"""
    .. math::
            F(x) = \sum_{z \in \mathcal{Z}, \|z\|_2 \uparrow} \limits \frac{x-z}{\|x-z\|_2^d} - \rho \kappa_d x

    Args:
        x (_type_): 1 times d array
        points (_type_): N times d array arrange by increasing Euclidean distance from the origine.
        intensity (_type_): _description_
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


def force_k(x, point_pattern, correction=True, p=None, kd_tree=None, q=0, k=None):
    """ Coulombic force exerted by the points in the point_pattern (deprived of x) on x.
    If p is not None, it corresponds to the force exerted by the points in an annulus centered at x with small radius q and large radius p.
    Args:
        x (_type_): _description_
        point_pattern (_type_): _description_
        correction (bool, optional): _description_. Defaults to True.
        p (_type_, optional): _description_. Defaults to None.
        kd_tree (_type_, optional): _description_. Defaults to None.
        q (int, optional): _description_. Defaults to 0.
        k (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    point_pattern = copy.deepcopy(point_pattern)
    points = point_pattern.points
    intensity = point_pattern.intensity
    assert k <= points.shape[0] - 1
    # removing the point x from point_pattern
    point_pattern.points = np.delete(points, k, axis=0)
    if p is not None:
            points = _select_point_in_annulus(x, idx_x=k, kd_tree=kd_tree, p=p, q=q)
    force_x = force_base(x, points, intensity=intensity, correction=correction)
    return force_x


def _select_point_in_annulus(x, idx_x, kd_tree, p, q=0):
    """Select the points in the annulus of small radius q and large radius p around x.

    Args:
        x (_type_): _description_
        idx_x (_type_): index of x in the kd_tree
        kd_tree (_type_): kd_tree of points
        p (_type_): annulus large radius.
        q (int, optional): annulus small radius. Defaults to 0.
    """
    if q==0:
        idx_points_in_annulus_ = kd_tree.query_ball_point(x=x.ravel(), r=p)
        idx_points_in_annulus_.remove(idx_x)
        idx_points_in_annulus = [i if i<idx_x else i-1 for i in idx_points_in_annulus_] #kd tree is built on all points (without removing k)
        points = points[idx_points_in_annulus]
    else:
        #! add warning that this method may have memory problems while using multiprocessing?
        window = AnnulusWindow(center=x.ravel(), large_radius=p, small_radius=q)
        idx_points_in_window = window.indicator_function(points)
        points = points[idx_points_in_window]
    return points
