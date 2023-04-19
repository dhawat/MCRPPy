import numpy as np
from rpppy.spatial_windows import UnitBallWindow, BallWindow

def sort_points_by_increasing_distance(points):
    norm_points = np.linalg.norm(points, axis=1)
    points = points[np.argsort(norm_points)]
    return points

def volume_unit_ball(d):
    center = np.full(shape=(d), fill_value=0)
    return UnitBallWindow(center=center).volume

def sort_output_push_point(x, epsilon):
    x_list = []
    if not isinstance (epsilon, list):
            epsilon = [epsilon]
    for e in range(len(epsilon)):
        x_e = np.vstack([x[i][e] for i in range(len(x))])
        x_list.append(x_e)
    return x_list

def _sort_point_pattern(point_pattern):
    point_pattern.points = sort_points_by_increasing_distance(point_pattern.points)
    return point_pattern

def indicator_annulus_window(points, center, small_radius, large_radius):
    large_window = BallWindow(center, large_radius)
    if small_radius == 0:
        indicator = large_window.indicator_function(points)
    else:
        small_window = BallWindow(center, small_radius)
    return indicator


# utils for monte_carlo_methods
def _find_sum_of_coef_of_cubic_term(poly, d):
    """Function used to find the sum of the coefficient of the quadratic terms in a polynomial regression of degree 2. Used to find the mean of the proposal in ``estimate_control_variate_proposal``.

    _extended_summary_

    Args:
        poly (_type_): _description_
        d (_type_): _description_

    Returns:
        _type_: _description_
    """
    eval_points = []
    for i in range(0,d):
        x = np.zeros((1,d))
        y = np.zeros((1,d))
        y[:,i] = -1
        eval_points.append(y)
        x[:,i] = 1
        eval_points.append(x)

    return (sum(poly(p) for p in eval_points) - 2*d*poly(np.zeros((1,d))))/2
