import numpy as np
from GPPY.spatial_windows import UnitBallWindow, BallWindow
from scipy.spatial import KDTree

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
