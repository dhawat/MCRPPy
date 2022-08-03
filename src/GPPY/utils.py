import numpy as np
from structure_factor.spatial_windows import UnitBallWindow


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
