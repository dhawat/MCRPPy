import numpy as np
from structure_factor.spatial_windows import UnitBallWindow


def sort_points_by_increasing_distance(points):
    norm_points = np.linalg.norm(points, axis=1)
    points = points[np.argsort(norm_points)]
    return points


def volume_unit_ball(d):
    center = np.full(shape=(d), fill_value=0)
    return UnitBallWindow(center=center).volume
