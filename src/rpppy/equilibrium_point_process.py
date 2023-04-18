from rpppy.point_pattern import PointPattern
import numpy as np
import copy
#from numba import jit
from multiprocessing import Pool, freeze_support, active_children
from functools import partial
from rpppy.coulomb_force import force_k
from rpppy.utils import sort_output_push_point, _sort_point_pattern, volume_unit_ball
from scipy.spatial import KDTree
from rpppy.spatial_windows import subwindow_parameter_max

class RepelledPointProcess:
    def __init__(self, point_pattern):
        r"""Initialize RepelledPointProcess from ``point_pattern``.

        Args:
            point_pattern (:py:class:`~rpppy.point_pattern.PointPattern`): Object of type point pattern which contains a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.window`` and (optionally) the intensity of the point process ``point_pattern.intensity``.
        """
        assert isinstance(point_pattern, PointPattern)
        self.point_pattern = _sort_point_pattern(point_pattern)

    @property
    def dimension(self):
        """Ambient dimension of the underlying point process."""
        return self.point_pattern.dimension

    @property
    def allocation_basin_volume(self):
        """Volume of each basin of attraction of the gravitational allocation"""
        return 1 / self.point_pattern.intensity

    @property
    def epsilon_critical(self):
        d = self.dimension
        intensity = self.point_pattern.intensity
        return 1/(2*d*volume_unit_ball(d)*intensity)

    def equilibrium_point_process(self, epsilon, stop_time, correction=True):
        point_pattern = copy.deepcopy(self.point_pattern)
        points = point_pattern.points.astype(float)
        points_nb = points.shape[0]
        for _ in range(0, stop_time):
            for k in range(0, points_nb):
                f_k = force_k(k=k, x=points[k], point_pattern=point_pattern, correction=correction)
                points[k] = points[k] - epsilon * f_k
                point_pattern.points = points
        return points

    def equilibrium_point_pattern(self, epsilon, stop_time):
        points = self.equilibrium_point_process(epsilon, stop_time)
        window = self.point_pattern.window
        point_pattern_new = PointPattern(points, window)
        return point_pattern_new



def epsilon_critical(d, intensity):
    return 1/(2*d*volume_unit_ball(d)*intensity)
# @jit
# def fast_push_point(points, intensity, epsilon, stop_time):
#     points_nb = points.shape[0]
#     push = []
#     for j in range(points_nb):
#         x = points[j]
#         for i in range(0, stop_time):
#             x = x - epsilon * force_k(k=j, x=x, points=points, intensity=intensity)
#         push.append(x)
#     return np.vstack(push)
