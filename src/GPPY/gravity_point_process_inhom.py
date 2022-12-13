from curses import window
from structure_factor.point_pattern import PointPattern
import numpy as np
from numba import jit
from multiprocessing import Pool, freeze_support
from functools import partial
from GPPY.gravitational_force import force_k, force, force_truncated_k
from GPPY.utils import sort_points_by_increasing_distance, sort_output_push_point, _sort_point_pattern, volume_unit_ball


class GravityPointProcessInhomogeneous:
    def __init__(self, point_pattern, intensity):
        assert isinstance(point_pattern, PointPattern)
        self.point_pattern = _sort_point_pattern(point_pattern)
        self.intensity = intensity

    @property
    def dimension(self):
        """Ambient dimension of the underlying point process."""
        return self.point_pattern.dimension

    @property
    def epsilon_critical(self):
        d = self.dimension
        return 1/(3*d*volume_unit_ball(d))

    def _pushed_point(self, k, epsilon, stop_time, betta):
        points = self.point_pattern.points
        intensity = self.intensity
        x = points[k]
        d = points.shape[1]
        if not isinstance (epsilon, list):
            epsilon = [epsilon]
        epsilon_matrix = [[epsilon[i]]*d for i in range(len(epsilon))]
        for _ in range(0, stop_time):
            x = x - epsilon_matrix * force_k(k=k, x=x, points=points, intensity=intensity, inhomogeneous=True, betta=betta)
        return x

    def pushed_point_process(self, epsilon, stop_time=1, betta=1):
        freeze_support()
        points = self.point_pattern.points
        points_nb = points.shape[0]

        new_points = [self._pushed_point(k, epsilon, stop_time, betta) for k in range(points_nb)]
        return sort_output_push_point(new_points, epsilon)

    def pushed_point_pattern(self, epsilon, stop_time=1, betta=1):
        points = self.pushed_point_process(epsilon, stop_time, betta)
        window = self.point_pattern.window
        point_pattern_new = [PointPattern(p, window) for p in points]
        return point_pattern_new

    def equilibrium_point_process(self, epsilon, stop_time):
        points = np.copy(self.point_pattern.points)
        points_nb = points.shape[0]
        intensity = self.point_pattern.intensity
        for _ in range(0, stop_time):
            for n in range(0, points_nb):
                f_k = force_k(k=n, x=points[n], points=points, intensity=intensity)
                points[n] = points[n] - epsilon * f_k
        return points

    def equilibrium_point_pattern(self, epsilon, stop_time):
        points = self.equilibrium_point_process(epsilon, stop_time)
        window = self.point_pattern.window
        point_pattern_new = PointPattern(points, window)
        return point_pattern_new




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
