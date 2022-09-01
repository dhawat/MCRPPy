from curses import window
from structure_factor.point_pattern import PointPattern
import numpy as np
from numba import jit
from multiprocessing import Pool, freeze_support
from functools import partial
from GPPY.gravitational_force import force_k, force, force_truncated_k
from GPPY.utils import sort_points_by_increasing_distance, sort_output_push_point, _sort_point_pattern, volume_unit_ball


class GravityPointProcess:
    def __init__(self, point_pattern):
        r"""Initialize StructureFactor from ``point_pattern``.

        Args:
            point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type point pattern which contains a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.window`` and (optionally) the intensity of the point process ``point_pattern.intensity``.
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

    #todo the force is bigger in dimension 2. it 's divergent for infinite number of points so chose a smaller epsilon for d=2 than for higher dimension
    @property
    def epsilon(self):
        return (self.allocation_basin_volume**(1/self.dimension))/100

    @property
    def epsilon_critical(self):
        d = self.dimension
        intensity = self.point_pattern.intensity
        return 1/(2*d*volume_unit_ball(d)*intensity)

    def _pushed_point(self, k, epsilon, stop_time, force_truncated=False, p=0, q=0, add_correction=False):
        points = self.point_pattern.points
        intensity = self.point_pattern.intensity
        x = points[k]
        d = points.shape[1]
        if not isinstance (epsilon, list):
            epsilon = [epsilon]
        epsilon_matrix = [[epsilon[i]]*d for i in range(len(epsilon))]
        for _ in range(0, stop_time):
            if force_truncated:
                if add_correction:
                    c=intensity #add to the force kappa*rho*x
                else :
                    c=0
                x = x - epsilon_matrix * force_truncated_k(p=p, q=q, k=k, x=x, points=points, intensity=c)
            else:
                x = x - epsilon_matrix * force_k(k=k, x=x, points=points, intensity=intensity)
        return x

    def pushed_point_process(self, epsilon, stop_time, force_truncated=False, core_number=7, p=0, q=0, add_correction=False):
        freeze_support()
        points_nb = self.point_pattern.points.shape[0]
        with Pool(core_number) as pool:
            new_points = pool.map(
                partial(self._pushed_point, epsilon=epsilon, stop_time=stop_time, force_truncated=force_truncated, p=p, q=q, add_correction=add_correction),
                list(range(points_nb)),
            )
        return sort_output_push_point(new_points, epsilon)

    def pushed_point_pattern(self, epsilon, stop_time, force_truncated=False, core_number=7, p=0, q=0, add_correction=False):
        points = self.pushed_point_process(epsilon, stop_time, force_truncated, core_number, p, q, add_correction)
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
