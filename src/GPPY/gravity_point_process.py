from structure_factor.point_pattern import PointPattern
import numpy as np
from structure_factor.spatial_windows import UnitBallWindow
from multiprocessing import Pool, freeze_support
from functools import partial
from GPPY.gravitational_force import force_k
from GPPY.utils import sort_points_by_increasing_distance


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

    def _pushed_point(self, k, epsilon, stop_time):
        points = self.point_pattern.points
        intensity = self.point_pattern.intensity
        x = points[k]
        for _ in range(0, stop_time):
            x = x - epsilon * force_k(k=k, x=x, points=points, intensity=intensity)
        return x

    def pushed_point_process(self, epsilon, stop_time, core_number=7):
        freeze_support()
        points_nb = self.point_pattern.points.shape[0]
        with Pool(core_number) as pool:
            new_points = pool.map(
                partial(self._pushed_point, epsilon=epsilon, stop_time=stop_time),
                list(range(points_nb)),
            )

        return np.vstack(new_points)

    def equilibrium_point_process(self, epsilon, stop_time):
        points = self.point_pattern.points
        points_nb = points.shape[0]
        intensity = self.point_pattern.intensity
        for _ in range(0, stop_time):
            for n in range(0, points_nb):
                f_k = force_k(k=n, x=points[n], points=points, intensity=intensity)
                points[n] = points[n] - epsilon * f_k
        return points


def _sort_point_pattern(point_pattern):
    point_pattern.points = sort_points_by_increasing_distance(point_pattern.points)
    return point_pattern
