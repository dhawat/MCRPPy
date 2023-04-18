import numpy as np
from multiprocessing import Pool, freeze_support
from functools import partial
from scipy.spatial import KDTree
from rpppy.coulomb_force import force_k
from rpppy.point_pattern import PointPattern
from rpppy.utils import sort_output_push_point, _sort_point_pattern, volume_unit_ball
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

    def _repelled_point(self, k, epsilon, stop_time, p=None, **kwargs):
        #! kdtree not none if p not none
        points = self.point_pattern.points.astype(float)
        window = self.point_pattern.window
        x = points[k]
        d = points.shape[1]
        if not isinstance (epsilon, list):
            epsilon = [epsilon]
        epsilon_matrix = [[epsilon[i]]*d for i in range(len(epsilon))]
        for _ in range(0, stop_time):
            #using trucated force
            if p is not None:
                max_radius = subwindow_parameter_max(window, "BallWindow")
                #don't consider points outside B(0, p/2)
                if np.linalg.norm(x) + p < max_radius:
                    x = x + epsilon_matrix * force_k(k=k, points=points, p=p, **kwargs)
                else:
                    x = np.atleast_2d(x)
            #using force with correction
            else:
                x = x + epsilon_matrix * force_k(k=k, points=points, **kwargs)
        return x

    def repelled_point_process(self, epsilon=None, p=None, stop_time=1, core_number=1, correction=True):
        points = self.point_pattern.points
        if epsilon is None:
            epsilon=epsilon_critical(self.dimension, self.point_pattern.intensity)
        if p is not None:
            points_kd_tree = KDTree(points)
        else:
            points_kd_tree=None
        points_nb = points.shape[0]
        freeze_support()
        if core_number>1:
            with Pool(processes=core_number) as pool:
                new_points = pool.map(
                partial(self._repelled_point, epsilon=epsilon, stop_time=stop_time, correction=correction, p=p, kd_tree=points_kd_tree),
                list(range(points_nb)),
                )
                pool.close()
                pool.join()
        else:
            new_points = [self._repelled_point(k, epsilon=epsilon, stop_time=stop_time, correction=correction, p=p, kd_tree=points_kd_tree) for k in range(points_nb)]
        return sort_output_push_point(new_points, epsilon)

    def repelled_point_pattern(self, epsilon=None, stop_time=1, core_number=1, correction=True, p=None):
        points = self.repelled_point_process(epsilon=epsilon, stop_time=stop_time, core_number=core_number, correction=correction, p=p)
        window = self.point_pattern.window
        repelled_point_pattern = [PointPattern(p, window) for p in points]
        return repelled_point_pattern[0] if len(repelled_point_pattern) == 1 else repelled_point_pattern

def epsilon_critical(d, intensity):
    return 1/(2*d*volume_unit_ball(d)*intensity)
