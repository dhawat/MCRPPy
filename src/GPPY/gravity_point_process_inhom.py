from curses import window
from GPPY.point_pattern import PointPattern
import numpy as np
import copy
#from numba import jit
from multiprocessing import Pool, freeze_support
from functools import partial
from GPPY.gravitational_force import force_k
from GPPY.utils import sort_output_push_point, _sort_point_pattern, volume_unit_ball
from scipy.spatial import KDTree
from GPPY.spatial_windows import subwindow_parameter_max

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
    #@property
    # def epsilon(self):
    #     return (self.allocation_basin_volume**(1/self.dimension))/100

    @property
    def epsilon_critical(self):
        d = self.dimension
        intensity = self.point_pattern.intensity
        return 1/(2*d*volume_unit_ball(d)*intensity)

    def _pushed_point(self, k, epsilon, stop_time, p=None, **kwargs):
        # todo add warning p>q
        #! kdtree not none if p not none
        point_pattern = copy.deepcopy(self.point_pattern)
        points = point_pattern.points.astype(float)
        window = point_pattern.window
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
                # todo periodic boudary to treat well this case?
                if np.linalg.norm(x) + p < max_radius:
                    x = x - epsilon_matrix * force_k(k=k, x=x, point_pattern=point_pattern, p=p, **kwargs)
                else:
                    x = np.atleast_2d(x)
            #using force with correction
            else:
                x = x - epsilon_matrix * force_k(k=k, x=x, point_pattern=point_pattern,  **kwargs)
                #print(x.shape)
        return x

    def pushed_point_process(self, epsilon=None, p=None, stop_time=1, core_number=7, correction=True, multiprocess=True, q=0):
        if epsilon is None:
            epsilon=self.epsilon_critical
        freeze_support()
        if p is not None:
            points_kd_tree = KDTree(self.point_pattern.points)
        else:
            points_kd_tree=None
        points_nb = self.point_pattern.points.shape[0]
        # change to 7000 for nb_core=8
        if multiprocess and points_nb>1000:
            print(core_number)
            with Pool(core_number) as pool:
                new_points = pool.map(
                    partial(self._pushed_point, epsilon=epsilon, stop_time=stop_time, correction=correction, p=p, kd_tree=points_kd_tree, q=q),
                    list(range(points_nb)),
                )
            #pool.close()
        else:
            new_points = [self._pushed_point(k, epsilon=epsilon, stop_time=stop_time, correction=correction, p=p, kd_tree=points_kd_tree, q=q) for k in range(points_nb)]
        return sort_output_push_point(new_points, epsilon)

    def pushed_point_pattern(self, epsilon=None, stop_time=1, core_number=7, correction=True, p=None, multiprocess=True, q=0):
        points = self.pushed_point_process(epsilon=epsilon, stop_time=stop_time, core_number=core_number, correction=correction, p=p, multiprocess=multiprocess, q=q)
        window = self.point_pattern.window
        point_pattern_new = [PointPattern(p, window) for p in points]
        # todo code the following in more accurent way
        if len(point_pattern_new)==1:
            return point_pattern_new[0]
        else:
            return point_pattern_new

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
