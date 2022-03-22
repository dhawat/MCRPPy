from structure_factor.point_pattern import PointPattern
import numpy as np
from structure_factor.spatial_windows import UnitBallWindow
from multiprocessing import Pool, freeze_support
from functools import partial


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

    def _pushed_point(self, point, epsilon, stop_time):
        points = self.point_pattern.points
        intensity = self.point_pattern.intensity
        for _ in range(0, stop_time):
            point = point - epsilon * _force(
                z=point, points=points, intensity=intensity
            )
        return point

    def pushed_point_process(self, epsilon, stop_time, core_number=7):
        freeze_support()
        list_points = self.point_pattern.points[
            :,
        ].tolist()
        with Pool(core_number) as pool:
            new_points = pool.map(
                partial(self._pushed_point, epsilon=epsilon, stop_time=stop_time),
                list_points,
            )

        return np.vstack(new_points)

    def equilibrium_point_process(self, epsilon, stop_time):
        points = self.point_pattern.points
        points_number = points.shape[0]
        intensity = self.point_pattern.intensity
        for _ in range(0, stop_time):
            for n in range(0, points_number):
                points[n] = points[n] - epsilon * _force(
                    z=points[n], points=points, intensity=intensity
                )
        return points


def _volume_unit_ball(d):
    center = np.full(shape=(d), fill_value=0)
    return UnitBallWindow(center=center).volume


def _force(z, points, intensity):
    r"""Gravitational force of a the point_pattern :math:`\mathcal{X}` applied to `z`  :math:`F(x) = \sum_{x \in \mathcal{X}, x \neq z \\ \|x\|_2 \uparrow} \limits \frac{x - z}{\|x-z\|_2^d} - \rho \kappa_d z`
    Args:
        z (np.ndarray): :math:`1\times d` array

    Returns:
        _type_: _description_
    """
    d = points.shape[1]
    # drop z from points
    # todo drop corresponding points without cheking inside array
    z = np.atleast_2d(z)
    index = points != z
    points = points[index[:, 1]]
    gravity = (points - z).astype("float")  # numerator
    denominator = np.linalg.norm(gravity, axis=1) ** d  # denominator
    np.divide(gravity, np.atleast_2d(denominator).T, out=gravity)
    kappa_d = _volume_unit_ball(d)  # volume of unit ball
    force_z = np.sum(gravity, axis=0) + (intensity) * kappa_d * z
    return force_z


def _sort_points_by_increasing_distance(points):
    norm_points = np.linalg.norm(points, axis=1)
    index = np.argsort(norm_points)
    points = points[index]
    return points


def _sort_point_pattern(point_pattern):
    point_pattern.points = _sort_points_by_increasing_distance(point_pattern.points)
    return point_pattern
