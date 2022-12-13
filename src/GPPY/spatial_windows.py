"""Collection of classes representing observation windows (box, ball, etc).

- :py:class:`~structure_factor.spatial_windows.BallWindow`: Ball window object.
- :py:class:`~structure_factor.spatial_windows.BoxWindow`: Box window object.

.. note::

    **Typical usage**

    - :py:class:`~structure_factor.point_pattern.PointPattern` has a :py:attr:`~structure_factor.point_pattern.PointPattern.window` argument/attribute.
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from structure_factor.utils import get_random_number_generator


class AbstractSpatialWindow(metaclass=ABCMeta):
    r"""Encapsulate the notion of spatial window in :math:`\mathbb{R}^d`."""

    @property
    @abstractmethod
    def dimension(self):
        """Return the ambient dimension of the corresponding window."""

    @property
    @abstractmethod
    def volume(self):
        """Compute the volume of the corresponding window."""

    @abstractmethod
    def __contains__(self, point):
        r"""Return True if :math:`d` dimensional vector ``point`` falls inside the corresponding :math:`d` dimensional window, otherwise return False.

        Args:
            point (numpy.ndarray): :math:`d` dimensional vector to test.
        """

    def indicator_function(self, points):
        r"""Return the indicator function of the corresponding window evaluated at each of the :math:`n` ``points``.

        Args:
            points (numpy.ndarray): Vector of size :math:`d` or array of size :math:`n \times d` containing the point(s) to be tested.

        Returns:
            bool or numpy.ndarray:
            - If :math:`n=1`, bool.
            - If :math:`n>1`, :math:`n` dimensional boolean array.
        """
        if points.ndim == 1 and points.size == self.dimension:
            return points in self
        return np.apply_along_axis(self.__contains__, axis=1, arr=points)

    # @abstractmethod
    # def rand(self, n=1, seed=None):
    #     r"""Generate `n` points uniformly at random in the corresponding spatial window.

    #     Args:
    #         n (int, optional): Number of points. Defaults to 1.
    #         seed (int or np.random.Generator, optional): Defaults to None.

    #     Returns:
    #         numpy.ndarray:
    #         - If :math:`n=1`, :math:`d` dimensional vector.
    #         - If :math:`n>1`, :math:`n \times d` array containing the points.
    #     """


class AnnulusWindow(AbstractSpatialWindow):
    r"""Create a :math:`d` dimensional annulus window :math:`A(c, q, p)`, where :math:`c \in \mathbb{R}^d` and :math:`p>q>0`.
    .. todo::

        list attributes

    # Example:
    #     .. plot:: code/spatial_window/ball_window.py
    #         :include-source: True
    #         :align: center

    # .. seealso::

    #     - :py:mod:`~structure_factor.point_pattern`
    #     - :py:class:`~structure_factor.spatial_windows.BoxWindow`
    """

    def __init__(self, center, small_radius=0.1, large_radius=1.0):
        """Initialize a :math:`d` dimensional annulus window :math:`B(c, q, p)` from the prescribed ``center`` c, ``small_radius`` q, and ``large_radius``p .

        Args:
            center (numpy.ndarray): Center :math:`c` of the ball.
            small_radius (float, optional): Radius :math:`q > 0` of the small ball of the annulus. Defaults to 0.1.
            small_radius (float, optional): Radius :math:`p> q > 0` of the large ball of the annulus. Defaults to 1.0.
        """
        center = np.asarray(center)
        if not center.ndim == 1:
            raise ValueError("center must be 1D numpy.ndarray")
        if not small_radius > 0:
            raise ValueError("Small radius must be positive")
        if not large_radius > small_radius:
            raise ValueError("Large radius must be bigger than small radius")
        self.center = center
        self.small_radius = float(small_radius)
        self.large_radius = float(large_radius)

    @property
    def dimension(self):
        return len(self.center)

    @property
    def surface(self):
        d, small_r, large_r = self.dimension, self.small_radius, self.large_radius
        if d == 1:
            return 0.0
        if d == 2:
            return 2 * np.pi * (large_r - small_r)
        if d == 3:
            return 4 * np.pi * (large_r ** 2 - small_r**2)
        return 2 * np.pi ** (d / 2) / sp.special.gamma(d / 2)* (large_r ** (d - 1) - small_r ** (d - 1))

    @property
    def volume(self):
        d, small_r, large_r = self.dimension, self.small_radius, self.large_radius

        if d == 1:
            return 2 * (large_r - small_r)
        if d == 2:
            return np.pi * (large_r ** 2 - small_r**2)
        if d == 3:
            return 4 / 3 * np.pi * (large_r ** 3 - small_r**3)
        return np.pi ** (d / 2) / sp.special.gamma(d / 2 + 1)  * (large_r ** d - small_r**d)

    def __contains__(self, point):
        point = np.asarray(point)
        assert point.ndim == 1 and point.size == self.dimension
        return self.indicator_function(point)

    def indicator_function(self, points):
        dist_points_center = np.linalg.norm(points - self.center, axis=-1)
        return np.logical_and(dist_points_center <= self.large_radius, dist_points_center >= self.small_radius)

    # def rand(self, n=1, seed=None):
    #     # Method of dropped coordinates
    #     # Efficiently sampling vectors and coordinates from the n-sphere and n-ball
    #     # Voelker, Aaron and Gosmann, Jan and Stewart, Terrence
    #     # doi: 10.13140/RG.2.2.15829.01767/1
    #     small_r, large_r = self.small_radius, self.large_radius
    #     rng = get_random_number_generator(seed)
    #     d = self.dimension
    #     points = rng.standard_normal(size=(n, d + 2))
    #     points /= np.linalg.norm(points, axis=-1, keepdims=True)
    #     idx = 0 if n == 1 else slice(0, n)
    #     return self.center + small_r + (large_r-small_r) * points[idx, :d]

    # def to_spatstat_owin(self, **params):
    #     """Convert the object to a ``spatstat.geom.disc`` R object of type ``disc``, which is a subtype of ``owin``.

    #     Args:
    #         params (dict): Optional keyword arguments passed to ``spatstat.geom.disc``.

    #     Returns:
    #         spatstat.geom.disc: R object.

    #     .. seealso::

    #         - `https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/disc <https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/disc>`_
    #     """
    #     spatstat = SpatstatInterface(update=False)
    #     spatstat.import_package("geom", update=False)
    #     r = self.radius
    #     c = robjects.vectors.FloatVector(self.center)
    #     return spatstat.geom.disc(radius=r, centre=c, **params)

    def plot(self, axis=None, **kwargs):
        """Display the window on matplotlib `axis`.

        Args:
            axis (plt.Axes, optional): Support axis of the plot. Defaults to None.

        Keyword Args:
            kwargs (dict): Keyword arguments of ``matplotlib.patches.Circle`` with default ``fill=False``.

        Returns:
            plt.Axes: Plot axis.
        """
        if self.dimension != 2:
            raise NotImplementedError("Method implemented only for 2D window")

        if axis is None:
            fig, axis = plt.subplots(figsize=(5, 5))

        kwargs.setdefault("fill", False)
        small_circle = Circle(self.center, self.small_radius, **kwargs)
        large_circle = Circle(self.center, self.large_radius, **kwargs)


        axis.add_patch(small_circle)
        axis.add_patch(large_circle)
        return axis
