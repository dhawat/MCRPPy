"""Collection of point processes at related properties, e.g., intensity, pair correlation function, structure factor.

- :py:class:`~mcrppy.point_processes.HomogeneousPoissonPointProcess`: The homogeneous Poisson point process.

- :py:class:`~mcrppy.point_processes.ThomasPointProcess`: The Thomas point process.

- :py:class:`~mcrppy.point_processes.GinibrePointProcess`: The Ginibre point process.

- :py:class:`~mcrppy.point_processes.ScrambledSobol`: The Ginibre point process.

- :py:func:`~mcrppy.point_processes.mutual_nearest_neighbor_matching`: The matching process of :cite:`KlaLasYog20`.

Based on `https://github.com/For-a-few-DPPs-more/structure-factor/blob/main/src/structure_factor/point_processes.py`

"""
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
import scipy as sp
import math
import warnings

from mcrppy.point_pattern import PointPattern
from mcrppy.spatial_windows import (
    AbstractSpatialWindow,
    BallWindow,
    BoxWindow,
)
from mcrppy.repelled_point_process import RepelledPointProcess

def get_random_number_generator(seed=None):
    """Turn seed into a np.random.Generator instance."""
    return np.random.default_rng(seed)

class HomogeneousPoissonPointProcess(object):
    """`Homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_.

    .. todo::

        list attributes
    """

    def __init__(self, intensity=1.0):
        """Create a `homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_ with prescribed (positive) ``intensity`` parameter.

        Args:
            intensity (float, optional): intensity of the homogeneous Poisson point process. Defaults to 1.0.
        """
        if not intensity > 0:
            raise TypeError("intensity argument must be 2positive")
        self._intensity = float(intensity)

    @property
    def intensity(self):
        r"""Return the intensity :math:`\rho_1(r) = \rho` of the homogeneous Poisson point process.

        Returns:
            float: Constant intensity.
        """
        return self._intensity

    def generate_sample(self, window, seed=None):
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")

        rng = get_random_number_generator(seed)
        rho = self.intensity
        nb_points = rng.poisson(rho * window.volume)
        return window.rand(nb_points, seed=rng)

    def generate_point_pattern(self, window, seed=None):
        points = self.generate_sample(window=window, seed=seed)
        point_pattern = PointPattern(
            points=points, window=window, intensity=self.intensity
        )
        return point_pattern
    def generate_repelled_point_pattern(self, window, seed=None, add_boundary=None, output=None, **repelled_params):
        warnings.warn("Method only available for centered ball or cubic window.")
        # simulation window
        simulation_window = _simulation_window_repelled_sample(window, add_boundary)
        # simulation sample
        point_pattern_simulation = self.generate_point_pattern(simulation_window, seed)
        # obtained repelled sample
        rpp = RepelledPointProcess(point_pattern_simulation)
        repelled_point_pattern_simulation = rpp.repelled_point_pattern(**repelled_params)
        if output=="simulation":
            return point_pattern_simulation, repelled_point_pattern_simulation
        else:
            # samples in support window
            point_pattern = point_pattern_simulation.restrict_to_window(window)
            repelled_point_pattern = repelled_point_pattern_simulation.restrict_to_window(window)
            return point_pattern, repelled_point_pattern


class ThomasPointProcess:
    """Homogeneous Thomas point process with Gaussian clusters.

    .. todo::

        list attributes
    """

    def __init__(self, kappa, mu, sigma):
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma

        self._intensity = float(kappa * mu)

    @property
    def intensity(self):
        return self._intensity
    #! seed is not working
    def generate_sample(self, window, seed=None):
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")

        rng = get_random_number_generator(seed)

        tol = 6 * self.sigma
        if isinstance(window, BoxWindow):
            extended_bounds = window.bounds + np.array([-tol, +tol])
            extended_window = BoxWindow(extended_bounds)
        elif isinstance(window, BallWindow):
            exented_radius = window.radius + tol
            extended_window = BallWindow(window.center, exented_radius)

        pp = HomogeneousPoissonPointProcess(self.kappa)
        centers = pp.generate_sample(extended_window, seed=rng)
        #! seed is not taking into consideration here
        n_per_cluster = np.random.poisson(self.mu, size=len(centers))
        d = window.dimension
        s = self.sigma
        points = np.vstack(
            [rng.normal(c, s, (n, d)) for (c, n) in zip(centers, n_per_cluster)]
        )
        mask = window.indicator_function(points)
        return points[mask]

    def generate_point_pattern(self, window, seed=None):
        points = self.generate_sample(window=window, seed=seed)
        point_pattern = PointPattern(
            points=points, window=window, intensity=self.intensity
        )
        return point_pattern
    def generate_repelled_point_pattern(self, window, seed=None, add_boundary=None, output=None, **repelled_params):
        warnings.warn("Method only available for centered ball or cubic window.")
        # simulation window
        simulation_window = _simulation_window_repelled_sample(window, add_boundary)
        # simulation sample
        point_pattern_simulation = self.generate_point_pattern(simulation_window, seed)
        # obtained repelled sample
        rpp = RepelledPointProcess(point_pattern_simulation)
        repelled_point_pattern_simulation = rpp.repelled_point_pattern(**repelled_params)
        if output=="simulation":
            return point_pattern_simulation, repelled_point_pattern_simulation
        else:
            # samples in support window
            point_pattern = point_pattern_simulation.restrict_to_window(window)
            repelled_point_pattern = repelled_point_pattern_simulation.restrict_to_window(window)
            return point_pattern, repelled_point_pattern


#! not convinced with the number of points in ginibre it seems if we set the number the points its not what we get
class GinibrePointProcess(object):
    """Ginibre point process corresponds to the complex eigenvalues of a standard complex Gaussian matrix.

    .. todo::

        list attributes
    """

    def __init__(self):
        self._intensity = 1.0 / np.pi

    @property
    def intensity(self):
        r"""Return the intensity :math:`\rho_1(r) = \frac{1}{\pi}` of the Ginibre point process.

        Returns:
            float: Constant intensity.
        """
        return self._intensity

    @staticmethod
    def pair_correlation_function(r_norm):
        return 1.0 - np.exp(-(r_norm ** 2))

    @staticmethod
    def structure_factor(k_norm):
        return 1.0 - np.exp(-0.25 * (k_norm ** 2))

    def generate_sample(self, window, nb_points=None, seed=None):
        if not isinstance(window, BallWindow):
            raise ValueError("The window should be a 2-d centered BallWindow.")
        if window.dimension != 2:
            raise ValueError("The window should be a 2-d window.")
        if not np.all(np.equal(window.center, 0.0)):
            raise ValueError("The window should be a centered window.")

        if nb_points is None:
            nb_points = int(window.volume * self.intensity)
        assert isinstance(nb_points, int)
        rng = get_random_number_generator(seed)

        A = np.zeros((nb_points, nb_points), dtype=complex)
        A.real = rng.standard_normal((nb_points, nb_points))
        A.imag = rng.standard_normal((nb_points, nb_points))
        eigvals = la.eigvals(A) / np.sqrt(2.0)

        points = np.vstack((eigvals.real, eigvals.imag)).T
        mask = window.indicator_function(points)

        return points[mask]

    def generate_point_pattern(self, window, nb_points=None, seed=None):
        r"""Generate a :math:`2`-dimensional :py:class:`~structure_factor.point_pattern.PointPattern` of the point process, with a centered :py:class:`~structure_factor.spatial_windows.BallWindow`.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): :math:`2`-dimensional observation centered :py:class:`~structure_factor.spatial_windows.BallWindow`.

            n (int, optional): Number of points of the output sample. If ``n`` is None (default), it is set to the integer part of :math:`\rho |W| = \frac{1}{\pi} |W|`. Defaults to None.

            seed (int, optional): Seed to initialize the points generator. Defaults to None.

        Returns:
            :py:class:`~structure_factor.point_pattern.PointPattern`:Object of type :py:class:`~structure_factor.point_pattern.PointPattern` containing a realization of the point process, the observation window, and (optionally) the intensity of the point process (see :py:class:`~structure_factor.point_pattern.PointPattern`).

        Example:
            .. plot:: code/point_processes/ginibre_pp.py
                :include-source: True
                :caption:
                :alt: code/point_processes/ginibre_pp.py
                :align: center

        .. seealso::

            - :py:class:`~structure_factor.spatial_windows.BallWindow`
            - :py:class:`~structure_factor.point_pattern.PointPattern`
        """
        points = self.generate_sample(window=window, nb_points=nb_points, seed=seed)
        point_pattern = PointPattern(
            points=points, window=window, intensity=self.intensity
        )
        return point_pattern

    def generate_repelled_point_pattern(self, window, nb_points=None, seed=None, add_boundary=None, output=None, **repelled_params):
        warnings.warn("Method only available for centered ball or cubic window.")
        # simulation window
        simulation_window = _simulation_window_repelled_sample(window, add_boundary)
        # simulation sample
        point_pattern_simulation = self.generate_point_pattern(simulation_window, nb_points=nb_points, seed=seed)
        # obtained repelled sample
        rpp = RepelledPointProcess(point_pattern_simulation)
        repelled_point_pattern_simulation = rpp.repelled_point_pattern(**repelled_params)
        if output=="simulation":
            return point_pattern_simulation, repelled_point_pattern_simulation
        else:
            # samples in support window
            point_pattern = point_pattern_simulation.restrict_to_window(window)
            repelled_point_pattern = repelled_point_pattern_simulation.restrict_to_window(window)
            return point_pattern, repelled_point_pattern
class ScrambleSobolPointProcess(object):

    def generate_point_pattern(self, nb_points, window, seed=None, **kwargs):
        """Generate scramble sobol point pattern in a centered box or ball window using ``https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html``.
        """
        #! TBC to get the needed number of points in both cases
        if isinstance(window, BallWindow):
            warnings.warn("The obtained number of points is not `nb_points`. The points are sampled in a BoxWindow of lenghtside equal the diameter of the window then a restriction is made to get the points in the support window.")
        rng = get_random_number_generator(seed)
        d = window.dimension
        sobol = sp.stats.qmc.Sobol(d=d, scramble=True, seed=rng, **kwargs)
        l, nb_points = _simulation_param_sobol_sequence(window=window, nb_points=nb_points)
        simulation_window = BoxWindow(bounds=[[-l/2, l/2]]*d)
        points_unit_box = sobol.random(n=nb_points)
        points = (points_unit_box - 0.5)*l
        point_pattern = PointPattern(points, simulation_window).restrict_to_window(window)
        point_pattern.intensity = point_pattern.points.shape[0]/window.volume
        return point_pattern

    def generate_sample(self, nb_points, window, seed=None, **kwargs):
        point_pattern = self.generate_point_pattern(window=window, nb_points=nb_points, seed=seed, **kwargs)
        return point_pattern.points

    def generate_repelled_point_pattern(self, nb_points, window, seed=None, add_boundary=None, output=None, **repelled_params):
        warnings.warn("Method only available for centered ball or box window.")
        # simulation window
        simulation_window = _simulation_window_repelled_sample(window, add_boundary)
        # simulation nb_points
        nb_points_simulation = int((nb_points*simulation_window.volume)/window.volume)
        # simulation sample
        point_pattern_simulation = self.generate_point_pattern(nb_points_simulation, simulation_window, seed)
        # obtained repelled sample
        rpp = RepelledPointProcess(point_pattern_simulation)
        repelled_point_pattern_simulation = rpp.repelled_point_pattern(**repelled_params)
        if output=="simulation":
            return point_pattern_simulation, repelled_point_pattern_simulation
        else:
            # samples in support window
            point_pattern = point_pattern_simulation.restrict_to_window(window)
            repelled_point_pattern = repelled_point_pattern_simulation.restrict_to_window(window)
            return point_pattern, repelled_point_pattern

class BinomialPointProcess(object):
    def generate_sample(self, nb_points, window, seed=None):
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")

        rng = get_random_number_generator(seed)
        return window.rand(nb_points, seed=rng)
    def generate_point_pattern(self, nb_points, window, seed=None):
        points = self.generate_sample(nb_points=nb_points, window=window, seed=seed)
        point_pattern = PointPattern(
            points=points, window=window
        )
        return point_pattern
    def generate_repelled_point_pattern(self, nb_points, window, seed=None, add_boundary=None, output=None, **repelled_params):
        warnings.warn("Method only available for centered ball or box window.")
        # simulation window
        simulation_window = _simulation_window_repelled_sample(window, add_boundary)
        # simulation nb_points
        nb_points_simulation = int((nb_points*simulation_window.volume)/window.volume)
        # simulation sample
        point_pattern_simulation = self.generate_point_pattern(nb_points_simulation, simulation_window, seed)
        # obtained repelled sample
        rpp = RepelledPointProcess(point_pattern_simulation)
        repelled_point_pattern_simulation = rpp.repelled_point_pattern(**repelled_params)
        if output=="simulation":
            return point_pattern_simulation, repelled_point_pattern_simulation
        else:
            # samples in support window
            point_pattern = point_pattern_simulation.restrict_to_window(window)
            repelled_point_pattern = repelled_point_pattern_simulation.restrict_to_window(window)
            return point_pattern, repelled_point_pattern

def _simulation_window_repelled_sample(window, add_boundary=None):
    """Repelled point process should be always sampled in a ball window containing the support window `window`. To reduce boundary effect `add_boundary` can be used to add additional factor to the radius of the simulation window.
    Args:
        window (_type_): _description_
        add_boundary (_type_, optional): _description_. Defaults to None.
    Returns:
        _type_: BallWindow where the sample should be generate.
    """
    if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")
    d = window.dimension
    if isinstance(window, BoxWindow):
        l = min(np.diff(window.bounds, axis=1)) #length side window
        r=math.sqrt(d)*l/2 #radius ball window containing box_window
    else :
        r = window.radius
    if add_boundary is not None:
        r += add_boundary
    window = BallWindow(center=[0]*d, radius=r)
    return window

def _simulation_param_sobol_sequence(window, nb_points):
    d = window.dimension
    if isinstance(window, BoxWindow):
        l = np.max(np.diff(window.bounds))
    elif isinstance(window, BallWindow):
        l = 2*window.radius
    nb_points = int(nb_points/window.volume*(l**d))
    return l, nb_points

def mutual_nearest_neighbor_matching(X, Y, **KDTree_params):
    r"""Match the set of points ``X`` with a subset of points from ``Y`` based on mutual nearest neighbor matching :cite:`KlaLasYog20`. It is assumed that :math:`|X| \leq |Y|` and that each point in ``X``, resp. ``Y``, can have only one nearest neighbor in ``Y``, resp. ``X``.

    The matching routine involves successive 1-nearest neighbor sweeps performed by `scipy.spatial.KDTree <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html>`_ with the euclidean distance.

    Args:
        X (numpy.ndarray): Array of size (m, d) collecting points to be matched with a subset of points from ``Y``.
        Y (numpy.ndarray): Array of size (n, d) of points satisfying :math:`m \leq n`.

    Keyword Args:
        see (documentation): keyword arguments of `scipy.spatial.KDTree <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html>`_.

    .. note::

        The ``boxsize`` keyword argument can be used **only** when points belong to a box :math:`\prod_{i=1}^{d} [0, L_i)` (upper boundary excluded). It allows to consider periodic boundaries, i.e., the toroidal distance is used for searching for nearest neighbors.

    Returns:
        numpy.ndarray: vector of indices ``matches`` such that ``X[i]`` is matched to ``Y[matches[i]]``.
    """
    if not (X.ndim == Y.ndim == 2):
        raise ValueError(
            "X and Y must be 2d numpy arrays with respective size (m, d) and (n, d), where d is the ambient dimension."
        )
    if X.shape[0] > Y.shape[0]:
        raise ValueError(
            "The sets of points represented by X and Y must satisfy |X| <= |Y|."
        )

    m, n = X.shape[0], Y.shape[0]
    idx_X_unmatched = np.arange(m, dtype=int)
    idx_Y_unmatched = np.arange(n, dtype=int)
    matches = np.zeros(m, dtype=int)

    for _ in range(m):  # at most |X| nearest neighbor sweeps are performed

        X_ = X[idx_X_unmatched]
        Y_ = Y[idx_Y_unmatched]

        knn = KDTree(Y_, **KDTree_params)
        X_to_Y = knn.query(X_, k=1, p=2)[1]  # p=2, i.e., euclidean distance

        knn = KDTree(X_, **KDTree_params)
        Y_to_X = knn.query(Y_, k=1, p=2)[1]

        identity = range(len(idx_X_unmatched))
        mask_X = np.equal(Y_to_X[X_to_Y], identity)

        matches[idx_X_unmatched[mask_X]] = idx_Y_unmatched[X_to_Y[mask_X]]

        if np.all(mask_X):  # all points from X got matched
            break

        idx_X_unmatched = idx_X_unmatched[~mask_X]
        mask_Y = np.full(len(idx_Y_unmatched), True, dtype=bool)
        mask_Y[X_to_Y[mask_X]] = False
        idx_Y_unmatched = idx_Y_unmatched[mask_Y]

    return matches
