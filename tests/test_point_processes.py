import numpy as np
import pytest
from scipy.stats import norm
import math

from mcrppy.point_processes import (
    GinibrePointProcess,
    HomogeneousPoissonPointProcess,
    ThomasPointProcess,
    BinomialPointProcess,
    ScrambleSobolPointProcess,
    _simulation_param_sobol_sequence,
    _simulation_window_repelled_sample
)
from mcrppy.spatial_windows import BallWindow, BoxWindow

#poisson and thomas
@pytest.mark.parametrize(
    "point_process, nb_samples, W",
    (
        [
            HomogeneousPoissonPointProcess(100),
            10_000,
            BoxWindow(np.array([[0, 2], [0, 2]])),
        ],
        [
            ThomasPointProcess(kappa=1 / (2 * np.pi), mu=2 * np.pi, sigma=2),
            10_000,
            BallWindow(radius=3, center=[1, 0]),
        ],
    ),
)
def test_intensity_estimation_of_poisson_process(point_process, nb_samples, W):
    """Estimate the intensity ``rho`` as mean number of points divided by the volume of the observation window ``W`` from ``nb_samples`` and check that the intensity falls inside a confidence interval constructed from the following central limit theorem-like result.

    As W -> R^d, we have

    2 sqrt(vol(W)) (sqrt(rho_hat) − sqrt(rho)) ~ N(0,1)
    """

    rho = point_process.intensity
    nb_points = [len(point_process.generate_sample(W)) for _ in range(nb_samples)]
    rho_hat = np.mean(nb_points) / W.volume
    # mean_rho_hat = rho
    # var_rho_hat = rho / W.volume
    alpha = 0.05
    center = np.sqrt(rho_hat)
    z_a2 = -norm.ppf(alpha / 2)
    width_2 = z_a2 / (2 * np.sqrt(W.volume))
    assert (center - width_2) <= np.sqrt(rho) <= (center + width_2)

def test_repelled_poisson_point_pattern():
    intensity=1/math.pi
    window = BallWindow([0,0], 2)
    seed = 2
    epsilon = 0.01
    expected_sample = np.array([[ 1.57050379,  0.99844945],
                                [ 0.4690294 , -0.92371741],
                                [-0.67386449, -1.62335999]])
    a_ = np.array([[0.41285263, 0.61175712],
                   [ 0.4120337 , -0.00202209],
                   [-0.82488633, -0.60973503]])
    a =  epsilon*a_ + (1-1*epsilon)*expected_sample
    expected_repelled =  a[np.linalg.norm(a, axis=1)<=2]
    poisson = HomogeneousPoissonPointProcess(intensity=intensity)
    sample, repelled = poisson.generate_repelled_point_pattern(window=window, seed=seed, epsilon=epsilon)
    np.testing.assert_array_almost_equal(expected_sample, sample.points)
    np.testing.assert_array_almost_equal(expected_repelled, repelled.points)


#ginibre
def test_pair_correlation_function_ginibre():
    r = np.array([[0], [1], [10 ^ 5]])
    pcf = GinibrePointProcess.pair_correlation_function(r)
    expected = np.array([[0], [1 - 1 / np.exp(1)], [1]])
    np.testing.assert_array_equal(pcf, expected)


def test_structure_factor_ginibre():
    k = np.array([[0], [1], [10 ^ 5]])
    sf = GinibrePointProcess.structure_factor(k)
    expected = np.array([[0], [1 - 1 / np.exp(1 / 4)], [1]])
    np.testing.assert_array_equal(sf, expected)

def test_repelled_ginibre_point_pattern():
    window = BallWindow([0,0], 2)
    seed = 1
    epsilon = 0.01
    expected_sample = np.array([[ 1.3029939 ,  0.4007133 ],
                                [ 0.89889012, -1.10197517],
                                [-1.14367331, -0.59972807],
                                [-0.05466552,  0.22429708]])
    a_ = np.array([[1.24139299, 0.85789918],
                   [ 0.65214605, -1.23116726],
                   [-1.39577084, -0.47151224],
                   [-0.49776821,  0.84478032]])
    a =  epsilon*a_ + (1-1*epsilon)*expected_sample
    expected_repelled =  a[np.linalg.norm(a, axis=1)<=2]
    ginibre = GinibrePointProcess()
    sample, repelled = ginibre.generate_repelled_point_pattern(window=window, seed=seed, epsilon=epsilon)
    np.testing.assert_array_almost_equal(expected_sample, sample.points)
    np.testing.assert_array_almost_equal(expected_repelled, repelled.points)

#sobol
@pytest.mark.parametrize(
    "window, nb_points, expected_l, expected_nb_points",
    [(BoxWindow([[-1, 1], [-1,1]]), 100, 2, 100),
     (BoxWindow([[-1, 1], [-1,1], [-1, 1]]), 100, 2, 100),
     (BoxWindow([[-1, 0], [-1,2]]), 100, 3, 300),
     (BallWindow([0, 0], 3), 50, 6, int(200/math.pi)) ]
)
def test_simulation_params_sobol(window, nb_points, expected_l, expected_nb_points):
    l, nb_points = _simulation_param_sobol_sequence(window, nb_points)
    np.testing.assert_equal(l, expected_l)
    np.testing.assert_equal(nb_points, expected_nb_points)

@pytest.mark.parametrize(
    "nb_points, window",
    [(100, BoxWindow([[-5,5], [-5, 5]])),
     (50, BoxWindow([[-1,1], [-1, 1], [-1, 1]]))]
)

def test_sobol_process_intensity_in_cube(nb_points,
    window):
    sobol = ScrambleSobolPointProcess()
    sobol_pp = sobol.generate_point_pattern(nb_points, window)
    expected = nb_points/window.volume
    np.testing.assert_equal(expected, sobol_pp.intensity)


@pytest.mark.parametrize(
    "nb_points, window",
    [(1000, BoxWindow([[-1,0], [-5, 0]])),
     (5000, BallWindow([0,0, 0], 5))]
)

def test_sobol_process_intensity_in_non_cube(nb_points,
    window):
    sobol = ScrambleSobolPointProcess()
    nb_sample=100
    sobol_pp = [sobol.generate_point_pattern(nb_points, window) for _ in range(nb_sample)]
    expected_list = [p.points.shape[0]/window.volume for p in sobol_pp]
    expected = sum(expected_list)/nb_sample
    result = sum([p.intensity for p in sobol_pp])/nb_sample
    np.testing.assert_equal(expected, result)


def test_repelled_sobol_point_pattern():
    nb_points = 4
    window = BallWindow([0,0], 1)
    seed = 1
    expected_sample = np.array([[-0.68906936,  0.17749465],
                                [ 0.22669113,  0.71838415],
                                [-0.15305269, -0.31300241]])
    a_ = np.array([[-1.82493606,  0.450981  ],
                 [1.12392998, 1.33198704],
                 [ 0.70100608, -1.78296804]])
    a =  a_ - 2*expected_sample
    expected_repelled =  a[np.linalg.norm(a, axis=1)<=1]
    sobol = ScrambleSobolPointProcess()
    sample, repelled = sobol.generate_repelled_point_pattern(nb_points=nb_points, window=window, seed=seed, epsilon=1)
    np.testing.assert_array_almost_equal(expected_sample, sample.points)
    np.testing.assert_array_almost_equal(expected_repelled, repelled.points)


@pytest.mark.parametrize(
    "window, add_boundary, expected_radius",
    [(BallWindow([0,0,0],10), 0, 10),
     (BallWindow([0,0],10), 2, 12),
     (BoxWindow([[-1,1], [-1,1]]), 0, math.sqrt(2)),
     (BoxWindow([[-1,1], [-1,1], [-1,1]]), 4.5, math.sqrt(3) + 4.5)]
)
def test_simulation_window_repelled_sample(window, add_boundary, expected_radius):
    simulation_window= _simulation_window_repelled_sample(window,add_boundary)
    radius = simulation_window.radius
    np.testing.assert_equal(window.dimension, simulation_window.dimension)
    np.testing.assert_equal(radius, expected_radius)


#binomial
def test_binomial_generate_sample_in_window():
    binomial = BinomialPointProcess()
    window = BallWindow(center=[0,0], radius=4)
    sample = binomial.generate_sample(nb_points=4, window=window, seed=1)
    np.testing.assert_equal(sample, sample[np.linalg.norm(sample, axis=1)<=4])

def test_repelled_binomial_point_pattern():
    nb_points = 3
    window = BallWindow([0,0], 1)
    seed = 1
    epsilon = 0.01
    expected_sample = np.array([[0.21424427, 0.50936062],
                                [0.70590254, 0.34803657],
                                [0.50599447, 0.40823008]])
    a_ = np.array([[-4.89616014,  1.66318383],
                   [6.42270409, -1.98352292],
                   [-1.52654395,  0.32033909]])
    a =  epsilon*a_ + (1-3*epsilon)*expected_sample
    expected_repelled =  a[np.linalg.norm(a, axis=1)<=1]
    binomial = BinomialPointProcess()
    sample, repelled = binomial.generate_repelled_point_pattern(nb_points=nb_points, window=window, seed=seed, epsilon=epsilon)
    np.testing.assert_array_almost_equal(expected_sample, sample.points)
    np.testing.assert_array_almost_equal(expected_repelled, repelled.points)
