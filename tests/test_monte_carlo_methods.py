from mcrppy.monte_carlo_methods import _repelled_binomial_samples, _binomial_samples, _scramble_sobol_samples, _dpp_samples
from mcrppy.spatial_windows import BallWindow
from mcrppy.integrand_test_functions import support_integrands
import numpy as np
import pytest
# test _binomial_samples respect the number of samples and of points

def test_binomial_samples():
    d=2
    window = BallWindow([0]*d, 4)
    nb_samples = 10
    nb_points = 20
    binomial_pp = _binomial_samples(nb_samples, nb_points, window)
    np.testing.assert_equal(len(binomial_pp), nb_samples)
    np.testing.assert_equal(binomial_pp[5].points.shape, (nb_points, d))

# test _repelled_binomial_samples respect the number of samples and the window

def test_repelled_binomial_samples():
    d=3
    r=4
    window = BallWindow([0]*d, 4)
    nb_samples = 12
    nb_points = 10
    repelled_pp = _repelled_binomial_samples(nb_samples, nb_points, window, nb_cores=1)
    sample = repelled_pp[5].points
    nb_points_out = sum(np.linalg.norm(sample, axis=1)>r)
    np.testing.assert_equal(len(repelled_pp), nb_samples)
    np.testing.assert_equal(sample.shape[1], d)
    np.testing.assert_equal(nb_points_out, 0)

# test _scramble_sobol_samples respect the number of samples, number of points and window

def test_scramble_sobol_samples():
    d=3
    window = support_integrands(d) #centered cubic window of unit volume
    nb_samples = 12
    nb_points = 10
    sobol_pp = _scramble_sobol_samples(nb_samples, nb_points, window)
    sample = sobol_pp[3].points
    nb_points_out = sum(abs(sample[i, 0])>1/2 for i in range(d))
    np.testing.assert_equal(len(sobol_pp), nb_samples)
    np.testing.assert_equal(sample.shape, (nb_points, d))
    np.testing.assert_equal(nb_points_out, 0)

# test _scramble_sobol_samples respect the number of samples, number of points and window

def test_dpp_sample():
    d=2
    nb_samples = 4
    nb_points = 6
    dpp_pp = _dpp_samples(nb_points, d, nb_samples, nb_cores=None, pool_dpp=False)
    sample = dpp_pp[3]
    nb_points_out = sum(abs(sample[i, 0] )>1 for i in range(d))
    np.testing.assert_equal(len(dpp_pp), nb_samples)
    np.testing.assert_equal(sample.shape, (nb_points, d))
    np.testing.assert_equal(nb_points_out, 0)
