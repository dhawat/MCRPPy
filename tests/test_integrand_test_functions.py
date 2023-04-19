import numpy as np
import pytest
from rpppy.integrand_test_functions import f_1, f_2, f_3, f_4, f_5

@pytest.mark.parametrize(
    "x, expected",
    [(np.array([[1/2, 0, 0]]), np.array([0])),
     (np.array([[1/2, -1, 0, 7]]), np.array([0])),
     (np.array([[1/4, 1/4]]), np.array([np.exp(-4)/4])),
    ],
)
def test_f_1(x, expected):
    np.testing.assert_array_almost_equal(f_1(x), expected)

def test_f_2():
    x = np.array([[1, 2, 2], [1/2, 0, 0], [1/2, 1/2, 1/2]])
    result = f_2(x)
    expected = np.array([0, 1, 0])
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize(
    "x, expected",
    [(np.array([[1/2, 0, 5, 6]]), np.array([0])),
     (np.array([[2.1, 3.3]]), np.array([0])),
     (np.array([[1/2, 0.04]]), np.array([0])),
     (np.array([[1/4, -1/4]]), np.array([-1/2**4]))
    ],
)
def test_f_3(x, expected):
    np.testing.assert_array_almost_equal(f_3(x), expected)
