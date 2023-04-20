import numpy as np
from scipy import stats
from rpppy.spatial_windows import UnitBallWindow, BoxWindow

def sort_points_by_increasing_distance(points):
    norm_points = np.linalg.norm(points, axis=1)
    points = points[np.argsort(norm_points)]
    return points

def volume_unit_ball(d):
    center = np.full(shape=(d), fill_value=0)
    return UnitBallWindow(center=center).volume

def sort_output_push_point(x, epsilon):
    x_list = []
    if not isinstance (epsilon, list):
            epsilon = [epsilon]
    for e in range(len(epsilon)):
        x_e = np.vstack([x[i][e] for i in range(len(x))])
        x_list.append(x_e)
    return x_list

def _sort_point_pattern(point_pattern):
    point_pattern.points = sort_points_by_increasing_distance(point_pattern.points)
    return point_pattern

def jaccobi_measure(x, jac_params):
    d = x.shape[1]
    support_window = BoxWindow([[-1,1]]*d)
    alpha, betta = jac_params[:, 0], jac_params[:, 1]
    a = np.zeros_like(x)
    for i in range(d):
        a[:,i]= (1-x[:,i])**alpha[i]*(1+x[:,i])**betta[i]
    result = np.prod(a, axis=1)
    support = support_window.indicator_function(x)*1
    return result*support
# def test_jaccobi_measure():
#     x = np.array([[1, 1/2, 0], [1/2, 0, 0], [0, 1.1, 0]])
#     #x= np.array([ [1/2, 0, 0]])
#     jac_params = np.array([[1, 1, 0], [2, 0, 1]]).T
#     expected = np.array([0, 9/8, 0])
#     if np.isclose(jaccobi_measure(x, jac_params), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, error=", jaccobi_measure(x,jac_params)- expected)

# utils for monte_carlo_methods
def _find_sum_of_coef_of_cubic_term(poly, d):
    """Function used to find the sum of the coefficient of the quadratic terms in a polynomial regression of degree 2. Used to find the mean of the proposal in ``estimate_control_variate_proposal``.

    _extended_summary_

    Args:
        poly (_type_): _description_
        d (_type_): _description_

    Returns:
        _type_: _description_
    """
    eval_points = []
    for i in range(0,d):
        x = np.zeros((1,d))
        y = np.zeros((1,d))
        y[:,i] = -1
        eval_points.append(y)
        x[:,i] = 1
        eval_points.append(x)

    return (sum(poly(p) for p in eval_points) - 2*d*poly(np.zeros((1,d))))/2

#utils for monte_carlo_setup

def error(approx, exact):
    if exact is not None:
        return np.array(approx) - exact
    else:
        return "NAN"


def mse(mean, std, exact, verbose=False):
    #print(mean)
    if exact is not None:
        var = np.square(std)
        bias_square = np.square(np.array(mean)- exact)
        if verbose:
            print( "Bias=", bias_square)
        return var + bias_square
    else:
        return "NAN"

def regression_line(x, y, log=True, residual=False, residual_normality_test="KS"):
    if log:
        x = np.log(x)
        y = np.log(y)
    else:
        x = np.array(x)
        y = np.array(y)
    #reg = LinearRegression().fit(x, y)
    #slope = reg.coef_
    reg_fit = stats.linregress(x,y)
    slope = reg_fit.slope
    std_slope = reg_fit.stderr
    reg_line = x*slope + reg_fit.intercept
    if residual:
        #Kolmogorov-Smirnov or Shapiro-Wilk test for the residual of linear regressian to determine if it follows a normal distribution
        residual = y - reg_line # residual r = y - estimated_y
        if residual_normality_test=="KS":
            test_result = stats.kstest(residual, 'norm')
        elif residual_normality_test=="SW":
            test_result = stats.shapiro(residual)
        else:
            test_result=None
        if test_result is not None:
            return reg_line, slope, std_slope, residual, test_result
        else:
            return reg_line, slope, std_slope, residual
    else:
        return reg_line, slope, std_slope
