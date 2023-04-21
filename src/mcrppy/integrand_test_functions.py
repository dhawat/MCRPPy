from mcrppy.spatial_windows import BallWindow, BoxWindow
import math
import numpy as np
import warnings

def indicator(x, window):
    return window.indicator_function(x)*1

# Support of all integrands
def support_integrands(d):
    return BoxWindow([[-1/2,1/2]]*d)

# Support of radial integrands
def support_integrands_ball(d):
    return BallWindow(center=[0]*d, radius=1/2)

# bump function times C^2 function
def f_1(x):
    warnings.filterwarnings('ignore')
    d = x.shape[1]
    support = support_integrands_ball(d)
    norm_x = np.linalg.norm(x, axis=1)
    a = 1 - 4*norm_x**2
    b = np.nan_to_num(np.exp(-2/a, where=a!=0))
    return a**2*b*indicator(x, support)

def exact_integral_f_1(d):
    return None

# indicator function
def f_2(x):
    d = x.shape[1]
    support = support_integrands_ball(d)
    return indicator(x, support)

def exact_integral_f_2(d):
    r= support_integrands_ball(d).radius
    return BallWindow(center=[0]*d, radius=r).volume

# sinusoidal C^2 function
def f_3(x):
    d = x.shape[1]
    support = support_integrands(d)
    a = np.sin(math.pi*x)*(np.cos(math.pi*x))**3
    return np.prod(a, axis=1)*indicator(x, support)
def exact_integral_f_3(d):
    return 0

def f_4(x, a=30, c=10):
    _, d = x.shape
    support = support_integrands(d)
    return c*np.prod(np.sin(a*math.pi*x), axis=1)*indicator(x, support)
def exact_integral_f_4(d):
    return 0

def f_5(x, a=10, c=10):
    _, d = x.shape
    support = support_integrands(d)
    return c*np.prod(np.cos(a*math.pi*x), axis=1)*indicator(x, support)
def exact_integral_f_5(d):
    return 0
