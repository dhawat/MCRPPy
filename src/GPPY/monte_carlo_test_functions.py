from GPPY.spatial_windows import BallWindow, BoxWindow, UnitBallWindow
import math
import numpy as np

def indicator(x, window):
    return window.indicator_function(x)*1

# Support of all integrands
def support_integrands(d):
    return BoxWindow([[-1/2,1/2]]*d)

def support_integrands_ball(d):
    # r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    r= 1/2
    return BallWindow(center=[0]*d, radius=r)

def f_1(x):
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
    #radius of the biggest BallWindow included in support_integrands
    support = support_integrands_ball(d)
    return indicator(x, support)

def exact_integral_f_2(d):
    r= support_integrands_ball(d).radius
    return BallWindow(center=[0]*d, radius=r).volume

# def test_f_2():
#     x = np.array([[1, 2, 2], [1/2, 0, 0], [1/2, 1/2, 1/2]])
#     if (f_1(x)== np.array([[0, 1, 0]])).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_2(x)=", f_2(x))
# test_f_2()

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

#old f_4
def f_5(x, a=10, c=10):
    _, d = x.shape
    support = support_integrands(d)
    return c*np.prod(np.cos(a*math.pi*x), axis=1)*indicator(x, support)
def exact_integral_f_5(d):
    return 0

#old fcts

# old f_1
def f_6(x):
    d = x.shape[1]
    support = support_integrands_ball(d)
    norm_x = np.linalg.norm(x, axis=1)
    return np.exp(-norm_x**d)*indicator(x, support)
def exact_integral_f_6(d):
    r= support_integrands_ball(d).radius
    kappa_d = UnitBallWindow(center=[0]*d).volume
    return kappa_d*(1 - np.exp(-r**(d)))

# def f_6(x):
#     d = x.shape[1]
#     support = support_integrands_ball(d)
#     norm_x = np.linalg.norm(x, axis=1)
#     return (1/(norm_x**d + 0.5**d) - 2**(d-1))*indicator(x, support)
# def exact_integral_f_6(d):
#     kappa_d = UnitBallWindow(center=[0]*d).volume
#     return kappa_d*(math.log(2) - 0.5)

# def cv_proposal_f_7(d):
#     r= min(np.diff(support_integrands(d).bounds, axis=1))/2
#     support = BallWindow(center=[0]*d, radius=r)
#     proposal = lambda x: 4*np.exp(np.linalg.norm(x, axis=1)**d)*indicator(x, support)
#     kappa_d = UnitBallWindow(center=[0]*d).volume
#     mean_proposal = 4*kappa_d*(np.exp(r**(d))-1) #mean(g(x))
#     return proposal, mean_proposal

#old f_4

# def f_5(x, a=25, c=10):
#     return f_5(x, a,c)
# def exact_integral_f_5(d, a=25,c=10):
#     return -c*(2/(math.pi*a))**d

#old f_3
def scale_f_9(d):
    return 10**(3*d/2)

def f_9(x):
    d = x.shape[1]
    #scale=scale_f_3(d)
    support = support_integrands(d)
    a = 2*np.sin(np.pi*x)**2*np.cos(np.pi*x)
    b = np.cos(np.pi*x - np.pi/4)
    return np.product(a-b, axis=1)*indicator(x, support)

def exact_integral_f_3(d):
    #scale=scale_f_3(d)
    return ((4/3 - np.sqrt(2))/np.pi)**d
