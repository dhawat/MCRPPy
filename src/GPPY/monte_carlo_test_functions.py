from structure_factor.spatial_windows import BallWindow, BoxWindow, UnitBallWindow
import math
import numpy as np

def indicator(x, window):
    return window.indicator_function(x)*1

# Support of all integrands
def support_integrands(d):
    return BoxWindow([[-1/2,1/2]]*d)


def f_1(x):
    d = x.shape[1]
    support = support_integrands(d)
    return (1/np.prod(x+1, axis=1)-1)*indicator(x, support)

def exact_integral_f_1(d):
    return math.log(3)**d - 1

# def test_f_1():
#     x = np.array([[0, 0, 0, 0], [1/2, 0, -1/2,-1/2], [1, 0, 0, 0]])
#     expected = np.array([[0, 5/3, 0]])
#     if np.isclose(f_1(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, error=", f_1(x) - expected)
# test_f_1()

#control variate setup for f_1
def cv_setup_f_1(d):
    support = support_integrands(d)
    proposal = lambda x: np.prod(x+1, axis=1)*indicator(x, support) #g(x)
    mean_proposal = 1 #mean(g(x))
    c = (math.log(3)**d)/((25/12)**d-1) #-Cov(g(x), f(x))/var(g(x))
    return c, proposal, mean_proposal

# indicator function
def f_2(x):
    d = x.shape[1]
    #radius of the biggest BallWindow included in support_integrands
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    support = BallWindow(center=[0]*d, radius=r)
    return indicator(x, support)
def exact_integral_f_2(d):
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    return BallWindow(center=[0]*d, radius=r).volume

# def test_f_2():
#     x = np.array([[1, 2, 2], [1/2, 0, 0], [1/2, 1/2, 1/2]])
#     if (f_2(x)== np.array([[0, 1, 0]])).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_2(x)=", f_2(x))
# test_f_2()

def scale_f_3(d):
    return 10**(3*d/2)

def f_3(x):
    d = x.shape[1]
    #scale=scale_f_3(d)
    support = support_integrands(d)
    a = 2*np.sin(np.pi*x)**2*np.cos(np.pi*x)
    b = np.cos(np.pi*x - np.pi/4)
    return np.product(a-b, axis=1)*indicator(x, support)

def exact_integral_f_3(d):
    #scale=scale_f_3(d)
    return ((4/3 - np.sqrt(2))/np.pi)**d

# def test_f_3():
#     x = np.array([[2.2, 3.5, 2], [1/2, 0, 0], [1/4, 0, 0]])
#     expected = np.array([[0, -(math.sqrt(2)/2)**3, (math.sqrt(2)/4 -1/2)]])
#     if np.isclose(f_3(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_3(x)=", f_3(x), "expected=", expected)
# test_f_3()


def f_4(x):
    d = x.shape[1]
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    support = BallWindow(center=[0]*d, radius=r)
    norm_x = np.linalg.norm(x, axis=1)
    return np.exp(-norm_x**d)*indicator(x, support)
def exact_integral_f_4(d):
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    kappa_d = UnitBallWindow(center=[0]*d).volume
    return kappa_d*(1 - np.exp(-r**(d)))

# def test_f_4():
#     x = np.array([[1/2, 1/2, 1/2], [0, 0, 0], [1/4, 1/3, 0]])
#     expected = np.array([0, 1, math.exp(- math.sqrt(1/16 + 1/9)**3)])
#     if np.isclose(f_4(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_4(x)=", f_4(x), "expected=", expected)
# test_f_4()

def f_5(x):
    d = x.shape[1]
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    support = BallWindow(center=[0]*d, radius=r)
    norm_x = np.linalg.norm(x, axis=1)
    return (1/(norm_x**d + 0.5**d) - 2**(d-1))*indicator(x, support)
def exact_integral_f_5(d):
    kappa_d = UnitBallWindow(center=[0]*d).volume
    return kappa_d*(math.log(2) - 0.5)

# def test_f_5():
#     x = np.array([[1/2, 0, 0], [1, 0, 0], [1/4, 1/4, 0], [-1/2, 0, 0]])
#     expected = np.array([0, 0, 1/(math.sqrt(8)**(-3) + 0.5**3)-4, 0])
#     if np.isclose(f_5(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_5(x)=", f_5(x), expected)
# test_f_5()

def f_6(x):
    N, d = x.shape
    results = np.zeros((N,))
    support = support_integrands(d)
    for n in range(N):
        results[n] = 1 if x[n,0]<np.min(x[n,1:]) else 0
    return results*indicator(x, support)
def exact_integral_f_6(d):
    return 1/d

# def test_f_6():
#     x = np.array([[1, 1/2, 1/2], [0, 0, 0],
#                   [1/4, 1/3, 1/2], [1/4, 1/3, 1]])
#     expected = np.array([0, 0, 1, 0])
#     if np.isclose(f_6(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_6(x)=", f_6(x), "expected=", expected)
# test_f_6()
