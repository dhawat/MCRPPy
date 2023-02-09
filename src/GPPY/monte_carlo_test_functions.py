from GPPY.spatial_windows import BallWindow, BoxWindow, UnitBallWindow
import math
import numpy as np

def indicator(x, window):
    return window.indicator_function(x)*1

# Support of all integrands
def support_integrands(d):
    return BoxWindow([[-1/2,1/2]]*d)

def f_1(x):
    d = x.shape[1]
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    support = BallWindow(center=[0]*d, radius=r)
    norm_x = np.linalg.norm(x, axis=1)
    return (1/(norm_x**d + 0.5**d) - 2**(d-1))*indicator(x, support)
def exact_integral_f_1(d):
    kappa_d = UnitBallWindow(center=[0]*d).volume
    return kappa_d*(math.log(2) - 0.5)

def cv_proposal_f_1(d):
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    support = BallWindow(center=[0]*d, radius=r)
    proposal = lambda x: (np.linalg.norm(x, axis=1)**d)*indicator(x, support)
    kappa_d = UnitBallWindow(center=[0]*d).volume
    mean_proposal = kappa_d/2*r**(2*d) #mean(g(x))
    return proposal, mean_proposal

# def test_f_1():
#     x = np.array([[1/2, 0, 0], [1, 0, 0], [1/4, 1/4, 0], [-1/2, 0, 0]])
#     expected = np.array([0, 0, 1/(math.sqrt(8)**(-3) + 0.5**3)-4, 0])
#     if np.isclose(f_3(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_1(x)=", f_1(x), expected)
# test_f_1()

def f_2(x):
    d = x.shape[1]
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    support = BallWindow(center=[0]*d, radius=r)
    norm_x = np.linalg.norm(x, axis=1)
    return np.exp(-norm_x**d)*indicator(x, support)
def exact_integral_f_2(d):
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    kappa_d = UnitBallWindow(center=[0]*d).volume
    return kappa_d*(1 - np.exp(-r**(d)))

def cv_proposal_f_2(d):
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    support = BallWindow(center=[0]*d, radius=r)
    proposal = lambda x: 4*np.exp(np.linalg.norm(x, axis=1)**d)*indicator(x, support)
    kappa_d = UnitBallWindow(center=[0]*d).volume
    mean_proposal = 4*kappa_d*(np.exp(r**(d))-1) #mean(g(x))
    return proposal, mean_proposal

# def test_f_2():
#     x = np.array([[1/2, 1/2, 1/2], [0, 0, 0], [1/4, 1/3, 0]])
#     expected = np.array([0, 1, math.exp(- math.sqrt(1/16 + 1/9)**3)])
#     if np.isclose(f_2(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_2(x)=", f_2(x), "expected=", expected)
# test_f_2()

# indicator function
def f_3(x):
    d = x.shape[1]
    #radius of the biggest BallWindow included in support_integrands
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    support = BallWindow(center=[0]*d, radius=r)
    return indicator(x, support)

def exact_integral_f_3(d):
    r= min(np.diff(support_integrands(d).bounds, axis=1))/2
    return BallWindow(center=[0]*d, radius=r).volume

#control variate setup for f_1
def cv_proposal_f_3(d):
    proposal = lambda x: f_2(x) #g(x)
    mean_proposal = exact_integral_f_2(d) #mean(g(x))
    return proposal, mean_proposal

# def test_f_3():
#     x = np.array([[1, 2, 2], [1/2, 0, 0], [1/2, 1/2, 1/2]])
#     if (f_1(x)== np.array([[0, 1, 0]])).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_3(x)=", f_3(x))
# test_f_1()

def f_4(x):
    d = x.shape[1]
    support = support_integrands(d)
    return (1/np.prod(x+1, axis=1)-1)*indicator(x, support)

def exact_integral_f_4(d):
    return math.log(3)**d - 1

# def test_f_4():
#     x = np.array([[0, 0, 0, 0], [1/2, 0, -1/2,-1/2], [1, 0, 0, 0]])
#     expected = np.array([[0, 5/3, 0]])
#     if np.isclose(f_4(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, error=", f_4(x) - expected)
# test_f_4()

#control variate setup for f_4
def cv_proposal_f_4(d):
    support = support_integrands(d)
    proposal = lambda x: np.prod(x+1, axis=1)*indicator(x, support) #g(x)
    mean_proposal = 1 #mean(g(x))
    #c = (math.log(3)**d)/((25/12)**d-1) #Cov(g(x), f(x))/var(g(x))
    return proposal, mean_proposal

def scale_f_5(d):
    return 10**(3*d/2)

def f_5(x):
    d = x.shape[1]
    #scale=scale_f_3(d)
    support = support_integrands(d)
    a = 2*np.sin(np.pi*x)**2*np.cos(np.pi*x)
    b = np.cos(np.pi*x - np.pi/4)
    return np.product(a-b, axis=1)*indicator(x, support)

def exact_integral_f_5(d):
    #scale=scale_f_3(d)
    return ((4/3 - np.sqrt(2))/np.pi)**d

#control variate setup for f_5
def cv_proposal_f_5(d):
    support = support_integrands(d)
    proposal = lambda x: np.prod(np.sin(x)**2*np.cos(x), axis=1)*indicator(x, support) #g(x)
    mean_proposal = (2/3*(np.sin(0.5))**3)**d #mean(g(x))
    #c = (math.log(3)**d)/((25/12)**d-1) #Cov(g(x), f(x))/var(g(x))
    return proposal, mean_proposal

# def test_f_5():
#     x = np.array([[2.2, 3.5, 2], [1/2, 0, 0], [1/4, 0, 0]])
#     expected = np.array([[0, -(math.sqrt(2)/2)**3, (math.sqrt(2)/4 -1/2)]])
#     if np.isclose(f_5(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_5(x)=", f_5(x), "expected=", expected)
# test_f_5()

# def f_8(x):
#     N, d = x.shape
#     results = np.zeros((N,))
#     support = support_integrands(d)
#     for n in range(N):
#         results[n] = 1 if x[n,0]<np.min(x[n,1:]) else 0
#     return results*indicator(x, support)
# def exact_integral_f_8(d):
#     return 1/d

#control variate setup for f_8

# def test_f_8():
#     x = np.array([[1, 1/2, 1/2], [0, 0, 0],
#                   [1/4, 1/3, 1/2], [1/4, 1/3, 1]])
#     expected = np.array([0, 0, 1, 0])
#     if np.isclose(f_8(x), expected, atol=1e-9).all():
#         print("test succeeded")
#     else:
#         print("test failed, f_8(x)=", f_8(x), "expected=", expected)
# test_f_8()


def f_6(x, a=10, c=10):
    _, d = x.shape
    support = support_integrands(d)
    return c*np.prod(np.sin(a*math.pi*x), axis=1)*indicator(x, support)
def exact_integral_f_6(d):
    return 0

def f_7(x, a=10, c=10):
    _, d = x.shape
    support = support_integrands(d)
    return c*np.prod(np.cos(a*math.pi*x), axis=1)*indicator(x, support)
def exact_integral_f_7(d):
    return 0

def f_8(x, a=15, c=10):
    return f_7(x, a,c)
def exact_integral_f_8(d, a=15,c=10):
    return -c*(2/(math.pi*a))**d
