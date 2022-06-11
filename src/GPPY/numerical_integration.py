import numpy as np

def monte_carlo_integration(f, points):
    points_nb = points.shape[0]
    return np.sum(f(points))/points_nb
