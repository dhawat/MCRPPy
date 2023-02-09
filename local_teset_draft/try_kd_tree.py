import numpy as np
import timeit
from scipy.spatial import KDTree
from structure_factor.spatial_windows import BallWindow
from GPPY.spatial_windows import AnnulusWindow

d=2
N=int(1e6)
L=200
r=4
centers = 4*(np.random.rand(50, d) - 0.5)
points = L*(np.random.rand(N, d)- 0.5)
#test in annulus
start = timeit.default_timer()
for c in centers:
    window = BallWindow(center=c, radius=r)
    points_in = window.indicator_function(points)

time_end = timeit.default_timer() - start
print("time=", int(time_end/60), "min", time_end%60, "s")
#test using kd_tree
start_2 = timeit.default_timer()
tree = KDTree(points)
for c in centers:
    points_in_2 =sorted(tree.query_ball_point(c, r))
time_end_2 = timeit.default_timer() - start_2
print("time=", int(time_end_2/60), "min", time_end_2%60, "s")

import matplotlib.pyplot as plt

fig, axis = plt.subplots(figsize=(4,4))
plt.plot(centers[-1,0], centers[-1,1], 'ko')
plt.plot(points[points_in][:,0], points[points_in][:,1], 'b*')
plt.plot(points[points_in_2][:,0], points[points_in_2][:,1], 'r.')
window.plot(axis=axis)
plt.show()
