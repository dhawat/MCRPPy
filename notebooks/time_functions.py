# Creat a poisson point process
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BallWindow, BoxWindow
from GPPY.gravity_point_process import GravityPointProcess

simu_window = BoxWindow([[-20, 20], [-20,20]])
volume_simu_window = simu_window.volume
support_window = BallWindow(center=[0,0], radius=20)
f = lambda x: np.sum(support_window.indicator_function(x))
exact_integration = support_window.volume
samples_nb = 10
intensity = [1]
mc_std_poisson, mc_std_push = [], []
for i in intensity:
    print(i, i*volume_simu_window)
    process = HomogeneousPoissonPointProcess(i)
    poisson = [process.generate_sample(window=simu_window) for _ in range(samples_nb)]
    poisson_pattern = [PointPattern(p, simu_window, i) for p in poisson]
    gpp_process = [GravityPointProcess(p) for p in poisson_pattern]
    volume_basin = gpp_process[0].allocation_basin_volume
    if __name__ == "__main__":
        push = [g.pushed_point_process(epsilon=volume_basin/100, stop_time=9) for g in gpp_process]
    mc_poisson = [f(p)/p.shape[0]*volume_simu_window for p in poisson]
    mc_push = [f(p)/p.shape[0]*volume_simu_window for p in push]
    mc_std_poisson.append(stat.stdev(mc_poisson))
    mc_std_push.append(stat.stdev(mc_push))
