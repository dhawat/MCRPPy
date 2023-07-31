from mcrppy.point_pattern import PointPattern
import numpy as np
import math
from mcrppy.coulomb_force import force_base
import matplotlib.pyplot as plt
from mcrppy.repelled_point_process import epsilon_critical

class GravitationalAllocation:
    def __init__(self, point_pattern):
        assert isinstance(point_pattern, PointPattern)
        self.point_pattern = point_pattern
        self.point_pattern.intensity = self.estimated_intensity

    @property
    def estimated_intensity(self):
        return self.point_pattern.points.shape[0]/self.point_pattern.window.volume
    @property
    def allocation_basin_volume(self):
        """Volume of each basin of attraction of the gravitational allocation"""
        return 1 / self.point_pattern.intensity

    def _starting_points_2d(self, nb_points=40, scale=0.02):
        #points will be diplayed on circles around each centers
        #The radius of the circle will be propostionel to basins volume and controle by `scale`
        #The number of angles on which the points will be placed is specidied by nb_points
        #output: list of 2d-arrays. each 2d-array contains teh starting points of a corresponding center
        centers = np.atleast_2d(self.point_pattern.points) #allocation centers
        intensity = self.point_pattern.intensity
        #setup for the points position
        theta = np.linspace(0, 1, nb_points)*2*math.pi #angles
        theta = theta.tolist()
        radius = 1/intensity*scale #distance from the centers; prop to the basin volume

        points = [centers[i,:] + radius*np.array([np.cos(theta), np.sin(theta)]).T for i in range(centers.shape[0])]
        return points

    def _trajectory(self, point, stoptime=500, stepsize=None):
        #trajectoy of the point `point` by (inverse) gravitational allocation from the point pattern
        #discritization scheme of the trajactory with fix step size `stepsize` and a stoping time `stoptime`.
        #return 2d array of the trajectories
        centers = self.point_pattern.points
        intensity = self.point_pattern.intensity
        if stepsize is None:
            d = centers.shape[1]
            stepsize = epsilon_critical(d, intensity)/100 #default steptize
        point_trajectory= [point]
        for t in range(stoptime):
            point_trajectory.append(point_trajectory[t] + stepsize*force_base(point_trajectory[t], centers, intensity=intensity) )
        return np.vstack(point_trajectory)

    def trajectory(self, points, stoptime=500, stepsize=None):
        #points shoud be (n,d) array
        #return list of arrays; trajectories of points
        #positions of points w.r.t. time
        nb_points = points.shape[0]
        trajectories = [self._trajectory(points[i],stoptime=stoptime, stepsize=stepsize) for i in range(nb_points)]
        return trajectories

    def plot_2D(self, axis=None, file_name="", nb_points=30, scale=0.02, stepsize=0.001, stoptime=800, size_centers=5, rasterized=True, label_centers="centers", linewidth_trajectory=0.5, end_trajectories=False):
        centers = self.point_pattern.points
        if axis is None:
            _, axis = plt.subplots(figsize=(4, 4))
        starting_points = self._starting_points_2d(nb_points, scale) #starting points
        for i in range(len(starting_points)):
            points_i = starting_points[i] #starting points for center_i
            ##trajectories of points
            trajectories_i = self.trajectory(points=points_i,stoptime=stoptime, stepsize=stepsize)
            color_i = np.random.rand(3,)*0.7 #color of basin_i
            for j in range(len(trajectories_i)):
                axis.plot(trajectories_i[j][:,0], trajectories_i[j][:,1], c=color_i, linewidth=linewidth_trajectory, zorder=0, rasterized=rasterized)#trajectories_i
                if end_trajectories:
                    axis.scatter(trajectories_i[j][-1, 0], trajectories_i[j][-1, 1], color="b", s=2, zorder=10) #end points
        axis.scatter(centers[:,0], centers[:,1], color="k", s=size_centers, label=label_centers, zorder=5) #allocation centers
        axis.set_xticks([])
        axis.set_yticks([])
        plt.legend(loc="upper right")
        plt.tight_layout()
        if file_name:
            fig = axis.get_figure()
            fig.savefig(file_name, bbox_inches="tight")
        plt.show()
        #return axis
