import numpy as np
import matplotlib.pyplot as plt

def _plot_proposal(f, proposal, dim=2):
    x = np.linspace(-1/2,1/2, 100)
    if dim == 2:
        X, Y = np.meshgrid(x, x)
        z = np.array([X.ravel(), Y.ravel()]).T
        fig = plt.figure(figsize=(14, 4))
        ax = fig.add_subplot(2, 6, 1, projection='3d')
        ax.set_title(r"$f$")
        ax.scatter3D(X.ravel(), Y.ravel(), f(z), c=f(z))
        ax = fig.add_subplot(2, 6, 2, projection='3d')
        ax.scatter3D(X.ravel(), Y.ravel(), proposal(z), c=proposal(z))
        ax.set_title("Control variate proposal")
        plt.show()
    else:
        raise ValueError("Actually, only available for 2D")
