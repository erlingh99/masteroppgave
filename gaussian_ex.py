from gaussian import MultiVarGauss
import matplotlib.pyplot as plt
import visgeom as vg
import numpy as np

mean = np.array([3*np.pi/4, 3, 3])
Sigma = np.array([[0.4, 0, 0],
                [0, 0.2, 0.1],
                [0, 0.1, 0.4]])

mvg = MultiVarGauss(mean, Sigma)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x, y, z = mvg.covar_ellipsis()
ax.plot(xs=x, ys=y, zs=z)
ax.plot_surface(x.reshape(21,21), y.reshape(21,21), z.reshape(21,21), alpha=0.5)
mvg.draw_significant_ellipses(ax)

plt.axis("equal")
plt.show()