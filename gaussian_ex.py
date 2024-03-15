from gaussian import MultiVarGauss
import matplotlib.pyplot as plt
import visgeom as vg
import numpy as np

mean = np.array([0, 0, 2.])
Sigma = np.array([[0.1,   0,   0], 
                  [  0,   2, 0.2], 
                  [  0, 0.2, 0.1]])

mvg = MultiVarGauss(mean, Sigma)
x, y, z = mvg.covar_ellipsis()

u, s, _ = np.linalg.svd(Sigma)
scale = np.sqrt(9 * s)
n = 20
x2, y2, z2 = vg.utils.generate_ellipsoid(n, pose=(u, np.zeros([3, 1])), scale=scale)
x2, y2, z2 = x2.flatten(), y2.flatten(), z2.flatten()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(xs=x, ys=y, zs=z)
ax.plot(xs=x2, ys=y2, zs=z2)

ax.plot3D(mvg.mean[0], mvg.mean[1], mvg.mean[2], "rx")
plt.show()