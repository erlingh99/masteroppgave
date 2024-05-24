from lie_theory import SE3_2, SE2
import numpy as np
from gaussian import ExponentialGaussian
import matplotlib.pyplot as plt
from plot_utils import plot_2d_frame, plot_as_SE2
from models import IMU_Model, IMU_Measurement


mean = SE3_2.Exp([0, 0, 0, 0, 0, 0, 0, 0, 0]).as_matrix()
cov = np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0])
cov4=cov2=cov

imu_noise_cov = np.diag([0, 0, 0.5, 0, 0, 0])**2
imu = IMU_Model(imu_noise_cov)

dt = 0.05

z = IMU_Measurement(np.array([0,0,0]), np.array([1, 0, 9.81]))

for _ in range(300):
    mean = imu.propegate_mean(mean, z, dt)
    cov4 = imu.propegate_cov(cov4, z, dt)
    # _, cov2 = imu.propegate_cov(cov2, z, dt)

dist = ExponentialGaussian(SE3_2.from_matrix(mean), cov2)
dist2 = ExponentialGaussian(SE3_2.from_matrix(mean), cov4)

fig, ax = plt.subplots(1, 2)
plot_as_SE2(ax[0], dist2, scale=1, color="blue", n_points=50)
plot_as_SE2(ax[1], dist, scale=1, color="green", n_points=50)

# dist.draw_significant_ellipses(ax, n_std=1)
# dist.draw_significant_ellipses(ax, n_std=2)
# dist.draw_significant_ellipses(ax, n_std=3)
# dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=1)
# dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=2)
# dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=3)
# dist.draw_significant_ellipses(ax, n_points=50, color="blue")
dist2.draw_significant_ellipses(ax[0], n_points=50, color="red")
dist.draw_significant_ellipses(ax[1], n_points=50, color="red")


# plot_as_SE2(ax, dist, scale=1)
# plot_2d_frame(ax, dist.mean)
# plot_2d_frame(ax[0], SE2.Exp([0, 0, 0]))

# mean = SE2.Exp([3*np.pi/4, 3, 3])
# cov = np.array([[0.4, 0.2, 0],
#                 [0.2, 0.2, 0.1],
#                 [0, 0.1, 0.4]])

# dist = ExponentialGaussian(mean, cov)
# dist.draw_significant_ellipses(ax, n_std=3, n_ellipsis=3, n_points=200)
# dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=1)

# ts = dist.project_translation(n_points=50)
# ax.plot(ts[:, 0], ts[:, 1])

ax[0].axis("equal")
ax[1].axis("equal")
plt.show()