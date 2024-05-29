import matplotlib.pyplot as plt
import numpy as np

from SE23.gaussian import ExponentialGaussian
from SE23.lie_theory import SE3_2
from SE23.models import IMU_Model
from SE23.measurements import IMU_Measurement
from SE23.plot_utils import plot_3d_frame

mean = SE3_2.Exp([0, 0, 0, 0, 0, 0, 0, 0, 0]).as_matrix()
cov = np.diag([0, 0, 0.2, 0.3, 0, 0, 2, 6, 0])

dt = 0.05
imu_noise_cov = np.diag([0, 0.01, 0.2, 1, 1, 1])**2
imu = IMU_Model(imu_noise_cov)
z = IMU_Measurement(np.array([0,0,0]), np.array([1, 0, 9.81]))

for _ in range(300):
    mean = imu.propegate_mean(mean, z, dt)
    cov = imu.propegate_cov(cov, z, dt)

mean = SE3_2.from_matrix(mean)
exp = ExponentialGaussian(mean, cov)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
exp.draw_significant_spheres(ax, n_spheres=4, n_points=20)
plot_3d_frame(ax)
plt.axis("equal")
plt.show()