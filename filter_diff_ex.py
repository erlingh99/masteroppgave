import numpy as np
import matplotlib.pyplot as plt

from SE23.lie_theory import SE3_2, SO3
from SE23.models import IMU_Model
from SE23.measurements import IMU_Measurement

init = SE3_2.Exp(np.array([0, -0.3, 0.4 , 0 , 0 ,0, 0 ,0,0]))
T0 = init.as_matrix()
R0 = T0[:3, :3]
v0 = T0[:3, 3]

dt = 2

omega = np.array([0, 0, 1])
acc = np.array([0, 0, 0]) #free fall
z = IMU_Measurement(omega, acc)

imu = IMU_Model(0) #only interested in mean, so cov does not matter
res1 = imu.propegate_mean(T0, z, dt)

g = np.array([0, 0, -9.81])
u = np.array([omega*dt, (acc + R0.T@g)*dt, R0.T@v0*dt + 0.5*(acc + R0.T@g)*(dt**2)]).flatten()
res2 = T0@SE3_2.Exp(u).as_matrix()

# Create figure and axis.
fig = plt.figure()
ax = plt.axes()
ax.set_xlabel('x')
ax.set_ylabel('z')


def plot_pose(ax, T, **kwargs):
    R = T[:3, :3]
    t = T[:3, 4].reshape((3,1))
    alpha = kwargs.get('alpha', 1)
    axis_colors = kwargs.get('axis_colors', ('r', 'g', 'b'))
    scale = kwargs.get('scale', 1)
    text = kwargs.get('text', '')


    # If R is a valid rotation matrix, the columns are the local orthonormal basis vectors in the global frame.
    for i in range(3):
        axis_line = np.column_stack((t, t + R[:, i, np.newaxis] * scale))
        #plot only x,z
        ax.plot(axis_line[0, :], axis_line[2, :], axis_colors[i] + '-', alpha=alpha)

    if text:
        ax.text(t[0, 0], t[2, 0], text)


# # Plot the poses.
plot_pose(ax, T0)
plot_pose(ax, res1)
plot_pose(ax, res2)

diff_h = init.inverse()@SE3_2.from_matrix(res1)
diff_R = diff_h.R.Log()

diff2 = (init.inverse()@SE3_2.from_matrix(res2)).Log()
# Plot the interpolated poses.
for alpha in np.linspace(0, 1, 10):
    ti = T0.copy()
    ti[:3, :3] = ti[:3, :3]@SO3.Exp(alpha*diff_R).as_matrix()
    ti[:3, 3] = T0[:3, 3] + alpha*(res1[:3, 3] - T0[:3, 3])
    ti[:3, 4] = T0[:3, 4] + alpha*(res1[:3, 4] - T0[:3, 4])
    plot_pose(ax, ti, alpha=0.4, scale=1)

    ti = init@SE3_2.Exp(alpha*diff2)
    plot_pose(ax, ti.as_matrix(), alpha=0.4, scale=1)

plt.xlim(-5, 5)
plt.axis("equal")
# plt.axis("off")
plt.show()