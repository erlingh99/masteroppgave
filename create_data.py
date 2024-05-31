from scipy.io import loadmat
import numpy as np
from SE23.quaternion import RotationQuaternion
from SE23.lie_theory import SE3, SO3
from tqdm import tqdm

import matplotlib.pyplot as plt
from SE23.plot_utils import plot_3d_frame

path = "./data/true_state.mat"

arr = loadmat(path)

dt = 0.01
N = 30_000
gt = arr["x_true"][:, :N]

ned_convert = np.diag([1, -1, -1])

pos = (ned_convert@gt[:3, :]).T
vel = (ned_convert@gt[3:6, :]).T
q = gt[6:, :]

R = np.zeros((N, 3, 3))
omega = np.zeros_like(pos)
acc = np.zeros_like(pos)
pos_ca = np.zeros_like(pos)
vel_ca = np.zeros_like(pos)
R_ca = np.zeros_like(R)

pos_ca[0] = pos[0]
vel_ca[0] = vel[0]
R_ca[0] = R[0] = ned_convert@RotationQuaternion(q[0, 0], q[1:, 0]).as_rotation_matrix()@ned_convert.T

for i in tqdm(range(1, N)):
    R[i] = ned_convert@RotationQuaternion(q[0, i], q[1:, i]).as_rotation_matrix()@ned_convert.T

    omega[i-1] = SO3(R[i-1].T@R[i]).Log()/dt
    acc[i-1] = R[i-1].T@(vel[i] - vel[i-1]) / dt

    R_ca[i] = R_ca[i-1]@SO3.Exp(omega[i-1]*dt).as_matrix()
    J = SO3.jac_left(omega[i-1]*dt)
    vel_ca[i] = vel_ca[i-1] + R_ca[i-1]@J@acc[i-1]*dt
    pos_ca[i] = pos_ca[i-1] + dt*vel_ca[i-1] + R_ca[i-1]@J@acc[i-1]*dt*dt/2


print(np.linalg.norm(pos - pos_ca, axis=1).mean())

save_dict = {'pos': pos_ca, 'vel': vel_ca, 'rot': R_ca, 'acc': acc, 'gyro': omega, 'dt': dt}
np.save("./data/example_trajectory.npy", save_dict)


ax = plt.axes(projection='3d', xlabel='north [m]', ylabel='west [m]', zlabel='up [m]')

ax.plot(xs=pos[:N, 0], ys=pos[:N, 1], zs=pos[:N, 2])
ax.plot(xs=pos_ca[:N, 0], ys=pos_ca[:N, 1], zs=pos_ca[:N, 2])
for i in range(0, N, 1000):
    plot_3d_frame(ax, SE3(SO3(R[i]), pos[i, :]), scale=50)
    plot_3d_frame(ax, SE3(SO3(R_ca[i]), pos_ca[i, :]), scale=50)

plt.axis("equal")
plt.show()