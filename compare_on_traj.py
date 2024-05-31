import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement
from SE23.lie_theory import SE3_2, SO3, SE3, SO3xR3xR3
from SE23.states import PlatformState
from SE23.plot_utils import plot_3d_frame
from SE23.gaussian import MultiVarGauss as mvg

from ESKF.states import EskfState, ImuMeasurement, ErrorState, NominalState
from ESKF.senfuslib.gaussian import MultiVarGauss
from ESKF.eskf import ESKF
from ESKF.models import ModelIMU
from ESKF.quaternion import RotationQuaterion


np.random.seed(42)

N = 2_000
N = min(N, 29999)

g = np.array([0, 0, 9.81])

gt = np.load("./data/example_trajectory.npy", allow_pickle=True).item()
dt = gt["dt"]
p = gt["pos"]
v = gt["vel"]
rot = gt["rot"]
acc = gt["acc"]
gyro = gt["gyro"]

acc_noise = np.array([3e-4, 3e-4, 3e-4])
gyro_noise = np.array([3e-4, 3e-4, 3e-4])


##Agent setup, noises
IMU_noise = np.diag([*gyro_noise, *acc_noise])**2 #imu noise, gyro, acc

#Init poses
T_pred = np.empty(N, dtype=PlatformState)
T_pred2 = np.empty(N, dtype=PlatformState)

init_cov = np.diag([0.1, 0.1, 0.2, 0, 0, 0, 0, 0, 0])**2

reorder_mat = np.block([[np.zeros((3, 6)), np.eye(3)],
                        [np.zeros((3,3)), np.eye(3), np.zeros((3,3))],
                        [np.eye(3), np.zeros((3, 6))]])




T0 = SO3xR3xR3(SO3(rot[0]), v[0], p[0]) #true start pos
T02 = SE3_2(SO3(rot[0]), v[0], p[0]) #true start pos

#draw random start perturbation
init_perturbs = multivariate_normal([0]*9, init_cov)
T02 = T02@SE3_2.Exp(init_perturbs)
T0 = SO3xR3xR3.from_matrix(T02.as_matrix())
init_state = PlatformState(T0, init_cov) #perturb our calculation starting point
init_state2 = PlatformState(T02, init_cov)
T_pred[0] = init_state
T_pred2[0] = init_state2

#create the agent
agent = Agent(IMU_noise, None, None, init_state)
agent2 = Agent(IMU_noise, None, None, init_state2)


imu_sim = ModelIMU(accm_std=acc_noise*dt**0.5, gyro_std=gyro_noise*dt**0.5)
x_est_init_nom_sim = NominalState(pos=init_state.pos, vel=init_state.vel, ori=RotationQuaterion.from_matrix(init_state.rot))
x_est_init_err_sim = MultiVarGauss[ErrorState](np.zeros(9), reorder_mat.T@init_cov@reorder_mat)
x_est_init = EskfState(x_est_init_nom_sim, x_est_init_err_sim)
eskf = ESKF(imu_sim, None) 

ESKF_pred = np.empty(N, EskfState)
ESKF_pred[0] = x_est_init


###Generate measurments
def generate_IMU_measurement(k):
    gyrom = multivariate_normal(gyro[k], np.diag(gyro_noise**2))
    accm = multivariate_normal(acc[k] + rot[k].T@g, np.diag(acc_noise**2))
    return ImuMeasurement(accm, gyrom), IMU_Measurement(gyrom, accm) #generate IMU measurement at time t


#propagate and simulate
for k in tqdm(range(N - 1)):
    z_imu_eskf, z_agent = generate_IMU_measurement(k)
    agent.propegate(z_agent, dt)
    agent2.propegate(z_agent, dt)
    T_pred[k+1] = agent.state
    T_pred2[k+1] = agent2.state

    x_est_pred = eskf.predict_from_imu(ESKF_pred[k], z_imu_eskf, dt)
    ESKF_pred[k+1] = x_est_pred


#plotting
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

x = np.empty((N, 3))
x2 = np.empty((N, 3))
x3 = np.empty((N, 3))
for i in range(N):
    x[i] = T_pred[i].mean.p
    x2[i] = T_pred2[i].mean.p
    x3[i] = ESKF_pred[i].nom.pos
ax.plot(*x[:].T, "g--", alpha=1)
ax.plot(*x2[:].T, "r--", alpha=1)
ax.plot(*x3[:].T, "b--", alpha=1)

#x2 er SE23
print(np.linalg.norm(x-x2))
print(np.linalg.norm(x2-x3))
print(np.linalg.norm(x-x3))

print(np.linalg.norm(p[:N] - x2))

cov = reorder_mat@ESKF_pred[-1].err.cov@reorder_mat.T
print(np.linalg.norm(T_pred[-1].cov - cov, ord="fro"))
print(np.linalg.norm(T_pred2[-1].cov - cov, ord="fro"))
print(np.linalg.norm(T_pred[-1].cov - T_pred2[-1].cov, ord="fro"))



for i in range(50, N+51, 400):
    idx = min(i, N-1)
    T_pred[idx].draw_significant_ellipses(ax, color="green")
    # T_pred[idx].draw_significant_spheres(ax, n_spheres=1, color="green")
    plot_3d_frame(ax, SE3(T_pred[idx].mean.R, T_pred[idx].mean.p), scale=10)

    T_pred2[idx].draw_significant_ellipses(ax, color="red")
    # T_pred2[idx].draw_significant_spheres(ax, n_spheres=1, color="red")
    plot_3d_frame(ax, SE3(T_pred2[idx].mean.R, T_pred2[idx].mean.p), scale=10)
    
    eskf_R = SO3(ESKF_pred[idx].nom.ori.as_rotmat())
    eskf_p = ESKF_pred[idx].nom.pos
    cov = reorder_mat@ESKF_pred[idx].err.cov@reorder_mat.T
    dist = mvg(eskf_p, cov[-3:, -3:])

    dist.draw_significant_ellipses(ax, color="blue")
    # dist.draw_significant_spheres(ax, n_spheres=1, color="blue")
    plot_3d_frame(ax, SE3(eskf_R, eskf_p), scale=10)
    
plot_3d_frame(ax, SE3(SO3(rot[N]), p[N]), scale=5)
plot_3d_frame(ax, SE3(SO3(rot[0]), p[0]), scale=5)
plot_3d_frame(ax, SE3(init_state.rot, init_state.pos), scale=5)

#plot gt
ax.plot(*p[:N].T, "k--")
ax.plot(*p[N].T, "ko")
plt.axis("equal")
plt.show()






