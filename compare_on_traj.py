import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement, GNSS_Measurement
from SE23.lie_theory import SE3_2, SO3, SE3, SO3xR3xR3
from SE23.states import PlatformState
from SE23.plot_utils import plot_3d_frame
from SE23.gaussian import MultiVarGauss as mvg

from ESKF.states import EskfState, ImuMeasurement, ErrorState, NominalState, GnssMeasurement
from ESKF.senfuslib.gaussian import MultiVarGauss
from ESKF.eskf import ESKF
from ESKF.models import ModelIMU
from ESKF.quaternion import RotationQuaterion
from ESKF.sensors import SensorGNSS

np.random.seed(100)

N = 30_000
N = min(N, 29999)

g = np.array([0, 0, 9.81])

gt = np.load("./data/example_trajectory.npy", allow_pickle=True).item()
dt = gt["dt"]
p = gt["pos"]
v = gt["vel"]
rot = gt["rot"]
acc = gt["acc"]
gyro = gt["gyro"]

acc_noise = np.array([1e-4, 1e-4, 1e-4])
gyro_noise = np.array([1e-4, 1e-4, 1e-4])
gnss_std = 2
gnss_std_up = 2
GNSS_dt = 5 #in seconds

IMU_noise = np.diag([*gyro_noise, *acc_noise])**2 #imu noise, gyro, acc
GNSS_noise = np.diag([gnss_std, gnss_std, gnss_std_up])**2

#Init poses
T_pred = np.empty(N, dtype=PlatformState)
T_pred2 = np.empty(N, dtype=PlatformState)

init_cov = np.diag([0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0])**2

reorder_mat = np.block([[np.zeros((3, 6)), np.eye(3)],
                        [np.zeros((3,3)), np.eye(3), np.zeros((3,3))],
                        [np.eye(3), np.zeros((3, 6))]])


#true start pos
T0 = SO3xR3xR3(SO3(rot[0]), v[0], p[0])
T02 = SE3_2(SO3(rot[0]), v[0], p[0])

#draw random start perturbation
init_perturbs = multivariate_normal([0]*9, init_cov)
T02 = T02@SE3_2.Exp(init_perturbs)
T0 = SO3xR3xR3.from_matrix(T02.as_matrix())
init_state = PlatformState(T0, init_cov) #perturb our calculation starting point
init_state2 = PlatformState(T02, init_cov)
T_pred[0] = init_state
T_pred2[0] = init_state2

#create the agent
agent = Agent(IMU_noise/dt, GNSS_noise, None, init_state)
agent2 = Agent(IMU_noise, GNSS_noise, None, init_state2)


imu_sim = ModelIMU(accm_std=acc_noise, gyro_std=gyro_noise, g=np.array([0, 0, -9.81]))
x_est_init_nom_sim = NominalState(pos=init_state.pos, vel=init_state.vel, ori=RotationQuaterion.from_matrix(init_state.rot))
x_est_init_err_sim = MultiVarGauss[ErrorState](np.zeros(9), reorder_mat.T@init_cov@reorder_mat)
x_est_init = EskfState(x_est_init_nom_sim, x_est_init_err_sim)
gnssSensor = SensorGNSS(gnss_std, gnss_std_up)
eskf = ESKF(imu_sim, gnssSensor) 

ESKF_pred = np.empty(N, EskfState)
ESKF_pred[0] = x_est_init


###Generate measurments
def generate_IMU_measurement(k):
    gyrom = multivariate_normal(gyro[k], np.diag(gyro_noise**2))
    accm = multivariate_normal(acc[k] + rot[k].T@g, np.diag(acc_noise**2))
    return ImuMeasurement(accm, gyrom), IMU_Measurement(gyrom, accm) #generate IMU measurement at time t

def generate_GNSS_measurement(k):
    gnss = multivariate_normal(p[k], GNSS_noise)
    return GnssMeasurement(gnss), GNSS_Measurement(gnss) #generate IMU measurement at time t

gnss_pos = []

#propagate and simulate
for k in tqdm(range(N - 1)):
    z_imu_eskf, z_agent = generate_IMU_measurement(k)
    agent.propegate(z_agent, dt)
    agent2.propegate(z_agent, dt)
    T_pred[k+1] = agent.state
    T_pred2[k+1] = agent2.state

    x_est_pred = eskf.predict_from_imu(ESKF_pred[k], z_imu_eskf, dt)
    ESKF_pred[k+1] = x_est_pred
    
    if (k*dt)%GNSS_dt == 0 and k > 0: #gnss measurement
        z_gnss_eksf, z_gnss = generate_GNSS_measurement(k)
        gnss_pos.append(z_gnss.pos)
        agent.platform_update(z_gnss)
        agent2.platform_update(z_gnss)
        T_pred[k+1] = agent.state
        T_pred2[k+1] = agent2.state

        x_up, _ = eskf.update_from_gnss(x_est_pred, z_gnss_eksf)
        ESKF_pred[k+1] = x_up


#plotting
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for gp in gnss_pos:
    ax.plot(*gp, "bx")

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



for i in range(500, N+500, 500):
    idx = min(i, N-2)
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

    idx = idx + 1
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






