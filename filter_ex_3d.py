import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement
from SE23.lie_theory import SE3_2, SO3, SO3xR3xR3
from SE23.states import PlatformState
from SE23.utils import find_mean, exp_cov
from SE23.plot_utils import *

np.random.seed(42)

n_steps = 300
n_random = 100 #number of simulations
dt = 0.05 #sec per step, imu rate
T = (n_steps-1)*dt #end time

g = np.array([0, 0, 9.81])

#define ground truth (world)
# p = lambda t: np.array([t**2, 5*np.sin(t), 0])
# v = lambda t: np.array([2*t, 5*np.cos(t), 0])
# a = lambda t: np.array([2, -5*np.sin(t), 0])
# w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
# Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])
p = lambda t: np.array([0.5*t**2, -4*np.cos(0.5*t)+4, -np.sin(t)])
v = lambda t: np.array([t, 2*np.sin(0.5*t), -np.cos(t)])
a = lambda t: np.array([1, np.cos(0.5*t), np.sin(t)])
w = lambda t: np.array([0, 0, 1])
Rot = lambda t: SO3.Exp([0, 0, t])


##Agent setup, noises
IMU_noise = np.diag([0, 0, 0.03/dt, 0, 0, 0.3])**2 #imu noise, gyro, acc

#Init poses
T_sim = np.empty((n_steps, n_random, 5, 5))
T_pred = np.empty(n_steps, dtype=PlatformState)
T_pred_2 = np.empty(n_steps, dtype=PlatformState)

init_cov = np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0])**2

T0 = SE3_2(Rot(0), v(0), p(0)) #true start pos
T0_2 = SO3xR3xR3(Rot(0), v(0), p(0)) #true start pos

#draw random start pertrbations
init_perturbs = multivariate_normal([0]*9, init_cov, n_random) #random perturbations
for i in range(n_random):
    T_sim[0, i] = (T0@SE3_2.Exp(init_perturbs[i])).as_matrix()

init_state = PlatformState(T0, init_cov) #also perturb our calculation starting point
init_state_2 = PlatformState(T0_2, init_cov) #also perturb our calculation starting point
T_pred[0] = init_state #save the initial state
T_pred_2[0] = init_state_2 #save the initial state

#create the agent
agent = Agent(IMU_noise, None, None, init_state)
agent_2 = Agent(IMU_noise, None, None, init_state_2)

###Generate measurments
#imu measurements
gyro = lambda t, n: multivariate_normal(w(t), IMU_noise[:3, :3], size=n) #generate n gyro measurements at time t
acc = lambda t, n: multivariate_normal(Rot(t).T@(a(t) + g), IMU_noise[3:, 3:], size=n) #imu measures g up, acc is in body
generate_IMU_measurements = lambda t, n: [IMU_Measurement(g_t, acc_t) for g_t, acc_t in zip(gyro(t, n), acc(t, n))] #generate n IMU measurments at time t

#propegate and simulate
for k in tqdm(range(n_steps - 1)):
    z_imu = generate_IMU_measurements(k*dt, n_random) #sample n_random different inputs
    z_true = IMU_Measurement(w(k*dt), Rot(k*dt).T@(a(k*dt) + g))
    agent.propegate(z_true, dt) #use one of the random inputs in the full filter
    agent_2.propegate(z_true, dt) #use one of the random inputs in the full filter
    T_pred[k+1] = agent.state #save the current state of the agent
    T_pred_2[k+1] = agent_2.state #save the current state of the agent
    for i in range(n_random):
        T_sim[k+1, i] = agent.inertialNavigation.model.propegate_mean(T_sim[k, i], z_imu[i], dt) #propegate the simulations with the corresponding random input


#calculate SE3_2 distribution of simulated samples
final_pose = np.empty(n_random, dtype=SE3_2) 
for i in range(n_random):
    final_pose[i] = SE3_2(SO3(T_sim[-1, i, :3, :3]), T_sim[-1, i, :3, 3], T_sim[-1, i, :3, 4])

mean = find_mean(final_pose, T_pred[-1].mean) #find mean, with the predicted mean as initial guess
sim_cov = exp_cov(final_pose, mean)
sim_pose = PlatformState(mean, sim_cov)

#calculate SO3xR3xR3 distribution of simulated samples
final_pose2 = np.empty(n_random, dtype=SO3xR3xR3) 
for i in range(n_random):
    final_pose2[i] = SO3xR3xR3(SO3(T_sim[-1, i, :3, :3]), T_sim[-1, i, :3, 3], T_sim[-1, i, :3, 4])

mean2 = find_mean(final_pose2, T_pred_2[-1].mean) #find mean, with the predicted mean as initial guess
sim_cov2 = exp_cov(final_pose2, mean2)
sim_pose2 = PlatformState(mean2, sim_cov2)


#plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


sim_pose.draw_significant_ellipses(ax, color="yellow")
sim_pose2.draw_significant_ellipses(ax, color="orange")
for i in range(n_random):
    ax.plot(T_sim[:, i, 0, 4], T_sim[:, i, 1, 4], T_sim[:, i, 2, 4], color='gray', alpha=0.1) 
ax.scatter(T_sim[-1, :, 0, 4], T_sim[-1, :, 1, 4], T_sim[-1, :, 2, 4], s=2, color='black')


x = np.empty((n_steps, 3))
x2 = np.empty((n_steps, 3))
for i in range(n_steps):
    x[i] = T_pred[i].mean.p
    x2[i] = T_pred_2[i].mean.p
ax.plot(*x.T, "g--", alpha=1)
ax.plot(*x2.T, "b--", alpha=1)

plot_3d_frame(ax, scale=5)
for i in range(100, n_steps+1, 50):
    idx = min(i, n_steps-1)
    T_pred[idx].draw_significant_ellipses(ax, color="green")
    plot_3d_frame(ax, SE3(T_pred[idx].rot, T_pred[idx].pos), scale=5)
    T_pred_2[idx].draw_significant_ellipses(ax, color="blue")
    plot_3d_frame(ax, SE3(T_pred_2[idx].rot, T_pred_2[idx].pos), scale=5)

plot_3d_frame(ax, SE3(Rot(T), p(T)), scale=5)

#plot gt
ts = np.linspace(0, T, n_steps)
ps = np.empty((n_steps, 3))
for i, tsi in enumerate(ts):
    ps[i] = p(tsi)
ax.plot(*ps.T, "r--")
ax.plot(*ps[-1], "ro")
plt.axis("equal")
plt.show()


print(np.linalg.norm(sim_cov - T_pred[-1].cov, ord="fro"))
print(np.linalg.norm(sim_cov2 - T_pred_2[-1].cov, ord="fro"))