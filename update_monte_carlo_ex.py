import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement, GNSS_Measurement
from SE23.lie_theory import SE3_2, SO3, SE2, SO2, SO3xR3xR3
from SE23.states import PlatformState
from SE23.utils import find_mean, exp_cov
from SE23.plot_utils import *

# np.random.seed(10)


n_steps = 300
n_random = 100 #number of simulations
dt = 0.05 #sec per step, imu rate
T = (n_steps-1)*dt #end time

g = np.array([0, 0, 9.81])

#define ground truth (world)
p = lambda t: np.array([0.5*t**2+2*t, 0, 0])
v = lambda t: np.array([t+2, 0, 0])
a = lambda t: np.array([1, 0, 0])
w = lambda t: np.array([0, 0, 0])
Rot = lambda t: SO3.Exp([0, 0, 0])
# p = lambda t: np.array([t**2, 5*np.sin(t), 0])
# v = lambda t: np.array([2*t, 5*np.cos(t), 0])
# a = lambda t: np.array([2, -5*np.sin(t), 0])
# w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
# Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])


##Agent setup, noises
init_cov = np.diag([0, 0, 0.1, 0, 0, 0, 0.1, 0.1, 0])**2
IMU_noise = np.diag([0, 0, 0.03/dt, 0.4, 0.4, 0])**2 #imu noise, gyro, acc
GNSS_noise = np.diag([2, 2, 0.001])**2
radar_noise = 0 # not used in this example

#create array to save the state at each timestep
agent_state = np.empty(n_steps, dtype=PlatformState)
agent_state_ESKF = np.empty(n_steps, dtype=PlatformState)
T_sim = np.zeros((n_steps, n_random, 5, 5))
T_sim2 = np.zeros((n_steps, n_random, 5, 5))

#Init poses
T0 = SE3_2(Rot(0), v(0), p(0)) #true start pos
T0_ESKF = SO3xR3xR3(Rot(0), v(0), p(0)) #true start pos

#draw random start pertrbations and create agents
init_perturbs = multivariate_normal([0]*9, init_cov, n_random) #random perturbations
for i in range(n_random):
    T_sim[0, i] = (T0@SE3_2.Exp(init_perturbs[i])).as_matrix()
    T_sim2[0, i] = (T0_ESKF@SO3xR3xR3.Exp(init_perturbs[i])).as_matrix()


agent = Agent(IMU_noise, GNSS_noise, radar_noise, PlatformState(T0, init_cov)) 
agent_ESKF = Agent(IMU_noise, GNSS_noise, radar_noise, PlatformState(T0_ESKF, init_cov)) 
agent_state[0] = PlatformState(T0, init_cov)
agent_state_ESKF[0] = PlatformState(T0_ESKF, init_cov)


###Generate measurments
#imu measurements
gyro = lambda t, n: multivariate_normal(w(t), IMU_noise[:3, :3], size=n) #generate n gyro measurements at time t
acc = lambda t, n: multivariate_normal(Rot(t).T@(a(t) + g), IMU_noise[3:, 3:], size=n) #imu measures g up, acc is in body
generate_IMU_measurements = lambda t, n: [IMU_Measurement(g_t, acc_t) for g_t, acc_t in zip(gyro(t, n), acc(t, n))] #generate n IMU measurments at time t
#GNSS measurements
pos_m = lambda t: multivariate_normal(p(t), GNSS_noise) #measured position
gnssMeasurement = lambda t: GNSS_Measurement(pos_m(t)) #create gnss measurement with noise

ws = np.empty(n_random)

#propegate and simulate
for k in tqdm(range(n_steps-1)):
    z_imu = generate_IMU_measurements(k*dt, n_random) #sample n_random different inputs
    z_true = IMU_Measurement(w(k*dt), Rot(k*dt).T@(a(k*dt) + g))
    #propegate all agents with the corresponding random IMU measurement, save the state
    agent.propegate(z_true, dt)
    agent_ESKF.propegate(z_true, dt)
    agent_state[k+1] = agent.state.copy()
    agent_state_ESKF[k+1] = agent_ESKF.state.copy()
    
    for i in range(n_random):
        T_sim[k+1, i] = agent.inertialNavigation.model.propegate_mean(T_sim[k, i], z_imu[i], dt, cls=SE3_2) 
        T_sim2[k+1, i] = agent.inertialNavigation.model.propegate_mean(T_sim2[k, i], z_imu[i], dt, cls=SO3xR3xR3)


z_gps = gnssMeasurement(T)
for i in range(n_random):
    #prob agent i
    m = z_gps.pos - T_sim[-1, i, :3, 4]
    ws[i] = np.exp(-0.5*m.T@np.linalg.solve(GNSS_noise, m))

    agent.platform_update(z_gps)
    agent_ESKF.platform_update(z_gps)


# calculate SE3_2 distribution of simulated samples
final_means = np.array([SE3_2.from_matrix(s) for s in T_sim[-1]])
#calculate the eperical mean
mean = find_mean(final_means, agent.state.mean, weights=ws) #find mean, with the predicted mean of one of the agents as initial guess
sim_cov = exp_cov(final_means, mean, weights=ws)
sim_pose = PlatformState(mean, sim_cov)

# calculate SO3xR3xR3 distribution of simulated samples
final_means_ESKF = np.array([SO3xR3xR3.from_matrix(s) for s in T_sim2[-1]])
#calculate the eperical mean
mean_ESKF = find_mean(final_means_ESKF, agent_ESKF.state.mean, weights=ws) #find mean, with the predicted mean of one of the agents as initial guess
sim_cov_ESKF = exp_cov(final_means_ESKF, mean_ESKF, weights=ws)
sim_pose_ESKF = PlatformState(mean_ESKF, sim_cov_ESKF)


#plotting
fig = plt.figure()
ax = fig.add_subplot(111)


plot_as_SE2(ax, sim_pose, color="yellow")
# plot_as_SO2xR2(ax, sim_pose_ESKF, color="orange")
for i in range(n_random):
    t = np.empty((n_steps, 2))
    for k in range(n_steps):
        t[k, :] = T_sim[k, i, :2, 4]
    ax.plot(*t.T, color='gray', alpha=0.1)
    ax.scatter(*final_means[i].p[:2] , s=2, color='black')
ax.plot(*z_gps.pos[:2], 'bx')

plot_as_SE2(ax, agent.state, color="red")
plot_as_SE2(ax, agent_state[-1], color="green")
# plot_as_SO2xR2(ax, agent_ESKF.state color="pink")
plot_2d_frame(ax, SE2(SO2(Rot(T).as_matrix()[:2, :2]), p(T)[:2]), scale=5)



#plot gt
ts = np.linspace(0, T, 101)
xs = []
ys = []
for tsi in ts:
    xsi, ysi, _ = p(tsi)
    xs.append(xsi)
    ys.append(ysi)
ax.plot(xs, ys, "r--")
ax.plot(xs[-1], ys[-1], "ro")
plt.axis("equal")
plt.show()