import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from SE23.agent import Agent
# from SE23.target import TargetBody2 as TargetBody
from SE23.target import TargetBody

from SE23.lie_theory import SE3_2, SO3, SE2, SO2
from SE23.states import PlatformState, TargetState
from SE23.measurements import IMU_Measurement, TargetMeasurement, GNSS_Measurement
from SE23.plot_utils import *

np.random.seed(42)

#meta vars
n_times = 10
n_steps = 100 #this is done n_times times
#imu rate
dt = 0.01 #sec per step
T = (n_times*n_steps-1)*dt

g = np.array([0, 0, 9.81])
#define ground truth of platform
p = lambda t: np.array([t**2+3*t, 5*np.sin(t)+3*t, 0])
v = lambda t: np.array([2*t+3, 5*np.cos(t)+3, 0])
a = lambda t: np.array([2, -5*np.sin(t), 0])
w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])

#define ground truth of target
# pt = lambda t: np.array([200*np.sin(t/8), 400*np.cos(t/8) - 100, 0])
# vt = lambda t: np.array([25*np.cos(t/8), -50*np.sin(t/8), 0])
pt_0 = np.array([0, 100, 0])
vt_0 = np.array([30, 0, 0])

#spawn agent
IMU_cov = np.diag([0, 0, 0.2, 0, 0, 0])**2
GNSS_cov = np.diag([2, 2, 0.001])**2
radar_cov = np.diag([5, 5, 0.001])**2

init_agent_cov = np.diag([0, 0, 0.2, 0, 0, 0, 0, 0, 0])**2
init_agent_gt = SE3_2(Rot(0), v(0), p(0))
init_agent_mean = init_agent_gt@SE3_2.Exp(multivariate_normal([0]*9, init_agent_cov))
init_agent_pose = PlatformState(init_agent_mean, init_agent_cov)

agent = Agent(IMU_cov, GNSS_cov, radar_cov, init_agent_pose)

#spawn target
init_target_cov = np.diag([3, 3, 0, 5, 5, 0])**2
init_target_mean = multivariate_normal(np.array([*pt_0, *vt_0]), init_target_cov)
init_target_pose = TargetState(init_target_mean, init_target_cov)

cv_velocity_variance = 2**2

Fcv = np.block([[       np.eye(3), dt*np.eye(3)],
                [np.zeros((3, 3)),    np.eye(3)]])

Q = np.block([[dt**3/3*np.eye(3), dt**2/2*np.eye(3)],
              [dt**2/2*np.eye(3), dt*np.eye(3)]])*cv_velocity_variance

target_state_gt = np.empty((n_times*n_steps+1, 6))
target_state_gt[0] = np.array([*pt_0, *vt_0])

TARGET_ID = 0
target = TargetBody(id=TARGET_ID, var_acc=cv_velocity_variance, state=init_target_pose)


#generate IMU measurements
#imu noise, gyro, acc
#imu measurements values
gyro = lambda t: multivariate_normal(w(t), IMU_cov[:3, :3]) #generate gyro measurement at time t
acc = lambda t: multivariate_normal(Rot(t).T@(a(t) + g), IMU_cov[3:, 3:]) #imu measures g up, acc is in body
#imu measurement
generate_IMU_measurement = lambda t: IMU_Measurement(gyro(t), acc(t)) #generate IMU measurment at time t

#GNSS measurement
pos_m = lambda t: multivariate_normal(p(t), GNSS_cov) #measured position
gnssMeasurement = lambda t: GNSS_Measurement(pos_m(t)) #create gnss measurement with noise


target_noise = lambda: multivariate_normal([0]*6, Q)


agent.add_target(target)

#plotting
fig = plt.figure()
ax = fig.add_subplot(111)

plot_as_SE2(ax, agent.state, color="green") #plot initial state
plot_as_SE2(ax, agent.targets[0].convert_state_to_world_manifold(agent.state), color="pink")
# plot_as_SE2(ax, agent.state@agent.targets[0].state, color="pink")

agent_pos = np.empty((n_steps*n_times + 1, 2))
agent_pos[0] = agent.state.pos[:2]
#sim
for n in tqdm(range(n_times)):
    #propegate
    for k in range(n_steps):
        z_imu = generate_IMU_measurement(dt*(k + n*n_steps)) #sample random inputs
        agent.propegate(z_imu, dt)
        agent_pos[k + n*n_steps + 1, :] = agent.state.pos[:2]
        target_state_gt[k + n*n_steps + 1, :] = Fcv@target_state_gt[k + n*n_steps, :] + target_noise()

    #current time
    t = dt*(n_steps*(n+1))

    #update platform
    z_gnss = gnssMeasurement(t)
    agent.platform_update(z_gnss)

    #update target
    # plot_as_2d(ax, agent.targets[0].state, color="yellow")
    #target measurement
    y = TargetMeasurement(multivariate_normal(Rot(t).T@(target_state_gt[(n+1)*n_steps, :3] - p(t)), radar_cov))
    agent.target_update(TARGET_ID, y)


    #plotting
    plot_2d_frame(ax, SE2(SO2(Rot(t).as_matrix()[:2, :2]), p(t)[:2]), color="red", scale=5) #gt frame

    plot_as_SE2(ax, agent.state, color="green")
    plot_as_SE2(ax, agent.targets[0].convert_state_to_world_manifold(agent.state), color="pink")



    ax.plot(z_gnss.pos[0], z_gnss.pos[1], "gx")

    y_world = target_state_gt[(n+1)*n_steps, :3] #this is the actual world position of the target plus noise
    y_world_hat = agent.state.mean@y.relative_pos #this is the estimated world position of the target plus noise

    ax.plot(y_world[0], y_world[1], "bx")
    ax.plot(y_world_hat[0], y_world_hat[1], "bo")


plot_as_SE2(ax, agent.state, color="green") #plot the last ellipsis
plot_as_SE2(ax, agent.targets[0].convert_state_to_world_manifold(agent.state), color="pink")

agent_pos[-1, :] = agent.state.pos[:2] #add final position
ax.plot(agent_pos[:, 0], agent_pos[:, 1], "g--")

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

#plot target
ax.plot(*target_state_gt[:, :2].T, "k--")
ax.plot(*target_state_gt[-1, :2], "ko")


plt.axis("equal")
plt.show()