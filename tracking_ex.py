import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent import Agent
from target import Target
from lie_theory import SE3_2, SO3, SE2, SO2
from states import PlatformState, TargetState
from models import CV_world
from measurements import IMU_Measurement, TargetMeasurement, GNSS_Measurement
from utils import plot_2d_frame

np.random.seed(42)

#meta vars
n_steps = 400
#imu rate
dt = 0.01 #sec per step
T = (4*n_steps-1)*dt

g = np.array([0, 0, 9.81])
#define ground truth of platform
p = lambda t: np.array([t**2, 5*np.sin(t), 0])
v = lambda t: np.array([2*t, 5*np.cos(t), 0])
a = lambda t: np.array([2, -5*np.sin(t), 0])
w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])

#define ground truth of target
pt = lambda t: np.array([30*t, -2*t**2 + 100, 0])
vt = lambda t: np.array([30, -4*t, 0])


#spawn agent
IMU_cov = np.diag([0, 0, 0.4, 0, 0, 0])**2

GNSS_cov = np.diag([5, 5, 0.001])**2
radar_cov = np.diag([0.1, 0.2, 0.001])**2

init_agent_mean = SE3_2(Rot(0), v(0), p(0))
init_agent_cov = np.diag([0, 0, 0.2, 0, 0, 0, 0, 0, 0])**2
init_agent_pose = PlatformState(init_agent_mean, init_agent_cov)

agent = Agent(IMU_cov, GNSS_cov, radar_cov, init_agent_pose)

#spawn target
init_target_mean = np.array([*pt(dt*n_steps), *vt(dt*n_steps)]) 
init_target_cov = np.diag([0, 0, 0, 0, 0, 0])**2
init_target_pose = TargetState(init_target_mean, init_target_cov)

cv_velocity_variance = 2**2

target = Target(id=0, motion_model=CV_world(cv_velocity_variance), state=init_target_pose)


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

#target measurement
y = lambda t: multivariate_normal(Rot(t).T@(pt(t) - p(t)), radar_cov)
targetMeasurement = lambda t: TargetMeasurement(y(t))


agent_states = np.empty(4*n_steps, dtype=PlatformState)
agent_states[0] = init_agent_pose

target_states = np.empty(3*n_steps + 1, dtype=TargetState)
target_states[0] = init_target_pose


#propegate platform without target
for k in tqdm(range(1, n_steps)):
    z_imu = generate_IMU_measurement(k*dt) #sample random inputs

    agent.propegate(z_imu, dt)

    agent_states[k] = agent.state #save the current state of the agent


#add target
agent.add_target(target)

#propegate platform and target
for k in tqdm(range(n_steps, 2*n_steps)):
    z_imu = generate_IMU_measurement(k*dt) #sample random inputs

    agent.propegate(z_imu, dt)

    agent_states[k] = agent.state #save the current state of the agent
    target_states[k - n_steps + 1] = agent.targets[0].state #save the current state of the target

#update platform
z_gnss = gnssMeasurement(dt*2*n_steps)
agent.platform_update(z_gnss)

#propegate platform and target
for k in tqdm(range(2*n_steps, 3*n_steps)):
    z_imu = generate_IMU_measurement(k*dt) #sample random inputs

    agent.propegate(z_imu, dt)

    agent_states[k] = agent.state #save the current state of the agent
    target_states[k - n_steps + 1] = agent.targets[0].state #save the current state of the targets (only 1)


#update target
y_target = targetMeasurement(dt*3*n_steps)
agent.target_update(0, y_target)

#propegate platform and target
for k in tqdm(range(3*n_steps, 4*n_steps)):
    z_imu = generate_IMU_measurement(k*dt) #sample random inputs

    agent.propegate(z_imu, dt)

    agent_states[k] = agent.state #save the current state of the agent
    target_states[k - n_steps + 1] = agent.targets[0].state #save the current state of the targets (only 1)


#plotting
fig = plt.figure()
ax = fig.add_subplot(111)


def plot_as_SE2(pose, color="red", z=None):
    extract = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0]])
    m = pose.mean.as_matrix()
    c = extract@pose.cov@extract.T
    pose2D = SE2(SO2(m[:2, :2]), m[:2, 4])
    exp = PlatformState(pose2D, c)
    plot_2d_frame(ax, pose2D, scale=5)
    exp.draw_2Dtranslation_covariance_ellipse(ax, "xy", 3, 50, color=color)
    ax.plot(m[0, 4], m[1, 4], color=color, marker="o")
    if z is not None:
        ax.plot(z[0], z[1], color=color, marker="x")

def plot_as_2d(pose, color="red", z=None, num_std=3):
    m = pose.mean[:2]
    c = pose.cov[:2, :2]
    pose2d = TargetState(m, c)
    x, y = pose2d.covar_ellipsis(resolution=50, num_std=num_std)
    ax.plot(x, y, label=f"{num_std}σ", color=color)
    ax.plot(m[0], m[1], color=color, marker="o")

    if z is not None:
        ax.plot(z[0], z[1], color=color, marker="x")



for i in range(49, 4*n_steps, 100):
    plot_as_SE2(agent_states[i], color="green")


# plot_2d_frame(ax, SE2.Exp([0, 0, 0]), scale=5)
ps = np.empty((4*n_steps, 2))
for i in range(0, 4*n_steps):
    ps[i] = agent_states[i].mean.p[:2]
ax.plot(ps[:,0], ps[:, 1], "g--")


# plot_as_SE2(agent_states[2*n_steps - 1], color="blue")
# plot_as_SE2(agent_states[2*n_steps + 1], color="blue")
ax.plot(z_gnss.pos[0], z_gnss.pos[1], "bx")

y_world = Rot(dt*3*n_steps)@y_target.relative_pos + p(dt*3*n_steps)
y_world_hat = agent_states[3*n_steps].mean@y_target.relative_pos

ax.plot(y_world[0], y_world[1], "bx")
ax.plot(y_world_hat[0], y_world_hat[1], "bo")


for i in range(0, 3*n_steps, 49):
    plot_as_2d(target_states[i], color="pink")


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
ts = np.linspace(dt*n_steps, 4*dt*n_steps, 101)
xs = []
ys = []
for tsi in ts:
    xsi, ysi, _ = pt(tsi)
    xs.append(xsi)
    ys.append(ysi)
ax.plot(xs, ys, "k--")
ax.plot(xs[-1], ys[-1], "ko")


plt.axis("equal")
plt.show()