import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from agent import Agent
from measurements import IMU_Measurement, GNSS_Measurement
from lie_theory import SE3_2, SO3, SE2, SO2
from states import PlatformState
from utils import plot_2d_frame, find_mean, exp_cov

np.random.seed(1)

n_steps = 300
n_random = 1000 #number of simulations
dt = 0.05 #sec per step, imu rate
T = (n_steps-1)*dt #end time

g = np.array([0, 0, 9.81])

#define ground truth (world)
p = lambda t: np.array([t**2, 5*np.sin(t), 0])
v = lambda t: np.array([2*t, 5*np.cos(t), 0])
a = lambda t: np.array([2, -5*np.sin(t), 0])
w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])


##Agent setup, noises
IMU_noise = np.diag([0, 0, 0.4, 0, 0, 0])**2 #imu noise, gyro, acc
GNSS_noise = np.diag([10, 10, 0.001])**2
radar_noise = 0 # not used in this example

#Init poses
T_sim = np.empty((n_steps, n_random, 5, 5))
T_pred = np.empty(n_steps, dtype=PlatformState)

init_cov = np.diag([0, 0, 0.0, 0, 0, 0, 0, 0, 0])**2
T0 = SE3_2(Rot(0), v(0), p(0)) #true start pos
#draw random start pertrbations
init_perturbs = multivariate_normal([0]*9, init_cov, n_random) #random perturbations
for i in range(n_random):
    T_sim[0, i] = (T0@SE3_2.Exp(init_perturbs[i])).as_matrix()

init_state = PlatformState(T0@SE3_2.Exp(init_perturbs[0]), init_cov) #also perturb our calculation starting point
T_pred[0] = init_state #save the initial state

#create the agent
agent = Agent(IMU_noise, GNSS_noise, radar_noise, init_state)


###Generate measurments
#imu measurements
gyro = lambda t, n: multivariate_normal(w(t), IMU_noise[:3, :3], size=n) #generate n gyro measurements at time t
acc = lambda t, n: multivariate_normal(Rot(t).T@(a(t) + g), IMU_noise[3:, 3:], size=n) #imu measures g up, acc is in body
generate_IMU_measurements = lambda t, n: [IMU_Measurement(g_t, acc_t) for g_t, acc_t in zip(gyro(t, n), acc(t, n))] #generate n IMU measurments at time t
#GNSS measurements
pos_m = lambda t: multivariate_normal(p(t), GNSS_noise) #measured position
gnssMeasurement = lambda t: GNSS_Measurement(pos_m(t)) #create gnss measurement with noise

#propegate and simulate
for k in tqdm(range(1, n_steps)):
    z_imu = generate_IMU_measurements(k*dt, n_random) #sample n_random different inputs
    # z_true = IMU_Measurement(w(k*dt), Rot(k*dt).T@(a(k*dt) + g))
    agent.propegate(z_imu[0], dt) #use one of the random inputs in the full filter
    T_pred[k] = agent.state #save the current state of the agent
    for i in range(n_random):
        T_sim[k, i] = agent.inertialNavigation.__propegate_mean__(T_sim[k-1, i], z_imu[i], dt) #propegate the simulations with the corresponding random input
#update 
z_gnss = gnssMeasurement(dt*n_steps)
agent.platform_update(z_gnss)
T_update = agent.state

#calculate SE3_2 distribution of simulated samples
final_pose = np.empty(n_random, dtype=SE3_2) 
for i in range(n_random):
    final_pose[i] = SE3_2(SO3(T_sim[-1, i, :3, :3]), T_sim[-1, i, :3, 3], T_sim[-1, i, :3, 4])

mean = find_mean(final_pose, T_pred[-1].mean) #find mean, with the predicted mean as initial guess
sim_cov = exp_cov(final_pose, mean)
sim_pose = PlatformState(mean, sim_cov)

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

plot_as_SE2(sim_pose, color="yellow")
for i in range(n_random):
    ax.plot(T_sim[:, i, 0, 4], T_sim[:, i, 1, 4], color='gray', alpha=0.1) 
ax.scatter(T_sim[-1, :, 0, 4], T_sim[-1, :, 1, 4], s=2, color='black')


x = np.empty((n_steps, 2))
for i in range(n_steps):
    x[i] = T_pred[i].mean.p[:2]
ax.plot(x[:, 0], x[:, 1], "g--", alpha=1)

plot_2d_frame(ax, SE2.Exp([0, 0, 0]), scale=5)
for i in range(100, n_steps+1, 50):
    idx = min(i, n_steps-1)
    plot_as_SE2(T_pred[idx], color="green")
plot_as_SE2(T_update, color="blue", z=z_gnss.pos)

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