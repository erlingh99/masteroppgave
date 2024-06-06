import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement
from SE23.lie_theory import SE3_2, SO3, SE2, SO2, SO3xR3xR3
from SE23.states import PlatformState
from SE23.plot_utils import plot_2d_frame, plot_as_SE2, plot_as_SO2xR2
from SE23.gaussian import ExponentialGaussian
from SE23.gaussian import MultiVarGauss as mvg

np.random.seed(42)

N = 300
dt = 0.05 #sec per step, imu rate
T = (N-1)*dt #end time

g = np.array([0, 0, 9.81])

#define ground truth (world)
p = lambda t: np.array([t**2, 5*np.sin(t), 0])
v = lambda t: np.array([2*t, 5*np.cos(t), 0])
a = lambda t: np.array([2, -5*np.sin(t), 0])
w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])
# p = lambda t: np.array([0.5*t**2, 0, 0])
# v = lambda t: np.array([t, 0, 0])
# a = lambda t: np.array([1, 0, 0])
# w = lambda t: np.array([0, 0, 0])
# Rot = lambda t: SO3.Exp([0, 0, 0])

acc_noise = np.array([0.2, 0.2, 0])*0
gyro_noise = np.array([0, 0, 0.03/dt])*0

##Agent setup, noises
IMU_noise = np.diag([*gyro_noise, *acc_noise])**2 #imu noise, gyro, acc
# IMU_noise = np.diag([0, 0, 0, 0, 0, 0])**2 #imu noise, gyro, acc

#Init poses
T_pred = np.empty(N, dtype=PlatformState)
T_pred2 = np.empty(N, dtype=PlatformState)

init_cov = np.diag([0, 0, 0.2, 0, 0, 0, 0, 0, 0])**2


T0 = SO3xR3xR3(Rot(0), v(0), p(0)) #true start pos
T02 = SE3_2(Rot(0), v(0), p(0)) #true start pos

#draw random start pertrbations
init_perturbs = multivariate_normal([0]*9, init_cov) #random perturbations
T02 = T02@SE3_2.Exp(init_perturbs)
init_state2 = PlatformState(T02, init_cov) #also perturb our calculation starting point
init_state = PlatformState(SO3xR3xR3.from_matrix(T02.as_matrix()), init_cov) #also perturb our calculation starting point
T_pred[0] = init_state #save the initial state
T_pred2[0] = init_state2 #save the initial state

#create the agent
agent = Agent(IMU_noise, None, None, init_state)
agent2 = Agent(IMU_noise, None, None, init_state2)



###Generate measurments
def generate_IMU_measurement(t):
    gyro = multivariate_normal(w(t), np.diag(gyro_noise**2))
    acc_null = multivariate_normal([0]*3, np.diag(acc_noise**2))
    # acc_ESKF = Rot(t).T@(a(t) - g) + acc_null
    acc_Lie = Rot(t).T@(a(t) + g) + acc_null
    return IMU_Measurement(gyro, acc_Lie) #generate IMU measurment at time t


#propagate and simulate
for k in tqdm(range(N - 1)):
    z_agent = generate_IMU_measurement(k*dt)
    agent.propegate(z_agent, dt)
    agent2.propegate(z_agent, dt)
    T_pred[k+1] = agent.state 
    T_pred2[k+1] = agent2.state


print(T_pred[-1].mean.p)
print(T_pred2[-1].mean.p)
print(T_pred[-1].mean.v)
print(T_pred2[-1].mean.v)
print(T_pred[-1].mean.R)
print(T_pred2[-1].mean.R)




#plotting
fig = plt.figure()
ax = fig.add_subplot(111)

x = np.empty((N, 2))
x2 = np.empty((N, 2))
for i in range(N):
    x[i] = T_pred[i].mean.p[:2]
    x2[i] = T_pred2[i].mean.p[:2]
ax.plot(x[:, 0], x[:, 1], "g--", alpha=1)
ax.plot(x2[:, 0], x2[:, 1], "b--", alpha=1)

#x2 er SE23
print(np.linalg.norm(x-x2))

print(T_pred[-1].cov)
print(T_pred2[-1].cov)

print(np.linalg.norm(T_pred[-1].cov - T_pred2[-1].cov, ord="fro"))


plot_2d_frame(ax, SE2.Exp([0, 0, 0]), scale=5)
for i in range(100, N+1, 50):
    idx = min(i, N-1)
    T_pred[idx].draw_significant_ellipses(ax, color="green")
    T_pred2[idx].draw_significant_ellipses(ax, color="red")
    plot_2d_frame(ax, SE2(SO2(T_pred[idx].mean.R.as_matrix()[:2, :2]), T_pred[idx].mean.p[:2]), scale=5)
    plot_2d_frame(ax, SE2(SO2(T_pred2[idx].mean.R.as_matrix()[:2, :2]), T_pred2[idx].mean.p[:2]), scale=5)
    
    
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