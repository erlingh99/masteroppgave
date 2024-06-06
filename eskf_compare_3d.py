import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement
from SE23.lie_theory import SE3_2, SO3, SE3, SO3, SO3xR3xR3
from SE23.states import PlatformState
from SE23.plot_utils import plot_3d_frame

# np.random.seed(42)

N = 300
dt = 0.05 #sec per step, imu rate
T = (N-1)*dt #end time

g = np.array([0, 0, 9.81])

#define ground truth (world)
p = lambda t: np.array([t**2, 5*np.sin(t), -100*np.cos(0.1*t)+100])
v = lambda t: np.array([2*t, 5*np.cos(t), 10*np.sin(0.1*t)])
a = lambda t: np.array([2, -5*np.sin(t), np.cos(0.1*t)])
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

init_cov = np.diag([0.2, 0, 0.2, 0, 0, 0, 0, 0, 0])**2



T0 = SO3xR3xR3(Rot(0), v(0), p(0)) #true start pos
T02 = SE3_2(Rot(0), v(0), p(0)) #true start pos

#draw random start pertrbation
init_perturbs = multivariate_normal([0]*9, init_cov)
# T02 = T02@SE3_2.Exp(init_perturbs)

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
    acc = multivariate_normal(Rot(t).T@(a(t) + g), np.diag(acc_noise**2))
    return IMU_Measurement(gyro, acc) #generate IMU measurment at time t


#propagate and simulate
for k in tqdm(range(N - 1)):
    z_agent = generate_IMU_measurement(k*dt)
    agent.propegate(z_agent, dt)
    agent2.propegate(z_agent, dt)
    T_pred[k+1] = agent.state 
    T_pred2[k+1] = agent2.state



#plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x = np.empty((N, 3))
x2 = np.empty((N, 3))

for i in range(N):
    x[i] = T_pred[i].mean.p
    x2[i] = T_pred2[i].mean.p

ax.plot(*x.T, "g--", alpha=1)
ax.plot(*x2.T, "b--", alpha=1)


#x2 er SE23
print(np.linalg.norm(x-x2))


print(np.linalg.norm(T_pred[-1].cov - T_pred2[-1].cov, ord="fro"))


plot_3d_frame(ax, scale=5)
for i in range(100, N+1, 100):
    idx = min(i, N-1)
    # T_pred[idx].draw_significant_spheres(ax, n_spheres=1, color="green")
    T_pred[idx].draw_significant_ellipses(ax, color="green")
    # T_pred2[idx].draw_significant_spheres(ax, n_spheres=1, color="red")
    T_pred2[idx].draw_significant_ellipses(ax, color="red")
    plot_3d_frame(ax, SE3(T_pred[idx].mean.R, T_pred[idx].mean.p), scale=5)
    plot_3d_frame(ax, SE3(T_pred2[idx].mean.R, T_pred2[idx].mean.p), scale=5)

    
plot_3d_frame(ax, SE3(Rot(T), p(T)), scale=5)

#plot gt
ts = np.linspace(0, T, N)
ps = np.empty((N, 3))
for i, tsi in enumerate(ts):
    ps[i] = p(tsi)

print(np.linalg.norm(ps - x, axis=1).mean())
print(np.linalg.norm(ps - x2, axis=1).mean())


ax.plot(*ps.T, "r--")
ax.plot(*ps[-1], "ro")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.show()






