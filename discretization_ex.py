import numpy as np
from numpy.random import multivariate_normal
from models import IMU_Model
from measurements import IMU_Measurement
from lie_theory import SE3_2, SO3
import matplotlib.pyplot as plt
from tqdm import tqdm

# np.random.seed(42)

T = 100
#imu rate
dt = 0.2 #sec per step

n_steps = int(T/dt)+1

g = np.array([0, 0, 9.81])
#define ground truth (world)

p = lambda t: np.array([0.1*t**2, 50*np.sin(0.1*t), 0])
v = lambda t: np.array([0.2*t, 5*np.cos(0.1*t), 0])
a = lambda t: np.array([0.2, -0.5*np.sin(0.1*t), 0])
w = lambda t: np.array([0, 0, 2])#2*np.sin(0.1*t)])#2
Rot = lambda t: SO3.Exp([0, 0, 2*t])#-20*np.cos(0.1*t)+20])#2*t

#imu noise, gyro, acc
covar = np.diag([0, 0, 0.0, 0, 0, 0])**2
#imu measurements values
gyro = lambda t: multivariate_normal(w(t), covar[:3, :3]) #generate n gyro measurements at time t
acc = lambda t: multivariate_normal(Rot(t).T@(a(t) + g), covar[3:, 3:]) #imu measures g up, acc is in body

#imu measurement
measurement = lambda t: IMU_Measurement(gyro(t), acc(t)) #generate n IMU measurments at time t


#filter init
filter = IMU_Model(covar)


T_simple = np.empty((n_steps, 5, 5))
T_medium = np.empty((n_steps, 5, 5))
T_hard = np.empty((n_steps, 5, 5))
#init pos
T0 = SE3_2(Rot(0), v(0), p(0)).as_matrix()
T_simple[0] = T0
T_medium[0] = T0
T_hard[0] = T0

for k in tqdm(range(1, n_steps)):
    meas = measurement(k*dt) #get gt measureument
    T_hard[k] = filter.propegate_mean(T_hard[k-1], meas, dt, mode=1)
    T_medium[k] = filter.propegate_mean(T_medium[k-1], meas, dt, mode=2)
    T_simple[k] = filter.propegate_mean(T_simple[k-1], meas, dt, mode=0)




#plotting
fig = plt.figure()
ax = fig.add_subplot(111)


x1 = np.empty((n_steps, 2))
x2 = np.empty((n_steps, 2))
x3 = np.empty((n_steps, 2))
for i in range(n_steps):
    x1[i] = T_hard[i, :2, 4]
    x2[i] = T_medium[i, :2, 4]
    x3[i] = T_simple[i, :2, 4]
ax.plot(x1[:, 0], x1[:, 1], "b--", alpha=1, label="hard")
ax.plot(x1[-1, 0], x1[-1, 1], "bo", alpha=1)
ax.plot(x2[:, 0], x2[:, 1], "g--", alpha=1, label="medium")
ax.plot(x2[-1, 0], x2[-1, 1], "go", alpha=1)
ax.plot(x3[:, 0], x3[:, 1], "k--", alpha=1, label="simple")
ax.plot(x3[-1, 0], x3[-1, 1], "ko", alpha=1)

# plot_2d_frame(ax, SE2.Exp([0, 0, 0]), scale=5)
# plot_2d_frame(ax, SE2(SO2(Rot(dt*(n_steps-1)).as_matrix()[:2, :2]), p(dt*(n_steps-1))[:2]), scale=5)

#plot gt
ts = np.linspace(0, T, n_steps)
ps = np.empty((len(ts), 2))
for ix, tsi in enumerate(ts):
    ps[ix, :] = p(tsi)[:2]


print("diff hard - medium")
print(np.linalg.norm(T_hard[:, :3, 4] - T_medium[:, :3, 4])/np.linalg.norm(T_hard[:, :3, 4])*100, "%")
print("diff hard gt")
print(np.linalg.norm(T_hard[:, :2, 4] - ps)/np.linalg.norm(T_hard[:, :2, 4])*100, "%")


ax.plot(ps[:, 0], ps[:, 1], "r--", label="ground truth")
ax.plot(ps[-1, 0], ps[-1, 1], "ro")
plt.legend()
plt.axis("equal")
plt.show()