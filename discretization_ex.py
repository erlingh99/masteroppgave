import numpy as np
from numpy.random import multivariate_normal
from SE23.models import IMU_Model
from SE23.measurements import IMU_Measurement
from SE23.lie_theory import SE3_2, SO3, SE2, SO2
from SE23.plot_utils import plot_2d_frame
import matplotlib.pyplot as plt
from tqdm import tqdm

# np.random.seed(42)

T = 20
#imu rate
dt = 0.1 #sec per step

n_steps = int(T/dt) + 1

g = np.array([0, 0, 9.81])
#define ground truth (world)

# p = lambda t: np.array([0.25*t**2+t, 50*np.sin(0.1*t), 0])
# v = lambda t: np.array([0.5*t+1, 5*np.cos(0.1*t), 0])
# a = lambda t: np.array([0.5, -0.5*np.sin(0.1*t), 0])

p = lambda t: np.array([0.25*t**2+t, 4*np.sin(5*t), 0])
v = lambda t: np.array([0.5*t+1, 20*np.cos(5*t), 0])
a = lambda t: np.array([0.5, -100*np.sin(5*t), 0])


# p = lambda t: np.array([t**2, 0, 0])
# v = lambda t: np.array([2*t, 0, 0])
# a = lambda t: np.array([2, 0, 0])


# alpha = 2
# w = lambda t: np.array([0, 0, alpha])
# Rot = lambda t: SO3.Exp([0, 0, alpha*t])

w = lambda t: np.array([0, 0, 0.1*np.sin(5*t)])
Rot = lambda t: SO3.Exp([0, 0, -0.1/5*np.cos(5*t)+0.1/5])
# w = lambda t: np.array([0, 0, np.sin(0.1*t)])
# Rot = lambda t: SO3.Exp([0, 0, -10*np.cos(0.1*t)+10])


# a = lambda t: Rot(t)@np.array([2, 0, 0]) #constant local acc
# v = lambda t: np.array([1/alpha*np.sin(alpha*t) + 0.1, -1/alpha*np.cos(alpha*t), 0])*2
# p = lambda t: np.array([-1/alpha/alpha*np.cos(alpha*t) + 0.1*t, -1/alpha/alpha*np.sin(alpha*t), 0])*2


#imu noise, gyro, acc
covar = np.diag([0, 0, 0, 0, 0, 0])**2
#imu measurements values
gyro = lambda t: multivariate_normal(w(t), covar[:3, :3]) #generate n gyro measurements at time t
acc = lambda t: multivariate_normal(Rot(t).T@(a(t) + g), covar[3:, 3:]) #imu measures g up, acc is in body

#imu measurement
measurement = lambda t: IMU_Measurement(gyro(t), acc(t)) #generate n IMU measurments at time t


#filter init
filter = IMU_Model(covar)

T_simple = np.empty((n_steps, 5, 5))
T_medium = np.empty((n_steps, 5, 5))
T_hard   = np.empty((n_steps, 5, 5))
#init pos
T0 = SE3_2(Rot(0), v(0), p(0)).as_matrix()
T_simple[0] = T0
T_medium[0] = T0
T_hard[0] = T0

for k in tqdm(range(n_steps-1)):
    meas = measurement(k*dt) #get gt measurement
    T_simple[k+1] = filter.propegate_mean(T_simple[k], meas, dt, mode=0)
    T_hard[k+1]   = filter.propegate_mean(T_hard[k]  , meas, dt, mode=1)
    T_medium[k+1] = filter.propegate_mean(T_medium[k], meas, dt, mode=2)



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
ax.plot(x1[:, 0], x1[:, 1], "b--", alpha=1, label="Constant local")
ax.plot(x1[-1, 0], x1[-1, 1], "bo", alpha=1)
ax.plot(x2[:, 0], x2[:, 1], "g--", alpha=1, label="Constant local approx")
ax.plot(x2[-1, 0], x2[-1, 1], "go", alpha=1)
ax.plot(x3[:, 0], x3[:, 1], "k--", alpha=1, label="Constant global")
ax.plot(x3[-1, 0], x3[-1, 1], "ko", alpha=1)

s = 5
for k in range(0, n_steps, 200):
    plot_2d_frame(ax, SE2(SO2(Rot(k*dt).as_matrix()[:2, :2]), p(k*dt)[:2]), scale=s)
    plot_2d_frame(ax, SE2(SO2(T_hard[k, :2, :2]), T_hard[k, :2, 4]), scale=s)
    plot_2d_frame(ax, SE2(SO2(T_medium[k, :2, :2]), T_medium[k, :2, 4]), scale=s)
    plot_2d_frame(ax, SE2(SO2(T_simple[k, :2, :2]), T_simple[k, :2, 4]), scale=s)

#plot gt
ts = np.linspace(0, T, n_steps)
ps = np.empty((len(ts), 2))
for ix, tsi in enumerate(ts):
    ps[ix, :] = p(tsi)[:2]

# calc p len
ts = np.linspace(0, T, n_steps*10)
ps_l = np.empty((len(ts), 2))
for ix, tsi in enumerate(ts):
    ps_l[ix, :] = p(tsi)[:2]
plen = sum(np.linalg.norm(ps_l[1:]-ps_l[:-1], axis=1))
print(plen)


print("diff hard - medium")
me = np.mean(np.linalg.norm(T_hard[:, :3, 4] - T_medium[:, :3, 4], axis=-1))
print(me, "mean error")
print(me/plen*100, "%")
print("diff hard gt")
me = np.mean(np.linalg.norm(T_hard[:, :2, 4] - ps, axis=-1))
print(me, "mean error")
print(me/plen*100, "%")
print("diff medium gt")
me = np.mean(np.linalg.norm(T_medium[:, :2, 4] - ps, axis=-1))
print(me, "mean error")
print(me/plen*100, "%")
print("diff simple gt")
me = np.mean(np.linalg.norm(T_simple[:, :2, 4] - ps, axis=-1))
print(me, "mean error")
print(me/plen*100, "%")


ax.plot(ps[:, 0], ps[:, 1], "r--", label="ground truth")
ax.plot(ps[-1, 0], ps[-1, 1], "ro")
plt.legend()
plt.axis("equal")
plt.show()