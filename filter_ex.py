import numpy as np
from numpy.random import multivariate_normal
from filter import Filter, IMU_Measurement, GNSS_Measurement, GNSS_Sensor
from lie_theory import ExponentialGaussian, SE3_2, SO3, SE2, SO2
import matplotlib.pyplot as plt
from utils import plot_2d_frame, find_mean, exp_cov
from tqdm import tqdm

# np.random.seed(42)

n_steps = 300
n_random = 1000
#imu rate
dt = 0.05 #sec per step

g = np.array([0, 0, 9.81])
#define ground truth (world)
p = lambda t: np.array([t**2, 5*np.sin(t), 0])
v = lambda t: np.array([2*t, 5*np.cos(t), 0])
a = lambda t: np.array([2, -5*np.sin(t), 0])
w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])

#imu noise, gyro, acc
covar = np.diag([0, 0, 0.4, 0, 0, 0])**2
#imu measurements values
gyro = lambda t: multivariate_normal(w(t), covar[:3, :3])
acc = lambda t: multivariate_normal(Rot(t).T@(a(t) + g), covar[3:, 3:]) #imu measures g up, acc is in body

#imu measurement
measurement = lambda t: IMU_Measurement(gyro(t), acc(t), covar)

#GNSS noise
R = np.diag([10, 10, 0.001])**2
#GNSS sensor
sensor = GNSS_Sensor()
pos_m = lambda t: multivariate_normal(p(t), R) #measured position
gnssMeasurement = lambda t: GNSS_Measurement(pos_m(t), R) #create gnss measurement with noise

#filter init
filter = Filter(sensor)

T_sim = np.empty((n_steps, n_random, 5, 5))
T_pred = np.empty(n_steps, dtype=ExponentialGaussian)
#init pos
T0 = SE3_2(Rot(0), v(0), p(0))
T_pred[0] = ExponentialGaussian(T0, np.zeros((9, 9)))
T_sim[0, :] = T0.as_matrix()
for k in tqdm(range(1, n_steps)):
    meas = measurement(k*dt)
    T_pred[k] = filter.propegate(T_pred[k-1], meas, dt)
    Qi = filter.Q(meas, dt)
    noise = multivariate_normal(mean=[0]*9, cov=Qi, size=n_random)
    for i in range(n_random):
        w_ki = SE3_2.Exp(noise[i]).as_matrix()
        T_sim[k, i] = filter.__propegate_mean__(T_sim[k-1, i], meas, dt)@w_ki   

z = gnssMeasurement(dt*n_steps)
T_update = filter.update(T_pred[-1], z)

final_pose = np.empty(n_random, dtype=SE3_2) 
for i in range(n_random):
    final_pose[i] = SE3_2(SO3(T_sim[-1, i, :3, :3]), T_sim[-1, i, :3, 3], T_sim[-1, i, :3, 4])

mean = find_mean(final_pose, T_pred[-1].mean)
sim_cov = exp_cov(final_pose, mean)
sim_pose = ExponentialGaussian(mean, sim_cov)

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
    exp = ExponentialGaussian(pose2D, c)
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
ax.plot(x[:, 0], x[:, 1], color='gray', alpha=1)

plot_2d_frame(ax, SE2.Exp([0, 0, 0]), scale=5)
for i in range(100, n_steps+1, 50):
    idx = min(i, n_steps-1)
    plot_as_SE2(T_pred[idx], color="green")
plot_as_SE2(T_update, color="blue", z=z.pos)

plot_2d_frame(ax, SE2(SO2(Rot(dt*n_steps).as_matrix()[:2, :2]), p(dt*n_steps)[:2]), scale=5)


#plot gt
ts = np.linspace(0, dt*n_steps, 101)
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