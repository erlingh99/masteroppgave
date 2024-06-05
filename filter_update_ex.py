import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement, GNSS_Measurement
from SE23.lie_theory import SE3_2, SO3, SE2, SO2, SO3xR3xR3, SO2xR2
from SE23.utils import exp_NEES
from scipy.stats import chi2
from SE23.states import PlatformState
from SE23.utils import find_mean, exp_cov
from SE23.plot_utils import plot_2d_frame, plot_as_SO2xR2, plot_as_SE2

# np.random.seed(100)
n_iter = 25
n_steps = 100
n_random = 0 #number of simulations
dt = 0.05 #sec per step, imu rate
T = (n_iter*n_steps-1)*dt #end time

g = np.array([0, 0, 9.81])

#define ground truth (world)
# p = lambda t: np.array([t**2, 5*np.sin(t), 0])
# v = lambda t: np.array([2*t, 5*np.cos(t), 0])
# a = lambda t: np.array([2, -5*np.sin(t), 0])
# w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
# Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])
p = lambda t: np.array([0.5*t**2, 0, 0])
v = lambda t: np.array([t, 0, 0])
a = lambda t: np.array([1, 0, 0])
w = lambda t: np.array([0, 0, 0])
Rot = lambda t: SO3.Exp([0, 0, 0])


##Agent setup, noises
IMU_noise = np.diag([0, 0, 0.03/dt, 0, 0, 0])**2 #imu noise, gyro, acc
GNSS_noise = np.diag([10, 10, 0.0001])**2
radar_noise = None # not used in this example

#Init poses
if n_random > 0:
    T_sim = np.empty((n_iter*n_steps, n_random, 5, 5))
T_pred = np.empty(n_iter*n_steps, dtype=PlatformState)
T_pred_ESKF = np.empty(n_iter*n_steps, dtype=PlatformState)

init_cov = np.diag([0, 0, 1, 0, 0, 0, 0.01, 0.01, 0])**2

T0 = SE3_2(Rot(0), v(0), p(0)) #true start pos

#draw random start pertrbations
init_perturbs = multivariate_normal([0]*9, init_cov, max(1, n_random)) #random perturbations
for i in range(n_random):
    T_sim[0, i] = (T0@SE3_2.Exp(init_perturbs[i])).as_matrix()

T0 = T0@SE3_2.Exp(init_perturbs[0])
T0_ESKF = SO3xR3xR3.from_matrix(T0.as_matrix())

init_state = PlatformState(T0, init_cov) #also perturb our calculation starting point
init_state_ESKF = PlatformState(T0_ESKF, init_cov) #also perturb our calculation starting point
T_pred[0] = init_state #save the initial state
T_pred_ESKF[0] = init_state_ESKF #save the initial state

#create the agent
agent = Agent(IMU_noise, GNSS_noise, radar_noise, init_state)
agent_ESKF = Agent(IMU_noise, GNSS_noise, radar_noise, init_state_ESKF)

###Generate measurments
#imu measurements
gyro = lambda t, n: multivariate_normal(w(t), IMU_noise[:3, :3], size=n) #generate n gyro measurements at time t
acc = lambda t, n: multivariate_normal(Rot(t).T@(a(t) + g), IMU_noise[3:, 3:], size=n) #imu measures g up, acc is in body
generate_IMU_measurements = lambda t, n: [IMU_Measurement(g_t, acc_t) for g_t, acc_t in zip(gyro(t, n), acc(t, n))] #generate n IMU measurments at time t
#GNSS measurements
pos_m = lambda t: multivariate_normal(p(t), GNSS_noise) #measured position
gnssMeasurement = lambda t: GNSS_Measurement(pos_m(t)) #create gnss measurement with noise

for l in tqdm(range(n_iter)):
    #propegate and simulate
    for k in range(max(0, l*n_steps - 1), (l+1)*n_steps - 1):
        z_imu = generate_IMU_measurements(k*dt, max(1, n_random)) #sample n_random different inputs
        z_true = z_imu[0]#IMU_Measurement(w(k*dt), Rot(k*dt).T@(a(k*dt) + g))
        agent.propegate(z_true, dt) #use one of the random inputs in the full filter
        agent_ESKF.propegate(z_true, dt) #use one of the random inputs in the full filter
        T_pred[k+1] = agent.state #save the current state of the agent
        T_pred_ESKF[k+1] = agent_ESKF.state #save the current state of the agent
        for i in range(n_random):
            T_sim[k+1, i] = agent.inertialNavigation.model.propegate_mean(T_sim[k, i], z_imu[i], dt) #propegate the simulations with the corresponding random input
    #update 
    z_gnss = gnssMeasurement(k*dt)
    agent.platform_update(z_gnss)
    agent_ESKF.platform_update(z_gnss)
    T_update = agent.state
    T_update_ESKF = agent_ESKF.state


#calculate SE3_2 distribution of simulated samples
if n_random > 0:
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

    mean2 = find_mean(final_pose2, T_pred_ESKF[-1].mean) #find mean, with the predicted mean as initial guess
    sim_cov2 = exp_cov(final_pose2, mean2)
    sim_pose2 = PlatformState(mean2, sim_cov2)


#plotting
fig = plt.figure()
ax = fig.add_subplot(111)

if n_random > 0:
    # plot_as_SE2(ax, sim_pose, color="yellow")
    # plot_as_SO2xR2(ax, sim_pose2, color="orange")
    for i in range(n_random):
        ax.plot(T_sim[:, i, 0, 4], T_sim[:, i, 1, 4], color='gray', alpha=0.1) 
    ax.scatter(T_sim[-1, :, 0, 4], T_sim[-1, :, 1, 4], s=2, color='black')


x = np.empty((n_iter*n_steps, 2))
for i in range(n_iter*n_steps):
    x[i] = T_pred[i].mean.p[:2]
ax.plot(x[:, 0], x[:, 1], "g--", alpha=1)

# plot_2d_frame(ax, SE2.Exp([0, 0, 0]), scale=5)
# for i in range(100, n_iter*n_steps+1, 50):
#     idx = min(i, n_iter*n_steps-1)
#     plot_as_SE2(ax, T_pred[idx], color="green")
#     plot_as_SO2xR2(ax, T_pred_ESKF[idx], color="red")

# plot_as_SE2(ax, T_update, color="blue", num_std=1, z=z_gnss.pos)
# plot_as_SE2(ax, T_update, color="blue", num_std=2)
plot_as_SE2(ax, T_update, color="blue", num_std=3, z=z_gnss.pos)
# plot_as_SO2xR2(ax, T_update_2, num_std=1, color="orange")
# plot_as_SO2xR2(ax, T_update_2, num_std=2, color="orange")
# plot_as_SO2xR2(ax, T_update_2, num_std=3, color="orange")
    


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
# plt.show()

# print(T_pred[-1].cov)
# print(np.linalg.norm(sim_cov - T_pred[-1].cov, ord="fro"))
# print(np.linalg.norm(sim_cov2 - T_pred_2[-1].cov, ord="fro"))



alpha = 0.05
#calc NEES
NEES = np.zeros(n_iter*n_steps, float)
NEES_ESKF = np.zeros(n_iter*n_steps, float)

def asSE2(pose):
    ext = np.block([[0, 0, 1, np.zeros((1, 6))],
                    [np.zeros((2, 6)), np.eye(2), np.zeros((2, 1))]])
    return PlatformState(SE2(SO2(pose.mean.R.as_matrix()[:2, :2]), pose.mean.p[:2]), ext@pose.cov@ext.T)
def asSO2xR2(pose):
    ext = np.block([[0, 0, 1, np.zeros((1, 6))],
                    [np.zeros((2, 6)), np.eye(2), np.zeros((2, 1))]])
    return PlatformState(SO2xR2(SO2(pose.mean.R.as_matrix()[:2, :2]), pose.mean.p[:2]), ext@pose.cov@ext.T)


for k in range(n_iter*n_steps):
    gt_mean = SE2(SO2(Rot(k*dt).as_matrix()[:2, :2]), p(k*dt)[:2])
    gt_mean_ESKF = SO2xR2(SO2(Rot(k*dt).as_matrix()[:2, :2]), p(k*dt)[:2])
    NEES[k] = exp_NEES(asSE2(T_pred[k]), gt_mean)
    NEES_ESKF[k] = exp_NEES(asSO2xR2(T_pred_ESKF[k]), gt_mean_ESKF)


fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)#, figsize=(8, 5), num=4, clear=True, sharex=True)
axs = axs.flatten()
# axs[0].set_yscale("log")
# axs[1].set_yscale("log")

df = 3

CI_NEES = chi2.interval(1 - alpha, df)
CI_ANEES = chi2.interval(1 - alpha, df * n_iter*n_steps, scale=1/(n_iter*n_steps))
print(f"CI NEES: {CI_NEES}")
print(f"CI ANEES: {CI_ANEES}")

axs[0].plot(NEES, lw=0.5)
axs[0].plot(np.full(n_iter*n_steps, CI_NEES[0]), "r--")
axs[0].plot(np.full(n_iter*n_steps, CI_NEES[1]), "r--")
axs[0].plot(np.full(n_iter*n_steps, CI_ANEES[0]), "g--")
axs[0].plot(np.full(n_iter*n_steps, CI_ANEES[1]), "g--")
# axs[0].set_ylim(0, CI_NEES[1]*1.1)

axs[1].plot(NEES_ESKF, lw=0.5)
axs[1].plot(np.full(n_iter*n_steps, CI_NEES[0]), "r--")
axs[1].plot(np.full(n_iter*n_steps, CI_NEES[1]), "r--")
axs[1].plot(np.full(n_iter*n_steps, CI_ANEES[0]), "g--")
axs[1].plot(np.full(n_iter*n_steps, CI_ANEES[1]), "g--")
# axs[1].set_ylim(0, CI_NEES[1]*1.1)

insideCI = (CI_NEES[0] <= NEES) * (NEES <= CI_NEES[1])
percents = insideCI.mean()*100
ANEES = NEES.mean()

insideCI_ESKF = (CI_NEES[0] <= NEES_ESKF) * (NEES_ESKF <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_ESKF.mean()


print("\n\nSE_2(3)")
print(f"Percentage of NEES inside bounds {percents}%")
print(f"ANEES: {ANEES}")


print("\n\nESKF")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")



# fig, axs = plt.subplots(3, 3)
# axs = axs.flatten()

# for i, ax in enumerate(axs):
#     if i < 3:
#         ax.plot(p[:N, i], "--", color="orange")
#         ax.plot(x2[:, i], color="blue")
#         # ax.plot(x3[:, i], color="green")
#         ax.set_title(f"position {'xyz'[i]}")
#     elif i < 6:
#         ax.plot(v[:N, i-3], "--", color="orange")
#         ax.plot(vel[:, i-3], color="blue")
#         # ax.plot(vel2[:, i-3], color="green")
#         ax.set_title(f"velocity {'xyz'[i-3]}")
#     else: #euler angles
#         ax.plot(euler_gt[:, i-6], "--", color="orange")
#         ax.plot(euler[:, i-6], color="blue")
#         # ax.plot(euler2[:, i-6], color="green")
#         ax.set_title(f"angle {'uvw'[i-6]}")

# print("\n\n")
# ep = p[:N]-x2
# print("RMSE pos", (1/N*np.trace(ep@ep.T))**0.5)
# ep = p[:N]-x3
# print("RMSE pos ESKF", (1/N*np.trace(ep@ep.T))**0.5)
# ev = v[:N]-vel
# print("RMSE vel", (1/N*np.trace(ev@ev.T))**0.5)
# ev = v[:N]-vel2
# print("RMSE vel ESKF", (1/N*np.trace(ev@ev.T))**0.5)
# ea = euler_gt - euler
# print("RMSE euler", (1/N*np.trace(ea@ea.T))**0.5)
# ea = euler_gt - euler2
# print("RMSE euler ESKF", (1/N*np.trace(ea@ea.T))**0.5)

# print(exp_NEES(T_pred2[500], SE3_2(SO3(rot[500]), v[500], p[500])))
# print(exp_NEES(T_pred2[501], SE3_2(SO3(rot[501]), v[501], p[501])))

plt.show()