import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2, norm

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement, GNSS_Measurement
from SE23.lie_theory import SE3_2, SO3, SE2, SO2, SO3xR3xR3, SO2xR2
from SE23.states import PlatformState
from SE23.utils import find_mean, exp_cov, exp_NEES
from SE23.plot_utils import *

# np.random.seed(10) #0 below, 1 both?
alpha = 0.05


n_steps = 100
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
IMU_noise = np.diag([0, 0, 0.03/dt, 0.4, 0.4, 0])**2 #imu noise, gyro, acc
GNSS_noise = np.diag([3, 3, 0.001])**2
radar_noise = 0 # not used in this example

#create array to save the state at each timestep
agent_state = np.empty((n_steps, n_random), dtype=PlatformState)
agent_state_ESKF = np.empty((n_steps, n_random), dtype=PlatformState)
#create array to keep all agents in
agents = np.empty(n_random, dtype=Agent)
agents_ESKF = np.empty(n_random, dtype=Agent)

#Init poses
# init_cov = np.diag([0, 0, 0.001, 0, 0, 0, 0.1, 0.1, 0])**2
init_cov = np.diag([0, 0, 0.1, 0, 0, 0, 0.1, 0.1, 0])**2
# init_cov = np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0])**2
T0 = SE3_2(Rot(0), v(0), p(0)) #true start pos
T0_ESKF = SO3xR3xR3(Rot(0), v(0), p(0)) #true start pos

#draw random start pertrbations and create agents
init_perturbs = multivariate_normal([0]*9, init_cov, n_random) #random perturbations

for i in range(n_random):
    init_state = PlatformState(T0@SE3_2.Exp(init_perturbs[i]), init_cov)
    init_state_ESKF = PlatformState(T0_ESKF@SO3xR3xR3.Exp(init_perturbs[i]), init_cov)
    agents[i] = Agent(IMU_noise, GNSS_noise, radar_noise, init_state.copy()) 
    agents_ESKF[i] = Agent(IMU_noise, GNSS_noise, radar_noise, init_state_ESKF.copy()) 
    agent_state[0, i] = init_state.copy()
    agent_state_ESKF[0, i] = init_state_ESKF.copy()


###Generate measurments
#imu measurements
gyro = lambda t, n: multivariate_normal(w(t), IMU_noise[:3, :3], size=n) #generate n gyro measurements at time t
acc = lambda t, n: multivariate_normal(Rot(t).T@(a(t) + g), IMU_noise[3:, 3:], size=n) #imu measures g up, acc is in body
generate_IMU_measurements = lambda t, n: [IMU_Measurement(g_t, acc_t) for g_t, acc_t in zip(gyro(t, n), acc(t, n))] #generate n IMU measurments at time t
#GNSS measurements
pos_m = lambda t: multivariate_normal(p(t), GNSS_noise) #measured position
gnssMeasurement = lambda t: GNSS_Measurement(pos_m(t)) #create gnss measurement with noise

ws = np.empty(n_random)
updated_agent_state = np.empty(n_random, dtype=PlatformState)
updated_agent_state_ESKF = np.empty(n_random, dtype=PlatformState)

#propegate and simulate
for k in tqdm(range(1, n_steps)):
    z_imu = generate_IMU_measurements(k*dt, n_random) #sample n_random different inputs
    #propegate all agents with the corresponding random IMU measurement, save the state
    for i in range(n_random):
        agents[i].propegate(z_imu[i], dt)
        agents_ESKF[i].propegate(z_imu[i], dt)
        agent_state[k,i] = agents[i].state.copy()
        agent_state_ESKF[k,i] = agents_ESKF[i].state.copy()

    if (k+1)%100 == 0:
        z_gps = gnssMeasurement(k*dt)
        for i in range(n_random):
            #prob agent i
            m = z_gps.pos - agents[i].state.pos
            ws[i] = np.exp(-0.5*m.T@np.linalg.solve(GNSS_noise, m))

            agents[i].platform_update(z_gps)
            agents_ESKF[i].platform_update(z_gps)
            updated_agent_state[i] = agents[i].state.copy()
            updated_agent_state_ESKF[i] = agents_ESKF[i].state.copy()



# calculate SE3_2 distribution of simulated samples
final_means = np.array([s.mean for s in agent_state[-1, :]])

#calculate the eperical mean
mean = find_mean(final_means, agents[-1].state.mean, weights=ws) #find mean, with the predicted mean of one of the agents as initial guess
sim_cov = exp_cov(final_means, mean, weights=ws)
sim_pose = PlatformState(mean, sim_cov)

# calculate SO3xR3xR3 distribution of simulated samples
final_means_ESKF = np.array([s.mean for s in agent_state_ESKF[-1, :]])
#calculate the eperical mean
mean_ESKF = find_mean(final_means_ESKF, agents_ESKF[-1].state.mean, weights=ws) #find mean, with the predicted mean of one of the agents as initial guess
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
        t[k, :] = agent_state[k, i].mean.t[:2]
    ax.plot(*t.T, color='gray', alpha=0.1)
    ax.scatter(*final_means[i].p[:2] , s=2, color='black')
ax.plot(*z_gps.pos[:2], 'bx')

plot_as_SE2(ax, updated_agent_state[0], color="red")
# plot_as_SO2xR2(ax, updated_agent_state_ESKF[0], color="pink")
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

def asSE2(pose: ExponentialGaussian):
    extract = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0]])
    m = pose.mean.as_matrix()
    c = extract@pose.cov@extract.T
    pose2D = SE2(SO2(m[:2, :2]), m[:2, 4])
    return ExponentialGaussian(pose2D, c)

def asSO2xR2(pose: ExponentialGaussian):
    extract = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0]])
    m = pose.mean.as_matrix()
    c = extract@pose.cov@extract.T
    pose2D = SO2xR2(SO2(m[:2, :2]), m[:2, 4])
    return ExponentialGaussian(pose2D, c)

#calc NEES
NEES = np.empty((n_steps, n_random), float)
NEES_ESKF = np.empty((n_steps, n_random), float)

for k in range(n_steps):

    gt_mean = SE2(SO2(Rot(k*dt).as_matrix()[:2, :2]), p(k*dt)[:2])
    gt_mean_ESKF = SO2xR2(SO2(Rot(k*dt).as_matrix()[:2, :2]), p(k*dt)[:2])
    # gt_mean = SE3_2(Rot(k*dt), v(k*dt), p(k*dt))

    for i in range(n_random):
        NEES[k,i] = exp_NEES(asSE2(agent_state[k,i]), gt_mean)
        NEES_ESKF[k,i] = exp_NEES(asSO2xR2(agent_state_ESKF[k,i]), gt_mean_ESKF)

fig, axs = plt.subplots(nrows=2, ncols=5)#, figsize=(8, 5), num=4, clear=True, sharex=True)
axs = axs.flatten()

# fig2, axs2 = plt.subplots(nrows=10, ncols=10)
# axs2 = axs2.flatten()

df = 3#9

CI_NEES = chi2.interval(1 - alpha, df)
CI_ANEES = chi2.interval(1 - alpha, df * n_steps, scale=1/n_steps)
CI_AANEES = chi2.interval(1 - alpha, df * n_steps * n_random, scale=1/(n_steps*n_random))
print(f"CI NEES: {CI_NEES}")
print(f"CI ANEES: {CI_ANEES}")
print(f"CI AANEES: {CI_AANEES}")

ANEES = np.empty(n_random)
percents = np.empty(n_random)

ANEES_ESKF = np.empty(n_random)
percents_ESKF = np.empty(n_random)

for i in range(n_random):
    NEESi = NEES[:, i]
    NEESi_ESKF = NEES_ESKF[:, i]
    
    if i < 5:
        axs[i].plot(NEESi, lw=0.5)
        axs[i].plot(np.full(n_steps, CI_NEES[0]), "r--")
        axs[i].plot(np.full(n_steps, CI_NEES[1]), "r--")
        axs[i].plot(np.full(n_steps, CI_ANEES[0]), "g--")
        axs[i].plot(np.full(n_steps, CI_ANEES[1]), "g--")
        axs[i].plot(np.full(n_steps, CI_AANEES[0]), "k--")
        axs[i].plot(np.full(n_steps, CI_AANEES[1]), "k--")
        axs[i].set_ylim(0, CI_NEES[1]*1.1)

        axs[i+5].plot(NEESi_ESKF, lw=0.5)
        axs[i+5].plot(np.full(n_steps, CI_NEES[0]), "r--")
        axs[i+5].plot(np.full(n_steps, CI_NEES[1]), "r--")
        axs[i+5].plot(np.full(n_steps, CI_ANEES[0]), "g--")
        axs[i+5].plot(np.full(n_steps, CI_ANEES[1]), "g--")
        axs[i+5].plot(np.full(n_steps, CI_AANEES[0]), "k--")
        axs[i+5].plot(np.full(n_steps, CI_AANEES[1]), "k--")
        axs[i+5].set_ylim(0, CI_NEES[1]*1.1)

    insideCI = (CI_NEES[0] <= NEESi) * (NEESi <= CI_NEES[1])
    percents[i] = insideCI.mean()*100
    ANEES[i] = NEESi.mean()

    insideCI_ESKF = (CI_NEES[0] <= NEESi_ESKF) * (NEESi_ESKF <= CI_NEES[1])
    percents_ESKF[i] = insideCI_ESKF.mean()*100
    ANEES_ESKF[i] = NEESi_ESKF.mean()

insideCIA = (CI_ANEES[0] <= ANEES) * (ANEES <= CI_ANEES[1])
underCIA = ANEES < CI_ANEES[0]
overCIA = ANEES > CI_ANEES[1]
AANEES = NEES.mean()

insideCIA_ESKF = (CI_ANEES[0] <= ANEES_ESKF) * (ANEES_ESKF <= CI_ANEES[1])
underCIA_ESKF = ANEES_ESKF < CI_ANEES[0]
overCIA_ESKF = ANEES_ESKF > CI_ANEES[1]
AANEES_ESKF = NEES_ESKF.mean()

print("\n\nSE2(3)")
print(f"Percentage of NEES inside bounds {percents.mean()}%")
print(f"Percentage of ANEES within bounds {insideCIA.mean()*100}%")
print(f"Percentage of ANEES above bounds {overCIA.mean()*100}%")
print(f"Percentage of ANEES under bounds {underCIA.mean()*100}%")
print(f"AANEES: {AANEES}")


print("\n\nESKF")
print(f"Percentage of NEES inside bounds {percents_ESKF.mean()}%")
print(f"Percentage of ANEES within bounds {insideCIA_ESKF.mean()*100}%")
print(f"Percentage of ANEES above bounds {overCIA_ESKF.mean()*100}%")
print(f"Percentage of ANEES under bounds {underCIA_ESKF.mean()*100}%")
print(f"AANEES: {AANEES_ESKF}")

plt.show()