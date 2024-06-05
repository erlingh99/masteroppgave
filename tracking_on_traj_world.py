import numpy as np
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2

from SE23.agent import Agent
from SE23.measurements import IMU_Measurement, GNSS_Measurement, TargetMeasurement
from SE23.target import TargetWorld, TargetBody, TargetWorldNaive
from SE23.lie_theory import SE3_2, SO3, SE3, SO3xR3xR3
from SE23.states import PlatformState, TargetState
from SE23.plot_utils import plot_3d_frame

np.random.seed(42)

alpha = 0.05

N = 10_001
N = min(N, 29_999)

g = np.array([0, 0, 9.81])

gt = np.load("./data/example_trajectory.npy", allow_pickle=True).item()
dt = gt["dt"]
p = gt["pos"]
v = gt["vel"]
rot = gt["rot"]
acc = gt["acc"]
gyro = gt["gyro"]

acc_noise_std = np.array([1e-2, 1e-2, 1e-2])
gyro_noise_std = np.array([1e-2, 1e-2, 1e-2])*10

gnss_std = 2
gnss_std_up = 2
GNSS_dt = 5 #in seconds

radar_noise_std = 2
radar_dt = 1 #in seconds

IMU_noise = np.diag([*gyro_noise_std, *acc_noise_std])**2 #imu noise, gyro, acc
GNSS_noise = np.diag([gnss_std, gnss_std, gnss_std_up])**2
radar_noise = np.diag([radar_noise_std]*3)**2

#Init poses
T_pred = np.empty(N, dtype=PlatformState)
T_pred_ESKF = np.empty(N, dtype=PlatformState)

init_cov = np.diag([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])**2

#true start pos
T0 = SE3_2(SO3(rot[0]), v[0], p[0])
T0_ESKF = SO3xR3xR3(SO3(rot[0]), v[0], p[0])

#draw random start perturbation
init_perturbs = multivariate_normal([0]*9, init_cov)
T0 = T0@SE3_2.Exp(init_perturbs)
T0_ESKF = SO3xR3xR3.from_matrix(T0.as_matrix())
init_state = PlatformState(T0, init_cov) #perturb our calculation starting point
init_state_ESKF = PlatformState(T0_ESKF, init_cov)
T_pred[0] = init_state
T_pred_ESKF[0] = init_state_ESKF

#create the agent
agent = Agent(IMU_noise, GNSS_noise, radar_noise, init_state)
agent_ESKF = Agent(IMU_noise, GNSS_noise, radar_noise, init_state_ESKF)

#create target
pt_0 = np.array([-100, 100, 100])
vt_0 = np.array([10, 0, 0])
init_target_cov = np.diag([2, 2, 2, 2, 2, 2])**2
init_target_mean = multivariate_normal(np.array([*pt_0, *vt_0]), init_target_cov)
init_target_pose = TargetState(init_target_mean, init_target_cov)
init_target_pose_ESKF = TargetState(init_target_mean.copy(), init_target_cov.copy())

cv_velocity_variance = 2**2

Fcv = np.block([[       np.eye(3), dt*np.eye(3)],
                [np.zeros((3, 3)),    np.eye(3)]])

Q = np.block([[dt**3/3*np.eye(3), dt**2/2*np.eye(3)],
              [dt**2/2*np.eye(3), dt*np.eye(3)]])*cv_velocity_variance

target_state_gt = np.empty((N, 6))
target_state_gt[0] = np.array([*pt_0, *vt_0])

TARGET_ID = 0
target = TargetBody(id=TARGET_ID, var_acc=cv_velocity_variance, state=init_target_pose, cls=SE3_2)
target_ESKF = TargetBody(id=TARGET_ID, var_acc=cv_velocity_variance, state=init_target_pose_ESKF, cls=SO3xR3xR3)

# target_state = np.empty(N, dtype=PlatformState)
target_state = np.empty(N, dtype=TargetState)
# target_state_ESKF = np.empty(N, dtype=PlatformState)
target_state_ESKF = np.empty(N, dtype=TargetState)
# target_state[0] = target.convert_state_to_world_manifold(init_state)
target_state[0] = target.convert_state_to_world_lin(init_state)
# target_state_ESKF[0] = target_ESKF.convert_state_to_world_manifold(init_state_ESKF)
target_state_ESKF[0] = target_ESKF.convert_state_to_world_lin(init_state_ESKF)

TARGET_ID_WORLD = 1
target_world = TargetWorld(id=TARGET_ID_WORLD, var_acc=cv_velocity_variance, state=target.convert_state_to_world_lin(init_state))
target_world_ESKF = TargetWorld(id=TARGET_ID_WORLD, var_acc=cv_velocity_variance, state=target_ESKF.convert_state_to_world_lin(init_state_ESKF))

target_world_state = np.empty(N, dtype=TargetState)
target_world_state_ESKF = np.empty(N, dtype=TargetState)
target_world_state[0] = target_world.state
target_world_state_ESKF[0] = target_world_ESKF.state


### assuming ground truth tracker (world)
TARGET_ID_WORLD_NAIVE = 2
target_naive = TargetWorldNaive(TARGET_ID_WORLD_NAIVE, var_acc=cv_velocity_variance, state=target.convert_state_to_world_lin(init_state))
target_naive_ESKF = TargetWorldNaive(TARGET_ID_WORLD_NAIVE, var_acc=cv_velocity_variance, state=target.convert_state_to_world_lin(init_state_ESKF))
target_naive_state = np.empty(N, dtype=TargetState)
target_naive_state_ESKF = np.empty(N, dtype=TargetState)
target_naive_state[0] = target_naive.state
target_naive_state_ESKF[0] = target_naive_ESKF.state

###Generate measurements
def generate_IMU_measurement(k):
    gyrom = multivariate_normal(gyro[k], IMU_noise[:3, :3])
    accm = multivariate_normal(acc[k] + rot[k].T@g, IMU_noise[3:, 3:])
    return IMU_Measurement(gyrom, accm) #generate IMU measurement at time t

def generate_GNSS_measurement(k):
    gnss = multivariate_normal(p[k], GNSS_noise)
    return GNSS_Measurement(gnss) #generate IMU measurement at time t

def generate_radar_measurement(k, target_pos_gt):
    rel_pos = multivariate_normal(rot[k].T@(target_pos_gt - p[k]), radar_noise)
    return TargetMeasurement(rel_pos)

gnss_pos = []
radar_pos = []

agent.add_target(target)
agent.add_target(target_world)
agent_ESKF.add_target(target_ESKF)
agent_ESKF.add_target(target_world_ESKF)

#propagate and simulate
for k in tqdm(range(N - 1)):
    if (k*dt)%GNSS_dt == 0 and k > 0: #gnss measurement
        z_gnss = generate_GNSS_measurement(k)
        gnss_pos.append(z_gnss.pos)

        agent.platform_update(z_gnss)
        T_pred[k] = agent.state

        agent_ESKF.platform_update(z_gnss)
        T_pred_ESKF[k] = agent_ESKF.state

    if (k*dt)%radar_dt == 0 and k > 0:
        y_target = generate_radar_measurement(k, target_state_gt[k, :3])
        # radar_pos.append((T_pred[k].mean@y_target.relative_pos, target_state_gt[k, :3]))
        radar_pos.append((T_pred[k].mean@y_target.relative_pos, T_pred_ESKF[k].mean@y_target.relative_pos, target_state_gt[k, :3]))

        agent.target_update(TARGET_ID, y_target)
        # target_state[k] = agent.targets[0].convert_state_to_world_manifold(agent.state)
        target_state[k] = agent.targets[0].convert_state_to_world_lin(agent.state)

        agent.target_update(TARGET_ID_WORLD, y_target)
        target_world_state[k] = agent.targets[1].state


        agent_ESKF.target_update(TARGET_ID, y_target)
        # target_state_ESKF[k] = agent_ESKF.targets[0].convert_state_to_world_manifold(agent_ESKF.state)
        target_state_ESKF[k] = agent_ESKF.targets[0].convert_state_to_world_lin(agent_ESKF.state)

        agent_ESKF.target_update(TARGET_ID_WORLD, y_target)
        target_world_state_ESKF[k] = agent_ESKF.targets[1].state

        target_naive.update(agent.state, y_target, radar_noise)
        target_naive_state[k] = target_naive.state

        target_naive_ESKF.update(agent_ESKF.state, y_target, radar_noise)
        target_naive_state_ESKF[k] = target_naive_ESKF.state

    z_imu = generate_IMU_measurement(k)

    agent.propegate(z_imu, dt)
    T_pred[k+1] = agent.state
    # target_state[k+1] = agent.targets[0].convert_state_to_world_manifold(agent.state)
    target_state[k+1] = agent.targets[0].convert_state_to_world_lin(agent.state)
    target_world_state[k+1] = agent.targets[1].state

    agent_ESKF.propegate(z_imu, dt)
    T_pred_ESKF[k+1] = agent_ESKF.state
    # target_state_ESKF[k+1] = agent_ESKF.targets[0].convert_state_to_world_manifold(agent_ESKF.state)
    target_state_ESKF[k+1] = agent_ESKF.targets[0].convert_state_to_world_lin(agent_ESKF.state)
    target_world_state_ESKF[k+1] = agent_ESKF.targets[1].state

    target_naive.propegate(dt)
    target_naive_state[k+1] = target_naive.state
    target_naive_ESKF.propegate(dt)
    target_naive_state_ESKF[k+1] = target_naive_ESKF.state

    target_state_gt[k+1] = Fcv@target_state_gt[k] + multivariate_normal([0]*6, Q)
    
#plotting
fig = plt.figure()
ax = fig.add_subplot(projection="3d")

for gp in gnss_pos:
    ax.plot(*gp, "bx")

# for ypw, ypwh in radar_pos:
for ypw, ypw_eskf, ypwh in radar_pos:
    ax.plot(*ypw, "bx")
    ax.plot(*ypw_eskf, "gx")
    ax.plot(*ypwh, "bo")

pos = np.empty((N, 3))
pos_ESKF = np.empty((N, 3))
vel = np.empty((N, 3))
vel_ESKF = np.empty((N, 3))
euler = np.empty((N, 3))
euler_ESKF = np.empty((N, 3))

euler_gt = np.empty((N, 3))

pos_t = np.empty((N, 3))
vel_t = np.empty((N, 3))
pos_t_ESKF = np.empty((N, 3))
vel_t_ESKF = np.empty((N, 3))

pos_t_w = np.empty((N, 3))
vel_t_w = np.empty((N, 3))
pos_t_ESKF_w = np.empty((N, 3))
vel_t_ESKF_w = np.empty((N, 3))

pos_t_n = np.empty((N, 3))
vel_t_n = np.empty((N, 3))
pos_t_ESKF_n = np.empty((N, 3))
vel_t_ESKF_n = np.empty((N, 3))

for i in range(N):
    pos[i] = T_pred[i].mean.p
    pos_ESKF[i] = T_pred_ESKF[i].mean.p

    vel[i] = T_pred[i].mean.v
    vel_ESKF[i] = T_pred_ESKF[i].mean.v

    euler[i] = T_pred[i].mean.R.as_euler()
    euler_ESKF[i] = T_pred_ESKF[i].mean.R.as_euler()

    euler_gt[i] = SO3.from_matrix(rot[i]).as_euler()

    pos_t[i] = target_state[i].pos
    pos_t_ESKF[i] = target_state_ESKF[i].pos
    vel_t[i] = target_state[i].vel
    vel_t_ESKF[i] = target_state_ESKF[i].vel
    
    pos_t_w[i] = target_world_state[i].pos
    pos_t_ESKF_w[i] = target_world_state_ESKF[i].pos
    vel_t_w[i] = target_world_state[i].vel
    vel_t_ESKF_w[i] = target_world_state_ESKF[i].vel

    pos_t_n[i] = target_naive_state[i].pos
    pos_t_ESKF_n[i] = target_naive_state_ESKF[i].pos
    vel_t_n[i] = target_naive_state[i].vel
    vel_t_ESKF_n[i] = target_naive_state_ESKF[i].vel


ax.plot(*pos.T, "r--", alpha=1)
ax.plot(*pos_ESKF.T, "g--", alpha=1)

ax.plot(*pos_t.T, "r--", alpha=1)
ax.plot(*pos_t_ESKF.T, "g--", alpha=1)

ax.plot(*pos_t_w.T, "r--", alpha=1)
ax.plot(*pos_t_ESKF_w.T, "g--", alpha=1)

ax.plot(*pos_t_n.T, "r--", alpha=1)
ax.plot(*pos_t_ESKF_n.T, "g--", alpha=1)

# print("\nMean norm of error between pos and gt pos:", np.linalg.norm(p[:N] - pos, axis=1).mean())
# print("Frobenius norm of difference of last covariances", np.linalg.norm(T_pred[-1].cov - T_pred_ESKF[-1].cov, ord="fro"))


for i in range(0, N, 400):
    target_state[i].draw_significant_ellipses(ax, color="orange")
    target_world_state[i].draw_significant_ellipses(ax, color="red")

    target_state_ESKF[i].draw_significant_ellipses(ax, color="green")
    target_world_state_ESKF[i].draw_significant_ellipses(ax, color="blue")

    target_naive_state[i].draw_significant_ellipses(ax, color="yellow")
    target_naive_state_ESKF[i].draw_significant_ellipses(ax, color="pink")

for i in range(499, N+500, 500):
    idx = min(i, N-2)

    T_pred[idx].draw_significant_ellipses(ax, color="red")
    plot_3d_frame(ax, SE3(T_pred[idx].mean.R, T_pred[idx].mean.p), scale=10)
    
    T_pred_ESKF[idx].draw_significant_ellipses(ax, color="green")
    plot_3d_frame(ax, SE3(T_pred_ESKF[idx].mean.R, T_pred_ESKF[idx].mean.p), scale=10)

    idx = idx + 1

    T_pred[idx].draw_significant_ellipses(ax, color="red")
    plot_3d_frame(ax, SE3(T_pred[idx].mean.R, T_pred[idx].mean.p), scale=10)

    T_pred_ESKF[idx].draw_significant_ellipses(ax, color="green")
    plot_3d_frame(ax, SE3(T_pred_ESKF[idx].mean.R, T_pred_ESKF[idx].mean.p), scale=10)

    
plot_3d_frame(ax, SE3(SO3(rot[N]), p[N]), scale=5)
plot_3d_frame(ax, SE3(SO3(rot[0]), p[0]), scale=5)
plot_3d_frame(ax, SE3(init_state.rot, init_state.pos), scale=5)

#plot gt
ax.plot(*p[:N].T, "k--")
ax.plot(*p[N].T, "ko")
ax.plot(*target_state_gt[:, :3].T, "--", color="orange")
plt.axis("equal")
# plt.show()



#calc NEES
NEES = np.empty(N, float)
NEES_vel = np.empty(N, float)
NEES_pos = np.empty(N, float)

NEES_world = np.empty(N, float)
NEES_world_vel = np.empty(N, float)
NEES_world_pos = np.empty(N, float)

NEES_ESKF = np.empty(N, float)
NEES_ESKF_vel = np.empty(N, float)
NEES_ESKF_pos = np.empty(N, float)

NEES_world_ESKF = np.empty(N, float)
NEES_world_ESKF_vel = np.empty(N, float)
NEES_world_ESKF_pos = np.empty(N, float)

NEES_naive = np.empty(N, float)
NEES_naive_vel = np.empty(N, float)
NEES_naive_pos = np.empty(N, float)

NEES_naive_ESKF = np.empty(N, float)
NEES_naive_ESKF_vel = np.empty(N, float)
NEES_naive_ESKF_pos = np.empty(N, float)

for k in range(N):
    # m, c = target_state[k].mean, target_state[k].cov[3:, 3:]
    # err = m.inverse().action2(target_state_gt[k])
    m, c = target_state[k].mean, target_state[k].cov
    err = target_state_gt[k] - m
    NEES[k] = err.T@np.linalg.solve(c, err)
    NEES_pos[k] = err[:3].T@np.linalg.solve(c[:3, :3], err[:3])
    NEES_vel[k] = err[3:].T@np.linalg.solve(c[3:, 3:], err[3:])
    
    m, c = target_world_state[k].mean, target_world_state[k].cov
    err = target_state_gt[k] - m
    NEES_world[k] = err.T@np.linalg.solve(c, err)
    NEES_world_pos[k] = err[:3].T@np.linalg.solve(c[:3, :3], err[:3])
    NEES_world_vel[k] = err[3:].T@np.linalg.solve(c[3:, 3:], err[3:])

    # m, c = target_state_ESKF[k].mean, target_state_ESKF[k].cov[3:, 3:]
    # err = m.inverse().action2(target_state_gt[k])
    m, c = target_state_ESKF[k].mean, target_state_ESKF[k].cov
    err = target_state_gt[k] - m
    NEES_ESKF[k] = err.T@np.linalg.solve(c, err)
    NEES_ESKF_pos[k] = err[:3].T@np.linalg.solve(c[:3, :3], err[:3])
    NEES_ESKF_vel[k] = err[3:].T@np.linalg.solve(c[3:, 3:], err[3:])

    m, c = target_world_state_ESKF[k].mean, target_world_state_ESKF[k].cov
    err = target_state_gt[k] - m
    NEES_world_ESKF[k] = err.T@np.linalg.solve(c, err)
    NEES_world_ESKF_pos[k] = err[:3].T@np.linalg.solve(c[:3, :3], err[:3])
    NEES_world_ESKF_vel[k] = err[3:].T@np.linalg.solve(c[3:, 3:], err[3:])

    m, c = target_naive_state[k].mean, target_naive_state[k].cov
    err = target_state_gt[k] - m
    NEES_naive[k] = err.T@np.linalg.solve(c, err)
    NEES_naive_pos[k] = err[:3].T@np.linalg.solve(c[:3, :3], err[:3])
    NEES_naive_vel[k] = err[3:].T@np.linalg.solve(c[3:, 3:], err[3:])


    m, c = target_naive_state_ESKF[k].mean, target_naive_state_ESKF[k].cov
    err = target_state_gt[k] - m
    NEES_naive_ESKF[k] = err.T@np.linalg.solve(c, err)
    NEES_naive_ESKF_pos[k] = err[:3].T@np.linalg.solve(c[:3, :3], err[:3])
    NEES_naive_ESKF_vel[k] = err[3:].T@np.linalg.solve(c[3:, 3:], err[3:])



CI_NEES = chi2.interval(1 - alpha, 3)
CI_ANEES = chi2.interval(1 - alpha, 3 * N, scale=1/N)
print(f"\nCI NEES: {CI_NEES}")
print(f"CI ANEES: {CI_ANEES}")

fig, ax = plt.subplots(1,1)#, figsize=(8, 5), num=4, clear=True, sharey=True)
ax.set_yscale("log")
ax.plot(NEES_pos, lw=0.5, label="SE_2(3) + body")
ax.plot(NEES_world_pos, lw=0.5, label="SE_2(3) + world")
ax.plot(NEES_ESKF_pos, lw=0.5, label="ESKF + body")
ax.plot(NEES_world_ESKF_pos, lw=0.5, label="ESKF + world")
ax.plot(NEES_naive_pos, lw=0.5, label="Naive SE_2(3) + world")
ax.plot(NEES_naive_ESKF_pos, lw=0.5, label="Naive ESKF + world")
ax.plot(np.full(N, CI_NEES[0]), "r--")
ax.plot(np.full(N, CI_NEES[1]), "r--")
ax.plot(np.full(N, CI_ANEES[0]), "g--")
ax.plot(np.full(N, CI_ANEES[1]), "g--")
ax.set_title(f"NEES position all filters")
ax.legend()

fig, ax = plt.subplots(1,1)#, figsize=(8, 5), num=4, clear=True, sharey=True)
ax.set_yscale("log")
ax.plot(NEES_vel, lw=0.5, label="SE_2(3) + body")
ax.plot(NEES_world_vel, lw=0.5, label="SE_2(3) + world")
ax.plot(NEES_ESKF_vel, lw=0.5, label="ESKF + body")
ax.plot(NEES_world_ESKF_vel, lw=0.5, label="ESKF + world")
ax.plot(NEES_naive_vel, lw=0.5, label="Naive SE_2(3) + world")
ax.plot(NEES_naive_ESKF_vel, lw=0.5, label="Naive ESKF + world")
ax.plot(np.full(N, CI_NEES[0]), "r--")
ax.plot(np.full(N, CI_NEES[1]), "r--")
ax.plot(np.full(N, CI_ANEES[0]), "g--")
ax.plot(np.full(N, CI_ANEES[1]), "g--")
ax.set_title(f"NEES velocity all filters")
ax.legend()


insideCI = (CI_NEES[0] <= NEES_pos) * (NEES_pos <= CI_NEES[1])
percents = insideCI.mean()*100
ANEES = NEES_pos.mean()
print("\nSE_2(3) + body, pos")
print(f"Percentage of NEES inside bounds {percents}%")
print(f"ANEES: {ANEES}")

insideCI_ESKF = (CI_NEES[0] <= NEES_ESKF_pos) * (NEES_ESKF_pos <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_ESKF_pos.mean()
print("\nESKF + body, pos")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")

insideCI = (CI_NEES[0] <= NEES_world_pos) * (NEES_world_pos <= CI_NEES[1])
percents = insideCI.mean()*100
ANEES = NEES_world_pos.mean()
print("\nSE_2(3) + world, pos")
print(f"Percentage of NEES inside bounds {percents}%")
print(f"ANEES: {ANEES}")

insideCI_ESKF = (CI_NEES[0] <= NEES_world_ESKF_pos) * (NEES_world_ESKF_pos <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_world_ESKF_pos.mean()
print("\nESKF + world, pos")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")

insideCI_ESKF = (CI_NEES[0] <= NEES_naive_pos) * (NEES_naive_pos <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_naive_pos.mean()
print("\nNaive SE_2(3) + world, pos")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")

insideCI_ESKF = (CI_NEES[0] <= NEES_naive_ESKF_pos) * (NEES_naive_ESKF_pos <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_naive_ESKF_pos.mean()
print("\nNaive ESKF + world, pos")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")


insideCI = (CI_NEES[0] <= NEES_vel) * (NEES_vel <= CI_NEES[1])
percents = insideCI.mean()*100
ANEES = NEES_vel.mean()
print("\nSE_2(3) + body, vel")
print(f"Percentage of NEES inside bounds {percents}%")
print(f"ANEES: {ANEES}")

insideCI_ESKF = (CI_NEES[0] <= NEES_ESKF_vel) * (NEES_ESKF_vel <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_ESKF_vel.mean()
print("\nESKF + body, vel")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")

insideCI = (CI_NEES[0] <= NEES_world_vel) * (NEES_world_vel <= CI_NEES[1])
percents = insideCI.mean()*100
ANEES = NEES_world_vel.mean()
print("\nSE_2(3) + world, vel")
print(f"Percentage of NEES inside bounds {percents}%")
print(f"ANEES: {ANEES}")

insideCI_ESKF = (CI_NEES[0] <= NEES_world_ESKF_vel) * (NEES_world_ESKF_vel <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_world_ESKF_vel.mean()
print("\nESKF + world, vel")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")

insideCI_ESKF = (CI_NEES[0] <= NEES_naive_vel) * (NEES_naive_vel <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_naive_vel.mean()
print("\nNaive SE_2(3) + world, vel")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")

insideCI_ESKF = (CI_NEES[0] <= NEES_naive_ESKF_vel) * (NEES_naive_ESKF_vel <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_naive_ESKF_vel.mean()
print("\nNaive ESKF + world, vel")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")



CI_NEES = chi2.interval(1 - alpha, 6)
CI_ANEES = chi2.interval(1 - alpha, 6 * N, scale=1/N)
print(f"\nCI NEES: {CI_NEES}")
print(f"CI ANEES: {CI_ANEES}")

fig, ax = plt.subplots(1,1)#, figsize=(8, 5), num=4, clear=True, sharey=True)
ax.set_yscale("log")
ax.plot(NEES, lw=0.5, label="SE_2(3) + body")
ax.plot(NEES_world, lw=0.5, label="SE_2(3) + world")
ax.plot(NEES_ESKF, lw=0.5, label="ESKF + body")
ax.plot(NEES_world_ESKF, lw=0.5, label="ESKF + world")
ax.plot(NEES_naive, lw=0.5, label="Naive SE_2(3) + world")
ax.plot(NEES_naive_ESKF, lw=0.5, label="Naive ESKF + world")
ax.plot(np.full(N, CI_NEES[0]), "r--")
ax.plot(np.full(N, CI_NEES[1]), "r--")
ax.plot(np.full(N, CI_ANEES[0]), "g--")
ax.plot(np.full(N, CI_ANEES[1]), "g--")
ax.set_title(f"NEES total all filters")
ax.legend()


insideCI = (CI_NEES[0] <= NEES) * (NEES <= CI_NEES[1])
percents = insideCI.mean()*100
ANEES = NEES.mean()
print("\nSE_2(3) + body, total")
print(f"Percentage of NEES inside bounds {percents}%")
print(f"ANEES: {ANEES}")

insideCI_ESKF = (CI_NEES[0] <= NEES_ESKF) * (NEES_ESKF <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_ESKF.mean()
print("\nESKF + body, total")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")

insideCI = (CI_NEES[0] <= NEES_world) * (NEES_world <= CI_NEES[1])
percents = insideCI.mean()*100
ANEES = NEES_world.mean()
print("\nSE_2(3) + world, total")
print(f"Percentage of NEES inside bounds {percents}%")
print(f"ANEES: {ANEES}")

insideCI_ESKF = (CI_NEES[0] <= NEES_world_ESKF) * (NEES_world_ESKF <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_world_ESKF.mean()
print("\nESKF + world, total")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")

insideCI = (CI_NEES[0] <= NEES_naive) * (NEES_naive <= CI_NEES[1])
percents = insideCI.mean()*100
ANEES = NEES_naive.mean()
print("\nNaive SE_2(3) + world, total")
print(f"Percentage of NEES inside bounds {percents}%")
print(f"ANEES: {ANEES}")

insideCI_ESKF = (CI_NEES[0] <= NEES_naive_ESKF) * (NEES_naive_ESKF <= CI_NEES[1])
percents_ESKF = insideCI_ESKF.mean()*100
ANEES_ESKF = NEES_naive_ESKF.mean()
print("\nNaive ESKF + world, total")
print(f"Percentage of NEES inside bounds {percents_ESKF}%")
print(f"ANEES: {ANEES_ESKF}")



fig, axs = plt.subplots(2, 3)
axs = axs.flatten()

for i, ax in enumerate(axs):
    if i < 3:
        ax.plot(target_state_gt[:, i], "--", color="orange")
        ax.plot(pos_t[:, i], color="red")
        ax.plot(pos_t_ESKF[:, i], color="green")
        ax.plot(pos_t_w[:, i], color="red")
        ax.plot(pos_t_ESKF_w[:, i], color="green")
        ax.plot(pos_t_n[:, i], color="red")
        ax.plot(pos_t_ESKF_n[:, i], color="green")
        ax.set_title(f"position {'xyz'[i]}")
    elif i < 6:
        ax.plot(target_state_gt[:, i], "--", color="orange")
        ax.plot(vel_t[:, i-3], color="red")
        ax.plot(vel_t_ESKF[:, i-3], color="green")
        ax.plot(vel_t_w[:, i-3], color="red")
        ax.plot(vel_t_ESKF_w[:, i-3], color="green")
        ax.plot(vel_t_n[:, i-3], color="red")
        ax.plot(vel_t_ESKF_n[:, i-3], color="green")
        ax.set_title(f"velocity {'xyz'[i-3]}")

plt.show()