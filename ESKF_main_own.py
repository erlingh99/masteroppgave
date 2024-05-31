from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from numpy.random import multivariate_normal

from ESKF.states import EskfState, ImuMeasurement, ErrorState, NominalState
from ESKF.senfuslib.gaussian import MultiVarGauss
from ESKF.senfuslib.timesequence import TimeSequence
from ESKF.eskf import ESKF
from ESKF.plotting import PlotterESKF
from ESKF.sensors import SensorGNSS
from ESKF.models import ModelIMU
from ESKF.quaternion import RotationQuaterion

from SE23.lie_theory import SO3


np.random.seed(42)

N = 300

p = lambda t: np.array([t**2, 5*np.sin(t), 0])
v = lambda t: np.array([2*t, 5*np.cos(t), 0])
a = lambda t: np.array([2, -5*np.sin(t), 0])
w = lambda t: np.array([0, 0, 2*np.sin(0.1*t)])
Rot = lambda t: SO3.Exp([0, 0, -20*np.cos(0.1*t)+20])
dt = 0.05

g = np.array([0, 0, 9.81])

acc_noise = np.array([0, 0, 0])
gyro_noise = np.array([0, 0, 0.03/dt])

imu_sim = ModelIMU(
    accm_std=acc_noise,   # Accelerometer standard deviation, TUNEABLE
    gyro_std=gyro_noise  # Gyro standard deviation
)

gnss_sim = SensorGNSS(
    gnss_std_ne=0.3,  # GNSS standard deviation in North and East
    gnss_std_d=0.5,  # GNSS standard deviation in Down
)

x_est_init_nom_sim = NominalState(
    pos=p(0),  # position
    vel=v(0),  # velocity
    ori=RotationQuaterion.from_matrix(Rot(0).as_matrix())  # orientation
)

x_err_init_std_sim = np.repeat(repeats=3, a=[
    0,  # position
    0,  # velocity
    np.deg2rad(0)  # angle vector
])

x_est_init_err_sim = MultiVarGauss[ErrorState](
    np.zeros(9),
    np.diag(x_err_init_std_sim**2))


eskf = ESKF(imu_sim, gnss_sim) 
x_est_init = EskfState(x_est_init_nom_sim, x_est_init_err_sim)


timeIMU = np.arange(N)*dt
x_true = np.empty((N, 10))
for i in range(N):
    q = RotationQuaterion.from_matrix(Rot(i*dt).as_matrix())
    x_true[i] = np.concatenate((p(i*dt), v(i*dt), q))

x_gt = TimeSequence((ts, NominalState.from_array(x))
                    for ts, x in zip(timeIMU, x_true))


gyro = lambda t: multivariate_normal(w(t), np.diag(gyro_noise)**2) #generate n gyro measurements at time t
acc = lambda t: multivariate_normal(Rot(t).T@(a(t) - g), np.diag(acc_noise)**2) #imu measures g up, acc is in body
generate_IMU_measurement = lambda t: ImuMeasurement(acc(t), gyro(t)) #generate IMU measurment at time t


z_imu_tseq = TimeSequence((t, generate_IMU_measurement(t)) for t in timeIMU)


t_prev = z_imu_tseq.times[0]
x_est_prev = x_est_init
x_est_pred_tseq = TimeSequence([(t_prev, x_est_init)])

for t_imu, z_imu in tqdm(z_imu_tseq.items()):
    dt = t_imu - t_prev
    if dt > 0:
        x_est_pred = eskf.predict_from_imu(x_est_prev, z_imu, dt)
        x_est_pred_tseq.insert(t_imu, x_est_pred)
        x_est_prev = x_est_pred
    t_prev = t_imu


print(x_est_pred)


PlotterESKF(x_gts=x_gt,
            z_imu=z_imu_tseq,
            x_preds=x_est_pred_tseq,
            ).show()

plt.show(block=True)












