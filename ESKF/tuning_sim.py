import numpy as np

from .eskf import ESKF
from .models import ModelIMU
from .sensors import SensorGNSS
from .states import EskfState, NominalState, ErrorState, RotationQuaterion
from .senfuslib import MultiVarGauss


"""Everything below here can be altered"""
start_time_sim = 0.  # Start time, set to None for full time
end_time_sim = 200  # End time in seconds, set to None to use all data

imu_min_dt_sim = None  # IMU is sampled at 100 Hz, use to downsample
gnss_min_dt_sim = 5  # GPS is sampled at 1 Hz, use this to downsample

imu_sim = ModelIMU(
    accm_std=3e-5,   # Accelerometer standard deviation
    gyro_std=3e-5  # Gyro standard deviation
)

gnss_sim = SensorGNSS(
    gnss_std_ne=2,  # GNSS standard deviation in North and East
    gnss_std_d=2,  # GNSS standard deviation in Down
)

x_est_init_nom_sim = NominalState(
    pos=np.array([0.2, 0, -5]),  # position
    vel=np.array([20, 0, 0]),  # velocity
    ori=RotationQuaterion.from_euler([0, 0, 0.05])  # orientation
)

x_err_init_std_sim = np.repeat(repeats=3, a=[
    0,  # position
    0,  # velocity
    0.2  # angle vector
])


"""Dont change anything below here"""
x_est_init_err_sim = MultiVarGauss[ErrorState](
    np.zeros(9),
    np.diag(x_err_init_std_sim**2))


eskf_sim = ESKF(imu_sim, gnss_sim) 
x_est_init_sim = EskfState(x_est_init_nom_sim, x_est_init_err_sim)