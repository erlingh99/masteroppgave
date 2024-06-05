from pathlib import Path
import numpy as np
from numpy.random import multivariate_normal
from scipy.io import loadmat
import pickle

from ESKF.states import NominalState
from ESKF.senfuslib import TimeSequence
from ESKF.states import NominalState, ImuMeasurement, GnssMeasurement
from ESKF.config import cache_dir
from ESKF.quaternion import RotationQuaterion


def load_custom(noise_gnss, noise_imu, dt_gnss = 1):
    gt = np.load("./data/example_trajectory.npy", allow_pickle=True).item()
    dt = gt["dt"]
    p = gt["pos"]
    v = gt["vel"]
    rot = gt["rot"]
    acc = gt["acc"]
    gyro = gt["gyro"]

    T = (len(p)-1)*dt

    imu_ts = np.arange(0, T, dt)
    gnss_ts = np.arange(dt_gnss, T, dt_gnss)

    gnss_step = int(dt_gnss/dt)

    correct_ned = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]])

    g = np.array([0, 0, 9.81])

    acc_noise  = np.array([multivariate_normal(acc[i] + rot[i].T@g, noise_imu[3:, 3:]) for i in range(len(p))])
    gyro_noise = np.array([multivariate_normal(gyro[i], noise_imu[:3, :3]) for i in range(len(p))])
    gnss_noise = np.array([multivariate_normal(p[gnss_step*i], noise_gnss) for i in range(1, len(p)//gnss_step)])

    
    x_gt = TimeSequence((ts, NominalState.from_array(np.concatenate([correct_ned@pk, correct_ned@vk, RotationQuaterion.from_matrix(correct_ned@rotk@correct_ned.T)])))
                        for ts, pk, vk, rotk in zip(imu_ts, p, v, rot)).zero(0)
    
    imu_measurements = TimeSequence((ts, ImuMeasurement(correct_ned@accm, correct_ned@gyrom))
                                    for ts, accm, gyrom
                                    in zip(imu_ts, acc_noise, gyro_noise)
                                    ).zero(0)
    
    gnss_measurements = TimeSequence((ts, GnssMeasurement(correct_ned@pos))
                                     for ts, pos
                                     in zip(gnss_ts, gnss_noise)
                                     ).zero(0)
    
    return x_gt, imu_measurements, gnss_measurements    

def load_data(file_name: Path):
    cached_file = cache_dir / f'cached_{file_name.stem}.pkl'
    if cached_file.exists():
        with open(cached_file, 'rb') as f:
            return pickle.load(f)
    print('Parsing data, it will be cached for future use')
    loaded_data = loadmat(file_name)

    timeGNSS = np.round(loaded_data["timeGNSS"].ravel(), 4)
    z_GNSS = loaded_data["zGNSS"].T
    if (gnss_acc := loaded_data.get('GNSSaccuracy', None)) is not None:
        accuracy_GNSS = gnss_acc.ravel()
    else:
        accuracy_GNSS = [None for _ in timeGNSS]
    timeIMU = np.round(loaded_data["timeIMU"].ravel(), 4)
    z_acc = loaded_data["zAcc"].T
    z_avel = loaded_data["zGyro"].T

    t0 = timeIMU[0]
    if 'xtrue' in loaded_data:
        x_true = loaded_data["xtrue"].T
        x_gt = TimeSequence((ts, NominalState.from_array(x))
                            for ts, x in zip(timeIMU, x_true)).zero(t0)
    else:
        x_gt = None

    imu_measurements = TimeSequence((ts, ImuMeasurement(acc, gyro))
                                    for ts, acc, gyro
                                    in zip(timeIMU, z_acc, z_avel)
                                    ).zero(t0)
    gnss_measurements = TimeSequence((ts, GnssMeasurement(pos, acsy))
                                     for ts, pos, acsy
                                     in zip(timeGNSS, z_GNSS, accuracy_GNSS)
                                     ).zero(t0)
    with open(cached_file, 'wb') as f:
        pickle.dump((x_gt, imu_measurements, gnss_measurements), f)

    return x_gt, imu_measurements, gnss_measurements


def load_drone_params(file_name: Path):
    loaded_data = loadmat(file_name)

    S_a = loaded_data["S_a"]  # accm_correction
    S_g = loaded_data["S_g"]  # gyro_correction
    lever_arm = loaded_data["leverarm"].ravel()
    return S_a, S_g, lever_arm
