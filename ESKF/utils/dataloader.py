from pathlib import Path
import numpy as np
from scipy.io import loadmat
import pickle

from states import NominalState
from senfuslib import TimeSequence
from quaternion import RotationQuaterion
from states import NominalState, ImuMeasurement, GnssMeasurement
from config import fname_data_sim, cache_dir


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
