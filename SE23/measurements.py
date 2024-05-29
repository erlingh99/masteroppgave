from dataclasses import dataclass
import numpy as np

@dataclass
class GNSS_Measurement:
    pos: np.ndarray[3]

@dataclass
class TargetMeasurement:
    relative_pos: np.ndarray[3]

@dataclass
class IMU_Measurement:
    gyro: np.ndarray[3]
    acc: np.ndarray[3]