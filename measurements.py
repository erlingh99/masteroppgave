from dataclasses import dataclass
import numpy as np


@dataclass
class IMU_Measurement:
    gyro: np.ndarray[3]
    acc: np.ndarray[3]
    noise: np.ndarray[6, 6]

@dataclass
class GNSS_Measurement:
    pos: np.ndarray[3]
    noise: np.ndarray[3, 3]
    

##add lever arm
class GNSS_Sensor:
    def h(self, T):
        return T.p
    
    def H(self, state):
        return np.block([np.zeros((3,6)), state.R.as_matrix()])