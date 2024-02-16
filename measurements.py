from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from error_state_kalman_filter import ESKF_State


@dataclass
class GNSS_Measurement:
    pos: np.ndarray[3]


@dataclass
class IMU_Measurement:
    acc: np.ndarray[3]
    gyro: np.ndarray[3]


@dataclass
class Sensor(ABC):
    lever_arm: np.ndarray[3]
    R: np.ndarray[3, 3]

    @abstractmethod
    def h(self, state):
        """
        The measurement model. Given a state, returns the predicted measurement.
        """
        pass

    @abstractmethod
    def H(self) -> ESKF_State:
        """
        The jacobian of the measurement model.
        """
        pass

    def lever_arm_compensate(self, state: ESKF_State) -> np.ndarray[3]:
        """
        Compensation of offset sensors compared to body frame.
        """
        return state.R@self.lever_arm

    
class GNSS_Sensor(Sensor):

    def h(self, state: ESKF_State) -> GNSS_Measurement: #should gnss measurement be a gaussian (include R?)
        return GNSS_Measurement(state.pos + self.lever_arm_compensate(state))
    
    def H(self):
        raise NotImplementedError

    