import numpy as np
from dataclasses import dataclass
from quaternion import RotationQuaternion
from measurements import IMU_Measurement, GNSS_Measurement
from gaussian import MultiVarGauss

@dataclass
class ESKF_State:
    pos: np.ndarray[3]
    vel: np.ndarray[3]
    ori: RotationQuaternion
    acc_bias: np.ndarray[3]
    gyro_bias: np.ndarray[3]

@dataclass
class ESKF_ErrorState(MultiVarGauss):
    mean: np.ndarray[6]
    cov: np.ndarray[9, 6]

@dataclass
class ESKF:
    nom_state: ESKF_State
    err_state: ESKF_ErrorState

    def predict(self,
                state:          ESKF_State, 
                measurment:     IMU_Measurement,
                dt:             float
                ) -> ESKF_State:
        g = np.array([0, 0, -9.81])

        acceleration = g + self.nom_state.ori.R@(measurment.acc - self.nom_state.acc_bias)

        self.nom_state.pos += self.nom_state.vel*dt + 0.5*acceleration*dt**2
        self.nom_state.vel +=  acceleration*dt
        self.nom_state.ori = self.nom_state.ori@RotationQuaternion.from_scaled_axis((measurment.gyro - self.nom_state.gyro_bias)*dt)

    def update(self, 
               state: ESKF_State, 
               measurement: GNSS_Measurement) -> ESKF_State:
        pass

    def inject(self, 
               nominal_state: ESKF_State,
               error_state: ESKF_State) -> ESKF_State:
        pass