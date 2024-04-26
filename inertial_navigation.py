import numpy as np
from dataclasses import dataclass

from lie_theory import SE3_2
from states import PlatformState
from sensors import ManifoldGNSS_Sensor
from measurements import GNSS_Measurement, IMU_Measurement
from models import IMU_Model


@dataclass
class InertialNavigation:
    model: IMU_Model
    sensor: ManifoldGNSS_Sensor

    def propegate(self, current_state: PlatformState, z: IMU_Measurement, dt: float):
        new_mean = self.model.propegate_mean(current_state.mean.as_matrix(), z, dt)
        new_mean = SE3_2.from_matrix(new_mean)
        new_cov = self.model.propegate_cov(current_state.cov, z, dt)
        return PlatformState(new_mean, new_cov)

    
    ### Kalman update
    def update(self, pose: PlatformState, z: GNSS_Measurement):
        zhat, S, H = self.sensor.predict_measurement(pose)
        innov = z.pos - zhat #innovation is z in world for GNSS
        K = pose.cov@np.linalg.solve(S.T, H).T #equiv to pose.cov@H.T@inv(S) #compute kalman gain
        err_hat = K@innov
        new_mean = pose.mean@SE3_2.Exp(err_hat) #inject
        err_J = SE3_2.jac_right(err_hat)
        new_cov = err_J@(np.eye(9)-K@H)@pose.cov@err_J.T #reset

        return PlatformState(new_mean, new_cov)
    