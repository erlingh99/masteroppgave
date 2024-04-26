from dataclasses import dataclass
import numpy as np
from models import CV_world
from states import TargetState, PlatformState, StackedState
from measurements import TargetMeasurement
from sensors import RelativePositionSensor


@dataclass
class Target:
    """
    class representing a target in the platform tracker
    """
    id: int
    motion_model: CV_world
    state: TargetState

    def propegate(self, dt):
        self.state = self.motion_model.propegate(self.state, dt)

    def update(self, platform_state: PlatformState, z: TargetMeasurement, sensor: RelativePositionSensor):
        """
        Target measurement is position in body of platform
        Platform state is exponetial gaussian on SE3_2

        this is eq ca 80
        """
        #need to stack the states
        stacked_state = StackedState(self.state, platform_state)
        zhat, S, H = sensor.predict_measurement(stacked_state)
        innov = z.relative_pos - zhat
        

        cov = stacked_state.cov
        #Kalman gain
        K = cov@np.linalg.solve(S.T, H).T #equiv to pose.cov@H.T@inv(S)

        #full error estimate
        full_err = K@innov
        full_cov = (np.eye(15)-K@H)@cov

        #marginalize
        marginal_err = full_err[:6]
        marginal_cov = full_cov[:6, :6]
        
        #update CV
        self.state = TargetState(self.state.mean + marginal_err, marginal_cov)
        return S