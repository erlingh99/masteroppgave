import numpy as np
from utils import cross_matrix
from states import PlatformState, StackedState


class NonLinearMeasurementModel:
    """
    Generic measurement model class.
    """
    def __init__(self, measurement_function, measurement_jacobian, cartesian_covariance):
        self.__measurement_function__ = measurement_function
        self.__measurement_jacobian__ = measurement_jacobian
        self.__cartesian_covariance__ = cartesian_covariance


    def get_measurement_function(self):
        return self.__measurement_function__

    def get_measurement_jacobian(self):
        return self.__measurement_jacobian__

    def predict_measurement(self, state):
        z_hat = self.__measurement_function__(state)
        H = self.__measurement_jacobian__(state)
        S = self.innovation_covariance(state, H)
        return z_hat, S, H
    
    def innovation_covariance(self, state, H):
        return H@state.cov@H.T + self.get_measurement_covariance(state.mean)

    def get_measurement_covariance(self, mean):
        return self.__cartesian_covariance__

#should maybe add polar (but 3d so spherical?) covariance support, for range bearing type measurements
   

class ManifoldGNSS_Sensor(NonLinearMeasurementModel):
    def __init__(self, cartesian_covariance):
        super().__init__(self.h, self.H, cartesian_covariance)

    def h(self, T: PlatformState):
        return T.pos
    
    def H(self, state: PlatformState):
        return np.block([np.zeros((3,6)), state.rot])
    

class RelativePositionSensor(NonLinearMeasurementModel):
    def __init__(self, cartesian_covariance):
        super().__init__(self.h, self.H, cartesian_covariance)

    def h(self, stacked_state: StackedState):
        target_state, platform_state_mean = stacked_state.mean
        return platform_state_mean.inverse()@target_state.pos
    
    def H(self, stacked_state: StackedState):
        _, platform_state_mean = stacked_state.mean
        R = platform_state_mean.R.as_matrix()
        return R.T@np.block([np.eye(3), np.zeros((3,3))])
    
    def innovation_covariance(self, state: StackedState, H):
        P, Sigma = state.cov 
        target_mean, platform_mean = state.mean
        J = np.block([cross_matrix(platform_mean.inverse()@target_mean.pos), np.zeros((3,3)), -np.eye(3)])
        return H@P@H.T + J@Sigma@J.T +  self.get_measurement_covariance(state.mean)