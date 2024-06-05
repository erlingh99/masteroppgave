from dataclasses import dataclass, InitVar
from abc import abstractmethod, ABC
import numpy as np

from .models import CV_world, CV_body, CV_body2
from .states import TargetState, PlatformState, StackedState
from .measurements import TargetMeasurement
from .sensors import RelativePositionSensor
from .utils import cross_matrix
from .lie_theory import LieGroup


@dataclass
class Target(ABC):
    """
    base target class
    """
    id: int
    state: TargetState
    var_acc: float

    @abstractmethod
    def propegate(self, *args):
        pass

    @abstractmethod
    def update(self, *args):
        pass

@dataclass
class TargetWorldNaive(Target):
    def __post_init__(self):
        self.motion_model = CV_world(self.var_acc)

    def propegate(self, dt, *args):
        self.state = self.motion_model.propegate(self.state, dt)

    def update(self, platform_state: PlatformState, z: TargetMeasurement, sensor_covar):
        H = np.block([platform_state.rot.T, np.zeros((3,3))])
        zhat = self.state.pos
        innov = z.relative_pos - zhat
        S = H@self.state.cov@H.T + sensor_covar
        K = self.state.cov@np.linalg.solve(S.T, H).T
        err = K@innov
        cov = (np.eye(6)-K@H)@self.state.cov
        
        self.state = TargetState(self.state.mean + err, cov)



@dataclass
class TargetWorld(Target):
    """
    class representing a target in the platform tracker
    """
    def __post_init__(self):
        self.motion_model = CV_world(self.var_acc)

    def propegate(self, dt, *args):
        self.state = self.motion_model.propegate(self.state, dt)

    def update(self, platform_state: PlatformState, z: TargetMeasurement, sensor: RelativePositionSensor):
        """
        Target measurement is position in body of platform
        Platform state is exponetial gaussian on SE3_2

        this is eq ca 80
        """
        #need to stack the states
        stacked_state = StackedState(self.state, platform_state)
        zhat, S, H, _ = sensor.predict_measurement(stacked_state)
        innov = z.relative_pos - zhat
        
        P, _  = stacked_state.cov
        #Kalman gain
        K = P@np.linalg.solve(S.T, H).T #equiv to pose.cov@H.T@inv(S)

        #full error estimate
        err = K@innov
        cov = (np.eye(6)-K@H)@P

        
        #update CV
        self.state = TargetState(self.state.mean + err, cov)

    
@dataclass
class TargetBody(Target):
    cls: InitVar[LieGroup]

    def __post_init__(self, cls):
        self.motion_model = CV_body(self.var_acc, cls)

    def propegate(self, dt, platform_state_k, platform_state_kp1, z, imu):
        self.state = self.motion_model.propegate(self.state, platform_state_k, platform_state_kp1, z, imu, dt)

    def update(self, _, z: TargetMeasurement, sensor: RelativePositionSensor):
        H = np.block([np.eye(3), np.zeros((3,3))])
        zhat = self.state.pos
        innov = z.relative_pos - zhat
        S = H@self.state.cov@H.T + sensor.get_measurement_covariance(self.state.mean)
        K = self.state.cov@np.linalg.solve(S.T, H).T
        err = K@innov
        cov = (np.eye(6)-K@H)@self.state.cov
        
        self.state = TargetState(self.state.mean + err, cov)


    def convert_state_to_world_lin(self, platform_state: PlatformState):
        Rhat = platform_state.rot
        J1 = np.block([[Rhat, np.zeros((3,3))],
                        [np.zeros((3, 3)), Rhat]])
        J2 = np.block([[-Rhat@cross_matrix(self.state.pos), np.zeros((3,3)), Rhat],
                        [-Rhat@cross_matrix(self.state.vel), Rhat, np.zeros((3,3))]])
        
        return TargetState(platform_state.mean.action2(self.state.mean), J1@self.state.cov@J1.T + J2@platform_state.cov@J2.T)

    def convert_state_to_world_manifold(self, platform_state: PlatformState):
        # local_mean = SE3_2(SO3.Exp([0, 0, 0]), self.state.vel, self.state.pos)
        # tau = local_mean.Log()
        tau = np.array([0,0,0,*self.state.vel, *self.state.pos])
        local_mean = platform_state.mean.__class__.Exp(tau)

        reorder_mat = np.block([[np.zeros((3,6))], [np.zeros((3,3)), np.eye(3)], [np.eye(3), np.zeros((3,3))]]) #need to reorder the states to match the SE3_2 convention
        local_cov = reorder_mat@self.state.cov@reorder_mat.T

        tot_mean = platform_state.mean@local_mean
        ad_inv = local_mean.inverse().adjoint()
        tot_cov = ad_inv@platform_state.cov@ad_inv.T + local_cov
        return PlatformState(tot_mean, tot_cov)
        


@dataclass
class TargetBody2(Target):
    cls: InitVar[LieGroup]

    def __post_init__(self, cls):
        self.motion_model = CV_body2(self.var_acc, cls)

        H = np.block([[np.zeros((3,6))],
                      [np.zeros((3,3)), np.eye(3)],
                      [np.eye(3), np.zeros((3,3))]])

        self.state = PlatformState(self.motion_model.cls.Exp(H@self.state.mean), H@self.state.cov@H.T)

    def propegate(self, dt, platform_state_k, platform_state_kp1, z, imu):
        self.state = self.motion_model.propegate(self.state, platform_state_k, platform_state_kp1, z, imu, dt)

    def update(self, _, z: TargetMeasurement, sensor: RelativePositionSensor):

        HH = np.block([[np.zeros((3,6))],
                       [np.zeros((3,3)), np.eye(3)],
                       [np.eye(3), np.zeros((3,3))]])

        cov = HH.T@self.state.cov@HH
        mean = np.array([*self.state.pos, *self.state.vel])

        H = np.block([np.eye(3), np.zeros((3,3))])
        zhat = self.state.pos
        innov = z.relative_pos - zhat
        S = H@cov@H.T + sensor.get_measurement_covariance(mean)
        K = cov@np.linalg.solve(S.T, H).T
        err = K@innov
        cov = HH@(np.eye(6)-K@H)@cov@HH.T
        
        self.state = PlatformState(self.state.mean@self.motion_model.cls.Exp(HH@err), cov)


    def convert_state_to_world_lin(self, platform_state: PlatformState):
        Rhat = platform_state.rot
        J1 = np.block([[Rhat, np.zeros((3,3))],
                        [np.zeros((3, 3)), Rhat]])
        J2 = np.block([[-Rhat@cross_matrix(self.state.pos), np.zeros((3,3)), Rhat],
                        [-Rhat@cross_matrix(self.state.vel), Rhat, np.zeros((3,3))]])
        
        return TargetState(platform_state.mean.action2(self.state.mean), J1@self.state.cov@J1.T + J2@platform_state.cov@J2.T)

    def convert_state_to_world_manifold(self, platform_state: PlatformState):

        tot_mean = platform_state.mean@self.state.mean
        ad_inv = self.state.mean.inverse().adjoint()
        tot_cov = ad_inv@platform_state.cov@ad_inv.T + self.state.cov
        return PlatformState(tot_mean, tot_cov)