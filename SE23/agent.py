import numpy as np
from typing import List

from .inertial_navigation import InertialNavigation
from .models import IMU_Model
from .states import PlatformState
from .measurements import TargetMeasurement, GNSS_Measurement, IMU_Measurement
from .sensors import ManifoldGNSS_Sensor, RelativePositionSensor
from .target import Target

class Agent:
    """
    An agent equipped with GNSS and IMU tracking targets and other agents
    This is known as platform in the thesis, but platform is a stdlib module in python, so agent here to avoid collision

    This runs the simplest possible tracker
    """
    state: PlatformState
    targets: List[Target] = []

    def __init__(self, IMU_cov: np.ndarray[6, 6], GNSS_sensor_cov: np.ndarray[3, 3], target_sensor_cov: np.ndarray[3, 3], init_state: PlatformState):
        self.state = init_state

        self.target_measurement_sensor = RelativePositionSensor(target_sensor_cov)

        self.imu = IMU_Model(IMU_cov)
        GNSS_sensor = ManifoldGNSS_Sensor(GNSS_sensor_cov)
        self.inertialNavigation = InertialNavigation(self.imu, GNSS_sensor)


    def propegate(self, imu_measurement: IMU_Measurement, dt: float):
        prev_state = self.state
        self.platform_propegate(imu_measurement, dt)

        self.targets_propegate(dt, prev_state, self.state, imu_measurement)

    def platform_propegate(self, imu_measurement: IMU_Measurement, dt: float):
       self.state = self.inertialNavigation.propegate(self.state, imu_measurement, dt)
           
    def platform_update(self, measurement: GNSS_Measurement):
        self.state = self.inertialNavigation.update(self.state, measurement)

    def targets_propegate(self, dt: float, prev_state, curr_state, z):
       for target in self.targets:
           target.propegate(dt, prev_state, curr_state, z, self.imu)

    def target_update(self, target_id: int, target_measurement: TargetMeasurement):
        for target in self.targets:
            if target.id == target_id:
                return target.update(self.state, target_measurement, self.target_measurement_sensor)
        
    def add_target(self, target: Target):
        self.targets.append(target)