from error_state_kalman_filter import ESKF
from dataclasses import dataclass


class Agent:
    """An agent equipped with GNSS and IMU tracking targets and other agents"""

    def __post_init__(self) -> None:
        self.eskf = ESKF()
        

    def predict(self, dt: float) -> None:
        """
        Run every time the 
        """
        self.eskf.predict(dt)
        

    def update(self, measurement) -> None:
        self.eskf.update(measurement)
        

    def __preintegrate_imu__(self):
        """
        This runs at tracker rate. Needed to formulate the movement of targets
        """
        pass