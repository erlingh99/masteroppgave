from filter import Filter


class Agent:
    """An agent equipped with GNSS and IMU tracking targets and other agents"""

    def __post_init__(self) -> None:
        self.filter = Filter()
        

    def predict(self, dt: float) -> None:
        """
        Run every time the 
        """
        pass
        

    def update(self, measurement) -> None:
        pass
        

    def __preintegrate_imu__(self):
        """
        This runs at tracker rate. Needed to formulate the movement of targets
        """
        pass