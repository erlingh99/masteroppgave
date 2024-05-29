from dataclasses import dataclass, fields
import numpy as np
from senfuslib import MultiVarGauss, TimeSequence
from typing import TypeVar, Generic

M = TypeVar('M', bound=np.ndarray)  # Measurement type
S = TypeVar('S', bound=np.ndarray)  # State type


@dataclass
class SensorModel(Generic[M]):
    def h(self, x: S) -> M:
        """Measurement function."""
        raise NotImplementedError

    def H(self, x: S) -> np.ndarray:
        """Jacobian of h."""
        raise NotImplementedError

    def R(self, x: S) -> np.ndarray:
        """Measurement noise covariance."""
        raise NotImplementedError

    def pred_from_est(self,
                      x_est: MultiVarGauss
                      ) -> MultiVarGauss[M]:
        """Get the predicted measurement distribution given a state estimate.
        """
        P = x_est.cov
        H = self.H(x_est.mean)
        R = self.R(x_est.mean)
        return MultiVarGauss(self.h(x_est.mean),
                             H @ P @ H.T + R)

    def pred_from_state(self, x: S) -> MultiVarGauss[M]:
        """Get the predicted measurement distribution given a state."""
        return MultiVarGauss(self.h(x), self.R(x))

    def sample_from_state(self, x: S) -> M:
        """Perform a measurement on x."""
        return self.pred_from_state(x).sample()

    def from_states(self, x_tseq: TimeSequence[S]) -> TimeSequence[M]:
        """Perform measurements on a time sequence of states."""
        return TimeSequence((t, self.sample_from_state(x))
                            for (t, x) in x_tseq.items())
