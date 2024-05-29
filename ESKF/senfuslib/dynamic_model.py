from typing import Any, Generic, Sequence, TypeVar
import numpy as np
from scipy.linalg import expm

from dataclasses import dataclass

import numpy as np
from senfuslib import MultiVarGauss, NamedArray
from scipy.integrate import solve_ivp

S = TypeVar('S', bound=np.ndarray)  # State type


class DynamicModel(Generic[S]):

    def f_c(self, x: S) -> S:
        """Continuous time dynamics"""
        raise NotImplementedError

    def A_c(self, x: S) -> np.ndarray:
        """Jacobian of continuous time dynamics"""
        raise NotImplementedError

    def Q_c(self, x: S) -> np.ndarray:
        """Continuous time process noise"""
        raise NotImplementedError

    def f_d(self, x: S, dt: float) -> S:
        """Discrete time dynamics"""
        return self.F_d(x, dt) @ x

    def F_d(self, x: S, dt: float) -> np.ndarray:
        """F in (4.58). Discretize A_c using (4.59) from the book."""
        return expm(self.A_c(x)*dt)

    def Q_d(self, x: S, dt: float) -> np.ndarray:
        """Deiscretize Q_d usin (4.63) from the book
        See https://en.wikipedia.org/wiki/Discretization for more info"""

        A_c = self.A_c(x)  # A in (4.60)
        Q_c = self.Q_c(x)  # Q is equivalent to G@D@G.T in (4.60)

        v_l = expm(dt*np.block([[-A_c, Q_c], [np.zeros_like(Q_c.T), A_c.T]]))
        F_d = v_l[A_c.shape[0]:, A_c.shape[1]:].T
        F_d_inv_Q_d = v_l[:A_c.shape[0], A_c.shape[1]:]
        Q_d = F_d @ F_d_inv_Q_d
        return Q_d

    def pred_from_est(self,
                      x_est: MultiVarGauss[S],
                      dt: float,
                      ) -> MultiVarGauss[S]:
        """Get the predicted state distribution given a state estimate."""
        P = x_est.cov
        F_d = self.F_d(x_est.mean, dt)
        Q_d = self.Q_d(x_est.mean, dt)
        x_est_pred = MultiVarGauss(self.f_d(x_est.mean, dt),
                                   F_d @ P @ F_d.T + Q_d)
        return x_est_pred

    def pred_from_state(self,
                        x: S,
                        dt: float,
                        ) -> MultiVarGauss[S]:
        """Get the predicted state distribution given a state."""
        Q_d = self.Q_d(x, dt)
        x_est_pred = MultiVarGauss(self.f_d(x, dt), Q_d)
        return x_est_pred

    def step_RK45(self, x: S, dt: float, add_noise) -> S:
        """Perform a simulation step using scipy.integrate.solve_ivp"""
        def rhs(t, x_):
            if isinstance(x, NamedArray):
                x_ = x.__class__.from_array(x_)
            dx_dt = self.f_c(x_)
            if add_noise:
                dx_dt += np.random.multivariate_normal(np.zeros_like(dx_dt),
                                                       self.Q_c(x_))
            return dx_dt
        out = solve_ivp(rhs, [0, dt], x, t_eval=[dt]).y[:, -1]
        if isinstance(x, NamedArray):
            out = x.with_new_data(out)
        return out

    def step_simulation(self, x: S, dt: float) -> S:
        """Perform a simulation step using the continuous time dynamics"""
        return self.step_RK45(x, dt, add_noise=True)
