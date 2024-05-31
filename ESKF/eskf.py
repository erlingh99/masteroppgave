import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .senfuslib import MultiVarGauss
from .states import ImuMeasurement, GnssMeasurement, EskfState
from .states import NominalState, ErrorState
from .quaternion import RotationQuaterion
from .utils.cross_matrix import get_cross_matrix
from .sensors import SensorGNSS
from .models import ModelIMU


@dataclass
class ESKF():
    model: ModelIMU
    sensor: SensorGNSS

    def predict_from_imu(self,
                         x_est_prev: EskfState,
                         z_imu: ImuMeasurement,
                         dt: float
                         ) -> EskfState:
        """Method called every time an IMU measurement is received

        Args:
            x_nom_prev: previous eskf state
            z_imu: raw IMU measurement
            dt: time since last imu measurement
        Returns:
            x_est_pred: predicted eskf state
        """
        if dt == 0:            
            return x_est_prev

        x_est_prev_nom = x_est_prev.nom

        x_est_pred_nom = self.model.predict_nom(x_est_prev_nom, z_imu, dt)
        x_est_pred_err = self.model.predict_err(x_est_prev, z_imu, dt)

        return EskfState(x_est_pred_nom, x_est_pred_err)        

    def update_err_from_gnss(self,
                             x_est_pred: EskfState,
                             z_est_pred: MultiVarGauss[GnssMeasurement],
                             z_gnss: GnssMeasurement
                             ) -> MultiVarGauss[ErrorState]:
        """Update the error state from a gnss measurement

        Hint: see (10.75)
        Due to numerical error its recomended use the robust calculation of 
        posterior covariance, somtimes called Joseph form:
            I_WH = np.eye(*P.shape) - W @ H
            x_err_cov_upd = (I_WH @ P @ I_WH.T + W @ R @ W.T)
        Remember that:
            S = H @ P @ H.T + R
        and that:
            np.linalg.solve(S, H.T) is faster than np.linalg.inv(S)

        Args:
            x_est_pred: predicted nominal and error state (gaussian)
            z_est_pred: predicted gnss measurement (gaussian)
            z_gnss: gnss measurement

        Returns:
            x_est_upd_err: updated error state gaussian
        """
        x_nom = x_est_pred.nom
        x_err = x_est_pred.err
        z_pred, S = z_est_pred

        innovation = z_gnss - z_pred
        H = self.sensor.H(x_nom)
        P = x_err.cov
        R = self.sensor.R
        
        W = P@np.linalg.solve(S.T, H).T
        x_err_upd = W@innovation
        I_WH = np.eye(*P.shape) - W@H
        x_err_cov_upd = I_WH@P@I_WH.T + W@R@W.T

        x_err_upd = ErrorState.from_array(x_err_upd)
        return MultiVarGauss[ErrorState](x_err_upd, x_err_cov_upd)

    def inject(self,
               x_est_nom: NominalState,
               x_est_err: MultiVarGauss[ErrorState],
               ) -> EskfState:
        """Perform the injection step

        Hint: see (10.85) and (10.72) on how to inject into nominal state.
        See (10.86) on how to find error covariance after injection

        Args:
            x_nom_prev: previous nominal state
            x_err_upd: updated error state gaussian

        Returns:
            x_est_inj: eskf state after injection
        """
        P = x_est_err.cov

        pos_inj = x_est_nom.pos + x_est_err.mean.pos
        vel_inj = x_est_nom.vel + x_est_err.mean.vel
        ori_inj = x_est_nom.ori@RotationQuaterion(1, 1/2*x_est_err.mean.avec)


        x_nom_inj = NominalState(pos_inj, vel_inj, ori_inj)

        G = np.eye(9)
        G[6:9, 6:9] -= get_cross_matrix(1/2*x_est_err.mean.avec)
        P_inj = G@P@G.T
        x_err_inj = MultiVarGauss[ErrorState](np.zeros(9), P_inj)
        return EskfState(x_nom_inj, x_err_inj)

    def update_from_gnss(self,
                         x_est_pred: EskfState,
                         z_gnss: GnssMeasurement,
                         ) -> Tuple[NominalState,
                                    MultiVarGauss[ErrorState],
                                    MultiVarGauss]:
        """Method called every time an gnss measurement is received.


        Args:
            x_est_pred: previous estimated esfk state
            z_gnss: gnss measurement

        Returns:
            x_est_upd: updated eskf state
            z_est_upd: predicted measurement gaussian

        """
        z_est_pred = self.sensor.pred_from_est(x_est_pred)
        x_est_upd_err = self.update_err_from_gnss(x_est_pred, z_est_pred, z_gnss)
        x_est_upd =  self.inject(x_est_pred.nom, x_est_upd_err)
    
        return x_est_upd, z_est_pred
