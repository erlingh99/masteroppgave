from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import scipy.linalg
from senfuslib import MultiVarGauss
from states import ErrorState, ImuMeasurement, CorrectedImuMeasurement, NominalState, GnssMeasurement, EskfState
from quaternion import RotationQuaterion
from utils.cross_matrix import get_cross_matrix


@dataclass
class ModelIMU:
    """The IMU is considered a dynamic model instead of a sensar. 
    This works as an IMU measures the change between two states, 
    and not the state itself.."""

    accm_std: float
    gyro_std: float
    g: 'np.ndarray[3]' = field(default=np.array([0, 0, 9.82]))
    Q_c: 'np.ndarray[12, 12]' = field(init=False, repr=False)


    def __post_init__(self):
        def diag3(x): return np.diag([x]*3)


        self.Q_c = scipy.linalg.block_diag(
            diag3(self.accm_std**2),
            diag3(self.gyro_std**2)
        )

    def correct_z_imu(self,
                      x_est_nom: NominalState,
                      z_imu: ImuMeasurement,
                      ) -> CorrectedImuMeasurement:
        """Correct IMU measurement so it gives a measurmenet of acceleration 
        and angular velocity in body.

        Hint: self.accm_correction and self.gyro_correction translates 
        measurements from IMU frame (probably not correct name) to body frame

        Args:
            x_est_nom: previous nominal state
            z_imu: raw IMU measurement

        Returns:
            z_corr: corrected IMU measurement
        """
        acc_est = z_imu.acc
        avel_est = z_imu.avel
        
        return CorrectedImuMeasurement(acc_est, avel_est)

    def predict_nom(self,
                    x_est_nom: NominalState,
                    z_corr: CorrectedImuMeasurement,
                    dt: float) -> NominalState:
        """Predict the nominal state, given a corrected IMU measurement and a 
        time step, by discretizing (10.58) in the book.

        We assume the change in orientation is negligable when caculating 
        predicted position and velicity, see assignment pdf.

        Hint: You can use: delta_rot = RotationQuaterion.from_avec(something)

        Args:
            x_est_nom: previous nominal state
            z_corr: corrected IMU measuremnt
            dt: time step
        Returns:
            x_nom_pred: predicted nominal state
        """
        a = x_est_nom.ori.R@z_corr.acc + self.g

        pos_pred = x_est_nom.pos + dt*x_est_nom.vel + dt**2/2*a
        vel_pred = x_est_nom.vel + dt*a

        delta_rot = RotationQuaterion.from_avec(dt*z_corr.avel)
        ori_pred = x_est_nom.ori@delta_rot    

        return NominalState(pos_pred, vel_pred, ori_pred)

    def A_c(self,
            x_est_nom: NominalState,
            z_corr: CorrectedImuMeasurement,
            ) -> 'np.ndarray[15, 15]':
        """Get the transition matrix, A, in (10.68)

        Hint: The S matrices can be created using get_cross_matrix. In the book
        a perfect IMU is expected (thus many I matrices). Here we have 
        to use the correction matrices, self.accm_correction and 
        self.gyro_correction, instead of som of the I matrices.  

        You can use block_3x3 to simplify indexing if you want to.
        ex: first I element in A can be set as A[block_3x3(0, 1)] = np.eye(3)

        Args:
            x_nom_prev: previous nominal state
            z_corr: corrected IMU measurement
        Returns:
            A (ndarray[15,15]): A
        """
        A_c = np.zeros((9, 9))
        Rq = x_est_nom.ori.as_rotmat()
        S_acc = get_cross_matrix(z_corr.acc)
        S_omega = get_cross_matrix(z_corr.avel)

        A_c[0:3, 3:6] = np.eye(3)
        A_c[3:6, 6:9] = -Rq@S_acc
        A_c[6:9, 6:9] = -S_omega               
        return A_c

    def get_error_G_c(self,
                      x_est_nom: NominalState,
                      ) -> 'np.ndarray[15, 15]':
        """The continous noise covariance matrix, G, in (10.68)

        Hint: you can use block_3x3 to simplify indexing if you want to.
        The first I element in G can be set as G[block_3x3(2, 1)] = -np.eye(3)

        Args:
            x_est_nom: previous nominal state
        Returns:
            G_c (ndarray[15, 15]): G in (10.68)
        """
        G_c = np.zeros((9, 6))
        Rq = x_est_nom.ori.as_rotmat()

        G_c[3:6, 0:3] = -Rq
        G_c[6:9, 3:6] = -np.eye(3)
        return G_c

    def get_discrete_error_diff(self,
                                x_est_nom: NominalState,
                                z_corr: CorrectedImuMeasurement,
                                dt: float
                                ) -> Tuple['np.ndarray[15, 15]',
                                           'np.ndarray[15, 15]']:
        """Get the discrete equivalents of A and GQGT in (4.63)

        Hint: Use scipy.linalg.expm to get the matrix exponential

        See (4.5 Discretization) and (4.63) for more information. 
        Or see "Discretization of process noise" in 
        https://en.wikipedia.org/wiki/Discretization

        Args:
            x_est_nom: previous nominal state
            z_corr: corrected IMU measurement
            dt: time step
        Returns:
            A_d (ndarray[15, 15]): discrede transition matrix
            GQGT_d (ndarray[15, 15]): discrete noise covariance matrix
        """
        A_c = self.A_c(x_est_nom, z_corr)
        G_c = self.get_error_G_c(x_est_nom)
        GQGT_c = G_c@self.Q_c@G_c.T

        
        exponent = np.block([[-A_c, GQGT_c], [np.zeros(A_c.shape), A_c.T]])*dt
        VanLoanMatrix = scipy.linalg.expm(exponent)

        A_d = VanLoanMatrix[9:, 9:].T
        GQGT_d = A_d@VanLoanMatrix[:9, 9:]

        return A_d, GQGT_d

    def predict_err(self,
                    x_est_prev: EskfState,
                    z_corr: CorrectedImuMeasurement,
                    dt: float,
                    ) -> MultiVarGauss[ErrorState]:
        """Predict the error state

        Hint: This is doing a discrete step of (10.68) where x_err 
        is a multivariate gaussian.

        Args:
            x_est_prev: previous estimated eskf state
            z_corr: corrected IMU measuremnt
            dt: time step
        Returns:
            x_err_pred: predicted error state gaussian
        """
        x_est_prev_nom = x_est_prev.nom
        x_est_prev_err = x_est_prev.err
        Ad, GQGTd = self.get_discrete_error_diff(x_est_prev_nom, z_corr, dt)
        
        P_pred = Ad@x_est_prev_err.cov@Ad.T + GQGTd        
        x_pred = Ad@x_est_prev_err.mean

        return MultiVarGauss(x_pred, P_pred)