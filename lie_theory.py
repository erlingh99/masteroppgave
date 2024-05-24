from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from utils import cross_matrix, from_cross_matrix
from quaternion import RotationQuaternion, Quaternion


class LieGroup(ABC):

    @abstractmethod
    def action(self, x):
        pass

    @abstractmethod
    def compose(self, other: "LieGroup"):
        pass

    @abstractmethod
    def inverse(self) -> "LieGroup":
        pass

    @abstractmethod
    def adjoint(self):
        pass

    @abstractmethod
    def Log(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def as_matrix(self):
        pass

    @classmethod
    @abstractmethod
    def Exp(cls, tau) -> "LieGroup":
        pass

    def __matmul__(self, other: "LieGroup"): #group composition if other of same type, else action
        if type(other) == type(self):
            return self.compose(other)
        return self.action(other)
    
    def __str__(self) -> str:
        return np.array2string(self.as_matrix(), precision=2)
    
@dataclass
class SO2(LieGroup):
    R: np.ndarray[2, 2]
    ndim = 1

    @classmethod
    def Exp(cls, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        return cls(R=np.array([[c, -s], [s, c]]))
    
    def Log(self):
        return np.arctan2(self.R[1, 0], self.R[0, 0])
    
    def action(self, x):
        return self.R@x
    
    def compose(self, other: LieGroup):
        return SO2(self.R@other.R)
    
    def adjoint(self):
        return 1
    
    def inverse(self) -> LieGroup:
        return SO2(self.R.T)
    
    @property
    def T(self):
        return self.inverse()
    
    @property
    def mat(self):
        return self.R
    
    @staticmethod
    def hat(theta):
        """Performs the hat operator on the tangent space vector theta_vec,
        which returns the corresponding skew symmetric Lie Algebra matrix theta_hat.

        :param theta_vec: 1D tangent space column vector.
        :return: The Lie Algebra (2x2 matrix).
        """
        return np.array([[0, -theta],
                         [theta, 0]])
    
    def copy(self):
        return SO2(self.R.copy())
    
    def as_matrix(self):
        return self.R
    

@dataclass
class SE2(LieGroup):
    R: SO2
    t: np.ndarray[2]

    ndim = 3

    @classmethod
    def Exp(cls, tau):
        tau = np.array(tau) #just to make sure

        rho_vec = tau[1:]
        theta = tau[0]

        if np.abs(theta) < 1e-10:
            return SE2(SO2.Exp(theta), rho_vec)

        V = (np.sin(theta) / theta) * np.identity(2) + ((1 - np.cos(theta)) / theta) * SO2.hat(1)

        return cls(SO2.Exp(tau[0]), V@tau[1:])
    
    def Log(self):
        theta = self.R.Log()

        if theta == 0:
            return [0, *self.t]

        a = np.sin(theta) / theta
        b = (1 - np.cos(theta)) / theta

        V_inv = (1.0 / (a**2 + b**2)) * np.array([[a, b], [-b, a]])
        rho_vec = V_inv @ self.t

        return np.array([theta, *rho_vec])
    
    def compose(self, other: LieGroup):
        return SE2(self.R@other.R, self.R@other.t + self.t)
    
    def inverse(self) -> LieGroup:
        return SE2(self.R.T, -(self.R.T@self.t))
    
    def action(self, x):
        return self.R@x + self.t.reshape(2,1)
    
    def adjoint(self):
        raise NotImplementedError()
        # return 0
    
    def rotation_matrix(self):
        return self.R.mat
    
    def copy(self):
        return SE2(self.R.copy(), self.t.copy())
    
    def as_matrix(self):
        mat = np.eye(3)
        mat[:2, :2] = self.R.as_matrix()
        mat[:2, 2] = self.t
        return mat

@dataclass
class SO3(LieGroup):
    R: np.ndarray[3, 3]    
    ndim = 3

    def __post_init__(self):
        assert np.allclose(np.linalg.det(self.R), 1), "Not a rotation matrix"
        assert np.allclose(self.R@self.R.T, np.eye(3)), "Not a rotation matrix"

    def action(self, x: np.ndarray[3]) -> np.ndarray[3]:
        return self.R@x
    
    def compose(self, other: "SO3") -> "SO3":
        return SO3(self.R@other.R)
    
    def inverse(self) -> "SO3":
        return SO3(self.R.T)
    
    def adjoint(self) -> np.ndarray[3, 3]:
        return self.R

    def Log(self) -> np.ndarray[3]:
        """
        SO3 logarithmic map from SO3 to R3
        """
        cTheta = 0.5*(np.trace(self.R) - 1)
        if np.abs(cTheta-1) < 1e-10:
            return np.zeros((3,))
        
        sTheta = (1 - cTheta**2)**0.5
        theta = np.arccos(cTheta)
        w = theta/(2*sTheta)*from_cross_matrix(self.R - self.R.T)
        return w
    
    @classmethod
    def Exp(cls, tau) -> "SO3":
        """
        SO3 exponential map from R3 to SO3
        """
        angle = np.linalg.norm(tau)
        if np.allclose(angle, 0):
            return SO3(np.eye(3))

        Tau = cross_matrix(tau)
        R = np.eye(3) + np.sin(angle)/angle*Tau + (1 - np.cos(angle))/(angle**2)*Tau@Tau    
        return cls(R)
    
    @property
    def mat(self):
        return self.R
    
    def as_matrix(self):
        return self.R
    
    @property
    def T(self):
        return SO3(self.R.T)
    
    def left_jacobian(self):
        return SO3.jac_left(self.Log())
    
    def right_jacobian(self):
        return SO3.jac_right(self.Log())

    @staticmethod
    def jac_left(theta_vec):
        """Compute the left derivative of Exp(theta_vec) with respect to theta_vec.

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        theta = np.linalg.norm(theta_vec)
        if theta < 1e-10:
            return np.identity(3)

        u = cross_matrix(theta_vec/theta)

        return np.identity(3) + ((1 - np.cos(theta)) / (theta )) * u + (
                (theta - np.sin(theta)) / (theta )) * u @ u
    
    @staticmethod
    def int_jac_left(theta_vec):
        """Compute the integral of the left derivative of Exp(theta_vec) with respect to the length of theta_vec.

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        theta = np.linalg.norm(theta_vec)
        if theta < 1e-10:
            return np.identity(3)

        u = cross_matrix(theta_vec/theta)

        return np.identity(3) + 2*((theta - np.sin(theta)) / (theta ** 2)) * u + (
                (theta**2 + 2*np.cos(theta)-2) / (theta ** 2)) * u @ u
    
    @staticmethod
    def jac_right(theta_vec):
        """Compute the right derivative of Exp(theta_vec) with respect to theta_vec.

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        theta = np.linalg.norm(theta_vec)
        if theta < 1e-10:
            return np.identity(3)

        theta_hat = cross_matrix(theta_vec)

        return np.identity(3) - ((1 - np.cos(theta)) / (theta ** 2)) * theta_hat + (
                (theta - np.sin(theta)) / (theta ** 3)) * theta_hat @ theta_hat


    def inverse_left_jacobian(self) -> np.ndarray[3, 3]:
        tau = self.Log()
        Tau = cross_matrix(tau)
        angle = np.linalg.norm(tau)
        if np.allclose(angle, 0):
            return np.eye(3)
        
        Jinv = np.eye(3) - 0.5*Tau + (angle**(-2) - (1 + np.cos(angle))/(2*angle*np.sin(angle)))*Tau@Tau
        return Jinv

    @staticmethod
    def inverse_jac_left(theta_vec):
        """Compute the left derivative of Exp(theta_vec) with respect to theta_vec.

        :param theta_vec: The tangent space 3D column vector.
        :return: The Jacobian (3x3 matrix)
        """
        theta = np.linalg.norm(theta_vec)
        if theta < 1e-10:
            return np.identity(3)

        theta_hat = cross_matrix(theta_vec)

        return np.identity(3) - 0.5*theta_hat + (theta**(-2) - (1 + np.cos(theta))/(2*theta*np.sin(theta)))*theta_hat@theta_hat

    def copy(self):
        return SO3(self.R.copy())

@dataclass
class S3(LieGroup):
    q: RotationQuaternion
    ndim = (4,)

    def action(self, x: np.ndarray[3]):
        x = Quaternion(0, x)
        return (self.q@x@self.q.T).epsilon
    
    def compose(self, other: "S3"):
        return S3(self.q@other.q)
    
    def inverse(self) -> "S3":
        return S3(self.q.T)
    
    def adjoint(self) -> np.ndarray[3, 3]:
        return self.q.as_rotation_matrix()

    @classmethod
    def Exp(cls, tau) -> "S3":
        return S3(RotationQuaternion.from_scaled_axis(tau))
    
    def Log(self) -> np.ndarray[3]:
        return self.q.as_angle_axis()
    
    def copy(self):
        return S3(self.q.copy())
    
    def as_matrix(self):
        return self.q.as_vector()

@dataclass
class SE3(LieGroup):
    R: SO3
    t: np.ndarray[3]
    
    ndim = 6
    size = (4, 4)

    def action(self, x: np.ndarray[3]) -> np.ndarray[3]:
        return self.R@x + self.t.reshape((3, 1))

    def compose(self, other: "SE3") -> "SE3":
        return SE3(self.R@other.R, self.t + self.R@other.t)
    
    def inverse(self) -> "SE3":
        return SE3(self.R.T, -(self.R.T@self.t))
    
    def adjoint(self) -> np.ndarray[6, 6]:
        return np.block([[self.R.mat, cross_matrix(self.t)@self.R.mat],[np.zeros((3, 3)), self.R.mat]])


    def Log(self):
        """Computes the tangent space vector xi_vec at the current element X.

        :return: The tangent space vector xi_vec = [rho_vec, theta_vec]^T.
        """
        Jinv = self.R.inverse_left_jacobian()
        rho = Jinv@self.t
        return np.array([*self.R.Log(), *rho])
    
    @staticmethod
    def hat(tau):
        """Performs the hat operator on the tangent space vector xi_vec,
        which returns the corresponding Lie Algebra matrix xi_hat.

        :param xi_vec: 6d tangent space column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Lie Algebra (4x4 matrix).
        """

        return np.block([[cross_matrix(tau[:3]), tau[3:].reshape(3, 1)],
                         [np.zeros((1, 4))]])
    
    @classmethod
    def from_matrix(cls, T):
        """Construct an SE(3) element corresponding from a pose matrix.
        The rotation is fitted to the closest rotation matrix, the bottom row of the 4x4 matrix is ignored.

        :param T: 4x4 or 3x4 pose matrix.
        :return: The SE(3) element.
        """

        return cls(SO3(T[:3, :3]), T[:3, -1])

    @classmethod
    def Exp(cls, tau: np.ndarray[6]) -> "SE3":
        """
        tau = [theta, rho]
        """
        xi_hat = SE3.hat(tau)
        theta = np.linalg.norm(tau[:3])

        if theta < 1e-10:
            return SE3.from_matrix(np.identity(4) + xi_hat)
        else:
            return SE3.from_matrix(
                np.identity(4) + xi_hat + ((1 - np.cos(theta)) / (theta ** 2)) * np.linalg.matrix_power(xi_hat, 2) +
                ((theta - np.sin(theta)) / (theta ** 3)) * np.linalg.matrix_power(xi_hat, 3))
        
    def copy(self):
        return SE3(self.R.copy(), self.t.copy())
        

    @staticmethod
    def _Q_left(xi_vec):
        theta_vec = xi_vec[:3]
        rho_vec = xi_vec[3:]
        theta = np.linalg.norm(theta_vec)

        rho_hat = cross_matrix(rho_vec)
        theta_hat = cross_matrix(theta_vec)

        if theta < 1e-10:
            return 0.5 * rho_hat

        return 0.5 * rho_hat + ((theta - np.sin(theta)) / theta ** 3) * \
               (theta_hat @ rho_hat + rho_hat @ theta_hat + theta_hat @ rho_hat @ theta_hat) - \
               ((1 - 0.5 * theta ** 2 - np.cos(theta)) / theta ** 4) * \
               (theta_hat @ theta_hat @ rho_hat + rho_hat @ theta_hat @ theta_hat -
                3 * theta_hat @ rho_hat @ theta_hat) - \
               0.5 * ((1 - 0.5 * theta ** 2 - np.cos(theta)) / theta ** 4 - 3 *
                      ((theta - np.sin(theta) - (theta ** 3 / 6)) / theta ** 5)) * \
               (theta_hat @ rho_hat @ theta_hat @ theta_hat + theta_hat @ theta_hat @ rho_hat @ theta_hat)
    

    
    @staticmethod
    def jac_left(xi_vec):
        """Compute the left derivative of Exp(xi_vec) with respect to xi_vec.

        :param xi_vec: The tangent space 6D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Jacobian (6x6 matrix)
        """
        theta_vec = xi_vec[:3]

        J_l_theta = SO3.jac_left(theta_vec)
        Q_l = SE3._Q_left(xi_vec)

        return np.block([[J_l_theta, np.zeros((3, 3))],
                         [Q_l, J_l_theta]])
    
    def as_matrix(self):
        mat = np.eye(4)
        mat[:3, :3] = self.R.as_matrix()
        mat[:3, 3] = self.t
        return mat


@dataclass
class SE3_2(LieGroup):
    R: SO3
    v: np.ndarray[3]
    p: np.ndarray[3]
    ndim = 9
    size = (5, 5) #5x5

    @classmethod
    def from_matrix(cls, matrix: np.ndarray[5, 5]) -> "SE3_2":
        return cls(SO3(matrix[:3, :3]), matrix[:3, 3], matrix[:3, 4])

    def action(self, x: np.ndarray[3]) -> np.ndarray[3]:
        return self.R@x + self.p
    
    def action2(self, x: np.ndarray[6]) -> np.ndarray[6]:
        return np.concatenate([self.R@x[:3] + self.p,
                               self.R@x[3:] + self.v])
    
    def compose(self, other: "SE3_2") -> "SE3_2":
        return SE3_2(self.R@other.R, self.v + self.R@other.v, self.p + self.R@other.p)
    
    def adjoint(self) -> np.ndarray[9, 9]:
        return np.block([[self.R.mat, np.zeros((3, 6))],
                         [cross_matrix(self.v)@self.R.mat, self.R.mat, np.zeros((3,3))],
                         [cross_matrix(self.p)@self.R.mat, np.zeros((3,3)), self.R.mat]])
    
    def inverse(self) -> "SE3_2":
        return SE3_2(self.R.T, -(self.R.T@self.v), -(self.R.T@self.p))

    def Log(self) -> np.ndarray[9]:
        Jinv = self.R.inverse_left_jacobian()
        nu = Jinv@self.v
        rho = Jinv@self.p
        return np.concatenate((self.R.Log(), nu, rho))
    
    def as_matrix(self):
        m = np.eye(5)
        m[:3, :3] = self.R.mat
        m[:3, 3] = self.v
        m[:3, 4] = self.p
        return m

    @classmethod
    def Exp(cls, tau: np.ndarray[9]) -> "SE3_2":
        """
        tau = [theta, nu, rho]
        """
        R = SO3.Exp(tau[:3])
        J = R.left_jacobian()
        return SE3_2(R, J@tau[3:6], J@tau[6:])
    
    @property
    def t(self):
        return self.p

    def left_jacobian(self):
        J = np.zeros((9, 9))
        J_R = self.R.left_jacobian()
        J[:3, :3] = J_R
        J[3:6, 3:6] = J_R
        J[6:9, 6:9] = J_R

        eps = self.Log()
        phi = eps[:3]
        nu = eps[3:6]
        rho = eps[6:9]

        J[3:6, :3] = self.__Q_phi__(phi, nu)
        J[6:, :3] = self.__Q_phi__(phi, rho)
        return J

    def inverse_left_jacobian(self):
        Jinv = np.zeros((9, 9))
        Jinv_R = self.R.inverse_left_jacobian()
        Jinv[:3, :3] = Jinv_R
        Jinv[3:6, 3:6] = Jinv_R
        Jinv[6:9, 6:9] = Jinv_R

        eps = self.Log()
        phi = eps[:3]
        nu = eps[3:6]
        rho = eps[6:9]

        Jinv[3:6, :3] = -Jinv_R@self.__Q_phi__(phi, nu)@Jinv_R
        Jinv[6:, :3] = -Jinv_R@self.__Q_phi__(phi, rho)@Jinv_R

        return Jinv
    

    def right_jacobian(self):
        """Compute the right derivative of Exp(xi_vec) with respect to xi_vec.

        :param xi_vec: The tangent space 9D column vector xi_vec = [rho_vec, theta_vec]^T.
        :return: The Jacobian (9x9 matrix)
        """
        J = np.zeros((9, 9))
        J_R = self.R.right_jacobian()
        J[:3, :3] = J_R
        J[3:6, 3:6] = J_R
        J[6:9, 6:9] = J_R

        eps = self.Log()
        phi = eps[:3]
        nu = eps[3:6]
        rho = eps[6:9]

        J[3:6, :3] = SE3_2.__Q_phi__(-phi, -nu)
        J[6:, :3] = SE3_2.__Q_phi__(-phi, -rho)
        
        return J
    

    @staticmethod
    def jac_right(tau):
        phi = tau[:3]
        nu = tau[3:6]
        rho = tau[6:9]

        J = np.zeros((9, 9))
        J_R = SO3.jac_right(phi)
        J[:3, :3] = J_R
        J[3:6, 3:6] = J_R
        J[6:9, 6:9] = J_R

        J[3:6, :3] = SE3_2.__Q_phi__(-phi, -nu)
        J[6:, :3] = SE3_2.__Q_phi__(-phi, -rho)
        
        return J

    @staticmethod
    def __Q_phi__(phi, nu_or_rho):
        """
        eq 95 in SE3_2 paper
        """
        return SE3._Q_left([*phi, *nu_or_rho])
    
    def copy(self):
        return SE3_2(self.R.copy(), self.v.copy(), self.p.copy())
    

    def as_matrix(self):
        mat = np.eye(5)
        mat[:3, :3] = self.R.as_matrix()
        mat[:3, 3] = self.v
        mat[:3, 4] = self.p 
        return mat   