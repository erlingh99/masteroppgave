import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from numpy import ndarray

from utils import cross_matrix, from_cross_matrix
from gaussian import MultiVarGauss
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

    @classmethod
    @abstractmethod
    def Exp(cls, tau) -> "LieGroup":
        pass

    def __matmul__(self, other: "LieGroup"): #group composition if other of same type, else action
        if type(other) == type(self):
            return self.compose(other)
        return self.action(other)
    


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
        axis = tau/angle

        Tau = cross_matrix(axis)
        R = np.eye(3) + Tau*np.sin(angle) + Tau@Tau*(1 - np.cos(angle))
        return SO3(R)
    
    @property
    def mat(self):
        return self.R
    
    @property
    def T(self):
        return SO3(self.R.T)

@dataclass
class S3(LieGroup):
    q: RotationQuaternion
    ndim = 3

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
    
    def Log(self) -> ndarray[3]:
        return self.q.as_angle_axis()

@dataclass
class SE3(LieGroup):
    R: SO3
    t: np.ndarray[3]
    
    ndim = 6

    def action(self, x: np.ndarray[3]) -> np.ndarray[3]:
        return self.R@x + self.t

    def compose(self, other: "SE3") -> "SE3":
        return SE3(self.R@other.R, self.t + self.R@other.t)
    
    def inverse(self) -> "SE3":
        return SE3(self.R.T, -(self.R.T@self.t))
    
    def adjoint(self) -> np.ndarray[6, 6]:
        return np.block([[self.R.mat, cross_matrix(self.t)@self.R.mat],[np.zeros((3, 3)), self.R.mat]])
    
    def Log(self) -> np.ndarray[6]:
        return np.concatenate((self.R.Log(), self.t))
    
    @classmethod
    def Exp(cls, tau: np.ndarray[6]) -> "SE3":
        """
        tau = [theta, rho]
        """
        theta = tau[:3]
        rho = tau[3:]
        return SE3(SO3.Exp(theta), rho)
    

@dataclass
class SE3_2(LieGroup):
    R: SO3
    v: np.ndarray[3]
    p: np.ndarray[3]
    ndim = 9

    def action(self, x: np.ndarray[3]):
        raise NotImplementedError("No action is defined on SE3_2.")
    
    def compose(self, other: "SE3_2") -> "SE3_2":
        return SE3_2(self.R@other.R, self.v + self.R@other.v, self.p + self.R@other.p)
    
    def adjoint(self) -> np.ndarray[9, 9]:
        return np.block([[self.R.mat, np.zeros((3, 6))],
                         [cross_matrix(self.v)@self.R.mat, self.R.mat, np.zeros((3,3))],
                         [cross_matrix(self.p)@self.R.mat, np.zeros((3,3)), self.R.mat]])
    
    def inverse(self) -> "SE3_2":
        return SE3_2(self.R.T, -(self.R.T@self.v), -(self.R.T@self.p))

    def Log(self) -> np.ndarray[6]:
        return np.concatenate((self.R.Log(), self.v, self.p))
    
    @classmethod
    def Exp(cls, tau: np.ndarray[9]) -> "SE3":
        """
        tau = [theta, nu, rho]
        """
        theta = tau[:3]
        nu = tau[3:6]
        rho = tau[6:]
        return SE3_2(SO3.Exp(theta), nu, rho)


@dataclass
class ExponentialGaussian(MultiVarGauss):
    mean: LieGroup
    cov: np.ndarray

    def __post_init__(self):
        assert isinstance(self.mean, LieGroup), "The mean of an ExponentialGaussian must be a LieGroup" 
        assert self.cov.shape == (self.mean.ndim, self.mean.ndim), f"The covariance must be of shape {(self.mean.ndim, self.mean.ndim)}."


    #how to handle distances? OK in exp-space
    

if __name__ == "__main__":
    T = SE3_2.Exp([0, 1, 0, 3, 2, 1, 2, 3, 1])
    exp = ExponentialGaussian(T, np.zeros((7, 9)))