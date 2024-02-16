from dataclasses import dataclass
from utils import cross_matrix
import numpy as np

@dataclass
class Quaternion:
    eta: float
    epsilon: np.ndarray[3]

    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        eta_new = self.eta*other.eta - self.epsilon.T@other.epsilon
        epsilon_new = other.eta*self.epsilon + self.eta*other.epsilon + np.cross(self.epsilon, other.epsilon)
        return Quaternion(eta_new, epsilon_new)

    def __matmul__(self, other: 'Quaternion') -> 'Quaternion':
        return self.multiply(other)

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.eta, -self.epsilon)

    @property
    def T(self) -> 'Quaternion':
        return self.conjugate()

    @property
    def vec(self) -> np.ndarray[4]:
        return np.array([self.eta, *self.epsilon])
    
    def as_vector(self) -> np.ndarray[4]:
        return self.vec
    
    
@dataclass
class RotationQuaternion(Quaternion):
    def __post_init__(self):
        norm = np.linalg.norm(self.vec)
        if np.allclose(norm, 1):
            self.eta = self.eta/norm
            self.epsilon = self.epsilon/norm
        if self.eta < 0:
            self.eta = -self.eta
            self.epsilon = -self.epsilon

    def as_rotation_matrix(self) -> np.ndarray[3, 3]:
        return (self.eta**2 - self.epsilon.T@self.epsilon)*np.eye(3) \
                + 2*np.outer(self.epsilon, self.epsilon) \
                + 2*self.eta*cross_matrix(self.epsilon)
    
    def as_angle_axis(self) -> np.ndarray[3]:
        norm = np.linalg.norm(self.epsilon)
        angle = 2*np.arctan2(norm, self.eta)
        vector = self.epsilon/norm
        return vector*angle
    
    @classmethod
    def from_scaled_axis(cls, angle_axis: np.ndarray[3]) -> 'RotationQuaternion':
        angle = np.linalg.norm(angle_axis)
        axis = angle_axis/angle
        return RotationQuaternion(eta=np.cos(angle/2), epsilon=np.sin(angle/2)*axis)
    
    def conjugate(self) -> 'RotationQuaternion':
        return RotationQuaternion(self.eta, -self.epsilon)
    
    @property
    def R(self) -> np.ndarray[3, 3]:
        return self.as_rotation_matrix()
