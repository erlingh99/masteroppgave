import numpy as np
from dataclasses import dataclass
from typing import Optional

from .quaternion import RotationQuaterion
from .senfuslib import NamedArray, AtIndex, MetaData
from .senfuslib import MultiVarGauss


@dataclass
class WithXYZ(NamedArray):
    x: AtIndex[0]
    y: AtIndex[1]
    z: AtIndex[2]
    xy: AtIndex[0:2]


@dataclass
class NominalState(NamedArray):
    """Class representing a nominal state. See (Table 10.1) in the book.

    Args:
        pos (ndarray[3]): position in NED
        vel (ndarray[3]): velocity in NED
        ori (RotationQuaterion): orientation as a quaternion in NED
        accm_bias (ndarray[3]): accelerometer bias
        gyro_bias (ndarray[3]): gyro bias
    """
    pos: AtIndex[0:3] | WithXYZ
    vel: AtIndex[3:6] | WithXYZ
    ori: AtIndex[6:10] | RotationQuaterion

    def diff(self, other: 'NominalState') -> 'ErrorState':
        """Calculate the difference between two nominal states.
        Used to calculate NEES.
        Returns:
            ErrorState: error state representing the difference
        """
        return NominalState(
            pos=self.pos - other.pos,
            vel=self.vel - other.vel,
            ori=self.ori.diff_as_avec(other.ori))

    @property
    def euler(self) -> np.ndarray:
        """Orientation as euler angles (roll, pitch, yaw) in NED"""
        return WithXYZ.from_array(self.ori.as_euler())


@dataclass
class ErrorState(NamedArray):
    """Class representing a nominal state. See (Table 10.1) in the book."""
    pos: AtIndex[0:3] | WithXYZ
    vel: AtIndex[3:6] | WithXYZ
    avec: AtIndex[6:9] | WithXYZ


@dataclass
class EskfState:
    """A combination of nominal and error state"""
    nom: NominalState
    err: MultiVarGauss[ErrorState]

    def get_err_gauss(self, gt: NominalState) -> MultiVarGauss[ErrorState]:
        """Used to calculate error and NEES"""
        err = ErrorState(
            pos=self.nom.pos - gt.pos,
            vel=self.nom.vel - gt.vel,
            avec=gt.ori.diff_as_avec(self.nom.ori))
        return MultiVarGauss[ErrorState](err, self.err.cov)


@dataclass
class ImuMeasurement(NamedArray):
    """Represents raw data received from the imu, see (10.53) in the book and 
    the note in the assignment pdf.
    Args:
        acc: accelerometer measurement
        avel: gyro measurement
    """
    acc: AtIndex[0:3] | WithXYZ
    avel: AtIndex[3:6] | WithXYZ


@dataclass
class CorrectedImuMeasurement(ImuMeasurement):
    """Represents processed data from the IMU.
    Corrected for axis alignmentand scale scale, and bias. 
    Not 'corrected' for gravity.
    """


@ dataclass
class GnssMeasurement(NamedArray):
    """Represents data received from gnss
    Args:
        pos(ndarray[:, 3]): GPS position measurement
        accuracy (Optional[float]): the reported accuracy from the gnss (not used)
    """
    pos: AtIndex[0:3] | WithXYZ
    accuracy: MetaData[Optional[float]] = None
