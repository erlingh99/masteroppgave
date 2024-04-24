import numpy as np
from dataclasses import dataclass
from gaussian import MultiVarGauss, ExponentialGaussian

#wrappers around gaussians with convenience properties
#should really use named array as MultiVarGauss[NamedArray] type


class TargetState(MultiVarGauss):
    @property
    def pos(self):
        return self.mean[:3]

    @property
    def vel(self):
        return self.mean[3:]


class PlatformState(ExponentialGaussian):
    @property
    def rot(self):
        return self.mean.R.as_matrix()

    @property
    def vel(self):
        return self.mean.v

    @property
    def pos(self):
        return self.mean.p
    

@dataclass
class StackedState: #representing the combined state of the above
    target_state: TargetState
    platform_state: PlatformState

    @property
    def mean(self):
        return (self.target_state, self.platform_state.mean)
    
    @property
    def cov(self):
        return np.block([[self.target_state.cov, np.zeros((6,9))],
                         [np.zeros((9,6)), self.platform_state.cov]])