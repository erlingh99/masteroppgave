from lie_theory import SE3_2, SO3, SE2
from gaussian import ExponentialGaussian, MultiVarGauss
from utils import cross_matrix
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_as_2d, plot_as_SE2, plot_2d_frame

T_mean = SE3_2.Exp([0, 0, np.pi/4, 0, 1, 0, 3, 8, 0])
T_cov = np.diag([0, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.1, 0])
T_state = ExponentialGaussian(T_mean, T_cov)

T_ish = MultiVarGauss(T_mean.t[:2], np.diag([0.1, 0.1]))

target_mean = np.array([1, 5, 0, 2, 5, 0])
target_cov = np.diag([0.3, 0.1, 0, 0.1, 0.3, 0])
target_body = MultiVarGauss(target_mean, target_cov)

def convert_state_to_world_lin(target_state: MultiVarGauss, platform_state: ExponentialGaussian):
    Rhat = platform_state.mean.R.as_matrix()
    J1 = np.block([[Rhat, np.zeros((3,3))],
                    [np.zeros((3, 3)), Rhat]])
    J2 = np.block([[-Rhat@cross_matrix(target_state.mean[:3]), np.zeros((3,3)), Rhat],
                    [-Rhat@cross_matrix(target_state.mean[3:]), Rhat, np.zeros((3,3))]])
    
    return MultiVarGauss(platform_state.mean.action2(target_state.mean), J1@target_state.cov@J1.T + J2@platform_state.cov@J2.T)


def convert_state_to_world_SE3_2(target_state: MultiVarGauss, platform_state: ExponentialGaussian):
    local_mean = SE3_2(SO3(np.eye(3)), target_state.mean[3:], target_state.mean[:3])
    reorder_mat = np.block([[np.zeros((3,6))], [np.zeros((3,3)), np.eye(3)], [np.eye(3), np.zeros((3,3))]]) #need to reorder the states to match the SE3_2 convention
    extended_cov = reorder_mat@target_state.cov@reorder_mat.T
    tot_mean = platform_state.mean@local_mean
    ad_inv = local_mean.inverse().adjoint()
    tot_cov = ad_inv@platform_state.cov@ad_inv.T + extended_cov
    return ExponentialGaussian(tot_mean, tot_cov)



target_world = convert_state_to_world_lin(target_body, T_state)
target_world_se = convert_state_to_world_SE3_2(target_body, T_state)


R2 = np.block([[T_state.mean.R.as_matrix(), np.zeros((3,3))], [np.zeros((3,3)), T_state.mean.R.as_matrix()]])
target_body_world = MultiVarGauss(T_state.mean.action2(target_body.mean), R2@target_body.cov@R2.T)

target_world2 = MultiVarGauss(target_body_world.mean[:2], target_body_world.cov[:2, :2] + T_ish.cov)

_, axs = plt.subplots(1,2)
plot_as_SE2(axs[0], T_state, scale=1)
plot_as_2d(axs[0], T_ish)
plot_as_2d(axs[0], target_body_world, color="orange")
# plot_as_2d(axs[0], target_body, color="orange")
plot_2d_frame(axs[0], SE2.Exp([0, 0, 0]))
plot_as_2d(axs[1], target_world, color="green")
plot_as_2d(axs[1], target_world2, color="blue")
plot_as_SE2(axs[1], target_world_se, scale=1)
plot_2d_frame(axs[1], SE2.Exp([0, 0, 0]))
for a in axs:
    a.axis("equal")
plt.show()