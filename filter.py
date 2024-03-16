import numpy as np
from lie_theory import SE3_2, SO3
from gaussian import ExponentialGaussian
from measurements import IMU_Measurement, GNSS_Sensor
from utils import op1, op2
   

class Filter:
    g = np.array([0, 0, -9.81])
    
    def __init__(self, measurement_sensor: GNSS_Sensor):
        self.measurement_sensor = measurement_sensor

    def Gamma(self, dt):
        """Compute Gamma"""
        Gamma = np.eye(5)
        Gamma[:3, 3] = self.g*dt
        Gamma[:3, 4] = 1/2*self.g*(dt**2)
        return Gamma

    def Phi(self, T, dt):
        """Compute Phi"""
        Phi = T.copy()
        Phi[:3, 4] += Phi[:3, 3]*dt
        return Phi
    
    def Upsilon(self, measurement: IMU_Measurement, dt):
        """Compute Upsilon (IMU)
        simplest possible integration"""
        deltaR = SO3.Exp(dt*measurement.gyro).as_matrix()
        upsilon = np.eye(5)
        upsilon[:3, :3] = deltaR
        upsilon[:3, 3] = measurement.acc*dt
        upsilon[:3, 4] = 1/2*measurement.acc*(dt**2)
        return upsilon

    def F(self, dt):
        f = np.eye(9)
        f[6:, 3:6] = dt * np.eye(3)
        return f
    
    def A(self, upsilon, dt):
        return SE3_2(SO3(upsilon[:3, :3]), upsilon[:3, 3], upsilon[:3, 4]).inverse().adjoint()@self.F(dt)

    def Q(self, measurement, dt):
        gyro = measurement.gyro
        G = np.zeros((9, 6))
        # G[:3, :3] = -SO3.inverse_jac_left(gyro*dt)*dt
        G[:3, :3] = -SO3.jac_right(gyro*dt)*dt
        G[3:6, 3:] = -SO3.Exp(-gyro*dt).as_matrix()*dt
        G[6:9, 3:] = -SO3.Exp(-gyro*dt).as_matrix()*dt**2/2
        return G@measurement.noise@G.T

    def propegate(self, current_state: ExponentialGaussian, imu_measurement: IMU_Measurement, dt):
        mean = current_state.mean.as_matrix()
        cov = current_state.cov

        upsilon = self.Upsilon(imu_measurement, dt)
        new_mean = self.Gamma(dt)@self.Phi(mean, dt)@upsilon
        new_mean = SE3_2.from_matrix(new_mean)
        
        Ai = self.A(upsilon, dt)
        Qi = self.Q(imu_measurement, dt)

        cov_tmp = Ai@cov@Ai.T
        cov_2nd = cov_tmp + Qi
        cov_4th = cov_2nd + self.__fourth_order__(cov_tmp, Qi)

        return ExponentialGaussian(new_mean, cov_4th)
    
    def __propegate_mean__(self, mean: np.ndarray, imu_measurement: IMU_Measurement, dt): #only for convinience when simulating
        upsilon = self.Upsilon(imu_measurement, dt)
        return self.Gamma(dt)@self.Phi(mean, dt)@upsilon

    
    ### Kalman update
    def update(self, pose, z):
        H = self.measurement_sensor.H(pose.mean)
        R = z.noise
        innov = z.pos - self.measurement_sensor.h(pose.mean) #innovation is z in world for GNSS
        S = H@pose.cov@H.T + R
        K = pose.cov@np.linalg.solve(S.T, H).T #equiv to H.T@inv(S)
        gain = K@innov
        err = pose.mean.__class__.Exp(gain)
        new_mean = pose.mean@err
        err_J = err.right_jacobian()
        new_cov = err_J@(np.eye(9)-K@H)@pose.cov@err_J.T
        return ExponentialGaussian(new_mean, new_cov)
    
    def __fourth_order__(self, Sigma, Q):
        Sigma_pp = Sigma[:3, :3]
        Sigma_vp = Sigma[3:6, :3]
        Sigma_vv = Sigma[3:6, 3:6]
        Sigma_vr = Sigma[3:6, 6:9]
        Sigma_rp = Sigma[6:9, :3]
        Sigma_rr = Sigma[6:9, 6:9]

        Qpp = Q[:3, :3]
        Qvp = Q[3:6, :3]
        Qvv = Q[3:6, 3:6]
        Qvr = Q[3:6, 6:9]
        Qrp = Q[6:9, :3]
        Qrr = Q[6:9, 6:9]

        A1 = np.zeros((9, 9))
        A1[:3, :3] = op1(Sigma_pp)
        A1[3:6, :3] = op1(Sigma_vp + Sigma_vp.T)
        A1[3:6, 3:6] = op1(Sigma_pp)
        A1[6:9, :3] = op1(Sigma_rp + Sigma_rp.T)
        A1[6:9, 6:9] = op1(Sigma_pp)

        A2 = np.zeros((9, 9))
        A2[:3, :3] = op1(Qpp)
        A2[3:6, :3] = op1(Qvp + Qvp.T)
        A2[3:6, 3:6] = op1(Qpp)
        A2[6:9, :3] = op1(Qrp + Qrp.T)
        A2[6:9, 6:9] = op1(Qpp)

        Bpp = op2(Sigma_pp, Qpp)
        Bvv = op2(Sigma_pp, Qvv) + op2(Sigma_vp.T, Qvp) +\
            op2(Sigma_vp, Qvp.T) + op2(Sigma_vv, Qpp)
        Brr = op2(Sigma_pp, Qrr) + op2(Sigma_rp.T, Qrp) +\
            op2(Sigma_rp, Qrp.T) + op2(Sigma_rr, Qpp)
        Bvp = op2(Sigma_pp, Qvp.T) + op2(Sigma_vp.T, Qpp)
        Brp = op2(Sigma_pp, Qrp.T) + op2(Sigma_rp.T, Qpp)
        Bvr = op2(Sigma_pp, Qvr) + op2(Sigma_vp.T, Qrp) +\
            op2(Sigma_rp, Qvp.T) + op2(Sigma_vr, Qpp)

        B = np.zeros((9, 9))
        B[:3, :3] = Bpp
        B[:3, 3:6] = Bvp.T
        B[:3, 6:9] = Brp.T
        B[3:6, :3] = B[:3, 3:6].T
        B[3:6, 3:6] = Bvv
        B[3:6, 6:9] = Bvr
        B[6:9, :3] = B[:3, 6:9].T
        B[6:9, 3:6] = B[3:6, 6:9].T
        B[6:9, 6:9] = Brr

        return (A1@Q + Q.T@A1.T + A2@Sigma + Sigma.T@A2.T)/12 + B/4