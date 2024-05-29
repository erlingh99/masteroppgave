import numpy as np
from dataclasses import dataclass
from .states import TargetState, PlatformState
from .lie_theory import SE3_2, SO3
from .utils import cross_matrix, op1, op2
from .measurements import IMU_Measurement

@dataclass
class IMU_Model:
    """
    bias and frame correction can be implemented here
    here no bias and IMU and body frame coincide
    """
    noise: np.ndarray[6, 6] #noise of IMU measurements
    
    g = np.array([0, 0, -9.81])
    
    def propegate_mean(self, mean: np.ndarray[5, 5], z: IMU_Measurement, dt: float, mode=1):
        return self.gravityMatrix(dt)@self.positionChange(mean, dt)@self.incrementMatrix(z, dt, mode=mode)


    def propegate_cov(self, cov: np.ndarray[9, 9], z: IMU_Measurement, dt: float, cls=SE3_2):
        inc = self.incrementMatrix(z, dt, mode=1)
        Ai = self.__A__(inc, dt, cls=cls)
        Qi = self.__Q__(z, dt, mode=2, cls=cls)

        cov_tmp = Ai@cov@Ai.T
        cov_2nd = cov_tmp + Qi
        cov_4th = cov_2nd + self.__fourth_order__(cov_tmp, Qi)
        return cov_4th#, cov_2nd


    def gravityMatrix(self, dt):
        """Compute gravity compensation"""
        Gamma = np.eye(5)
        Gamma[:3, 3] = self.g*dt
        Gamma[:3, 4] = 1/2*self.g*(dt**2)
        return Gamma

    def positionChange(self, T, dt):
        """Compute Phi"""
        Phi = T.copy()
        Phi[:3, 4] += Phi[:3, 3]*dt
        return Phi
    
    def incrementMatrix(self, measurement: IMU_Measurement, dt, mode=1):
        """Compute Upsilon (IMU)"""
        # g = measurement.gyro
        # a = measurement.acc
        # return SE3_2.Exp(np.concatenate([dt*g, dt*a, self.__C__(g*dt)@a*0.5*dt*dt])).as_matrix()
        deltaR = SO3.Exp(dt*measurement.gyro).as_matrix()
        upsilon = np.eye(5)
        upsilon[:3, :3] = deltaR
        upsilon[:3, 3] = measurement.acc*dt
        upsilon[:3, 4] = 1/2*(dt**2)*measurement.acc
        
        if mode == 1:
            Jl = SO3.jac_left(dt*measurement.gyro)
            Hl = Jl@self.__C__(measurement.gyro*dt)
            upsilon[:3, 3] = Jl@upsilon[:3, 3]
            upsilon[:3, 4] = Hl@upsilon[:3, 4]
        elif mode == 2:
            Jl = SO3.jac_left(dt*measurement.gyro)
            upsilon[:3, 3] = Jl@upsilon[:3, 3]
            upsilon[:3, 4] = Jl@upsilon[:3, 4] #np.zeros(3)

        return upsilon


    def __F__(self, dt):
        f = np.eye(9)
        f[6:, 3:6] = dt * np.eye(3)
        return f
    
    def __A__(self, inc, dt, cls=SE3_2):
        return cls.from_matrix(inc).inverse().adjoint()@self.__F__(dt)
        # return SO3xR3xR3.from_matrix(inc).inverse().adjoint()@self.__F__(dt)
    
    def __C__(self, angle_axis): #correction matrix
        angle = np.linalg.norm(angle_axis)
        if angle < 1e-8:
            return np.eye(3)
        
        return np.eye(3) + ((1 + np.cos(angle))/np.sin(angle) - 2/angle)*cross_matrix(angle_axis/angle)

    def __Q__(self, measurement, dt, mode=100, cls=SE3_2):
        gyro = measurement.gyro
        acc = measurement.acc

        if mode == 0:
            G = np.zeros((9, 6))

            G[:3, :3] = -SO3.jac_right(gyro*dt)*dt
            G[3:6, 3:] = -SO3.Exp(-gyro*dt).as_matrix()*dt
            G[6:9, 3:] = -SO3.Exp(-gyro*dt).as_matrix()*dt**2/2
            return G@self.noise@G.T
        elif mode == 1:
            G = np.zeros((9, 6))

            G[:3, :3] = -SO3.jac_right(gyro*dt)*dt
            G[3:6, 3:] = G[:3, :3]
            G[6:9, 3:] = G[:3, :3]*dt/2
            return G@self.noise@G.T
    

        correction_mat = self.__C__(gyro*dt)

        tangent_increments = np.concatenate([gyro*dt, acc*dt, correction_mat@acc*dt*dt*0.5])
        Jr = cls.jac_right(tangent_increments)

        temp = np.zeros((9, 6)) #right part of last step leading to eq 46
        temp[:3, :3] = dt*np.eye(3)
        temp[3:6, 3:] = dt*np.eye(3)
        temp[6:, 3:] = 0.5*dt*dt*correction_mat
        G = Jr@temp

        return G@self.noise@G.T

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



@dataclass
class CV_world:
    var_acc: float #velocity variance, assumed same for x y z

    def propegate(self, state: TargetState, dt: float):
        F = self.F(dt)
        new_mean = F@state.mean
        new_cov = F@state.cov@F.T + self.Q(dt)

        return TargetState(new_mean, new_cov)
        

    def F(self, dt: float):
        return np.block([[       np.eye(3), dt*np.eye(3)],
                         [np.zeros((3, 3)),    np.eye(3)]])
    

    def Q(self, dt: float):
        return np.block([[dt**3/3*np.eye(3), dt**2/2*np.eye(3)],
                         [dt**2/2*np.eye(3), dt*np.eye(3)]])*self.var_acc


@dataclass
class CV_body:
    var_acc: float

    def propegate(self, state: TargetState, platform_state_k: PlatformState, platform_state_kp1: PlatformState, z: IMU_Measurement, imu: IMU_Model, dt: float):


        Rhatk = platform_state_k.rot
        Rhatkp1 = platform_state_kp1.rot

        dR = Rhatkp1.T@Rhatk
        A = np.block([[dR, dt*dR], [np.zeros((3,3)), dR]])
        b = np.concatenate([Rhatkp1.T@(platform_state_k.pos - platform_state_kp1.pos + dt*platform_state_k.vel),
                            Rhatkp1.T@(platform_state_k.vel - platform_state_kp1.vel)])


        n_mean = A@state.mean + b 

        lin_point = platform_state_kp1.mean.action2(n_mean)
        
        J_xb_Tinv = np.block([[-Rhatkp1.T@cross_matrix(lin_point[:3]), np.zeros((3,3)), Rhatkp1.T],
                              [-Rhatkp1.T@cross_matrix(lin_point[3:]), Rhatkp1.T, np.zeros((3,3))]])

        J_Tinv_T = -platform_state_kp1.mean.adjoint()

        J_xb_xw = np.block([[Rhatkp1.T, np.zeros((3,3))],
                            [np.zeros((3, 3)), Rhatkp1.T]])

        J_xw_T =  np.block([[-Rhatk@cross_matrix(state.pos), np.zeros((3,3)), Rhatk],
                            [-Rhatk@cross_matrix(state.vel), Rhatk, np.zeros((3,3))]])
    
        Fcv = self.Fcv(dt)
        Fdt = imu.__F__(dt)
        Ad_inc_inv = SE3_2.from_matrix(imu.incrementMatrix(z, dt)).inverse().adjoint()

        J1 = J_xb_Tinv@J_Tinv_T@Ad_inc_inv@Fdt + J_xb_xw@Fcv@J_xw_T
        J2 = J_xb_Tinv@J_Tinv_T
        J3 = J_xb_xw
        
        cv_noise = self.Q(dt)
        imu_noise = imu.__Q__(z, dt)

        n_cov = J1@platform_state_k.cov@J1.T + J2@imu_noise@J2.T + J3@cv_noise@J3.T

        return TargetState(n_mean, A@state.cov@A.T + n_cov)
                                                            
    def Fcv(self, dt: float):
        return np.block([[       np.eye(3), dt*np.eye(3)],
                         [np.zeros((3, 3)),    np.eye(3)]])
    
    def Q(self, dt: float):
        return np.block([[dt**3/3*np.eye(3), dt**2/2*np.eye(3)],
                         [dt**2/2*np.eye(3), dt*np.eye(3)]])*self.var_acc

@dataclass
class CV_body2:
    var_acc: float

    def propegate(self, state: PlatformState, platform_state_k: PlatformState, platform_state_kp1: PlatformState, z: IMU_Measurement, imu: IMU_Model, dt: float):

        
        F = imu.__F__(dt)

        ad1 = self.f(state.mean, dt).inverse().adjoint()
        ad2 = self.f(platform_state_k.mean@state.mean, dt).inverse().adjoint()
        ad3 = platform_state_kp1.mean.adjoint()
        ad4 = SE3_2.from_matrix(imu.incrementMatrix(z, dt)).inverse().adjoint()

        J1 = (ad1 - ad2@ad3@ad4)@F
        J2 = ad2@ad3

        imu_noise = imu.__Q__(z, dt)

        cov = J1@platform_state_k.cov@J1.T + F@state.cov@F.T + self.Q_ext(dt) + J2@imu_noise@J2.T

        mean = platform_state_kp1.mean.inverse()@self.f(platform_state_k.mean@state.mean, dt)     
        
        #
        ad = SE3_2(mean.R, np.zeros((3)), np.zeros((3))).adjoint()
        cov = ad@cov@ad.T
        mean.R = SO3(np.eye(3))

        return PlatformState(mean, cov)

    def f(self, T: SE3_2, dt):
        T = T.copy()
        T.p = T.p + dt*T.v
        return T
    

    def Q(self, dt: float):
        return np.block([[dt**3/3*np.eye(3), dt**2/2*np.eye(3)],
                         [dt**2/2*np.eye(3), dt*np.eye(3)]])*self.var_acc
    
    def Q_ext(self, dt):
        H = np.block([[np.zeros((3,6))],
                      [np.zeros((3,3)), np.eye(3)],
                      [np.eye(3), np.zeros((3,3))]])
        return H@self.Q(dt)@H.T