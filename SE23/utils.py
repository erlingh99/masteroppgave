import numpy as np

def cross_matrix(vec: np.ndarray[3]) -> np.ndarray[3, 3]:
    x, y, z = vec
    return np.array([[ 0, -z,  y],
                     [ z,  0, -x],
                     [-y,  x,  0]])

def from_cross_matrix(mat: np.ndarray[3, 3]) -> np.ndarray[3]:
    return np.array([-mat[1, 2],
                      mat[0, 2],
                     -mat[0, 1]])

def exp_NEES(pose, gt_mean):
    m, c = pose.mean, pose.cov
    err = (m.inverse()@gt_mean).Log()
    if np.linalg.norm(err) < 1e-5:
        return 0, 0, 0, 0
    
    if np.linalg.det(c) == 0:
        raise Exception("singular covariance, NEES not possible")
    
    nees = err.T@np.linalg.solve(c, err)
    nees_ori = err[:3].T@np.linalg.solve(c[:3, :3], err[:3])
    nees_vel = err[3:6].T@np.linalg.solve(c[3:6, 3:6], err[3:6])
    nees_pos = err[6:].T@np.linalg.solve(c[6:, 6:], err[6:])
    
    if nees < 0:
        raise ValueError("NEES cant be negative")
    return (nees, nees_ori, nees_vel, nees_pos)

def NEES(pose, gt_mean):
    m, c = pose.mean, pose.cov
    err = gt_mean-m
    if np.linalg.norm(err) < 1e-8:
        return 0, 0, 0
    
    if np.linalg.det(c) == 0:
        raise Exception("singular covariance, NEES not possible")
    
    nees = err.T@np.linalg.solve(c, err)
    nees_pos = err[:3].T@np.linalg.solve(c[:3, :3], err[:3])
    nees_vel = err[3:].T@np.linalg.solve(c[3:, 3:], err[3:])

    if nees < 0:
        raise ValueError("NEES cant be negative")
    return nees, nees_pos, nees_vel

    
def exp_cov(poses, mean, weights=None):
    N = len(poses)

    m = np.empty((poses[0].ndim, N))
    minv = mean.inverse()
    for i, p in enumerate(poses):
        m[:, i] = (minv@p).Log()

    # Determine the normalization
    if weights is None:
        fact = N - 1
        mT = m.T
    else:
        w_sum = sum(weights)
        fact = w_sum - weights.T@weights/w_sum
        mT = (m*weights).T
    return m@mT/fact


def find_mean(poses, init_guess, n_iter=100, eps=1e-5, weights=None):
    """
    Find the mean of a set of poses using iterating average on tangent plane
    """
    N = poses.shape[0]
    curr_mean = init_guess.copy()
    n = curr_mean.ndim

    if weights is None:
        weights = np.full(N, 1/N)

    if sum(weights) != 1:
        weights = weights/sum(weights)

    for _ in range(n_iter):
        inv = curr_mean.inverse()
        avg_chg = np.zeros((n, ))
        for i in range(N):
            avg_chg += (inv@poses[i]).Log()*weights[i]

        curr_mean = curr_mean@init_guess.__class__.Exp(avg_chg)
        if np.linalg.norm(avg_chg) < eps:
            break
    return curr_mean


def cov(m):
    '''Estimate a covariance matrix given data.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.

    Returns:
        The covariance matrix of the variables.
    '''

    fact = 1.0 / (m.shape[1] - 1)
    mean = np.mean(m, axis=1, keepdims=True)
    m = m - mean
    return fact * m@m.T

def op1(A):
    """
    <<.>> operator.
    
    <<A>> = - tr(A) Id + A
    """
    return -np.trace(A)*np.eye(3) + A

def op2(A, B):
    """
    <<., .>> operator.
    
    <<A, B>> = <<A>> <<B>> + <<BA>>
    """
    return op1(A)@op1(B) + op1(B@A)