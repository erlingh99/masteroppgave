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
    if np.linalg.det(c) == 0:
        print("singular covariance, NEES not possible")
        return 0
    err = (m.inverse()@gt_mean).Log()
    return err.T@np.linalg.solve(c, err)

def NEES(pose, gt_mean):
    m, c = pose.mean, pose.cov
    if np.linalg.det(c) == 0:
        print("singular covariance, NEES not possible")
        return 0
    err = gt_mean-m
    return err.T@np.linalg.solve(c, err)

    
def exp_cov(poses, mean):
    N = len(poses)
    fact = 1.0 / (N - 1)

    m = np.empty((poses[0].ndim, N))
    for i, p in enumerate(poses):
        m[:, i] = (mean.inverse()@p).Log()
    return fact * m@m.T


def find_mean(poses, init_guess, n_iter=100, eps=1e-5):
    """
    Find the mean of a set of poses using iterating average on tangent plane
    """
    N = poses.shape[0]
    curr_mean = init_guess.copy()
    n = curr_mean.ndim
    for _ in range(n_iter):
        inv = curr_mean.inverse()
        s = np.empty((n, ))
        for i in range(N):
            s += (inv@poses[i]).Log()
        avg_chg = s/N
        curr_mean = curr_mean@init_guess.__class__.Exp(avg_chg)
        if np.linalg.norm(avg_chg) < eps:
            return curr_mean
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