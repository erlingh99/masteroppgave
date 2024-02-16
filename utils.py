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