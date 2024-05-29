import numpy as np


def get_cross_matrix(vec: np.ndarray) -> 'np.ndarray[3, 3]':
    """Get the matrix equivalent of cross product. S() in (10.68)

    cross_product_matrix(vec1)@vec2 == np.cross(vec1, vec2)

    Hint: see (10.5)

    Args:
        vec (ndarray[3]): vector

    Returns:
        S (ndarray[3,3]): cross product matrix equivalent
    """
    S = np.array([
        [0, 	  -vec[2],  vec[1]],
        [vec[2],   0, 	   -vec[0]],
        [-vec[1],  vec[0],  0]
    ])
    return S
