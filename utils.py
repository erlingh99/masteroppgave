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

def plot_2d_frame(ax, pose, **kwargs):
    """Plot the pose (R, t) in the global frame.
    Keyword Arguments
        * *alpha* -- Alpha value (transparency), default 1
        * *axis_colors* -- List of colors for each axis, default ('r', 'g')
        * *scale* -- Scale factor, default 1.0
        * *text* -- Text description plotted at pose origin, default ''
    :param ax: Current axes
    :param pose: The pose (R, t) of the local frame relative to the global frame,
        where R is a 2x2 rotation matrix and t is a 2D column vector.
    :param kwargs: See above
    :return: List of artists.
    """
    alpha = kwargs.get('alpha', 1)
    axis_colors = kwargs.get('axis_colors', ('r', 'g'))
    scale = kwargs.get('scale', 1)
    text = kwargs.get('text', '')

    artists = []

    pts = scale * np.array([[0, 1, 0],
                            [0, 0, 1]])
    # If R is a valid rotation matrix, the columns are the local orthonormal basis vectors in the global frame.
    # Use the group action to transform between frames
    t_pts = pose@pts
    for i in range(0, 2):
        artists.extend(
            ax.plot([t_pts[0, 0], t_pts[0, i+1]], [t_pts[1, 0], t_pts[1, i+1]], axis_colors[i] + '-', alpha=alpha))

    if text:
        artists.extend([ax.text(t_pts[0, 0], t_pts[1, 0], text, fontsize='large')])

    return artists
    
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
    for _ in range(n_iter):
        inv = curr_mean.inverse()
        s = np.empty((9, ))
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