import numpy as np
from states import PlatformState, TargetState
from lie_theory import SE2, SO2


def plot_as_SE2(ax, pose: PlatformState, color: str ="red", z: np.ndarray = None):
    extract = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0]])
    m = pose.mean.as_matrix()
    c = extract@pose.cov@extract.T
    pose2D = SE2(SO2(m[:2, :2]), m[:2, 4])
    exp = PlatformState(pose2D, c)
    plot_2d_frame(ax, pose2D, scale=5)
    exp.draw_2Dtranslation_covariance_ellipse(ax, "xy", 3, 50, color=color)
    ax.plot(m[0, 4], m[1, 4], color=color, marker="o")
    if z is not None:
        ax.plot(z[0], z[1], color=color, marker="x")

def plot_as_2d(ax, pose: TargetState, color: str ="red", z: np.ndarray = None, num_std: float = 3):
    m = pose.mean[:2]
    c = pose.cov[:2, :2]
    pose2d = TargetState(m, c)
    x, y = pose2d.covar_ellipsis(resolution=50, num_std=num_std)
    ax.plot(x, y, label=f"{num_std}σ", color=color)
    ax.plot(m[0], m[1], color=color, marker="o")

    if z is not None:
        ax.plot(z[0], z[1], color=color, marker="x")


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