from lie_theory import SE2
import numpy as np
from gaussian import ExponentialGaussian
import matplotlib.pyplot as plt
from plot_utils import plot_2d_frame


# mean = SE2.Exp([np.pi/4, 5, 5])
# cov = np.array([[0.4, 0.2, 0],
                # [0.2, 0.2, 0],
                # [0, 0, 0.4]])
# dist = ExponentialGaussian(mean, cov)


fig, ax = plt.subplots(1, 1)

# dist.draw_significant_ellipses(ax, n_std=1)
# dist.draw_significant_ellipses(ax, n_std=2)
# dist.draw_significant_ellipses(ax, n_std=3)
# dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=1)
# dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=2)
# dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=3)
# plot_2d_frame(ax, dist.mean)
plot_2d_frame(ax, SE2.Exp([0, 0, 0]))

mean = SE2.Exp([3*np.pi/4, 3, 3])
cov = np.array([[0.4, 0, 0],
                [0, 0.2, 0.1],
                [0, 0.1, 0.4]])

dist = ExponentialGaussian(mean, cov)
dist.draw_significant_ellipses(ax, n_std=3, n_ellipsis=2, n_points=200)
dist.draw_significant_ellipses(ax, n_std=2, n_ellipsis=2, n_points=200)
dist.draw_significant_ellipses(ax, n_std=1, n_ellipsis=2, n_points=200)
dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=1)
dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=2)
dist.draw_2Dtranslation_covariance_ellipse(ax, n_points=50, num_std=3)

ts = dist.project_translation(n_points=50)
ax.plot(ts[:, 0], ts[:, 1])

plt.axis("equal")
plt.show()