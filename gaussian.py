from dataclasses import dataclass
import numpy as np
import scipy.stats


@dataclass
class MultiVarGauss:
    """A class for using Gaussians"""
    mean: np.ndarray
    cov: np.ndarray

    @property
    def ndim(self) -> int:
        return self.mean.shape[0]

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """
        Calculate the mahalanobis distance between self and x.
        """
        err = x.reshape(-1, 1) - self.mean.reshape(-1, 1)
        mahalanobis_distance = float(err.T @ np.linalg.solve(self.cov, err))
        return mahalanobis_distance

    def mahal_dist(self, x: np.ndarray) -> float:
        return self.mahalanobis_distance(x)

    def pdf(self, x: np.ndarray) -> float:
        """Calculate the likelihood of x given the Gaussian"""
        return scipy.stats.multivariate_normal(self.mean, self.cov).pdf(x)

    def logpdf(self, x: np.ndarray) -> float:
        """Calculate the log likelihood of x given the Gaussian"""
        return scipy.stats.multivariate_normal(self.mean, self.cov).logpdf(x)

    def sample(self) -> np.ndarray:
        """Sample from the Gaussian"""
        noise = np.random.multivariate_normal(
            np.zeros_like(self.mean), self.cov, 1).reshape(-1)
        return self.mean + noise

    def get_marginalized(self, indices):
        i_idx, j_idx = np.meshgrid(indices, indices,
                                   sparse=True, indexing='ij')
        mean = self.mean[i_idx.ravel()]
        cov = self.cov[i_idx, j_idx]
        return MultiVarGauss(mean, cov)

    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.cov)

    def __iter__(self):
        """
        Enable iteration over the mean and covariance. i.e.
            mvg = MultiVarGauss(mean=m, cov=c)
            mean, cov = mvg
        """
        return iter((self.mean, self.cov))

    def __repr__(self) -> str:
        """Used for pretty printing"""
        def sci(x): return f'{x: .23}'
        out = '\n'
        for i in range(self.mean.shape[0]):
            mline = sci(self.mean[i])
            cline = ' |'.join(sci(self.cov[i, j])
                              for j in range(self.cov.shape[1]))
            out += f"|{mline} |      |{cline} |\n"
        return out

    def __str__(self) -> str:
        return str(self)
