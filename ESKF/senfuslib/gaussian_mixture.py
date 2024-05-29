from dataclasses import dataclass
import numpy as np
from typing import Sequence, TypeVar, Generic

from senfuslib import MultiVarGauss
from .named_array import NamedArray
from config import DEBUG


S = TypeVar('S', bound=np.ndarray)  # State type


@dataclass
class GaussianMixture(Generic[S]):
    weights: np.ndarray
    gaussians: Sequence[MultiVarGauss[S]]

    _cache_mean = None
    _cache_cov = None

    def __post_init__(self):
        if DEBUG:
            self._debug()

    @property
    def mean(self):
        """Find the mean of the gaussian mixture.
        Hint: Use (6.15) from the book."""
        if self._cache_mean is not None:
            return self._cache_mean
        mean = np.average([g.mean for g in self.gaussians],
                          axis=0, weights=self.weights)
        if issubclass(cls := self.gaussians[0].mean.__class__, NamedArray):
            mean = mean.view(cls)
        self._cache_mean = mean
        return mean

    @property
    def cov(self):
        """Find the covariance of the gaussian mixture.
        Hint: Use (6.16) from the book."""
        if self._cache_cov is not None:
            return self._cache_cov
        covs = np.array([g.cov for g in self.gaussians])
        means = np.array([g.mean for g in self.gaussians])
        cov = (np.average(covs, axis=0, weights=self.weights)
               + (self.weights[:, None] * means).T @ means
               - self.mean[:, None] @ self.mean[None, :])
        self._cache_cov = cov
        return cov

    def reduce(self) -> MultiVarGauss[S]:
        """Recude the gaussian mixture to a single gaussian."""

        gauss = MultiVarGauss(self.mean, self.cov)
        return gauss

    def reduce_partial(self, indices: Sequence[int]):
        weights_to_reduce = np.array([self.weights[i] for i in indices])
        gauss_to_reduce = [self.gaussians[i] for i in indices]
        reduced = GaussianMixture(weights_to_reduce/np.sum(weights_to_reduce),
                                  gauss_to_reduce).reduce()

        keep_indices = list(set(range(len(self))) - set(indices))
        weights_to_keep = [self.weights[i] for i in keep_indices]
        gauss_to_keep = [self.gaussians[i] for i in keep_indices]
        out = GaussianMixture(
            np.array([sum(weights_to_reduce), *weights_to_keep]),
            [reduced, *gauss_to_keep])
        return out

    def pdf(self, x):
        return np.sum(self.weights * np.array([g.pdf(x)
                                              for g in self.gaussians]))

    def __len__(self):
        return len(self.gaussians)

    def _debug(self):
        assert self.weights.ndim == 1
        assert self.weights.shape[0] == len(self.gaussians)
        assert np.isclose(np.sum(self.weights), 1)

    def __getitem__(self, idx):
        return GaussianMixture(self.weights[idx], self.gaussians[idx])
