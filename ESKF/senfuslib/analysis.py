from functools import cache
from operator import attrgetter
from typing import Any, Sequence, Tuple, TypeVar, Union
import numpy as np
from scipy.stats import chi2
from dataclasses import dataclass, field
from senfuslib import TimeSequence, MultiVarGauss, NamedArray

S = TypeVar('S', bound=np.ndarray)  # State type
M = TypeVar('M', bound=np.ndarray)  # Measurement type


@cache
def chi2_interval(alpha, dof):
    return chi2.interval(alpha, dof)


@cache
def chi2_mean(dof):
    return chi2.mean(dof)


@dataclass
class ConsistencyData:
    mahal_dist_tseq: TimeSequence[MultiVarGauss[S]]
    low_med_upp_tseq: TimeSequence[MultiVarGauss[Tuple[float, float, float]]]
    above_median: float
    in_interval: float
    alpha: float
    dofs: list[int]
    a: float
    adof: int
    aconf: Tuple[float, float]


@dataclass
class ConsistencyAnalysis:
    x_gts: TimeSequence[S]
    zs: TimeSequence[M]
    x_ests: TimeSequence[Union[MultiVarGauss[S], Any]]
    z_preds: TimeSequence[Union[MultiVarGauss[S], Any]]

    x_err_gauss: TimeSequence[MultiVarGauss[S]] = field(init=False)
    z_err_gauss: TimeSequence[MultiVarGauss[M]] = field(init=False)

    def __post_init__(self):
        def get_err_tseq(gts: TimeSequence, ests: TimeSequence):
            err_gauss_tseq = TimeSequence()
            for t, est in ests.items():
                if t not in gts:
                    continue
                gt = gts.get_t(t)
                if isinstance(est, MultiVarGauss):
                    err = MultiVarGauss(est.mean - gt, est.cov)
                else:
                    err = est.get_err_gauss(gt)
                err_gauss_tseq.insert(t, err)
            return err_gauss_tseq

        if self.x_gts is not None:
            self.x_err_gauss = get_err_tseq(self.x_gts, self.x_ests)
        self.z_err_gauss = get_err_tseq(self.zs, self.z_preds)

    def get_nis(self, indices: Sequence[Union[int, str]] = None,
                alpha=0.95) -> ConsistencyData:
        err_gauss_tseq = self._get_err(self.z_err_gauss, indices)
        return self._get_nisornees(err_gauss_tseq, alpha)

    def get_nees(self, indices: Sequence[Union[int, str]] = None,
                 alpha=0.95) -> ConsistencyData:
        err_gauss_tseq = self._get_err(self.x_err_gauss, indices)
        return self._get_nisornees(err_gauss_tseq, alpha)

    def get_x_err(self, indices: Sequence[Union[int, str]] = None):
        return self._get_err(self.x_err_gauss, indices)

    def get_z_err(self, indices: Sequence[Union[int, str]] = None):
        return self._get_err(self.z_err_gauss, indices)

    @staticmethod
    def _get_err(err_gauss_tseq: TimeSequence[MultiVarGauss[NamedArray]],
                 indices: Sequence[Union[int, str]]
                 ) -> TimeSequence[MultiVarGauss[NamedArray]]:
        if indices is None:
            indices = np.arange(err_gauss_tseq.values[0].ndim)

        elif isinstance(indices, (int, str)):
            indices = [indices]

        def marginalize(err_gauss: MultiVarGauss[NamedArray]):
            def idx_map(idx):
                if isinstance(idx, str):
                    idx = attrgetter(idx)(err_gauss.mean.indices)[0]
                return idx
            _indices = np.r_[tuple(idx_map(idx) for idx in indices)]
            return err_gauss.get_marginalized(_indices)

        return err_gauss_tseq.map(marginalize)

    def _get_nisornees(self,
                       err_gauss_tseq: TimeSequence[MultiVarGauss[NamedArray]],
                       alpha: float,
                       ) -> ConsistencyData:

        def get_mahal(x: MultiVarGauss[NamedArray]):
            return x.mahalanobis_distance(np.zeros_like(x.mean))

        mahal_dist_tseq = err_gauss_tseq.map(get_mahal)

        low_med_upp_tseq = TimeSequence()
        dofs = []
        for t, err in err_gauss_tseq.items():
            dofs.append(err.mean.shape[0])
            lower, upper = chi2_interval(alpha, dofs[0])
            low_med_upp_tseq.insert(t, (lower, chi2_mean(dofs[0]), upper))

        n = len(mahal_dist_tseq)
        above_median = 0
        in_interval = 0
        for mahal_dist, lmu in zip(mahal_dist_tseq.values, low_med_upp_tseq.values):
            above_median += mahal_dist > lmu[1]
            in_interval += lmu[0] <= mahal_dist <= lmu[2]

        above_median = above_median / n
        in_interval = in_interval / n

        a = np.mean(mahal_dist_tseq.values)
        adof = sum(dofs)
        aconf = tuple(i/n for i in chi2_interval(alpha, adof))
        return ConsistencyData(mahal_dist_tseq, low_med_upp_tseq,
                               above_median, in_interval, alpha, dofs,
                               a, adof, aconf)
