from dataclasses import dataclass, field
from operator import attrgetter
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl

from .senfuslib import (TimeSequence, ConsistencyAnalysis,
                       MultiVarGauss, plot_field, scatter_field,
                       ax_config, fig_config, show_consistency)
from .states import EskfState, NominalState, GnssMeasurement, ImuMeasurement
from .config import PLOT_MIN_DT


class Mystring(str):
    bold = '\\mathbf'
    _mapping = {
        'pos':          [bold+'{\\rho}',        'm'],
        'pos.x':        ['x (north)',         'm'],
        'pos.y':        ['y (east)',          'm'],
        'pos.z':        ['z (down)',          'm'],

        'vel':          [bold+'{v}',            'm/s'],
        'vel.x':        ['u',                 'm/s'],
        'vel.y':        ['v',                 'm/s'],
        'vel.z':        ['w',                 'm/s'],

        'euler.x':      ['\\phi',             'rad'],
        'euler.y':      ['\\theta',           'rad'],
        'euler.z':      ['\\psi',             'rad'],

        'avec':         [bold+'{\\omega}',      'rad/s'],
        'avec.x':       ['\\omega_{\\phi}',   'rad/s'],
        'avec.y':       ['\\omega_{\\theta}', 'rad/s'],
        'avec.z':       ['\\omega_{\\psi}',   'rad/s'],

        'acc.x':        ['a_x',               'm/s^2'],
        'acc.y':        ['a_y',               'm/s^2'],
        'acc.z':        ['a_z',               'm/s^2'],
        'avel.x':       ['\\omega_x',         'rad/s'],
        'avel.y':       ['\\omega_y',         'rad/s'],
        'avel.z':       ['\\omega_z',         'rad/s'],
    }

    def __format__(self, format):
        return f'${self._mapping[self][0]}$'

    @property
    def unit(self):
        return f'${self._mapping[self][1]}$'


@dataclass
class PlotterESKF:
    x_gts: TimeSequence[NominalState]
    z_imu: TimeSequence[ImuMeasurement]
    x_preds: TimeSequence[EskfState]
    z_preds: TimeSequence[MultiVarGauss[GnssMeasurement]] = field(default=None)
    z_gnss: TimeSequence[GnssMeasurement] = field(default=None)
    x_upds: TimeSequence[EskfState] = field(default=None)

    dt_min: float = field(default=PLOT_MIN_DT)
    x_est: TimeSequence[EskfState] = field(init=False)
    consistency: ConsistencyAnalysis = field(init=False)

    def __post_init__(self):
        self.x_preds = self.x_preds.slice_time(min_dt=self.dt_min)
        self.z_imu = self.z_imu.slice_time(min_dt=self.dt_min)

        self.x_ests = self.x_preds.copy()
        if self.x_upds is not None:
            for t, x_upd in self.x_upds.items():
                self.x_ests.set_t(t, x_upd)

        if self.x_gts is not None:
            self.x_gts = TimeSequence((t, self.x_gts.get_t(t))
                                      for t in self.x_ests.times)

        if self.z_gnss is not None:
            self.consistency = ConsistencyAnalysis(x_gts=self.x_gts,
                                                    zs=self.z_gnss,
                                                    x_ests=self.x_ests,
                                                    z_preds=self.z_preds)
            
        mpl.rcParams['legend.loc'] = 'lower right'
        mpl.rcParams['legend.fontsize'] = 'small'
        mpl.rcParams['legend.labelspacing'] = '0.3'
        mpl.rcParams['legend.handletextpad'] = '0.5'
        mpl.rcParams['legend.columnspacing'] = '1'
        mpl.rcParams['legend.handlelength'] = '1.5'

        mpl.rcParams['axes.grid'] = True

    def imu_plots(self):
        """Here you can add extra plots"""
        for typ, title in zip(['acc', 'avel'],
                              ['Accelerometer', 'Gyroscope']):
            fig, axs = plt.subplots(3, 1, sharex=True)
            for i, n in enumerate('xyz'):
                f = Mystring(f'{typ}.{n}')
                plot_field(axs[i], self.z_imu, f, label=f'{f}')
                ax_config(axs[i], y_label=f'{f} [{f.unit}]')
            ax_config(axs[0], title=title)
            ax_config(axs[-1], x_label='Time [s]')
            fig_config(fig, title)

    def show(self):

        def gt_kwr():
            return dict(label='gt', linestyle=(0, (5, 1)), alpha=0.7, color='C1')

        def est_kwr():
            return dict(label='est', linestyle='-', alpha=0.7, color='C0')

        gt = self.x_gts
        est = self.x_ests.map(attrgetter('nom'))

        self.plot3d(gt, est, gt_kwr, est_kwr)
        self.imu_plots()
        it = zip(
            ('pos', 'vel', 'euler'),
            ('Position', 'Velocity', 'Orientation'))
        for thing, title in it:
            fig, ax = plt.subplots(3, 1, sharex=True)
            fields = [Mystring(f'{thing}.{n}') for n in 'xyz']
            for i, f in enumerate(fields):
                if thing == 'pos':
                    scatter_field(ax[i], self.z_gnss, f, label=f'gps',
                                  marker='o', s=7, c='C6', alpha=0.5)
                if gt is not None:
                    plot_field(ax[i], gt, f, **gt_kwr())
                plot_field(ax[i], est, f, **est_kwr())
                ax_config(ax[i], y_label=f'{f} [{f.unit}]')
            ax_config(ax[0], title=title)
            ax_config(ax[-1], x_label='Time [s]')
            fig_config(fig, title)

            con_fields = [Mystring(f.replace('euler', 'avec'))
                          for f in [thing, *fields]]

            fields_nis = con_fields if thing == 'pos' else []
            fields_nees = con_fields if gt is not None else []
            fields_err = con_fields[1:] if gt is not None else []
            show_consistency(self.consistency, fields_nis,
                             fields_nees, fields_err,
                             title=title)

    def plot3d(self, gt, est, gt_kwr, est_kwr):
        fig = plt.figure()
        ned_corr = np.array([[1, -1, -1]])
        est_pos_arr = ned_corr * est.field_as_array('pos')
        zs_pos_arr = None
        if self.z_gnss is not None:
            zs_pos_arr = ned_corr * self.z_gnss.field_as_array('pos')
        minmax = (np.amin(est_pos_arr), np.amax(est_pos_arr))
        try:
            ax = plt.axes(projection='3d', aspect='equal',
                          xlabel='north [m]', ylabel='west [m]', zlabel='up [m]',
                          xlim=minmax, ylim=minmax, zlim=minmax)
        except:
            ax = plt.axes(projection='3d',
                          xlabel='north [m]', ylabel='west [m]', zlabel='up [m]',
                          xlim=minmax, ylim=minmax, zlim=minmax)

        if gt is not None:
            gt_pos_arr = ned_corr * gt.field_as_array('pos')
            ax.plot(*gt_pos_arr.T, **gt_kwr())
        ax.plot(*est_pos_arr.T, **est_kwr())
        ax.scatter(*est_pos_arr[0], label='start', marker='x', color='red')
        if zs_pos_arr is not None:
            ax.scatter(*zs_pos_arr.T, label='gps', marker='o', s=7, c='C6', alpha=0.5)
        ax.legend()
        fig_config(fig, 'Position 3d')
