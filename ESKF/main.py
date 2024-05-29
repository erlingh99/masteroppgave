from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.dataloader import load_data
from states import EskfState, ImuMeasurement, GnssMeasurement
from senfuslib import MultiVarGauss, TimeSequence
from eskf import ESKF
from plotting import PlotterESKF

from tuning_sim import (eskf_sim, x_est_init_sim,
                        start_time_sim, end_time_sim,
                        imu_min_dt_sim, gnss_min_dt_sim)
from config import fname_data_sim


def run_eskf(eskf: ESKF,
             x_est_init: EskfState,
             z_imu_tseq: TimeSequence[ImuMeasurement],
             z_gnss_tseq: TimeSequence[GnssMeasurement],
             ) -> tuple[TimeSequence[EskfState],
                        TimeSequence[EskfState],
                        TimeSequence[MultiVarGauss[ImuMeasurement]]]:

    t_prev = z_imu_tseq.times[0]
    x_est_prev = x_est_init
    x_est_pred_tseq = TimeSequence([(t_prev, x_est_init)])
    z_est_pred_tseq = TimeSequence()
    x_est_upd_tseq = TimeSequence()

    gnss_copy = z_gnss_tseq.copy()
    for t_imu, z_imu in tqdm(z_imu_tseq.items()):

        # Handle gnss measurements that has arrived since last imu measurement
        while gnss_copy and gnss_copy.times[0] <= t_imu:
            t_gps, z_gnss = gnss_copy.pop_idx(0)
            dt = t_gps - t_prev
            x_est_pred = eskf.predict_from_imu(x_est_prev, z_imu, dt)
            x_est_upd, z_est_pred = eskf.update_from_gnss(x_est_pred, z_gnss)
            x_est_prev = x_est_upd
            x_est_pred_tseq.insert(t_gps, x_est_pred)
            z_est_pred_tseq.insert(t_gps, z_est_pred)
            x_est_upd_tseq.insert(t_gps, x_est_upd)
            t_prev = t_gps

        dt = t_imu - t_prev
        if dt > 0:
            x_est_pred = eskf.predict_from_imu(x_est_prev, z_imu, dt)
            x_est_pred_tseq.insert(t_imu, x_est_pred)
            x_est_prev = x_est_pred
        t_prev = t_imu

    return x_est_upd_tseq, x_est_pred_tseq, z_est_pred_tseq


def main():

    fname = fname_data_sim
    esfk = eskf_sim
    x_est_init = x_est_init_sim
    tslice = (start_time_sim, end_time_sim, imu_min_dt_sim)
    gnss_min_dt = gnss_min_dt_sim

    x_gt, z_imu_tseq, z_gnss_tseq = load_data(fname)

    z_imu_tseq = z_imu_tseq.slice_time(*tslice)
    z_gnss_tseq = z_gnss_tseq.slice_time(z_imu_tseq.times[0],
                                         z_imu_tseq.times[-1],
                                         gnss_min_dt)

    out = run_eskf(esfk, x_est_init, z_imu_tseq, z_gnss_tseq)
    x_est_upd_tseq, x_est_pred_tseq, z_est_pred_tseq = out

    PlotterESKF(x_gts=x_gt,
                z_imu=z_imu_tseq,
                z_gnss=z_gnss_tseq,
                x_preds=x_est_pred_tseq,
                z_preds=z_est_pred_tseq,
                x_upds=x_est_upd_tseq,
                ).show()
    plt.show(block=True)


if __name__ == '__main__':
    main()
