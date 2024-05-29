from dataclasses import dataclass, field, fields
from typing import Callable, Optional, Sequence
import numpy as np
from zlib import crc32
from pathlib import Path
import pickle
from typing import ClassVar, TypeVar
from senfuslib import DynamicModel, SensorModel, MultiVarGauss, TimeSequence
import logging
try:
    from config import sim_output_dir as out_dir
except ImportError:
    out_dir = None
import tqdm


S = TypeVar('S', bound=np.ndarray)  # State type
M = TypeVar('M', bound=np.ndarray)  # Measurement type


@dataclass
class Simulator:
    dynamic_model: DynamicModel[S]
    sensor_model: SensorModel[M]
    init_state: MultiVarGauss[S]

    end_time: float
    dt: float

    seed: Optional[str] = field(default=None)
    sensor_setter: Optional[Callable[[SensorModel[M], TimeSequence[S]],
                                     None]] = field(default=None, repr=False)

    _gt_data: TimeSequence[S] = field(init=False, default=None)
    _rand_state: ClassVar = None

    datapath: Optional[Path] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if out_dir is None:
            raise ImportError('Please create a config.py file in the root ')
        if self.seed is not None:
            id_number = crc32(repr(self).encode())
            self.datapath = out_dir / f'gt_{self.seed}_{id_number:010d}.pkl'

    def set_random_state(self):
        self._rand_state = np.random.get_state()

        if self.seed:
            np.random.seed(crc32(self.seed.encode('utf-8')))

    def restore_random_state(self):
        np.random.set_state(self._rand_state)

    def simulate(self) -> TimeSequence[S]:
        if self.datapath and self.datapath.is_file():
            logging.info('Loading ground truth data from file. '
                         '(Delete data/cache or change seed to regenerate)')
            with open(self.datapath, 'rb')as f:
                self._gt_data = pickle.load(f)
                return self._gt_data
        self.set_random_state()

        logging.info('Generating ground truth data...')
        self._gt_data = TimeSequence[np.ndarray]()
        if isinstance(self.init_state, MultiVarGauss):
            self._gt_data.insert(0, self.init_state.sample())
        else:
            self._gt_data.insert(0, self.init_state)

        for t in tqdm.tqdm(np.arange(self.dt, self.end_time+self.dt, self.dt)):
            t = np.round(t, 9)
            state_prv = self._gt_data[-1][1]
            state_nxt = self.dynamic_model.step_simulation(state_prv, self.dt)
            if 'time' in [f.name for f in fields(state_nxt)]:
                state_nxt.time = t
            self._gt_data.insert(t, state_nxt)

        if self.datapath:
            self.datapath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.datapath, 'wb') as f:
                pickle.dump(self._gt_data, f)

        self.restore_random_state()
        return self._gt_data

    def get_measurements(self, sensor_model: SensorModel[M]
                         ) -> TimeSequence[M]:

        if self.datapath:
            logging.info('Loading measurements from file. '
                         '(Delete data/cache or change seed to regenerate)')
            id_number = crc32(repr(sensor_model).encode())
            meas_path = self.datapath.with_name(
                f'Sensor_{id_number}_{self.datapath.name}')
            if meas_path.is_file():
                with open(meas_path, 'rb')as f:
                    return pickle.load(f)

        logging.info('Generating measurements...')
        self._gt_data = self._gt_data or self.simulate()
        meas_data = sensor_model.from_states(self._gt_data[1:])

        if self.datapath:
            with open(meas_path, 'wb') as f:
                pickle.dump(meas_data, f)

        return meas_data

    def get_gt_and_meas(self):
        """Returns ground truth and measurements"""
        self._gt_data = self._gt_data or self.simulate()
        if self.sensor_setter is not None:
            self.sensor_setter(self.sensor_model, self._gt_data)
        return self._gt_data, self.get_measurements(self.sensor_model)
