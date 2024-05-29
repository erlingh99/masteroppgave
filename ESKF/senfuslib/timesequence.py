from bisect import bisect_left, insort_right, bisect_right
from dataclasses import InitVar, dataclass, field
from typing import (Callable, ClassVar, Generic, Iterable, Optional, Sequence, TypeVar,
                    Union, Any)
from itertools import islice
import numpy as np
from collections.abc import Mapping
from operator import attrgetter


T = TypeVar("T")


class NoDefault:
    pass


class IterPeekable(Generic[T]):
    def __init__(self, iterator: Iterable[T], length: Optional[int] = None):
        self._it = iter(iterator)
        self._len_iter = length
        self._peeked = []

    def __next__(self) -> T:
        self._len_iter -= 1
        if self._peeked:
            return self._peeked.pop(0)
        else:
            return next(self._it)

    def peek(self, n: int = 1) -> T:
        while len(self._peeked) < n:
            self._peeked.append(next(self._it))
        return self._peeked[n-1]

    def peek_until(self, until: Callable[[T], bool]) -> T:
        for i in range(len(self)):
            if until(self.peek(i+1)):
                return self.peek(i+1)
        raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self._len_iter + len(self._peeked)


@dataclass(repr=False)
class TimeSequence(Mapping, Generic[T]):
    """A class for storing a sequence of objects in time order"""

    times: list[float] = field(default_factory=list, init=False)
    t_min: Optional[float] = field(default=None, init=False)
    t_max: Optional[float] = field(default=None, init=False)
    _value_dict: ClassVar[dict[float, T]]

    init_iter: InitVar[Iterable[tuple[float, T]]] = None

    def __post_init__(self, init_iter: Optional[Iterable[tuple[float, T]]]):
        """Create a new TimeSeries from an iterable of (ts, value) pairs"""
        self._value_dict = dict()
        for ts, value in init_iter or []:
            self.insert(ts, value)

    def zero(self, zero=None) -> 'TimeSequence':
        zero = zero if zero is not None else self.t_min
        self.t_max = self.t_max - zero
        self.t_min = self.t_min - zero
        new_times = [t - zero for t in self.times]

        self._value_dict = {t: v for t, v in zip(new_times, self.values)}
        self.times = new_times
        return self

    def copy(self) -> 'TimeSequence':
        out = TimeSequence()
        out.t_min = self.t_min
        out.t_max = self.t_max
        out.times = self.times.copy()
        out._value_dict = self._value_dict.copy()
        return out

    @property
    def values(self) -> list[T]:
        """Get the values of the time series, in time order"""
        return [self._value_dict[ts] for ts in self.times]

    def items(self) -> IterPeekable[tuple[float, T]]:
        return IterPeekable(((t, self._value_dict[t])
                             for t in self.times),
                            len(self))

    def insert(self, ts: float, value: T):
        """Insert a new value into the time series"""
        ts = float(ts)
        if ts in self._value_dict:
            raise ValueError(f"Timestamp {ts} already exists")
        self._value_dict[ts] = value

        lo, hi = (0, len(self.times))
        if self.t_max is None or ts > self.t_max:
            self.t_max = ts
            lo = hi
        if self.t_min is None or ts < self.t_min:
            self.t_min = ts
            hi = lo
        insort_right(self.times, ts, lo=lo, hi=hi)

    def pop_idx(self, idx: int) -> tuple[float, T]:
        t = self.times.pop(idx)
        val = self._value_dict.pop(t)
        if t == self.t_min:
            self.t_min = self.times[0] if self.times else None
        if t == self.t_max:
            self.t_max = self.times[-1] if self.times else None
        return t, val

    def pop_t(self, ts: float) -> tuple[float, T]:
        idx = bisect_left(self.times, ts)
        return self.pop_idx(idx)

    def get_idx(self, idx: int) -> tuple[float, T]:
        return self.times[idx], self._value_dict[self.times[idx]]

    def get_t(self, t: float, default=NoDefault) -> T:
        if default is NoDefault:
            return self._value_dict[t]
        return self._value_dict.get(t, default)

    def set_t(self, t: float, value: T):
        if t not in self._value_dict:
            self.insert(t, value)
        self._value_dict[t] = value

    def combine_with(self, *other: 'TimeSequence') -> IterPeekable:
        def gen():
            tseqs: Sequence[TimeSequence] = [self, *other]
            iters = [iter(ts.times) for ts in tseqs if ts]
            times = [next(it) for it in iters]

            while any(times):
                argmin = np.argmin(times)
                yield (times[argmin], tseqs[argmin].get_t(times[argmin]))
                if (time := next(iters[argmin], None)) is not None:
                    times[argmin] = time
                else:
                    tseqs.pop(argmin)
                    iters.pop(argmin)
                    times.pop(argmin)
        return IterPeekable(gen(),
                            len(self) + sum(len(s) for s in other))

    def first_matching(self, matcher: Callable[[T], bool], default=NoDefault,
                       after: float = -np.inf) -> Optional[T]:

        start = bisect_left(self.times, after)

        for ts in self.times[start:]:
            if matcher(self.get_t(ts)):
                return ts, self.get_t(ts)
        if isinstance(default, NoDefault):
            raise ValueError(f"No matching value found")
        else:
            return default

    def map(self, f: Callable[[T], T]) -> 'TimeSequence[T]':
        return TimeSequence((t, f(v)) for t, v in self.items())

    def filter(self, f: Callable[[T], bool]) -> 'TimeSequence':
        return TimeSequence((t, v) for t, v in self.items() if f(v))

    def field_as_array(self, field: Union[str, int, None]) -> np.ndarray:
        if isinstance(field, str):
            arr = np.stack([attrgetter(field)(v) if field else v
                            for v in self.values])
        elif isinstance(field, int):
            arr = np.stack([v[field] for v in self.values])
        elif field is None:
            arr = np.stack(self.values)
        else:
            raise TypeError(f"field must be str or int, not {type(field)}")
        return arr

    def values_as_array(self) -> np.ndarray:
        return self.field_as_array(None)

    def slice_idx(self, start=0, stop=None, step=None
                  ) -> 'TimeSequence':
        stop = stop or len(self.times)
        stop = stop if stop >= 0 else len(self.times) + stop
        return TimeSequence(islice(self.items(), start, stop, step))

    def slice_time(self, start=None, stop=None, min_dt=None,
                   lopen=False, ropen=True) -> 'TimeSequence[T]':
        def gen():
            _start = start or self.t_min
            _stop = stop or self.t_max
            start_idx = (bisect_right if lopen else bisect_left)(
                self.times, _start)
            stop_idx = (bisect_left if ropen else bisect_right)(
                self.times, _stop)
            prev = float('-inf')

            for ts, value in self.slice_idx(start_idx, stop_idx).items():
                if (min_dt is None
                        or ts - prev >= min_dt
                        or np.isclose(ts - prev, min_dt)):
                    yield ts, value
                    prev = ts
        return TimeSequence(gen())

    def get_min_max(self, key: Callable[[T], Any], return_time=False):
        vals = [key(v) for v in self.values]
        argmin = np.argmin(vals)
        argmax = np.argmax(vals)
        if return_time:
            return self.times[argmin], self.times[argmax]
        else:
            return vals[argmin], vals[argmax]

    def __iter__(self) -> Iterable[float]:
        return self.times

    def __getitem__(self, idx: Union[slice, int, float]) -> tuple[float, T]:
        if isinstance(idx, int):
            return self.get_idx(idx)
        elif isinstance(idx, float):
            return self.get_t(idx)
        elif isinstance(idx, slice):
            if any(isinstance(i, float) for i in (idx.start, idx.stop, idx.step)):
                return self.slice_time(idx.start, idx.stop, idx.step)
            else:
                return self.slice_idx(idx.start, idx.stop, idx.step)
        else:
            raise TypeError(f"Invalid index type {type(idx)}")

    def __contains__(self, t: float) -> bool:
        return t in self._value_dict

    def __len__(self) -> int:
        return len(self.times)

    def __bool__(self) -> bool:
        return bool(self.times)
