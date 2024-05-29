from dataclasses import fields, dataclass, Field, field
import numpy as np
from typing import Any, ClassVar, Generic, Union, Sequence, TypeVar, Type
from itertools import zip_longest
import re
from itertools import chain

T = TypeVar('T', bound='NamedArray')


class Foo:
    def __class_getitem__(cls, indices, hello=None):
        return indices, hello


class AtIndex(tuple):
    _type = None
    _min_shape = tuple()

    def __class_getitem__(cls, indices):
        return cls(indices)

    def __new__(cls, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        _min_shape = []
        for idx in indices:
            if idx is None:
                _min_shape.append(1)
            elif isinstance(idx, slice):
                def f(x): return 0 if x is None else (abs(x)+int(x < 0))
                _min_shape.append(max(map(f, (idx.start, idx.stop))))
            elif isinstance(idx, list):
                _min_shape.append(max(idx)+1)
            elif isinstance(idx, int):
                _min_shape.append(idx+1)
        obj = super().__new__(cls, indices)
        obj._min_shape = tuple(_min_shape)
        return obj

    def __getattr__(self, att: str):
        try:
            val = next(f.type for f in fields(self._type) if f.name == att)
            return AtIndex(np.r_[self][val])
        except Exception:
            raise AttributeError(f"Type {self._type} has no attribute {att}")

    def __call__(cls, *args, **kwargs):
        raise TypeError("AtIndex is not callable, use AtIndex[0, 2:3]")

    def __repr__(self):
        def mapper(item):
            if isinstance(item, slice):
                s = f'{item.start}:{item.stop}'
                s += f':{item.step}' if item.step is not None else ''
                return re.sub('None', '', s)
            else:
                return str(item)
        return f"AtIndex[{','.join(map(mapper, self))}]"

    def __or__(self, other: T) -> T:
        self._type = other
        return self

    def __ror__(self, other: T) -> T:
        self._type = other
        return self

    def __str__(self):
        return repr(self)

    def __hash__(self):
        """To work with Union"""
        return hash(str(self))


class MetaData(Generic[T]):
    ...


class NamedArray(np.ndarray):

    _cls_idx_view: ClassVar['IndexView']
    _cls_idx_field_dict: ClassVar[dict[str, AtIndex]]
    _cls_min_shape: ClassVar[tuple[int, ...]]
    _cls_meta_fields: ClassVar[set[Field]]
    _cls_meta_names: ClassVar[set[str]]

    _cls_initialized: ClassVar[set[Type[T]]] = set()

    def __new__(cls: Type[T], *args, **kwargs):
        if cls not in NamedArray._cls_initialized:
            cls._cls_init()
        obj = np.zeros(cls._cls_min_shape).view(cls)
        return obj

    def with_new_data(self, data):
        out = np.asarray(data).view(self.__class__)
        for name in self._cls_meta_names:
            setattr(out, name, getattr(self, name))
        return out

    def with_new_meta(self, **meta):
        out = np.asarray(self).view(self.__class__)
        for name, val in meta.items():
            assert name in self._cls_meta_names, f"{name} is not a meta field"
            setattr(out, name, val)
        return out

    @ classmethod
    def from_array(cls, arr: np.ndarray, **kwargs):
        obj = np.asarray(arr).view(cls)
        return obj

    @property
    def indices(self: T) -> T:
        return self._cls_idx_view

    @classmethod
    def _cls_init(cls):
        cls._cls_idx_view = IndexView(cls)

        cls._cls_idx_field_dict = dict()
        cls._cls_meta_fields = set()
        cls._cls_meta_names = set()
        for f in fields(cls):
            if isinstance(f.type, AtIndex):
                cls._cls_idx_field_dict[f.name] = f.type
            elif getattr(f.type, '__origin__', None) is MetaData:
                cls._cls_meta_fields.add(f)
                cls._cls_meta_names.add(f.name)
            else:
                tname = f.type.__name__
                raise TypeError(f"Invalid type for field {f.name}, "
                                f"use MetaDate[{tname}] or AtIndex[<indices>]")

        itr = (i._min_shape for i in cls._cls_idx_field_dict.values())
        itzip = zip_longest(*itr, fillvalue=0)
        cls._cls_min_shape = tuple(max(i) for i in itzip)

        NamedArray._cls_initialized.add(cls)

    def __array_finalize__(self, obj):
        if self.__class__ not in NamedArray._cls_initialized:
            self.__class__._cls_init()

        if isinstance(obj, NamedArray):
            for name in self._cls_meta_names:
                setattr(self, name, getattr(obj, name))

    @staticmethod
    def get_meta_dict(obj: T) -> dict[str, Any]:
        out = {}
        for fld in obj._cls_meta_fields:
            if hasattr(obj, fld.name):
                out[fld.name] = getattr(obj, fld.name)
            elif fld.default is not None:
                out[fld.name] = fld.name, fld.default
            elif fld.default_factory is not None:
                out[fld.name] = fld.name, fld.default_factory()
        return out

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        args = [np.asarray(arg) if isinstance(arg, NamedArray) else arg
                for arg in args]
        if kwargs.get('out', None) is not None:
            kwargs['out'] = tuple(np.asarray(a) if isinstance(a, NamedArray)
                                  else a for a in kwargs['out'])

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if ufunc.nout == 1 and np.shape(results) == self.shape:
            results = self.with_new_data(results)
        return results

    def __getitem__(self, indices):
        out = np.asarray(self)[indices]
        if isinstance(out, np.ndarray) and out.shape == self.shape:
            out = self.with_new_data(out)
        return out

    def __getattr__(self, att):
        if idx := self._cls_idx_field_dict.get(att, None):
            value = self[idx]
            if callable(idx._type):
                value = idx._type.from_array(value)
            return value
        if att in self._cls_meta_names:
            return getattr(self, att)
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute {att}")

    def __setattr__(self, att, value):
        if idx := self._cls_idx_field_dict.get(att, None):
            self[idx] = value
        elif att in self._cls_meta_names:
            super().__setattr__(att, value)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {att}")

    def __str__(self):
        return repr(self)

    def __reduce__(self):
        arr_state = super(NamedArray, self).__reduce__()
        extra_states = tuple((att, getattr(self, att))
                             for att in self._cls_meta_names)
        return (*arr_state[:2], (arr_state[2], extra_states))

    def __setstate__(self, state):
        np.ndarray.__setstate__(self, state[0])
        for att, val in state[1]:
            setattr(self, att, val)


class IndexView:
    _parent: NamedArray

    def __init__(self, parent: Type[NamedArray]):
        super().__setattr__('_parent', parent)

    def __getattr__(self, att: str) -> AtIndex:
        if att := self._parent._cls_idx_field_dict.get(att, None):
            return att
        else:
            raise AttributeError

    def __setattr__(self, att: str, value: AtIndex):
        raise AttributeError("IndexView is read-only")


if __name__ == '__main__':
    @dataclass
    class Foo(NamedArray):
        x: AtIndex[0]
        y: AtIndex[1]
    a = Foo(1, 2)
    a += 1
