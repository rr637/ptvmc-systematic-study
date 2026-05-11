from typing import Any, Callable
from netket.utils.history import History
import numpy as np
from netket.utils.types import Array, DType

from numbers import Number

from netket.utils.numbers import is_scalar
from netket.utils.history.accum import (
    AccumulatorFunTypeRegistry,
    init_history_from_data,
)

try:
    from netket.utils.history.history_dict import (
        register_historydict_deserialization_fun,
    )
    from netket.utils.history.history import replace_none_with_nan
except ImportError:

    def register_historydict_deserialization_fun(*args, **kwargs):
        pass

    def replace_none_with_nan(item):
        return item


from netket_pro._src.history.history_2d import History2D, compute_inner_length


def _safe_list(val):
    if not hasattr(val, "__len__"):
        return [val]
    return list(val)


class History3D:
    def __init__(
        self,
        values: History2D | list[History2D],
        iters: list | None = None,
        iter_dtype: DType | None = None,
    ):
        if iters is None:
            iters = [0]

        if is_scalar(iters):
            iters = np.array([iters], dtype=iter_dtype)
        elif isinstance(iters, list):
            iters = np.array(iters, dtype=iter_dtype)

        other_values = ((), ())
        if not isinstance(values, History2D):
            if isinstance(values, (list, tuple)):
                if not len(values) == len(iters):
                    raise ValueError("Length Mismatch between values and iters")
                other_values = (iters[1:], values[1:])
                values = values[0]
                iters = iters[:1]
            else:
                raise TypeError(
                    "values should be a History object or a list of History objects"
                )

        value_dict = values.to_dict().copy()
        for k, v in value_dict.items():
            value_dict[k] = np.expand_dims(np.array(v), 0)

        self._iters_2d = value_dict.pop("axis")

        self._value_dict = value_dict
        self._value_name = values._value_name
        self._single_value = values._single_value
        self._iters = iters
        self._lengths_inner = np.array([values._lengths_inner])

        for i, v in zip(*other_values):
            self.append(v, it=i)

    @classmethod
    def from_serialized_data(
        cls,
        value_dict: dict,
        iters: np.ndarray = None,
        iters_2d: np.ndarray = None,
        *,
        value_name: str | None = None,
        lengths_inner: list[int] | None = None,
    ):
        self = cls.__new__(cls)
        value_dict = value_dict.copy()

        if "axis0" in value_dict:
            _val = value_dict.pop("axis0")
            if iters is None:
                iters = _val
        elif iters is None:
            raise IOError(
                "You are loading some older version of a checkpoint. We can fix it but let me know"
            )

        if "axis1" in value_dict:
            _val = value_dict.pop("axis1")
            if iters_2d is None:
                iters_2d = _val
        elif iters_2d is None:
            raise IOError(
                "You are loading some older version of a checkpoint. We can fix it but let me know"
            )

        for k, v in value_dict.items():
            if isinstance(v, list):
                value_dict[k] = np.array(replace_none_with_nan(v))

        self._value_dict = value_dict
        self._iters = np.array(iters)
        self._iters_2d = np.array(iters_2d)

        if lengths_inner is None:
            lengths_inner = compute_inner_length(value_dict["iters"], axis=2)
        self._lengths_inner = np.array(lengths_inner, dtype=np.int64)

        if len(set(self.keys()) - {"axis0", "axis1", "iters"}) == 1:
            self._single_value = True
            if value_name is None:
                value_name = list(set(self.keys()) - {"axis0", "axis1", "iters"})[0]
        else:
            self._single_value = False

        self._value_name = value_name
        return self

    @property
    def iters(self) -> Array:
        return self._iters

    @property
    def iters_inner(self) -> Array:
        return self._iters_2d

    @property
    def iters_inner_inner(self) -> Array:
        return self._value_dict["iters"]

    @property
    def values(self) -> Array:
        return self._value_dict[self._value_name]

    @property
    def main_value_name(self):
        return self._value_name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.iters_inner_inner.shape[:3]

    def to_dict(self) -> dict:
        result = self._value_dict.copy()
        result["axis0"] = self._iters
        result["axis1"] = self._iters_2d
        return result

    def __len__(self) -> int:
        return len(self.iters)

    def __getattr__(self, attr):
        if attr in self._value_dict:
            return self._value_dict[attr]
        raise AttributeError

    def __iter__(self):
        return ((it, self[i]) for i, it in enumerate(self.iters))

    def __getitem__(self, key) -> Array:
        if isinstance(key, str):
            if key == "iters":
                return self.iters
            else:
                return self._value_dict[key]

        if not isinstance(key, tuple):
            key = (key,)

        if len(key) == 1 and isinstance(key[0], int):
            return self._get_single(key[0])
        elif len(key) <= 0:
            raise ValueError("Must have at least 1 index.")
        elif len(key) <= 3:
            if isinstance(key[0], int):
                return self._get_single(key[0])[key[1:]]
            else:
                return self._get_slice(key)
        else:
            raise ValueError("Can index at most 3 dimensions in a History3D object.")

    def _get_slice(self, slce: slice) -> History:
        hist = History3D.__new__(History3D)

        if len(slce) == 1:
            hist._lengths_inner = self._lengths_inner[slce[0]]
        else:
            hist._lengths_inner = self._lengths_inner[slce[0], slce[1]]

        max_len = self.shape[1]
        max_inner_len = np.max(hist._lengths_inner)

        if len(slce) == 1:
            slce = (slce[0], slice(max_len), slice(max_inner_len))
        elif len(slce) == 2:
            slce1, slce2 = slce
            if isinstance(slce2, slice):
                if slce2.stop is None or slce2.stop > max_len:
                    slce2 = slice(slce2.start, max_len, slce2.step)
            else:
                slce2 = slice(slce2, slce2 + 1)
            slce = (slce1, slce2)
        elif len(slce) == 3:
            slce1, slce2, slce3 = slce
            if isinstance(slce2, slice):
                if slce2.stop is None or slce2.stop > max_len:
                    slce2 = slice(slce2.start, max_len, slce2.step)
            else:
                slce2 = slice(slce2, slce2 + 1)

            if isinstance(slce3, slice):
                if slce3.stop is None or slce3.stop > max_inner_len:
                    slce3 = slice(slce3.start, max_inner_len, slce3.step)
            else:
                slce3 = slice(slce3, slce3 + 1)

            slce = (slce1, slce2, slce3)
        else:
            raise ValueError(f"wrong length {len(slce)}")

        values_sliced = {}
        for key in self._value_dict.keys():
            values_sliced[key] = self._value_dict[key][*slce]

        hist._value_dict = values_sliced
        hist._value_name = self._value_name
        hist._single_value = self._single_value
        hist._iters = self.iters[slce[0]]
        hist._iters_2d = self._iters_2d[slce[0], slce[1]]
        return hist

    def _get_single(self, i: slice) -> History2D:
        values_sliced = {}
        lengths_hist2d = self._lengths_inner[i]
        real_len = np.max(lengths_hist2d)

        real_len_2d = len(lengths_hist2d)

        for key in self.keys():
            values_sliced[key] = self._value_dict[key][i, :real_len_2d, :real_len]

        hist = History2D.__new__(History2D)
        hist._value_dict = values_sliced
        hist._value_name = self._value_name
        hist._single_value = self._single_value
        hist._iters = self._iters_2d[i]
        hist._lengths_inner = lengths_hist2d.tolist()
        return hist

    def __contains__(self, key: str) -> bool:
        return key in self._value_dict

    def keys(self) -> list:
        return list(self._value_dict.keys())

    def _pad_array(self, arr, target_shape):
        pad_shape = []
        for old, new in zip(arr.shape, target_shape):
            pad_shape.append((0, max(0, new - old)))

        if np.issubdtype(arr.dtype, np.floating):
            return np.pad(arr, pad_shape, mode="constant", constant_values=np.nan)
        else:
            return np.pad(arr, pad_shape, mode="edge")

    def append(
        self, val: History | dict, it: Number | None = None, it1: Number | None = None
    ):
        """
        Append another History2D object to this History3D object.

        This version supports variable second and third dimensions across
        appended History2D objects by padding ragged entries.
        """
        if it is None:
            it = self.iters[-1] + 1
        it0 = it

        if not isinstance(val, History2D):
            raise TypeError()

        old_outer_len = self.shape[1]
        old_inner_len = self.shape[2]

        new_outer_len = len(val)
        new_inner_len = int(np.max(val._lengths_inner))

        max_outer_len = max(old_outer_len, new_outer_len)
        max_inner_len = max(old_inner_len, new_inner_len)

        # Pad stored inner lengths if number of inner histories changed.
        if self._lengths_inner.shape[1] < max_outer_len:
            self._lengths_inner = np.pad(
                self._lengths_inner,
                ((0, 0), (0, max_outer_len - self._lengths_inner.shape[1])),
                mode="constant",
                constant_values=0,
            )

        new_lengths = np.asarray(val._lengths_inner, dtype=self._lengths_inner.dtype)
        if new_lengths.shape[0] < max_outer_len:
            new_lengths = np.pad(
                new_lengths,
                (0, max_outer_len - new_lengths.shape[0]),
                mode="constant",
                constant_values=0,
            )

        self._lengths_inner = np.concatenate(
            [self._lengths_inner, new_lengths[None, :]],
            axis=0,
        )

        # Pad stored second-level iteration axis if number of inner histories changed.
        if self._iters_2d.shape[1] < max_outer_len:
            self._iters_2d = np.pad(
                self._iters_2d,
                ((0, 0), (0, max_outer_len - self._iters_2d.shape[1])),
                mode="edge",
            )

        new_iters = np.asarray(val.iters, dtype=self._iters_2d.dtype)
        if new_iters.shape[0] < max_outer_len:
            new_iters = np.pad(
                new_iters,
                (0, max_outer_len - new_iters.shape[0]),
                mode="edge",
            )

        self._iters_2d = np.concatenate(
            [self._iters_2d, new_iters[None, :]],
            axis=0,
        )

        for key in self.keys():
            old_vals = self._value_dict[key]
            new_val = np.asarray(val._value_dict[key])

            target_shape = (
                old_vals.shape[0] + 1,
                max_outer_len,
                max_inner_len,
            ) + old_vals.shape[3:]

            padded_vals = self._pad_array(old_vals, target_shape)

            # If _pad_array padded axis 0, the last entry may be edge-copied.
            # Replace it with a clean placeholder before inserting new data.
            if np.issubdtype(padded_vals.dtype, np.floating):
                padded_vals[-1, ...] = np.nan
            else:
                padded_vals[-1, ...] = 0

            shp = new_val.shape

            padded_vals[
                -1,
                : shp[0],
                : shp[1],
                ...
            ] = new_val

            self._value_dict[key] = padded_vals

        try:
            self.iters.resize(len(self.iters) + 1)
        except ValueError:
            self._iters = np.resize(self.iters, (len(self.iters) + 1))

        self.iters[-1] = it0

    def __repr__(self):
        if len(self.iters) < 5:
            iters_repr = repr(self.iters)
        else:
            iters_repr = (
                f"[{self.iters[0]}, {self.iters[1]}, ..."
                f" {self.iters[-2]}, {self.iters[-1]}] "
                f"({len(self.iters)} steps)"
            )

        keys = list(set(self.keys()) - {"iters", "axis0", "axis1"})
        return (
            "History3D("
            + f"\n   keys  = {keys}, "
            + f"\n   shape  = {self.shape}, "
            + f"\n   iters = {iters_repr},"
            + "\n)"
        )

    def __str__(self):
        keys = list(set(self.keys()) - {"axis0", "axis1", "iters"})
        return f"History3d(keys={keys}, shape  = {self.shape},)"


def _accum_histories2d_history3d(
    fun: Callable[[Any, Any], Any],
    tree_accum: History,
    tree: History,
    **kwargs,
):
    return fun(tree_accum, tree, **kwargs)


AccumulatorFunTypeRegistry[History2D] = _accum_histories2d_history3d


@init_history_from_data.register
def init_history_from_data_history(val: History2D, step: Any):
    return History3D(val, step)


def is_history3d(hist_dict):
    return "axis0" in hist_dict and "axis1" in hist_dict


def reconstruct_history3d(hist_dict):
    axis0 = hist_dict.pop("axis0")
    axis1 = hist_dict.pop("axis1")
    value_name = "Mean" if "Mean" in hist_dict else None

    return History3D.from_serialized_data(
        hist_dict,
        axis0,
        axis1,
        value_name=value_name,
    )


register_historydict_deserialization_fun(
    is_history3d,
    reconstruct_history3d,
    precedence=10,
)