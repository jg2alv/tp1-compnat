"""Loading and validation of the numeric regression datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Dataset:
    """A fixed train/test split; each target is the original CSV's last column."""

    name: str
    X_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    X_test: NDArray[np.float64]
    y_test: NDArray[np.float64]

    def __post_init__(self) -> None:
        for split in ("train", "test"):
            X = np.asarray(getattr(self, f"X_{split}"), dtype=float)
            y = np.asarray(getattr(self, f"y_{split}"), dtype=float)
            if X.ndim != 2 or y.ndim != 1:
                raise ValueError(f"{split}: X must be a matrix and y must be a vector")
            if X.shape[0] == 0 or X.shape[1] == 0:
                raise ValueError(f"{split}: data must contain samples and at least one feature")
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"{split}: X and y must have the same number of samples")
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError(f"{split}: all values must be finite")
            # np.asarray may create new float arrays; store the normalized values.
            # The frozen dataclass requires object.__setattr__ during initialization.
            object.__setattr__(self, f"X_{split}", X)
            object.__setattr__(self, f"y_{split}", y)
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("train and test must have the same number of features")

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def _load_split(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
            values = np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)
    except ValueError as error:
        raise ValueError(f"{path}: expected a numeric CSV without a header") from error
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(f"{path}: expected samples, at least one feature, and a target column")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: all values must be finite")
    return values[:, :-1], values[:, -1]


def load_dataset(
    name: str,
    data_dir: str | Path = Path("bases-TP1/datasets"),
) -> Dataset:
    """Load ``synth1``, ``synth2``, or ``concrete`` (alias ``concrete1``)."""
    aliases = {"synth1": "synth1", "synth2": "synth2", "concrete": "concrete", "concrete1": "concrete"}
    if name not in aliases:
        raise ValueError(f"unknown dataset {name!r}; expected synth1, synth2, or concrete")
    canonical = aliases[name]
    directory = Path(data_dir) / canonical
    X_train, y_train = _load_split(directory / f"{canonical}-train.csv")
    X_test, y_test = _load_split(directory / f"{canonical}-test.csv")
    return Dataset(canonical, X_train, y_train, X_test, y_test)
