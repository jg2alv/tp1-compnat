"""Validate the supplied splits and reject malformed regression data."""

from pathlib import Path

import numpy as np
import pytest

from gp.data import Dataset, load_dataset


DATA_DIR = Path(__file__).resolve().parents[1] / "bases-TP1" / "datasets"


@pytest.mark.parametrize(
    ("name", "train_rows", "test_rows", "features"),
    [("synth1", 60, 600, 2), ("synth2", 300, 1000, 2), ("concrete", 824, 206, 8)],
)
def test_supplied_dataset_shapes_and_targets(name, train_rows, test_rows, features):
    data = load_dataset(name, DATA_DIR)
    assert data.X_train.shape == (train_rows, features)
    assert data.y_train.shape == (train_rows,)
    assert data.X_test.shape == (test_rows, features)
    assert data.y_test.shape == (test_rows,)
    assert data.n_features == features
    original = np.loadtxt(DATA_DIR / name / f"{name}-train.csv", delimiter=",")
    np.testing.assert_array_equal(data.X_train, original[:, :-1])
    np.testing.assert_array_equal(data.y_train, original[:, -1])


def test_concrete1_alias_uses_canonical_files():
    data = load_dataset("concrete1", DATA_DIR)
    assert data.name == "concrete"
    assert data.n_features == 8


def _write_dataset(tmp_path, train, test):
    directory = tmp_path / "synth1"
    directory.mkdir()
    (directory / "synth1-train.csv").write_text(train)
    (directory / "synth1-test.csv").write_text(test)


def test_single_row_split_is_kept_as_matrix(tmp_path):
    _write_dataset(tmp_path, "1,2,3\n", "4,5,6\n")
    data = load_dataset("synth1", tmp_path)
    assert data.X_train.shape == (1, 2)
    np.testing.assert_array_equal(data.y_test, [6.0])


@pytest.mark.parametrize(
    "train",
    ["", "\n", "1\n2\n", "x,y,z\n1,2,3\n", "1,2,3\n4,5\n", "1,nan,3\n", "1,2,inf\n"],
)
def test_malformed_csv_is_rejected(tmp_path, train):
    _write_dataset(tmp_path, train, "1,2,3\n")
    with pytest.raises(ValueError):
        load_dataset("synth1", tmp_path)


def test_inconsistent_split_width_is_rejected(tmp_path):
    _write_dataset(tmp_path, "1,2,3\n", "1,2\n")
    with pytest.raises(ValueError, match="same number of features"):
        load_dataset("synth1", tmp_path)


def test_unknown_dataset_and_missing_files_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("../../outside", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_dataset("synth1", tmp_path)


def test_dataset_constructor_validates_sample_alignment():
    with pytest.raises(ValueError, match="same number of samples"):
        Dataset("invalid", np.ones((2, 1)), np.ones(1), np.ones((1, 1)), np.ones(1))
