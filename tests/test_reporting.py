"""Check sample statistics and generation of standalone PNG figures."""

import csv
import json
from math import sqrt
import struct

import pytest

from gp.reporting import plot_runs, summarize_runs


def write_run(root, name, *, seed, configuration_id="config-a", stage="selection", dataset="synth1", offset=0.0):
    directory = root / name
    directory.mkdir(parents=True)
    summary = {
        "dataset": dataset,
        "stage": stage,
        "configuration_id": configuration_id,
        "seed": seed,
        "train_rmse": 1.0 + offset,
        "test_rmse": 2.0 + offset,
        "duration_seconds": 3.0 + offset,
        "best_size": 4.0 + offset,
        "best_depth": 2.0 + offset,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (directory / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["generation", "best_rmse", "mean_size"])
        writer.writeheader()
        writer.writerows([
            {"generation": 0, "best_rmse": 3.0 + offset, "mean_size": 3.0 + offset},
            {"generation": 1, "best_rmse": 1.0 + offset, "mean_size": 4.0 + offset},
        ])
    return directory


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_summary_uses_sample_standard_deviation_for_two_repetitions(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "seed0", seed=0)
    write_run(runs, "seed1", seed=1, offset=2.0)
    # An interrupted run without summary.json must not enter the aggregate.
    (runs / "unfinished").mkdir()
    output = summarize_runs(runs, tmp_path / "summaries")
    assert output.name == "summary.csv"
    [row] = read_csv(output)
    assert row["dataset"] == "synth1"
    assert row["stage"] == "selection"
    assert row["configuration_id"] == "config-a"
    assert row["repetitions"] == "2"
    assert row["seeds"] == "0|1"
    expected_means = {"train_rmse": 2.0, "test_rmse": 3.0, "duration_seconds": 4.0, "best_size": 5.0, "best_depth": 3.0}
    for metric, expected in expected_means.items():
        assert float(row[f"{metric}_mean"]) == pytest.approx(expected)
        assert float(row[f"{metric}_std"]) == pytest.approx(sqrt(2.0))


def test_summary_keeps_groups_separate_and_single_repetition_deviation_empty(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "selection-a", seed=0)
    write_run(runs, "selection-b", seed=0, configuration_id="config-b")
    write_run(runs, "different-stage", seed=0, stage="elitism")
    write_run(runs, "different-dataset", seed=0, dataset="concrete")
    rows = read_csv(summarize_runs(runs, tmp_path / "summaries"))
    assert len(rows) == 4
    assert len({(row["dataset"], row["stage"], row["configuration_id"]) for row in rows}) == 4
    for row in rows:
        assert row["repetitions"] == "1"
        for metric in ("train_rmse", "test_rmse", "duration_seconds", "best_size", "best_depth"):
            assert row[f"{metric}_std"] == ""


def test_summary_rejects_duplicate_seed_within_configuration(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "first", seed=8)
    write_run(runs, "second", seed=8, offset=1.0)
    with pytest.raises(ValueError, match="Repeated seeds"):
        summarize_runs(runs, tmp_path / "summaries")


@pytest.mark.parametrize("operation", [summarize_runs, plot_runs])
def test_empty_runs_directory_reports_no_complete_execution(tmp_path, operation):
    with pytest.raises(ValueError, match="No complete runs"):
        operation(tmp_path / "missing", tmp_path / "output")


def test_plot_runs_writes_readable_png_for_each_dataset_and_stage(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "seed0", seed=0)
    write_run(runs, "seed1", seed=1, offset=2.0)
    write_run(runs, "alternative", seed=0, configuration_id="config-b", offset=1.0)
    write_run(runs, "transfer", seed=0, stage="transfer", dataset="synth2")
    outputs = plot_runs(runs, tmp_path / "figures")
    assert {path.name for path in outputs} == {"synth1-selection.png", "synth2-transfer.png"}
    for path in outputs:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", data[16:24])
        assert width >= 1000
        assert height >= 400
        assert len(data) > 10000
