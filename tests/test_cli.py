"""Exercise the public command entry point with temporary result directories."""

import csv
import json
from pathlib import Path

import pytest

from gp.__main__ import main


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "baseline.toml"
PLAN = ROOT / "configs" / "experiments.toml"
DATA_DIR = ROOT / "bases-TP1" / "datasets"


def run_arguments(output, seed=17):
    return [
        "run",
        "--dataset", "synth1",
        "--config", str(BASELINE),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(output),
        "--seed", str(seed),
        "--population-size", "8",
        "--generations", "2",
        "--max-depth", "2",
        "--min-initial-depth", "1",
        "--selection", "epsilon_lexicase",
    ]


def test_run_applies_overrides_and_persists_complete_results(tmp_path, capsys):
    destination = tmp_path / "runs"
    assert main(run_arguments(destination)) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed.parent == destination
    assert printed.is_dir()
    assert {path.name for path in printed.iterdir()} == {
        "config.json", "best_tree.json", "history.csv", "operators.csv", "summary.json",
    }
    config = json.loads((printed / "config.json").read_text())
    assert {name: config[name] for name in (
        "seed", "population_size", "generations", "max_depth", "min_initial_depth", "selection",
    )} == {
        "seed": 17,
        "population_size": 8,
        "generations": 2,
        "max_depth": 2,
        "min_initial_depth": 1,
        "selection": "epsilon_lexicase",
    }
    summary = json.loads((printed / "summary.json").read_text())
    assert summary["dataset"] == "synth1"
    assert summary["seed"] == 17
    assert summary["parameters"] == config
    assert summary["best_depth"] <= 2
    assert summary["n_train"] == 60
    assert summary["n_test"] == 600
    with (printed / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    assert [int(row["generation"]) for row in history] == [0, 1, 2]
    with (printed / "operators.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 2 * (8 - config["elite_size"])


@pytest.mark.parametrize(("seed_options", "count"), [([], 180), (["--seeds", "0"], 6)])
def test_experiment_dry_run_expands_plan_without_creating_output(tmp_path, capsys, seed_options, count):
    destination = tmp_path / "dry-runs"
    assert main([
        "experiment", "--plan", str(PLAN), "--stage", "population",
        "--output-dir", str(destination), "--dry-run", *seed_options,
    ]) == 0
    output = capsys.readouterr().out
    assert f"Stage population: {count} runs, 3 configurations" in output
    assert output.count("  synth1:") == 3
    assert output.count("  concrete:") == 3
    assert not destination.exists()


def test_experiment_dataset_filter_selects_only_concrete(tmp_path, capsys):
    destination = tmp_path / "filtered-runs"
    assert main([
        "experiment", "--plan", str(PLAN), "--stage", "population",
        "--datasets", "concrete", "--seeds", "0",
        "--output-dir", str(destination), "--dry-run",
    ]) == 0
    output = capsys.readouterr().out
    assert "Stage population: 3 runs, 3 configurations" in output
    assert output.count("  concrete:") == 3
    assert "  synth1:" not in output
    assert not destination.exists()


@pytest.mark.parametrize("case", ["dataset", "negative_generations", "unknown_stage", "missing_runs"])
def test_invalid_arguments_exit_with_status_two(tmp_path, capsys, case):
    if case == "dataset":
        arguments = ["run", "--dataset", "invalid"]
    elif case == "negative_generations":
        arguments = run_arguments(tmp_path / "runs") + ["--generations", "-1"]
    elif case == "unknown_stage":
        arguments = ["experiment", "--plan", str(PLAN), "--stage", "missing", "--dry-run"]
    else:
        arguments = ["analyze", "--runs-dir", str(tmp_path / "missing")]
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert capsys.readouterr().err


def test_analyze_creates_summary_and_png_from_two_seed_runs(tmp_path, capsys):
    runs = tmp_path / "runs"
    for seed in (0, 1):
        assert main(run_arguments(runs, seed=seed)) == 0
    capsys.readouterr()
    summaries = tmp_path / "summaries"
    figures = tmp_path / "figures"
    assert main([
        "analyze", "--runs-dir", str(runs),
        "--summaries-dir", str(summaries), "--figures-dir", str(figures),
    ]) == 0
    output = capsys.readouterr().out
    summary_path = summaries / "summary.csv"
    assert str(summary_path) in output
    with summary_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["dataset"] == "synth1"
    assert rows[0]["repetitions"] == "2"
    assert rows[0]["seeds"] == "0|1"
    assert float(rows[0]["train_rmse_mean"]) >= 0
    assert float(rows[0]["test_rmse_std"]) >= 0
    images = list(figures.glob("*.png"))
    assert len(images) == 1
    assert images[0].name == "synth1-single.png"
    assert images[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert images[0].stat().st_size > 1000
    assert str(images[0]) in output
