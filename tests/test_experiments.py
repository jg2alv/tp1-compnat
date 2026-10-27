"""Persistence and experiment planning on small, reproducible problems."""

from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gp.config import GPConfig
from gp.data import Dataset
from gp.experiments import configuration_id, plan_runs, run_once
from gp.fitness import evaluate_tree, rmse
from gp.tree import Node


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "configs" / "experiments.toml"


@pytest.fixture
def tiny_problem():
    X_train = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    X_test = np.array([[-0.5], [0.5], [1.5]])
    dataset = Dataset("tiny", X_train, X_train[:, 0] * 2, X_test, X_test[:, 0] * 2)
    config = GPConfig(
        population_size=12,
        generations=2,
        min_initial_depth=1,
        max_depth=3,
        pc=0.6,
        pm=0.3,
        seed=19,
    )
    return dataset, config


def test_run_once_persists_reconstructible_champion_and_training_history(tmp_path, tiny_problem):
    dataset, config = tiny_problem
    directory = run_once(dataset, config, tmp_path, stage="smoke")
    assert directory.parent == tmp_path
    assert {path.name for path in directory.iterdir()} == {
        "config.json", "best_tree.json", "history.csv", "operators.csv", "summary.json"
    }
    saved_config = json.loads((directory / "config.json").read_text())
    assert GPConfig.from_dict(saved_config) == config
    tree = Node.from_dict(json.loads((directory / "best_tree.json").read_text()))
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["train_rmse"] == pytest.approx(rmse(dataset.y_train, evaluate_tree(tree, dataset.X_train)))
    assert summary["test_rmse"] == pytest.approx(rmse(dataset.y_test, evaluate_tree(tree, dataset.X_test)))
    assert summary["best_expression"] == tree.expression()
    assert summary["best_depth"] == tree.depth <= config.max_depth
    assert summary["best_size"] == tree.size
    assert summary["configuration_id"] == configuration_id(config)
    assert summary["dataset"] == "tiny"
    assert summary["stage"] == "smoke"
    assert summary["seed"] == config.seed
    assert (summary["n_train"], summary["n_test"], summary["n_features"]) == (5, 3, 1)
    assert summary["duration_seconds"] >= 0
    assert summary["initialization"] == "ramped_half_and_half"
    with (directory / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    assert [int(row["generation"]) for row in history] == [0, 1, 2]
    assert min(float(row["best_rmse"]) for row in history) == pytest.approx(summary["train_rmse"])
    assert all(float(row["max_depth"]) <= config.max_depth for row in history)
    assert {"duplicates", "mutation_better", "mutation_equal", "mutation_worse", "mean_size", "max_size"} <= history[0].keys()
    with (directory / "operators.csv").open(newline="") as stream:
        operators = list(csv.DictReader(stream))
    assert len(operators) == config.generations * (config.population_size - config.elite_size)


def test_test_targets_do_not_affect_evolution_and_same_seed_runs_have_unique_paths(tmp_path, tiny_problem):
    dataset, config = tiny_problem
    changed = replace(dataset, y_test=dataset.y_test + 1000.0)
    first = run_once(dataset, config, tmp_path)
    second = run_once(changed, config, tmp_path)
    assert first != second
    assert (first / "best_tree.json").read_bytes() == (second / "best_tree.json").read_bytes()
    assert (first / "history.csv").read_bytes() == (second / "history.csv").read_bytes()
    summaries = [json.loads((path / "summary.json").read_text()) for path in (first, second)]
    assert summaries[0]["train_rmse"] == summaries[1]["train_rmse"]
    assert summaries[0]["test_rmse"] != summaries[1]["test_rmse"]


def test_configuration_identifier_groups_seeds_but_preserves_parameters():
    baseline = GPConfig(seed=0)
    assert configuration_id(baseline) == configuration_id(replace(baseline, seed=29))
    assert configuration_id(baseline) != configuration_id(replace(baseline, population_size=101))
    assert configuration_id(baseline) != configuration_id(replace(baseline, pm=0.1))


def test_shipped_population_plan_expands_thirty_seeds_per_dataset_and_variant():
    specs = plan_runs(PLAN_PATH, "population")
    assert len(specs) == 2 * 3 * 30
    assert {spec.dataset for spec in specs} == {"synth1", "concrete"}
    assert {spec.config.population_size for spec in specs} == {50, 100, 500}
    for dataset in ("synth1", "concrete"):
        for size in (50, 100, 500):
            repetitions = [spec for spec in specs if spec.dataset == dataset and spec.config.population_size == size]
            assert {spec.config.seed for spec in repetitions} == set(range(30))
            assert len({spec.configuration_id for spec in repetitions}) == 1
            assert {spec.stage for spec in repetitions} == {"population"}


def test_plan_fixed_parameters_explicit_pairs_and_transfer_baseline():
    mutation = plan_runs(PLAN_PATH, "mutation", seeds=[7, 11])
    assert len(mutation) == 2 * 2 * 2
    assert {spec.config.pc for spec in mutation} == {0.6}
    assert {spec.config.pm for spec in mutation} == {0.05, 0.3}
    assert {spec.config.seed for spec in mutation} == {7, 11}
    pairs = plan_runs(PLAN_PATH, "operators", seeds=[3])
    assert {(spec.config.pc, spec.config.pm) for spec in pairs} == {(0.9, 0.05), (0.6, 0.3)}
    transfer = plan_runs(PLAN_PATH, "transfer")
    assert len(transfer) == 30
    assert {spec.dataset for spec in transfer} == {"synth2"}
    assert len({spec.configuration_id for spec in transfer}) == 1


def test_minimal_plan_defaults_to_thirty_seeds_and_accepts_baseline_override(tmp_path):
    baseline = tmp_path / "selected.toml"
    baseline.write_text("[gp]\npopulation_size = 22\nselection = 'roulette'\n", encoding="utf-8")
    plan = tmp_path / "plan.toml"
    plan.write_text(
        "[experiment]\ndatasets = ['synth2']\n[[stages]]\nname = 'transfer'\n",
        encoding="utf-8",
    )
    specs = plan_runs(plan, "transfer", baseline_path=baseline)
    assert len(specs) == 30
    assert {spec.config.seed for spec in specs} == set(range(30))
    assert all(spec.config.population_size == 22 and spec.config.selection == "roulette" for spec in specs)


@pytest.mark.parametrize("seeds", [[], [1, 1]])
def test_plan_rejects_empty_or_duplicate_seeds(seeds):
    with pytest.raises(ValueError):
        plan_runs(PLAN_PATH, "population", seeds=seeds)


def test_plan_rejects_unknown_stage():
    with pytest.raises(ValueError):
        plan_runs(PLAN_PATH, "missing")
