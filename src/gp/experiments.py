"""Reproducible repetitions, TOML plans, and run persistence."""

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import tomllib
import uuid

import numpy as np

from .config import GPConfig, load_config
from .data import Dataset, load_dataset
from .evolution import evolve
from .fitness import evaluate_tree, rmse


OPERATOR_FIELDS = [
    "generation", "event", "operator", "child_index", "parent_indices",
    "parent_mean_rmse", "child_rmse", "improvement", "outcome", "rejected",
]


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    config: GPConfig
    stage: str
    configuration_id: str


def configuration_id(config: GPConfig) -> str:
    """Group repetitions of the same configuration regardless of the seed."""
    values = config.to_dict()
    values.pop("seed")
    serialized = json.dumps(values, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()[:12]


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields if fields is not None else list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_once(
    dataset: Dataset,
    config: GPConfig,
    output_dir: str | Path = "results/runs",
    stage: str = "single",
) -> Path:
    """Evolve on training data only; evaluate the best tree on test data afterward."""
    start = perf_counter()
    result = evolve(dataset.X_train, dataset.y_train, config)
    test_rmse = rmse(dataset.y_test, evaluate_tree(result.best_tree, dataset.X_test))
    elapsed = perf_counter() - start
    identifier = configuration_id(config)
    directory = Path(output_dir) / f"{dataset.name}-{identifier}-seed{config.seed}-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "config.json", config.to_dict())
    _write_json(directory / "best_tree.json", result.best_tree.to_dict())
    write_csv(directory / "history.csv", result.history)
    write_csv(directory / "operators.csv", result.operator_records, OPERATOR_FIELDS)
    # Written last: only directories containing summary.json are complete runs.
    _write_json(directory / "summary.json", {
        "schema_version": 1, "dataset": dataset.name, "stage": stage,
        "parameters": config.to_dict(),
        "configuration_id": identifier, "seed": int(config.seed),
        "train_rmse": result.train_rmse, "test_rmse": test_rmse,
        "duration_seconds": elapsed, "best_expression": result.best_tree.expression(),
        "best_size": result.best_tree.size, "best_depth": result.best_tree.depth,
        "n_features": dataset.n_features, "n_train": len(dataset.y_train), "n_test": len(dataset.y_test),
        "initialization": "ramped_half_and_half", "depth_convention": "root=0; depth=edges",
        "case_error": "absolute", "epsilon_comparison": "<= best + candidate MAD",
        "operator_probabilities": "exclusive events; reproduction=1-pc-pm",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
    })
    return directory


def plan_runs(
    plan_path: str | Path,
    stage_name: str,
    seeds: list[int] | None = None,
    baseline_path: str | Path | None = None,
    datasets: list[str] | None = None,
) -> list[RunSpec]:
    """Expand one stage without searching automatically using test data."""
    path = Path(plan_path)
    with path.open("rb") as stream:
        plan = tomllib.load(stream)
    experiment = plan.get("experiment", {})
    stages = [stage for stage in plan.get("stages", []) if stage.get("name") == stage_name]
    if len(stages) != 1:
        raise ValueError(f"Missing or duplicate stage: {stage_name!r}")
    stage = stages[0]
    baseline = load_config(baseline_path or path.parent / experiment.get("baseline", "baseline.toml"))
    datasets = stage.get("datasets", experiment.get("datasets", [])) if datasets is None else datasets
    selected_seeds = experiment.get("seeds", list(range(30))) if seeds is None else seeds
    if not datasets or not selected_seeds:
        raise ValueError("The plan requires at least one dataset and one seed")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Seeds within a stage must not be repeated")
    if any(name not in {"synth1", "synth2", "concrete", "concrete1"} for name in datasets):
        raise ValueError("Unknown dataset in the plan")
    if "parameter" in stage:
        variants = [{stage["parameter"]: value} for value in stage.get("values", [])]
    else:
        variants = stage.get("configurations", [{}])
    if not variants:
        raise ValueError("The stage requires at least one configuration")
    runs = []
    for dataset in datasets:
        for variant in variants:
            values = {**baseline.to_dict(), **stage.get("fixed", {}), **variant}
            for seed in selected_seeds:
                config = GPConfig.from_dict({**values, "seed": seed})
                runs.append(RunSpec(dataset, config, stage_name, configuration_id(config)))
    return runs


def run_plan(
    specs: list[RunSpec],
    data_dir: str | Path = "bases-TP1/datasets",
    output_dir: str | Path = "results/runs",
) -> list[Path]:
    datasets: dict[str, Dataset] = {}
    directories = []
    for index, spec in enumerate(specs, start=1):
        if spec.dataset not in datasets:
            datasets[spec.dataset] = load_dataset(spec.dataset, data_dir)
        directory = run_once(datasets[spec.dataset], spec.config, output_dir, spec.stage)
        directories.append(directory)
        print(f"[{index}/{len(specs)}] {directory}", flush=True)
    return directories

