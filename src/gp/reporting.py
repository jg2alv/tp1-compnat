"""Result tables and figures"""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .experiments import write_csv


def _read_runs(runs_dir: str | Path) -> list[tuple[Path, dict]]:
    runs = [(path.parent, json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(Path(runs_dir).glob("*/summary.json"))]
    if not runs:
        raise ValueError(f"No complete runs found in {runs_dir}")
    return runs


def summarize_runs(runs_dir: str | Path, output_dir: str | Path = "results/summaries") -> Path:
    """Mean and sample standard deviation across runs (ddof=1).

    With a single repetition, the standard deviation is undefined and left blank.
    Repeating a seed within the same group raises an error to avoid pseudoreplication.
    """
    groups = defaultdict(list)
    for directory, summary in _read_runs(runs_dir):
        groups[(summary["dataset"], summary["stage"], summary["configuration_id"])].append(summary)
    rows = []
    for (dataset, stage, identifier), summaries in sorted(groups.items()):
        seeds = [summary["seed"] for summary in summaries]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Repeated seeds in {dataset}/{stage}/{identifier}; use separate directories")
        row = {"dataset": dataset, "stage": stage, "configuration_id": identifier,
               "repetitions": len(summaries), "seeds": "|".join(map(str, sorted(seeds)))}
        for parameter in ("population_size", "generations", "selection", "pc", "pm", "elite_size", "max_depth"):
            row[parameter] = summaries[0].get("parameters", {}).get(parameter, "")
        for metric in ("train_rmse", "test_rmse", "duration_seconds", "best_size", "best_depth"):
            values = np.asarray([summary[metric] for summary in summaries], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else ""
        rows.append(row)
    target = Path(output_dir) / "summary.csv"
    write_csv(target, rows)
    return target


def plot_runs(runs_dir: str | Path, output_dir: str | Path = "results/figures") -> list[Path]:
    """Training and mean tree growth curves, plus final training/test RMSE."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for directory, summary in _read_runs(runs_dir):
        groups[(summary["dataset"], summary["stage"])].append((directory, summary))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for (dataset, stage), runs in sorted(groups.items()):
        by_config = defaultdict(list)
        for directory, summary in runs:
            with (directory / "history.csv").open(encoding="utf-8", newline="") as stream:
                history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
            by_config[summary["configuration_id"]].append((summary, history))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
        comparison_parameters = ("population_size", "generations", "selection", "pc", "pm", "elite_size", "max_depth")
        varying = [key for key in comparison_parameters if
                   len({str(repetitions[0][0].get("parameters", {}).get(key, ""))
                        for repetitions in by_config.values()}) > 1]
        for identifier, repetitions in sorted(by_config.items()):
            parameters = repetitions[0][0].get("parameters", {})
            label = ", ".join(f"{key}={parameters.get(key)}" for key in varying) or identifier
            # Configurations with different run lengths do not extrapolate beyond the last generation.
            length = min(len(history) for _, history in repetitions)
            generations = np.arange(length)
            for axis, key in ((axes[0], "best_rmse"), (axes[1], "mean_size")):
                values = np.array([[row[key] for row in history[:length]] for _, history in repetitions])
                mean = values.mean(axis=0)
                deviation = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(length)
                axis.plot(generations, mean, label=label)
                axis.fill_between(generations, np.maximum(0, mean-deviation), mean+deviation, alpha=0.15)
            train = np.array([summary["train_rmse"] for summary, _ in repetitions])
            test = np.array([summary["test_rmse"] for summary, _ in repetitions])
            axes[2].plot([0, 1], [train.mean(), test.mean()], marker="o", label=label)
        axes[0].set(title="Training convergence", xlabel="Generation", ylabel="Best RMSE (mean ± SD)")
        axes[1].set(title="Tree growth", xlabel="Generation", ylabel="Mean number of nodes")
        axes[2].set(title="Final evaluation", xticks=[0, 1], xticklabels=["Training", "Test"], ylabel="Mean RMSE")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(fontsize="small")
        fig.suptitle(f"{dataset} / {stage}")
        output = destination / f"{dataset}-{stage}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        outputs.append(output)
    return outputs

