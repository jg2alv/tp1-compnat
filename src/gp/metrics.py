"""Training statistics and operator contributions for each generation."""

from collections import Counter

import numpy as np

from .tree import Node


def generation_metrics(
    generation: int,
    population: list[Node] | tuple[Node, ...],
    fitness: np.ndarray,
    invalid: np.ndarray,
    operator_records: list[dict],
) -> dict:
    """Count duplicates as excess individuals with identical structural signatures."""
    counts = Counter(tree.signature() for tree in population)
    sizes = np.asarray([tree.size for tree in population])
    depths = np.asarray([tree.depth for tree in population])
    row = {
        "generation": generation,
        "best_rmse": float(np.min(fitness)),
        "worst_rmse": float(np.max(fitness)),
        "mean_rmse": float(np.mean(fitness)),
        "duplicates": len(population) - len(counts),
        "mean_size": float(np.mean(sizes)),
        "max_size": int(np.max(sizes)),
        "mean_depth": float(np.mean(depths)),
        "max_depth": int(np.max(depths)),
        "invalid_individuals": int(np.sum(invalid)),
        "rejected_children": sum(record["rejected"] for record in operator_records),
    }
    for operator in ("crossover", "mutation", "reproduction"):
        records = [record for record in operator_records if record["operator"] == operator]
        for outcome in ("better", "equal", "worse"):
            row[f"{operator}_{outcome}"] = sum(r["outcome"] == outcome for r in records)
    return row

