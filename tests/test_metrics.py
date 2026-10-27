"""Generation summaries must retain duplicate, size, and operator accounting."""

import numpy as np
import pytest

from gp.metrics import generation_metrics
from gp.tree import Node


def example_population():
    x = Node.variable(0)
    addition = Node.function("add", x, Node.constant(1))
    sine = Node.function("sin", addition)
    # Construct a separate but structurally identical x for duplicate counting.
    return [x, Node.variable(0), addition, sine]


def test_summary_reports_fitness_structural_duplicates_and_tree_growth():
    row = generation_metrics(
        4,
        example_population(),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([False, False, True, False]),
        [],
    )
    assert row["generation"] == 4
    assert row["best_rmse"] == 1.0
    assert row["worst_rmse"] == 4.0
    assert row["mean_rmse"] == 2.5
    assert row["duplicates"] == 1
    assert row["mean_size"] == pytest.approx((1 + 1 + 3 + 4) / 4)
    assert row["max_size"] == 4
    assert row["mean_depth"] == pytest.approx((0 + 0 + 1 + 2) / 4)
    assert row["max_depth"] == 2
    assert row["invalid_individuals"] == 1
    assert row["rejected_children"] == 0
    assert all(row[f"{operator}_{outcome}"] == 0 for operator in ("crossover", "mutation", "reproduction") for outcome in ("better", "equal", "worse"))


def test_operator_outcomes_and_rejections_are_counted_per_generation():
    records = [
        {"operator": "crossover", "outcome": "better", "rejected": 1},
        {"operator": "crossover", "outcome": "equal", "rejected": 0},
        {"operator": "crossover", "outcome": "worse", "rejected": 0},
        {"operator": "mutation", "outcome": "better", "rejected": 2},
        {"operator": "mutation", "outcome": "better", "rejected": 0},
        {"operator": "mutation", "outcome": "worse", "rejected": 0},
        {"operator": "reproduction", "outcome": "equal", "rejected": 0},
    ]
    row = generation_metrics(1, example_population(), np.ones(4), np.zeros(4, dtype=bool), records)
    assert row["crossover_better"] == row["crossover_equal"] == row["crossover_worse"] == 1
    assert row["mutation_better"] == 2
    assert row["mutation_equal"] == 0
    assert row["mutation_worse"] == 1
    assert row["reproduction_better"] == row["reproduction_worse"] == 0
    assert row["reproduction_equal"] == 1
    assert row["rejected_children"] == 3


def test_all_identical_individuals_count_only_excess_copies():
    population = [Node.constant(1) for _ in range(5)]
    row = generation_metrics(0, population, np.ones(5), np.zeros(5, dtype=bool), [])
    assert row["duplicates"] == 4
    assert row["mean_depth"] == row["max_depth"] == 0
    assert row["mean_size"] == row["max_size"] == 1
