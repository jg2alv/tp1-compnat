"""Small evolutionary runs exercise population invariants and event semantics."""

from dataclasses import replace

import numpy as np
import pytest

from gp.config import GPConfig
from gp.evolution import evolve
from gp.operators import VariationResult
from gp.tree import Node


@pytest.fixture
def regression_data():
    X = np.array([[-1.0, 2.0], [0.0, 0.5], [0.5, -1.0], [1.0, 1.5], [2.0, 3.0]])
    return X, X[:, 0] + X[:, 1]


def small_config(**changes):
    values = dict(
        population_size=16,
        generations=3,
        min_initial_depth=1,
        max_depth=3,
        pc=0.6,
        pm=0.3,
        seed=27,
    )
    values.update(changes)
    return GPConfig(**values)


@pytest.mark.parametrize("selection", ["tournament", "roulette", "epsilon_lexicase"])
def test_all_selection_methods_preserve_population_and_depth(regression_data, selection):
    config = small_config(selection=selection)
    result = evolve(*regression_data, config)
    assert len(result.population) == config.population_size
    assert result.fitness.shape == (config.population_size,)
    assert np.isfinite(result.fitness).all()
    assert all(tree.depth <= config.max_depth for tree in result.population)
    assert result.best_tree.depth <= config.max_depth
    assert [row["generation"] for row in result.history] == [0, 1, 2, 3]
    assert all(row["max_depth"] <= config.max_depth for row in result.history)
    for generation in range(1, config.generations + 1):
        records = [row for row in result.operator_records if row["generation"] == generation]
        assert len(records) == config.population_size - config.elite_size


def test_elitism_preserves_best_training_score(regression_data):
    result = evolve(*regression_data, small_config(generations=6, elite_size=2))
    best = [row["best_rmse"] for row in result.history]
    assert all(after <= before for before, after in zip(best, best[1:]))
    assert result.train_rmse == min(best)
    assert result.train_rmse == np.min(result.fitness)
    assert result.best_tree in result.population


def test_zero_generations_evaluates_initial_population(regression_data):
    config = small_config(generations=0)
    result = evolve(*regression_data, config)
    assert len(result.population) == config.population_size
    assert len(result.history) == 1
    row = result.history[0]
    assert row["generation"] == 0
    assert row["best_rmse"] == result.train_rmse
    assert result.operator_records == []
    assert row["rejected_children"] == 0
    for operator in ("crossover", "mutation", "reproduction"):
        for outcome in ("better", "equal", "worse"):
            assert row[f"{operator}_{outcome}"] == 0


def test_same_seed_reproduces_population_metrics_and_events(regression_data):
    config = small_config()
    first = evolve(*regression_data, config)
    second = evolve(*regression_data, config)
    assert first.best_tree == second.best_tree
    assert first.train_rmse == second.train_rmse
    assert first.population == second.population
    np.testing.assert_array_equal(first.fitness, second.fitness)
    assert first.history == second.history
    assert first.operator_records == second.operator_records


@pytest.mark.parametrize(
    ("pc", "pm", "expected"),
    [(1.0, 0.0, "crossover"), (0.0, 1.0, "mutation"), (0.0, 0.0, "reproduction")],
)
def test_operator_probabilities_choose_exclusive_alternatives(regression_data, pc, pm, expected):
    config = small_config(population_size=9, elite_size=0, pc=pc, pm=pm)
    result = evolve(*regression_data, config)
    assert len(result.operator_records) == config.population_size * config.generations
    assert {row["operator"] for row in result.operator_records} == {expected}
    assert all(tree.depth <= config.max_depth for tree in result.population)
    for row in result.history[1:]:
        assert sum(row[f"{expected}_{outcome}"] for outcome in ("better", "equal", "worse")) == 9
    if expected == "reproduction":
        initial = evolve(*regression_data, replace(config, generations=0))
        original_signatures = {tree.signature() for tree in initial.population}
        assert all(tree.signature() in original_signatures for tree in result.population)
        assert all(row["outcome"] == "equal" for row in result.operator_records)
        assert all(row["improvement"] == 0 for row in result.operator_records)


@pytest.mark.parametrize("children, outcomes", [((0.5, 1.5), ("better", "worse")), ((1.0, 1.0), ("equal", "equal"))])
def test_crossover_outcomes_compare_against_mean_parent_fitness(monkeypatch, children, outcomes):
    # For y=0 and x=2 the two parents have RMSEs 2 and 0, averaging 1.
    import gp.evolution as evolution

    population = [Node.variable(0), Node.constant(0)]
    monkeypatch.setattr(evolution, "ramped_half_and_half", lambda *args, **kwargs: population)
    choices = iter([0, 1])
    monkeypatch.setattr(evolution, "select_parent", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(
        evolution,
        "crossover",
        lambda *args, **kwargs: VariationResult(tuple(Node.constant(value) for value in children), 2),
    )
    config = small_config(population_size=2, generations=1, elite_size=0, pc=1.0, pm=0.0)
    result = evolve(np.array([[2.0]]), np.array([0.0]), config)
    assert [record["parent_mean_rmse"] for record in result.operator_records] == [1.0, 1.0]
    assert [record["parent_indices"] for record in result.operator_records] == ["0|1", "0|1"]
    assert tuple(record["outcome"] for record in result.operator_records) == outcomes
    assert [record["improvement"] for record in result.operator_records] == [1 - value for value in children]
    # A rejected crossover attempt belongs to the event, so its second child
    # must not double-count the same rejected proposals.
    assert [record["rejected"] for record in result.operator_records] == [2, 0]
    assert result.history[-1]["rejected_children"] == 2


def test_historical_champion_is_returned_without_elitism(regression_data):
    result = evolve(*regression_data, small_config(elite_size=0))
    assert result.train_rmse == min(row["best_rmse"] for row in result.history)
