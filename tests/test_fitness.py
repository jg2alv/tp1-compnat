"""Tests for expression interpretation and numerical penalties."""

import numpy as np
import pytest

from gp.fitness import INVALID_PENALTY, evaluate_population, evaluate_tree, rmse
from gp.tree import Node


def test_evaluate_known_expression_and_broadcast_constant():
    tree = Node.function(
        "add", Node.function("mul", Node.variable(0), Node.variable(0)), Node.constant(2)
    )
    X = np.array([[1.0, 10.0], [-2.0, 4.0], [0.0, 3.0]])
    np.testing.assert_array_equal(evaluate_tree(tree, X), [3.0, 6.0, 2.0])
    np.testing.assert_array_equal(evaluate_tree(Node.constant(7), X), [7.0] * 3)


def test_population_returns_rmse_and_absolute_case_errors():
    X = np.array([[1.0], [2.0], [3.0]])
    result = evaluate_population([Node.variable(0), Node.constant(0)], X, X[:, 0])
    np.testing.assert_allclose(result.fitness, [0.0, np.sqrt(14 / 3)])
    np.testing.assert_array_equal(result.errors, [[0, 0, 0], [1, 2, 3]])
    np.testing.assert_array_equal(result.invalid, [False, False])


def test_protected_division_produces_finite_fitness():
    expression = Node.function("div", Node.variable(0), Node.constant(0))
    result = evaluate_population([expression], np.array([[1.0], [-2.0]]), np.zeros(2))
    assert not result.invalid[0]
    assert np.isfinite(result.fitness[0])
    assert np.all(np.isfinite(result.errors))


def test_overflow_penalizes_entire_individual():
    expression = Node.function("mul", Node.variable(0), Node.variable(0))
    result = evaluate_population([expression], np.array([[1.0], [1e200]]), np.ones(2))
    assert result.invalid[0]
    assert result.fitness[0] == INVALID_PENALTY
    np.testing.assert_array_equal(result.errors, [[INVALID_PENALTY] * 2])


def test_finite_extreme_errors_are_capped_without_overflow():
    with np.errstate(all="raise"):
        result = evaluate_population(
            [Node.variable(0)], np.array([[1e200], [0.0]]), np.zeros(2)
        )
        assert rmse(np.zeros(2), np.array([1e200, 0.0])) == INVALID_PENALTY
    assert not result.invalid[0]
    assert result.fitness[0] == INVALID_PENALTY
    assert np.all(np.isfinite(result.errors))


def test_rmse_penalty_for_nonfinite_predictions_and_overflowing_subtraction():
    assert rmse(np.zeros(2), np.array([0.0, np.nan])) == INVALID_PENALTY
    assert rmse(np.array([-1e308]), np.array([1e308])) == INVALID_PENALTY


def test_rmse_retains_precision_for_small_residuals():
    assert rmse(np.zeros(2), np.array([3e-200, 4e-200])) == pytest.approx(
        np.sqrt(12.5) * 1e-200, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize(
    "X,y", [(np.array([1, 2]), np.zeros(2)), (np.ones((2, 1)), np.zeros(3))]
)
def test_population_rejects_incompatible_data(X, y):
    with pytest.raises(ValueError):
        evaluate_population([Node.constant(0)], X, y)


def test_invalid_targets_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        rmse(np.array([np.nan]), np.array([0.0]))


def test_variable_index_is_checked():
    with pytest.raises(ValueError, match="Variable"):
        evaluate_tree(Node.variable(1), np.ones((2, 1)))
