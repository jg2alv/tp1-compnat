"""Numerical tree evaluation and symbolic regression error."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from .primitives import PRIMITIVES
from .tree import Node


# Limits the influence of extreme results and keeps all selectors finite.
# An invalid tree never gains an advantage over a tree with a very large error.
INVALID_PENALTY = 1e100


@dataclass(frozen=True)
class Evaluation:
    """Results aligned with the order of the population and training cases."""

    fitness: np.ndarray
    errors: np.ndarray
    invalid: np.ndarray


def _features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a nonempty matrix of samples and variables.")
    return values


def _targets(y: np.ndarray, n_samples: int | None = None) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("y must be a nonempty vector.")
    if n_samples is not None and values.shape[0] != n_samples:
        raise ValueError("X and y must contain the same number of samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Target values must be finite.")
    return values


def evaluate_tree(tree: Node, X: np.ndarray) -> np.ndarray:
    """Interpret the tree directly over the columns of X, without using ``eval``.

    Operations are vectorized. Nonfinite outputs remain visible to
    ``evaluate_population``, which penalizes the entire individual.
    """

    features = _features(X)
    n_samples, n_features = features.shape

    def visit(node: Node) -> np.ndarray:
        if node.kind == "constant":
            if node.children:
                raise ValueError("A constant cannot have children.")
            return np.full(n_samples, float(node.value), dtype=float)
        if node.kind == "variable":
            if node.children:
                raise ValueError("A variable cannot have children.")
            index = int(node.value)
            if index != node.value or not 0 <= index < n_features:
                raise ValueError(f"Variable index out of bounds for X: {node.value!r}.")
            return features[:, index]
        if node.kind != "function" or node.value not in PRIMITIVES:
            raise ValueError(f"Unknown node or function: {node.value!r}.")
        primitive = PRIMITIVES[node.value]
        if len(node.children) != primitive.arity:
            raise ValueError(f"Incorrect arity for {node.value!r}.")
        arguments = [visit(child) for child in node.children]
        result = np.asarray(primitive.function(*arguments), dtype=float)
        if result.ndim == 0:
            return np.full(n_samples, float(result), dtype=float)
        if result.shape != (n_samples,):
            raise ValueError(f"Function {node.value!r} returned an unexpected shape.")
        return result

    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        return visit(tree)


def _stable_rmse(absolute_errors: np.ndarray) -> float:
    scale = float(np.max(absolute_errors))
    if scale == 0.0:
        return 0.0
    # Scaling avoids directly squaring very large residuals.
    normalized = absolute_errors / scale
    value = scale * float(np.sqrt(np.mean(normalized * normalized)))
    return min(value, INVALID_PENALTY)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return stable RMSE, capped at the finite penalty for invalid results.

    Invalid targets are input errors; invalid predictions or subtractions that
    exceed the numerical representation receive ``INVALID_PENALTY``.
    """

    targets = _targets(y_true)
    predictions = np.asarray(y_pred, dtype=float)
    if predictions.shape != targets.shape:
        raise ValueError("Targets and predictions must have the same one-dimensional shape.")
    with np.errstate(over="ignore", invalid="ignore"):
        absolute_errors = np.abs(targets - predictions)
    if not np.all(np.isfinite(absolute_errors)):
        return INVALID_PENALTY
    return _stable_rmse(absolute_errors)


def evaluate_population(
    population: Iterable[Node], X: np.ndarray, y: np.ndarray
) -> Evaluation:
    """Compute RMSE and absolute errors per case for every individual.

    Any nonfinite result invalidates the entire individual. All of its cases
    receive the penalty, preventing lexicase selection from favoring a partially
    invalid expression. Extreme finite errors are capped at the same numerical
    limit; fitness is computed before applying this limit to individual cases.
    """

    features = _features(X)
    targets = _targets(y, features.shape[0])
    individuals = list(population)
    if not individuals:
        raise ValueError("The population cannot be empty.")
    errors = np.empty((len(individuals), targets.size), dtype=float)
    fitness = np.empty(len(individuals), dtype=float)
    invalid = np.zeros(len(individuals), dtype=bool)
    for index, tree in enumerate(individuals):
        predictions = evaluate_tree(tree, features)
        with np.errstate(over="ignore", invalid="ignore"):
            absolute_errors = np.abs(targets - predictions)
        invalid[index] = not np.all(np.isfinite(absolute_errors))
        if invalid[index]:
            errors[index].fill(INVALID_PENALTY)
            fitness[index] = INVALID_PENALTY
        else:
            errors[index] = np.minimum(absolute_errors, INVALID_PENALTY)
            fitness[index] = _stable_rmse(absolute_errors)
    return Evaluation(fitness=fitness, errors=errors, invalid=invalid)
