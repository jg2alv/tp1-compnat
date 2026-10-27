"""Parent selection using roulette, tournament, and epsilon-lexicase with MAD."""

from __future__ import annotations

import numpy as np


def _fitness_values(fitness: np.ndarray) -> np.ndarray:
    values = np.asarray(fitness, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Fitness must be a nonempty vector.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Fitness values must be finite and nonnegative.")
    return values


def roulette(fitness: np.ndarray, rng: np.random.Generator) -> int:
    """Select with weight 1/(1+RMSE), including when the error is zero."""

    values = _fitness_values(fitness)
    weights = 1.0 / (1.0 + values)
    # Normalization in two steps avoids subnormal sums for enormous errors.
    weights /= np.max(weights)
    probabilities = weights / np.sum(weights)
    return int(rng.choice(values.size, p=probabilities))


def tournament(
    fitness: np.ndarray, rng: np.random.Generator, size: int = 2
) -> int:
    """Sample with replacement and randomly choose among the best in a tie."""

    values = _fitness_values(fitness)
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Tournament size must be a positive integer.")
    contestants = rng.integers(0, values.size, size=int(size))
    scores = values[contestants]
    winners = contestants[scores == np.min(scores)]
    return int(rng.choice(winners))


def _median(values: np.ndarray) -> float:
    """Compute the median without adding two potentially enormous middle values."""

    middle = values.size // 2
    if values.size % 2:
        return float(np.partition(values, middle)[middle])
    ordered = np.partition(values, (middle - 1, middle))
    lower, upper = float(ordered[middle - 1]), float(ordered[middle])
    return lower + (upper - lower) / 2.0


def epsilon_lexicase(errors: np.ndarray, rng: np.random.Generator) -> int:
    """Apply epsilon-lexicase, recalculating MAD among each case's candidates.

    Errors <= best error + MAD survive. The non-strict inequality preserves
    the best candidates when MAD=0. Cases are shuffled for every selection.
    """

    values = np.asarray(errors, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Errors must form a nonempty matrix of individuals by cases.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Errors per case must be finite and nonnegative.")
    candidates = np.arange(values.shape[0])
    for case in rng.permutation(values.shape[1]):
        if candidates.size == 1:
            return int(candidates[0])
        case_errors = values[candidates, case]
        median = _median(case_errors)
        mad = _median(np.abs(case_errors - median))
        best = np.min(case_errors)
        # Equivalent to <= best+MAD, without the risk of overflow in the sum.
        candidates = candidates[(case_errors - best) <= mad]
    return int(rng.choice(candidates))


def select_parent(
    fitness: np.ndarray,
    errors: np.ndarray,
    rng: np.random.Generator,
    method: str = "tournament",
    tournament_size: int = 2,
) -> int:
    """Dispatch the configured method and return the selected parent's index."""

    if method == "roulette":
        return roulette(fitness, rng)
    if method == "tournament":
        return tournament(fitness, rng, size=tournament_size)
    if method in {"epsilon_lexicase", "epsilon-lexicase", "lexicase"}:
        values = _fitness_values(fitness)
        cases = np.asarray(errors, dtype=float)
        if cases.ndim != 2 or cases.shape[0] != values.size:
            raise ValueError("Fitness and errors must describe the same population.")
        return epsilon_lexicase(cases, rng)
    raise ValueError(f"Unknown selection method: {method!r}.")
