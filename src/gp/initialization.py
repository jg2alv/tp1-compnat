"""Grow, full and balanced ramped half-and-half initialization."""

from math import isfinite
from numbers import Integral

import numpy as np

from .primitives import DEFAULT_FUNCTIONS, PRIMITIVES
from .tree import Node


def _validate_tree_parameters(n_features, max_depth, functions, constant_range):
    if isinstance(n_features, bool) or not isinstance(n_features, Integral) or n_features < 1:
        raise ValueError("n_features must be a positive integer.")
    if isinstance(max_depth, bool) or not isinstance(max_depth, Integral) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer.")
    functions = tuple(functions)
    if not functions or any(name not in PRIMITIVES for name in functions):
        raise ValueError("functions must contain known primitive names.")
    if len(constant_range) != 2:
        raise ValueError("constant_range must have two endpoints.")
    low, high = map(float, constant_range)
    if not isfinite(low) or not isfinite(high) or low > high:
        raise ValueError("constant_range endpoints must be finite and ordered.")
    return functions, (low, high)


def generate_tree(
    rng: np.random.Generator,
    n_features: int,
    max_depth: int,
    method: str,
    functions=DEFAULT_FUNCTIONS,
    constant_range=(-5.0, 5.0),
) -> Node:
    """Sample an expression with a maximum depth in edges.

    Full uses functions until the target depth. Grow samples uniformly from
    function names, variables and one ephemeral-constant terminal; therefore
    its realized depth may be below the target.
    """
    functions, constant_range = _validate_tree_parameters(
        n_features, max_depth, functions, constant_range
    )
    if method not in {"grow", "full"}:
        raise ValueError("method must be 'grow' or 'full'.")

    def terminal(index=None):
        index = int(rng.integers(n_features + 1)) if index is None else index
        if index < n_features:
            return Node.variable(index)
        return Node.constant(float(rng.uniform(*constant_range)))

    def build(remaining_depth):
        if remaining_depth == 0:
            return terminal()
        if method == "grow":
            index = int(rng.integers(len(functions) + n_features + 1))
            if index >= len(functions):
                return terminal(index - len(functions))
            name = functions[index]
        else:
            name = functions[int(rng.integers(len(functions)))]
        return Node.function(
            name,
            *(build(remaining_depth - 1) for _ in range(PRIMITIVES[name].arity)),
        )

    return build(max_depth)


def ramped_half_and_half(
    population_size: int,
    n_features: int,
    rng: np.random.Generator,
    min_depth: int = 2,
    max_depth: int = 7,
    functions=DEFAULT_FUNCTIONS,
    constant_range=(-5.0, 5.0),
) -> list[Node]:
    """Balance target depths and split each depth group between grow and full.

    When the population does not divide evenly, depth groups differ by at most
    one individual. Odd groups alternate their extra grow/full member, so the
    overall methods also differ by at most one. The final order is shuffled.
    """
    functions, constant_range = _validate_tree_parameters(
        n_features, max_depth, functions, constant_range
    )
    if isinstance(population_size, bool) or not isinstance(population_size, Integral) or population_size < 1:
        raise ValueError("population_size must be a positive integer.")
    if isinstance(min_depth, bool) or not isinstance(min_depth, Integral) or not 0 <= min_depth <= max_depth:
        raise ValueError("min_depth must be between zero and max_depth.")
    depths = range(min_depth, max_depth + 1)
    base, extra = divmod(population_size, len(depths))
    population = []
    odd_to_grow = True
    for index, depth in enumerate(depths):
        count = base + (index < extra)
        grow_count = count // 2
        if count % 2:
            grow_count += int(odd_to_grow)
            odd_to_grow = not odd_to_grow
        for method, method_count in (("grow", grow_count), ("full", count - grow_count)):
            population.extend(
                generate_tree(rng, n_features, depth, method, functions, constant_range)
                for _ in range(method_count)
            )
    rng.shuffle(population)
    return population
