"""Subtree variation operators that preserve immutable parent expressions."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .initialization import generate_tree
from .primitives import DEFAULT_FUNCTIONS
from .tree import Node


@dataclass(frozen=True)
class VariationResult:
    children: tuple[Node, ...]
    rejected: int = 0


def _validate_parent(parent, max_depth):
    if isinstance(max_depth, bool) or not isinstance(max_depth, Integral) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer.")
    if parent.depth > max_depth:
        raise ValueError("Parent already exceeds max_depth.")


def crossover(
    parent1: Node,
    parent2: Node,
    rng: np.random.Generator,
    max_depth: int,
) -> VariationResult:
    """Exchange uniformly sampled subtrees, including possible root swaps.

    A child exceeding the depth limit is replaced by its corresponding parent;
    the other child is still accepted if valid.
    """
    _validate_parent(parent1, max_depth)
    _validate_parent(parent2, max_depth)
    paths1, paths2 = tuple(parent1.iter_paths()), tuple(parent2.iter_paths())
    path1 = paths1[int(rng.integers(len(paths1)))]
    path2 = paths2[int(rng.integers(len(paths2)))]
    offspring = (
        parent1.replace_subtree(path1, parent2.subtree(path2)),
        parent2.replace_subtree(path2, parent1.subtree(path1)),
    )
    rejected = sum(child.depth > max_depth for child in offspring)
    children = tuple(
        parent if child.depth > max_depth else child
        for parent, child in zip((parent1, parent2), offspring)
    )
    return VariationResult(children, rejected)


def mutate(
    parent: Node,
    rng: np.random.Generator,
    n_features: int,
    max_depth: int,
    functions=DEFAULT_FUNCTIONS,
    constant_range=(-5.0, 5.0),
) -> VariationResult:
    """Replace a uniformly sampled subtree with a grow-generated subtree."""
    _validate_parent(parent, max_depth)
    paths = tuple(parent.iter_paths())
    path = paths[int(rng.integers(len(paths)))]
    # Path length is the insertion depth, so this construction cannot overflow.
    replacement = generate_tree(
        rng, n_features, max_depth - len(path), "grow", functions, constant_range
    )
    return VariationResult((parent.replace_subtree(path, replacement),))
