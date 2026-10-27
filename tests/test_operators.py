"""Variation must preserve parents, arity and configured depth limits."""

import numpy as np
import pytest

from gp.initialization import ramped_half_and_half
from gp.operators import crossover, mutate
from gp.tree import Node


class FixedChoices:
    """Select concrete crossover points without coupling tests to NumPy seeds."""

    def __init__(self, *choices):
        self.choices = iter(choices)

    def integers(self, high):
        choice = next(self.choices)
        assert 0 <= choice < high
        return choice


def test_crossover_exchanges_selected_subtrees_without_changing_parents():
    first = Node.function("add", Node.variable(0), Node.constant(2))
    second = Node.function("mul", Node.variable(1), Node.constant(3))
    result = crossover(first, second, FixedChoices(1, 2), max_depth=2)
    assert result.rejected == 0
    assert result.children == (
        Node.function("add", Node.constant(3), Node.constant(2)),
        Node.function("mul", Node.variable(1), Node.variable(0)),
    )
    assert first.children[0] == Node.variable(0)
    assert second.children[1] == Node.constant(3)


def test_crossover_rejects_only_the_child_that_exceeds_the_depth():
    first = Node.function("sin", Node.function("sin", Node.variable(0)))
    second = Node.function("cos", Node.function("cos", Node.variable(1)))
    result = crossover(first, second, FixedChoices(2, 0), max_depth=2)
    assert result.rejected == 1
    assert result.children[0] is first
    assert result.children[1] == Node.variable(0)


def test_root_crossover_and_terminal_mutation():
    first, second = Node.variable(0), Node.constant(9)
    result = crossover(first, second, FixedChoices(0, 0), max_depth=0)
    assert result.children == (second, first)
    mutation = mutate(first, np.random.default_rng(4), n_features=2, max_depth=0)
    assert len(mutation.children) == 1
    assert mutation.children[0].depth == 0
    assert mutation.rejected == 0


def test_repeated_variation_preserves_limits_and_is_reproducible():
    initial = ramped_half_and_half(40, 3, np.random.default_rng(21), max_depth=4)
    originals = tuple(tree.signature() for tree in initial)

    def vary(seed):
        rng = np.random.default_rng(seed)
        children = []
        for first, second in zip(initial[::2], initial[1::2]):
            crossed = crossover(first, second, rng, max_depth=4)
            for child in crossed.children:
                mutated = mutate(child, rng, n_features=3, max_depth=4)
                assert mutated.rejected == 0
                children.extend(mutated.children)
        return children

    children = vary(83)
    assert children == vary(83)
    assert all(child.depth <= 4 for child in children)
    assert any(child != parent for child, parent in zip(children, initial))
    assert originals == tuple(tree.signature() for tree in initial)


def test_invalid_parents_and_limits_fail_early():
    parent = Node.function("sin", Node.variable(0))
    with pytest.raises(ValueError):
        crossover(parent, parent, np.random.default_rng(0), max_depth=0)
    with pytest.raises(ValueError):
        mutate(parent, np.random.default_rng(0), n_features=1, max_depth=-1)
