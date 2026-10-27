"""Tree and primitive behavior needed by all evolutionary components."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from gp.primitives import DIVISION_EPSILON, PRIMITIVES, protected_division
from gp.tree import Node


@pytest.fixture
def expression_tree():
    return Node.function(
        "add", Node.variable(0), Node.function("mul", Node.constant(2), Node.variable(1))
    )


def test_size_depth_and_paths(expression_tree):
    assert Node.constant(2).depth == 0
    assert expression_tree.size == 5
    assert expression_tree.depth == 2
    assert list(expression_tree.iter_paths()) == [(), (0,), (1,), (1, 0), (1, 1)]
    assert expression_tree.subtree((1, 0)) == Node.constant(2)


def test_replace_is_persistent_and_parents_are_immutable(expression_tree):
    changed = expression_tree.replace_subtree((1, 0), Node.constant(8))
    assert changed.expression() == "(x0 + (8.0 * x1))"
    assert expression_tree.expression() == "(x0 + (2.0 * x1))"
    assert changed.children[0] is expression_tree.children[0]
    with pytest.raises(FrozenInstanceError):
        expression_tree.value = "sub"
    assert expression_tree.replace_subtree((), Node.constant(3)) == Node.constant(3)


def test_json_round_trip_and_structural_signature(expression_tree):
    restored = Node.from_dict(json.loads(json.dumps(expression_tree.to_dict())))
    assert restored == expression_tree
    assert restored.signature() == expression_tree.signature()
    assert len({expression_tree.signature(), restored.signature()}) == 1
    assert expression_tree.replace_subtree((0,), Node.variable(1)).signature() != restored.signature()


@pytest.mark.parametrize("path", [(-1,), (2,), (0, 0), (True,), (1.5,)])
def test_invalid_paths_fail(expression_tree, path):
    with pytest.raises(IndexError):
        expression_tree.subtree(path)
    with pytest.raises(IndexError):
        expression_tree.replace_subtree(path, Node.constant(0))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Node.function("missing", Node.constant(1)),
        lambda: Node.function("add", Node.constant(1)),
        lambda: Node.variable(-1),
        lambda: Node.variable(0.5),
        lambda: Node.variable(True),
        lambda: Node.constant(float("inf")),
        lambda: Node.constant(float("nan")),
        lambda: Node("constant", 1, (Node.constant(2),)),
        lambda: Node("unknown", 0),
    ],
)
def test_malformed_nodes_fail_early(factory):
    with pytest.raises(ValueError):
        factory()


def test_protected_division_broadcasts_and_handles_threshold():
    numerator = np.array([3.0, 4.0, 6.0, 8.0, 10.0])
    denominator = np.array([0.0, -DIVISION_EPSILON, 2.0, -4.0, DIVISION_EPSILON * 2])
    with np.errstate(divide="raise", invalid="raise"):
        actual = protected_division(numerator, denominator)
    np.testing.assert_allclose(actual, [1.0, 1.0, 3.0, -2.0, 5e12])
    np.testing.assert_allclose(protected_division(6, np.array([2, 0])), [3, 1])
    assert protected_division(4, 2) == 2


def test_invalid_numeric_results_reach_fitness_instead_of_being_hidden():
    assert np.isnan(protected_division(1.0, np.nan))
    assert PRIMITIVES["div"].arity == 2
    assert Node.function("div", Node.variable(0), Node.constant(0)).expression() == "protected_div(x0, 0.0)"
