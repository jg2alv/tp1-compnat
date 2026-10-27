"""Verify initialization limits and ramp allocation"""

from collections import Counter

import numpy as np
import pytest

import gp.initialization as initialization
from gp.initialization import generate_tree, ramped_half_and_half
from gp.tree import Node


def test_full_attains_target_depth_and_grow_can_stop_early():
    rng = np.random.default_rng(45)
    for depth in range(8):
        full = generate_tree(rng, 3, depth, "full")
        assert full.depth == depth
        assert all(len(path) == depth for path in full.iter_paths() if not full.subtree(path).children)
    grown = [generate_tree(rng, 3, 5, "grow") for _ in range(50)]
    assert all(tree.depth <= 5 for tree in grown)
    assert any(tree.depth < 5 for tree in grown)


@pytest.mark.parametrize("size", [1, 3, 11, 50, 53, 120])
def test_ramped_allocates_balanced_depths_and_methods(monkeypatch, size):
    calls = []

    def record_tree(rng, n_features, max_depth, method, functions, constant_range):
        calls.append((max_depth, method))
        return Node.constant(max_depth)

    monkeypatch.setattr(initialization, "generate_tree", record_tree)
    population = ramped_half_and_half(size, 2, np.random.default_rng(3), min_depth=2, max_depth=6)
    assert len(population) == size
    depth_counts = [sum(depth == d for depth, _ in calls) for d in range(2, 7)]
    assert max(depth_counts) - min(depth_counts) <= 1
    methods = Counter(method for _, method in calls)
    assert abs(methods["grow"] - methods["full"]) <= 1
    for depth in range(2, 7):
        counts = Counter(method for d, method in calls if d == depth)
        assert abs(counts["grow"] - counts["full"]) <= 1


def test_population_is_reproducible_and_uses_requested_terminals():
    settings = dict(population_size=51, n_features=4, max_depth=5, constant_range=(2.0, 3.0))
    first = ramped_half_and_half(rng=np.random.default_rng(2026), **settings)
    second = ramped_half_and_half(rng=np.random.default_rng(2026), **settings)
    assert first == second
    assert any(a != b for a, b in zip(first, ramped_half_and_half(rng=np.random.default_rng(27), **settings)))
    for tree in first:
        assert tree.depth <= 5
        for path in tree.iter_paths():
            node = tree.subtree(path)
            if node.kind == "variable":
                assert 0 <= node.value < 4
            elif node.kind == "constant":
                assert 2 <= node.value <= 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"population_size": 0},
        {"n_features": 0},
        {"min_depth": 5, "max_depth": 2},
        {"max_depth": -1},
        {"functions": ()},
        {"functions": ("unknown",)},
        {"constant_range": (5, -5)},
    ],
)
def test_invalid_initialization_parameters_are_rejected(kwargs):
    settings = dict(population_size=10, n_features=2, rng=np.random.default_rng(0))
    settings.update(kwargs)
    with pytest.raises(ValueError):
        ramped_half_and_half(**settings)


def test_invalid_generation_method_is_rejected():
    with pytest.raises(ValueError):
        generate_tree(np.random.default_rng(0), 2, 4, "invalid")
