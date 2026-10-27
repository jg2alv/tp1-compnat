"""Selector tests, including ties and MAD equal to zero."""

import numpy as np
import pytest

from gp.selection import epsilon_lexicase, roulette, select_parent, tournament


class OrderedRng:
    """Cases in order and the last survivor, to test the exact filtering."""

    def permutation(self, n):
        return np.arange(n)

    def choice(self, candidates, p=None):
        self.probabilities = p
        return candidates - 1 if np.isscalar(candidates) else candidates[-1]


def test_lexicase_keeps_ties_when_mad_is_zero():
    assert epsilon_lexicase(np.zeros((3, 4)), OrderedRng()) == 2


def test_lexicase_recalculates_mad_among_current_candidates():
    errors = np.array([[0, 0], [0, 1], [0, 2], [100, 100], [100, 100]])
    # Case 0 keeps 0,1,2; case 1 has MAD 1, keeping only 0,1.
    assert epsilon_lexicase(errors, OrderedRng()) == 1


def test_lexicase_can_find_unique_survivor_for_either_case_order():
    errors = np.array([[0, 2], [0, 1], [4, 0]])
    rng = np.random.default_rng(12)
    assert {epsilon_lexicase(errors, rng) for _ in range(30)} == {1}


def test_lexicase_handles_extreme_finite_errors_without_overflow():
    with np.errstate(all="raise"):
        selected = epsilon_lexicase(np.full((4, 2), 1e308), OrderedRng())
    assert selected == 3


def test_roulette_weights_decrease_and_accept_zero_and_penalty():
    rng = OrderedRng()
    roulette(np.array([0, 1, 10, 1e100]), rng)
    probabilities = rng.probabilities
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities > 0)
    assert np.all(np.diff(probabilities) < 0)
    assert np.sum(probabilities) == pytest.approx(1)


def test_roulette_equal_extreme_errors_have_equal_probabilities():
    rng = OrderedRng()
    roulette(np.full(3, 1e308), rng)
    np.testing.assert_allclose(rng.probabilities, [1 / 3] * 3)


def test_tournament_chooses_best_among_contestants():
    class TournamentRng(OrderedRng):
        def integers(self, low, high, size):
            return np.array([0, 2, 1])

    assert tournament(np.array([5, 1, 3]), TournamentRng(), size=3) == 1


@pytest.mark.parametrize("method", ["roulette", "tournament", "epsilon_lexicase"])
def test_selection_is_reproducible_for_same_seed(method):
    fitness = np.array([0.0, 0.5, 2.0])
    errors = np.array([[0.0, 2.0], [0.5, 0.5], [2.0, 0.0]])
    left = np.random.default_rng(101)
    right = np.random.default_rng(101)
    assert [select_parent(fitness, errors, left, method) for _ in range(20)] == [
        select_parent(fitness, errors, right, method) for _ in range(20)
    ]


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_tournament_size_is_rejected(size):
    with pytest.raises(ValueError):
        tournament(np.array([1.0]), np.random.default_rng(0), size=size)


@pytest.mark.parametrize("values", [[], [np.nan], [-1]])
def test_invalid_fitness_is_rejected(values):
    with pytest.raises(ValueError):
        roulette(np.asarray(values), np.random.default_rng(0))


def test_unknown_method_and_incompatible_lexicase_shape_are_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unknown"):
        select_parent(np.ones(2), np.ones((2, 1)), rng, method="unknown")
    with pytest.raises(ValueError, match="same population"):
        select_parent(np.ones(2), np.ones((3, 1)), rng, method="epsilon_lexicase")
