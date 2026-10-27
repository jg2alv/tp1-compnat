"""Generational loop; the test set is never passed to this module."""

from dataclasses import dataclass

import numpy as np

from .config import GPConfig
from .fitness import Evaluation, evaluate_population
from .initialization import ramped_half_and_half
from .metrics import generation_metrics
from .operators import VariationResult, crossover, mutate
from .selection import select_parent
from .tree import Node


@dataclass(frozen=True)
class EvolutionResult:
    best_tree: Node
    train_rmse: float
    history: list[dict]
    operator_records: list[dict]
    population: tuple[Node, ...]
    fitness: np.ndarray


def evolve(X: np.ndarray, y: np.ndarray, config: GPConfig) -> EvolutionResult:
    """Run G generations after the initial population (history 0..G).

    Each event selects exactly one of crossover, mutation, or reproduction.
    Crossover produces two children; if only one slot remains, use the first.
    Without elitism, the best individual ever found is still stored outside the population.
    """
    config.validate()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or y.shape != (X.shape[0],):
        raise ValueError("X must be a nonempty matrix and y a vector with one target per row")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Training data must be finite")
    rng = np.random.default_rng(config.seed)
    population = ramped_half_and_half(
        config.population_size, X.shape[1], rng,
        min_depth=config.min_initial_depth, max_depth=config.max_depth,
        functions=config.functions, constant_range=config.constant_range,
    )
    evaluation = evaluate_population(population, X, y)
    best_index = int(np.argmin(evaluation.fitness))
    best_tree = population[best_index]
    best_fitness = float(evaluation.fitness[best_index])
    history = [generation_metrics(0, population, evaluation.fitness, evaluation.invalid, [])]
    records: list[dict] = []

    for generation in range(1, config.generations + 1):
        elite_indices = np.argsort(evaluation.fitness, kind="stable")[:config.elite_size]
        next_population = [population[int(index)] for index in elite_indices]
        scores = [float(evaluation.fitness[index]) for index in elite_indices]
        errors = [evaluation.errors[index] for index in elite_indices]
        invalids = [bool(evaluation.invalid[index]) for index in elite_indices]
        generation_records: list[dict] = []
        event = 0

        def parent() -> int:
            return select_parent(
                evaluation.fitness, evaluation.errors, rng,
                method=config.selection, tournament_size=config.tournament_size,
            )

        while len(next_population) < config.population_size:
            event += 1
            choice = rng.random()
            first = parent()
            parents = [first]
            if choice < config.pc:
                operator = "crossover"
                parents.append(parent())
                variation = crossover(population[first], population[parents[1]], rng, config.max_depth)
            elif choice < config.pc + config.pm:
                operator = "mutation"
                variation = mutate(
                    population[first], rng, X.shape[1], config.max_depth,
                    config.functions, config.constant_range,
                )
            else:
                operator = "reproduction"
                variation = VariationResult((population[first],), 0)
            remaining = config.population_size - len(next_population)
            children = variation.children[:remaining]
            child_evaluation = evaluate_population(children, X, y)
            parent_mean = float(np.mean(evaluation.fitness[parents]))
            for child_index, child in enumerate(children):
                score = float(child_evaluation.fitness[child_index])
                outcome = "better" if score < parent_mean else "worse" if score > parent_mean else "equal"
                generation_records.append({
                    "generation": generation, "event": event, "operator": operator,
                    "child_index": child_index,
                    "parent_indices": "|".join(str(index) for index in parents),
                    "parent_mean_rmse": parent_mean, "child_rmse": score,
                    "improvement": parent_mean - score, "outcome": outcome,
                    "rejected": variation.rejected if child_index == 0 else 0,
                })
                next_population.append(child)
                scores.append(score)
                errors.append(child_evaluation.errors[child_index])
                invalids.append(bool(child_evaluation.invalid[child_index]))
        population = next_population
        evaluation = Evaluation(np.asarray(scores), np.stack(errors), np.asarray(invalids))
        current = int(np.argmin(evaluation.fitness))
        if evaluation.fitness[current] < best_fitness:
            best_tree = population[current]
            best_fitness = float(evaluation.fitness[current])
        history.append(generation_metrics(
            generation, population, evaluation.fitness, evaluation.invalid, generation_records,
        ))
        records.extend(generation_records)

    return EvolutionResult(best_tree, best_fitness, history, records, tuple(population), evaluation.fitness.copy())

