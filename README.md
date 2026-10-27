# Darwin's Math: genetic programming for symbolic regression

A Python framework that evolves mathematical expression trees to fit regression datasets. It implements genetic operators, three parent-selection strategies, and configurable experiments with detailed evolutionary logs.

## 1. Project information

- Institution: Universidade Federal de Minas Gerais
- Course: Computação Natural
- Assignment: Trabalho Prático 1
- Academic period: 2026-02

## 2. Features and datasets

- Tree-based expressions with variables, ephemeral constants, and configurable mathematical primitives.
- Ramped half-and-half initialization, subtree crossover, subtree mutation, reproduction, and elitism.
- Roulette, tournament, and epsilon-lexicase parent selection.
- RMSE fitness with protected arithmetic and handling of non-finite evaluations.
- Seeded experiments, generation statistics, operator tracking, and reporting tools.

The supplied datasets use numeric CSV files without headers. The final column is the target; preceding columns are input features.

| Dataset | Input features | Training samples | Test samples |
| --- | ---: | ---: | ---: |
| synth1 | 2 | 60 | 600 |
| synth2 | 2 | 300 | 1000 |
| concrete | 8 | 824 | 206 |

`concrete1`, the name used in the assignment, is accepted as an alias for `concrete`.

## 3. Installation

Python 3.11 or newer is required. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Runtime dependencies are NumPy and Matplotlib. The development extra includes pytest.

## 4. Usage examples

Run a small symbolic-regression experiment using the versioned synth1 dataset:

```sh
python -m gp run --dataset synth1 --population-size 20 --generations 5 --max-depth 4 --seed 0
```

This reads:

- `bases-TP1/datasets/synth1/synth1-train.csv`
- `bases-TP1/datasets/synth1/synth1-test.csv`

The command loads `configs/baseline.toml` and overrides the supplied parameters.

Inspect an experimental stage without running its evolutionary searches:

```sh
python -m gp experiment --stage population --datasets synth1 --dry-run
```

Generate summaries and figures from saved runs:

```sh
python -m gp analyze
```

Use `python -m gp --help` for the command interface. The installed `gp` command is equivalent to `python -m gp`.

## 5. Implementation decisions

Expressions are immutable trees. Crossover exchanges subtrees, while mutation replaces a selected subtree. Depth constraints apply during initialization and variation; the root has depth zero, and the default maximum depth is seven.

Roulette selection uses weights proportional to `1 / (1 + RMSE)`. Tournament selection samples candidates with replacement. Epsilon-lexicase shuffles training cases and recalculates median absolute deviation among the remaining candidates, retaining the best candidates even when that deviation is zero.

Each variation event selects crossover, mutation, or reproduction exclusively. The configuration requires crossover and mutation probabilities to sum to at most one.

Evolution uses training data only. The best training expression found throughout the run is retained and evaluated on test data after evolution.

## 6. Experiments and tests

Run the automated tests with:

```sh
python -m pytest -q
```

The test modules address configuration, data loading, trees, initialization, fitness, selection, operators, evolution, metrics, experiment handling, reporting, and the CLI.

Runs are saved under `results/runs/`, including:

- `config.json`: effective parameters and random seed.
- `best_tree.json`: serialized expression tree.
- `history.csv`: generation-level statistics.
- `operators.csv`: variation events and offspring outcomes.
- `summary.json`: training/test performance and run metadata.

Reporting produces summary tables and figures for convergence, tree growth, and final training/test RMSE.

<!--
### Verified results
-->

## 7. Learning and limitations

The project connects evolutionary computation with experimental design, numerical robustness, and reproducibility.

- Tree depth and node count are distinct measures; both matter when investigating bloat.
- Duplicate counts identify structural repetition, not algebraic equivalence.
- A lower training error does not necessarily imply better generalization.
- Results depend on the primitive set, parameters, random seed, and numerical environment.
- Timing and generated run-directory identifiers can vary between executions.
