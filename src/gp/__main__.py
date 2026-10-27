import argparse
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .data import load_dataset
from .experiments import plan_runs, run_once, run_plan
from .reporting import plot_runs, summarize_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genetic programming for symbolic regression")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one dataset with one random seed")
    run.add_argument("--dataset", choices=["synth1", "synth2", "concrete", "concrete1"], required=True)
    run.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    run.add_argument("--data-dir", type=Path, default=Path("bases-TP1/datasets"))
    run.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    for parameter in ("seed", "population-size", "generations", "max-depth", "min-initial-depth"):
        run.add_argument(f"--{parameter}", type=int)
    run.add_argument("--selection", choices=["tournament", "roulette", "epsilon_lexicase"])
    experiment = commands.add_parser("experiment", help="Run one stage of the experimental plan")
    experiment.add_argument("--plan", type=Path, default=Path("configs/experiments.toml"))
    experiment.add_argument("--stage", required=True)
    experiment.add_argument("--baseline", type=Path, help="Baseline selected for this dataset/stage")
    experiment.add_argument("--seeds", type=int, nargs="+")
    experiment.add_argument("--datasets", nargs="+", choices=["synth1", "synth2", "concrete", "concrete1"],
                            help="Restrict datasets to use the selected baseline for each problem")
    experiment.add_argument("--data-dir", type=Path, default=Path("bases-TP1/datasets"))
    experiment.add_argument("--output-dir", type=Path, default=Path("results/runs"))
    experiment.add_argument("--dry-run", action="store_true", help="Show run counts and configurations without running")
    report = commands.add_parser("analyze", help="Generate tables and figures without writing a report")
    report.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    report.add_argument("--summaries-dir", type=Path, default=Path("results/summaries"))
    report.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            overrides = {name: getattr(args, name) for name in
                         ("seed", "population_size", "generations", "max_depth", "min_initial_depth", "selection")
                         if getattr(args, name) is not None}
            config = replace(load_config(args.config), **overrides)
            directory = run_once(load_dataset(args.dataset, args.data_dir), config, args.output_dir)
            print(directory)
        elif args.command == "experiment":
            specs = plan_runs(args.plan, args.stage, args.seeds, args.baseline, args.datasets)
            print(f"Stage {args.stage}: {len(specs)} runs, {len({spec.configuration_id for spec in specs})} configurations")
            if args.dry_run:
                for dataset, identifier in sorted({(spec.dataset, spec.configuration_id) for spec in specs}):
                    print(f"  {dataset}: {identifier}")
            else:
                run_plan(specs, args.data_dir, args.output_dir)
        else:
            print(summarize_runs(args.runs_dir, args.summaries_dir))
            for figure in plot_runs(args.runs_dir, args.figures_dir):
                print(figure)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

