import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
"""Example script: benchmark solver configurations on the knapsack problem.

Demonstrates the framework's benchmark mechanism: several ``Evolver``
configurations are run against the *same* problem instance (same seed →
same instance, same initial populations) and compared by

* the best value each configuration reached,
* relative to the known optimum (dynamic programming) and to the best
  achieved by any configuration,
* the number of evolution rounds needed to reach milestone targets —
  absolute or fractional references
  (see ``evolutionary_algorithm.benchmark``).
* one configuration uses the *slipped repair* variant: a configurable share
  of variation operations may leave slightly overweight children, and the
  ``FeasibilityAwareSelector`` fills a protected minority of the survivor
  slots from the best of the rejected specimens on a mild secondary scale.
"""

import argparse
import random

from evolutionary_algorithm.benchmark import (
    AbsoluteValue,
    BenchmarkRunner,
    BenchmarkScenario,
    FractionOfBestAchieved,
    FractionOfKnownOptimum,
    SolverConfig,
    render_report,
)
from evolutionary_algorithm.examples.knapsack.problem import (
    DEFAULT_SEED,
    compute_dp_optimum,
    create_example_problem,
)
from evolutionary_algorithm.examples.knapsack.solver import create_knapsack_solver
from evolutionary_algorithm.examples.knapsack.specimen_generator import (
    create_specimen_generator,
)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments of the benchmark example."""
    parser = argparse.ArgumentParser(description="Benchmark knapsack solver configurations.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for the problem instance and the evolution (reproducible).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Evolution rounds per configuration.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=80,
        help="Population size (held identical across configurations).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the knapsack benchmark example."""
    args = parse_args()
    random.seed(args.seed)  # one stream for instance and operators

    print("=" * 90)
    print("0/1-Knapsack Benchmark – comparing solver configurations")
    print("=" * 90)
    print(f"Seed: {args.seed} | rounds per run: {args.rounds} | "
          f"population: {args.population}")
    print()

    # ---------------------------------------------------------------------
    # Shared problem instance (same for every configuration)
    # ---------------------------------------------------------------------
    items, capacity = create_example_problem(seed=args.seed)
    optimum = compute_dp_optimum(items, capacity)
    print(f"Items: {len(items)} | capacity: {capacity} | "
          f"optimum (dynamic programming): {optimum}")
    print()

    create_specimen_generator(items, capacity)

    # ---------------------------------------------------------------------
    # Scenario with per-problem target definitions
    # ---------------------------------------------------------------------
    scenario = BenchmarkScenario(
        name=f"knapsack-hard-{args.seed}",
        specimen_generator=lambda: create_specimen_generator(items, capacity)(
            args.population
        ),
        iterations=args.rounds,
        ground_truth=float(optimum),
        # targets:
        # - an arbitrary absolute value that matters to the user,
        # - 90% of the known optimum (per problem reference),
        # - 95% of the best value any configuration achieves (relative),
        targets=[
            AbsoluteValue(0.9 * optimum, label="abs>=0.9·opt"),
            FractionOfKnownOptimum(0.90),
            FractionOfKnownOptimum(0.95),
            FractionOfBestAchieved(0.95),
        ],
    )

    # ---------------------------------------------------------------------
    # Solver configurations to compare (same problem, same seed)
    # ---------------------------------------------------------------------
    configs = [
        SolverConfig(
            "full operator set",
            lambda: create_knapsack_solver(items, capacity, with_greedy_fill=True),
        ),
        SolverConfig(
            "without greedy fill",
            lambda: create_knapsack_solver(items, capacity, with_greedy_fill=False),
        ),
        SolverConfig(
            "slipped: 60% repair, 5% overweight max, 10% overweight lane",
            lambda: create_knapsack_solver(
                items, capacity,
                repair_rate=0.6,          # 40% of operations skip strict repair
                max_overweight=0.05,      # ... and keep ≤ 5% overweight
                feasible_quota=0.9,       # 90% of survivors on the main scale,
                                          # 10% via the mild secondary lane
            ),
        ),
        SolverConfig(
            "slipped: 50% repair, 15% overweight max, 20% overweight lane",
            lambda: create_knapsack_solver(
                items, capacity,
                repair_rate=0.5,  # 50% of operations skip strict repair
                max_overweight=0.15,  # ... and keep ≤ 15% overweight
                feasible_quota=0.8,  # 80% of survivors on the main scale,
                # 20% via the mild secondary lane
            ),
        ),
        SolverConfig(
            "slipped: 10% repair, 5% overweight max, 40% overweight lane",
            lambda: create_knapsack_solver(
                items, capacity,
                repair_rate=0.1,  # 50% of operations skip strict repair
                max_overweight=0.05,  # ... and keep ≤ 15% overweight
                feasible_quota=0.6,  # 60% of survivors on the main scale,
                # 40% via the mild secondary lane
            ),
        ),
    ]

    runner = BenchmarkRunner()
    result = runner.run(scenario, configs)

    print()
    print(render_report(result))
    print()

    # Trajectories (best value per round) for the record
    print("Full histories (best value per round):")
    for run in result.runs:
        traj = ", ".join(f"{v:g}" for v in run.history[::20])
        print(f"  {run.config_name}: {traj} ...")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
