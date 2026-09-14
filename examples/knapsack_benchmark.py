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
* the number of evolution rounds (sequence: 1 catch: evolve) needed to
  reach milestone targets — absolute or fractional references
  (see ``evolutionary_algorithm.benchmark``).
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
from evolutionary_algorithm.cleanup import remove_duplicates
from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.examples.knapsack.evaluator import evaluate_knapsack
from evolutionary_algorithm.examples.knapsack.mutation import (
    FlipItemMutator,
    GreedyFillMutator,
    SwapInOutMutator,
)
from evolutionary_algorithm.examples.knapsack.phenotype import (
    create_phenotype_generator,
)
from evolutionary_algorithm.examples.knapsack.problem import (
    DEFAULT_SEED,
    Item,
    compute_dp_optimum,
    create_example_problem,
)
from evolutionary_algorithm.examples.knapsack.recombination import (
    UniformCrossoverRecombinator,
)
from evolutionary_algorithm.examples.knapsack.specimen_generator import (
    create_specimen_generator,
)
from evolutionary_algorithm.mutation import CollectionMutator
from evolutionary_algorithm.recombination import CollectionRecombinator
from evolutionary_algorithm.selection import SimpleSelector


def build_evolver(items: list[Item], capacity: int, *, with_greedy_fill: bool = True) -> Evolver:
    """Build a knapsack solver variant with a configurable operator set."""
    mutators = [FlipItemMutator(items, capacity), SwapInOutMutator(items, capacity)]
    if with_greedy_fill:
        mutators.append(GreedyFillMutator(items, capacity))

    evolver = Evolver(
        phenotype_generator=create_phenotype_generator(items, capacity),
        evaluator=evaluate_knapsack,
        selector=SimpleSelector(survival_rate=0.3, remove_duplicates=True),
        mutators=[CollectionMutator(m) for m in mutators],
        recombinators=[
            CollectionRecombinator(UniformCrossoverRecombinator(items, capacity))
        ],
        cleanup=remove_duplicates,
    )
    return evolver


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
            lambda: build_evolver(items, capacity, with_greedy_fill=True),
        ),
        SolverConfig(
            "without greedy fill",
            lambda: build_evolver(items, capacity, with_greedy_fill=False),
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
