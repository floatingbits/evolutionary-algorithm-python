"""Benchmark module for comparing solver configurations.

The module lets a user run several solver configurations (``Evolver``
factories) against the *same* problem instance and compare them by:

* the best fitness reached,
* the number of evolution rounds needed to reach milestone targets —
  either absolute values or fractions of a reference value,
* the full best-fitness trajectory over rounds.

Reference values are **per problem**: a scenario may declare a known
optimum (e.g. computed by an exact solver), or targets can be defined
against the best value achieved by any solver in the same benchmark, or
against an arbitrary absolute value that matters to the user.

Granularity is *evolution rounds* for the time being; benchmark users keep
the population size comparable between configurations.

Typical usage::

    scenario = BenchmarkScenario(
        name="knapsack-2024",
        specimen_generator=lambda: specimen_gen(80),
        ground_truth=6449,
        iterations=400,
        targets=[FractionOfKnownOptimum(0.9), FractionOfBestAchieved(0.95)],
    )

    runner = BenchmarkRunner()
    result = runner.run(scenario, [
        SolverConfig("default", lambda: evolver_a),
        SolverConfig("no-greedy-fill", lambda: evolver_b),
    ])
    print(render_report(result))
"""

from evolutionary_algorithm.benchmark.model import (
    BenchmarkResult,
    BenchmarkScenario,
    SolverConfig,
    SolverRun,
)
from evolutionary_algorithm.benchmark.report import render_report
from evolutionary_algorithm.benchmark.runner import BenchmarkRunner
from evolutionary_algorithm.benchmark.targets import (
    AbsoluteValue,
    FractionOfBestAchieved,
    FractionOfKnownOptimum,
    TargetStrategy,
)

__all__ = [
    "BenchmarkScenario",
    "BenchmarkResult",
    "SolverConfig",
    "SolverRun",
    "TargetStrategy",
    "AbsoluteValue",
    "FractionOfKnownOptimum",
    "FractionOfBestAchieved",
    "BenchmarkRunner",
    "render_report",
]
