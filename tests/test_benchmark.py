"""Tests for the benchmark module (rounds-history + milestones + report)."""

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
from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.genotype import SymbolArrayGenotype
from evolutionary_algorithm.mutation import (
    CollectionMutator,
    SimpleSymbolArrayMutator,
)
from evolutionary_algorithm.recombination import (
    CollectionRecombinator,
    SymbolArrayCrossoverRecombinator,
)
from evolutionary_algorithm.selection import SimpleSelector
from evolutionary_algorithm.specimen import Specimen, SpecimenCollection

LENGTH = 12  # bit-string length of the toy problem


class _Phenotype:
    """Phenotype of the toy problem: vector of bits (its count is the value)."""

    def __init__(self, n_ones: int):
        self.n_ones = n_ones


def _toy_evolver(length: int) -> Evolver:
    """Evolver maximizing the number of ones in a bit string."""
    return Evolver(
        phenotype_generator=lambda g: _Phenotype(sum(g.symbols)),
        evaluator=lambda p: Fitness(float(p.n_ones)),
        selector=SimpleSelector(survival_rate=0.3),
        mutators=[CollectionMutator(SimpleSymbolArrayMutator(0.2, lambda: random.randint(0, 1)))],
        recombinators=[
            CollectionRecombinator(SymbolArrayCrossoverRecombinator(crossover_points=2))
        ],
        cleanup=remove_duplicates,
    )


def _toy_generator(length: int, size: int):
    def generate() -> SpecimenCollection:
        specimens = [
            Specimen(
                genotype=SymbolArrayGenotype(
                    [random.randint(0, 1) for _ in range(length)]
                )
            )
            for _ in range(size)
        ]
        return SpecimenCollection(specimens)

    return generate


def test_benchmark_history_length_and_best():
    random.seed(11)
    length = 12
    scenario = BenchmarkScenario(
        name="toy-ones",
        specimen_generator=_toy_generator(length, size=20),
        iterations=30,
    )
    runner = BenchmarkRunner()
    result = runner.run(
        scenario,
        [SolverConfig("toy", lambda: _toy_evolver(length))],
    )

    run = result.runs[0]
    # Round 0 (initial population) + one entry per evolution round
    assert len(run.history) == scenario.iterations + 1
    assert run.best == max(run.history)
    # Monotone: history never decreases (population retains best via selection)
    assert run.history == sorted(run.history)


def test_absolute_target_milestone():
    run_threshold = 5  # absolute value that the toy problem should reach quickly
    scenario = BenchmarkScenario(
        name="toy-abs",
        specimen_generator=_toy_generator(LENGTH, size=40),
        iterations=50,
        targets=[AbsoluteValue(run_threshold)],
    )
    random.seed(23)
    result = BenchmarkRunner().run(scenario, [SolverConfig("toy", lambda: _toy_evolver(LENGTH))])
    run = result.runs[0]
    rounds = run.milestones["abs=5"]
    assert rounds is not None
    # milestone semantics: round index whose history entry first met the target
    assert run.history[rounds] >= run_threshold
    assert all(v < run_threshold for v in run.history[:rounds])


def test_fraction_of_known_optimum_unresolved_without_truth():
    target = FractionOfKnownOptimum(0.9)
    assert target.resolve(ground_truth=None, best_achieved=100) is None
    assert target.resolve(ground_truth=200, best_achieved=100) == 180


def test_fraction_of_best_achieved():
    target = FractionOfBestAchieved(0.5)
    assert target.resolve(ground_truth=None, best_achieved=100) == 50
    assert target.resolve(ground_truth=None, best_achieved=None) is None


def test_report_lists_all_configs_and_targets():
    random.seed(31)
    scenario = BenchmarkScenario(
        name="toy-report",
        specimen_generator=_toy_generator(LENGTH, size=40),
        iterations=30,
        ground_truth=float(LENGTH),
        targets=[FractionOfKnownOptimum(0.9), FractionOfBestAchieved(0.9)],
    )
    configs = [
        SolverConfig("alpha", lambda: _toy_evolver(LENGTH)),
        SolverConfig("beta", lambda: _toy_evolver(LENGTH)),
    ]
    result = BenchmarkRunner().run(scenario, configs)
    text = render_report(result)

    assert "toy-report" in text
    assert "alpha" in text and "beta" in text
    assert "90% of optimum" in text
    assert "90% of best achieved" in text
