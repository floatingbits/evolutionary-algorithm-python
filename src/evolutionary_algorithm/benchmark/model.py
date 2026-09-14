"""Data model for benchmarks.

A *scenario* describes one problem instance plus the benchmark protocol
(number of evolution rounds, milestone targets, optional ground truth for
the problem). One scenario is run against several *solver configurations*;
all configurations get the same problem and the same seed, so differences
in the results come from the solver configuration alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from evolutionary_algorithm.benchmark.targets import TargetStrategy
from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.specimen import SpecimenCollection


@dataclass
class BenchmarkScenario:
    """One problem instance plus the protocol for benchmarking it.

    Attributes
    ----------
    name: str
        Human readable name of the scenario/problem.
    specimen_generator: Callable[[], SpecimenCollection]
        Returns a fresh initial population. Called once per solver run so
        every configuration starts from an identically sized, equally
        randomised population.
    iterations: int
        Number of evolution rounds the runner performs.
    ground_truth: Optional[float]
        Fitness value of the true/known optimum of the problem instance,
        when available (e.g. via dynamic programming for the knapsack).
        Optional — only targets referencing it need this.
    cleanup_interval: Optional[int]
        Apply the evolver cleanup every N rounds (mirrors ``Tournament``).
    """

    name: str
    specimen_generator: Callable[[], SpecimenCollection]
    iterations: int
    ground_truth: float | None = None
    cleanup_interval: int | None = None
    targets: list[TargetStrategy] = field(default_factory=list)


@dataclass
class SolverConfig:
    """One solver configuration to benchmark.

    Attributes
    ----------
    name: str
        Human readable name of the configuration.
    evolver_factory: Callable[[], Evolver]
        Builds a fresh ``Evolver`` for each run (an Evolver holds no
        population state, so a factory is used for clarity and future
        configurability).
    """

    name: str
    evolver_factory: Callable[[], Evolver]


@dataclass
class SolverRun:
    """Result of one solver configuration on one scenario.

    Attributes
    ----------
    config_name: str
        Name of the solver configuration that produced this run.
    history: List[float]
        Best fitness after each evolution round; ``history[0]`` is the best
        fitness of the *initial* population, so the list has
        ``iterations + 1`` entries.
    milestones: Dict[str, Optional[int]]
        For each target (keyed by label) the number of evolution rounds
        needed to first reach it. ``history``-index semantics: a value of
        ``r`` means the target was reached by the end of round ``r``; value
        ``None`` means the target was never reached.
    """

    config_name: str
    history: list[float] = field(default_factory=list)
    milestones: dict[str, int | None] = field(default_factory=dict)

    @property
    def best(self) -> float | None:
        """Best fitness over the whole run."""
        return max(self.history) if self.history else None


@dataclass
class BenchmarkResult:
    """Complete outcome of a benchmark: the scenario and all solver runs."""

    scenario: BenchmarkScenario
    runs: list[SolverRun] = field(default_factory=list)

    def best_achieved(self) -> float | None:
        """Best fitness over all runs — the "best achieved" reference."""
        values = [run.best for run in self.runs if run.best is not None]
        return max(values) if values else None


__all__ = ["BenchmarkScenario", "BenchmarkResult", "SolverConfig", "SolverRun"]
