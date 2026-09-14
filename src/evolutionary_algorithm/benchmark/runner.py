"""Benchmark runner: executes solver configurations on a shared scenario.

The runner reproduces the exact round semantics of ``Tournament`` (evolve,
then periodic cleanup) but executes its rounds one at a time so the
best-fitness history can be recorded at round granularity.

All solver configurations in a benchmark share the same problem instance
and are started from identically generated populations (the scenario's
specimen generator is deterministic under the caller's seeded random
state — see the example scripts).
"""

from __future__ import annotations

from evolutionary_algorithm.benchmark.model import (
    BenchmarkResult,
    BenchmarkScenario,
    SolverConfig,
    SolverRun,
)
from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.specimen import SpecimenCollection


class BenchmarkRunner:
    """Runs solver configurations against a benchmark scenario.

    One call produces one :class:`BenchmarkResult`; call the runner again
    with new configurations to extend a comparison, so that e.g.
    ``FractionOfBestAchieved`` targets cover the whole grid later on.
    """

    def run(
        self,
        scenario: BenchmarkScenario,
        configs: list[SolverConfig],
    ) -> BenchmarkResult:
        """Run every configuration on the scenario's problem.

        Args:
            scenario: The shared problem and protocol.
            configs: The solver configurations to compare.

        Returns:
            A result containing one ``SolverRun`` per configuration, with
            milestones resolved against the benchmark context (known
            optimum of the scenario, best value achieved by any config).
        """
        result = BenchmarkResult(scenario=scenario)

        for config in configs:
            result.runs.append(self._run_config(scenario, config))

        best_achieved = result.best_achieved()
        ground_truth = scenario.ground_truth

        for run in result.runs:
            run.milestones = self._resolve_milestones(
                scenario.targets, ground_truth, best_achieved, run.history
            )
        return result

    def _run_config(
        self,
        scenario: BenchmarkScenario,
        config: SolverConfig,
    ) -> SolverRun:
        """Execute one configuration and record its best-fitness history."""
        evolver: Evolver = config.evolver_factory()
        population: SpecimenCollection = scenario.specimen_generator()

        history: list[float] = []

        # Round 0: best of the initial population (not yet evolved)
        self._evaluate_population(evolver, population)
        best = population.get_best_fitness()
        history.append(best.value if best else float("-inf"))

        for round_num in range(1, scenario.iterations + 1):
            # Same order of operations as Tournament.run()
            population = evolver.evolve(population)
            if (
                scenario.cleanup_interval
                and round_num % scenario.cleanup_interval == 0
            ):
                evolver.apply_cleanup(population)

            best = population.get_best_fitness()
            history.append(best.value if best else float("-inf"))

        return SolverRun(config_name=config.name, history=history)

    def _evaluate_population(
        self,
        evolver: Evolver,
        population: SpecimenCollection,
    ) -> None:
        """Evaluate all specimens (mirrors ``Evolver._evaluate_population``)."""
        for specimen in population:
            if specimen.fitness is None:
                if specimen.phenotype is None:
                    specimen.phenotype = evolver.phenotype_generator(specimen.genotype)
                specimen.fitness = evolver.evaluator(specimen.phenotype)

    def _resolve_milestones(
        self,
        targets: list,
        ground_truth: float | None,
        best_achieved: float | None,
        history: list[float],
    ) -> dict[str, int | None]:
        """Compute rounds-needed-to-target for one run's history.

        A milestone value of ``r`` means the target was reached at the end
        of evolution round ``r`` (``0`` would mean it was already met by
        the initial population); ``None`` means it was never reached.
        """
        milestones: dict[str, int | None] = {}
        for target in targets:
            threshold = target.resolve(ground_truth, best_achieved)
            if threshold is None:
                milestones[target.label] = None
                continue
            reached: int | None = None
            for round_num, best in enumerate(history):
                if best >= threshold:
                    reached = round_num
                    break
            milestones[target.label] = reached
        return milestones


__all__ = ["BenchmarkRunner"]
