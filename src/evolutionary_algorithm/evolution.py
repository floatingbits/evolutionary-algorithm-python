"""Evolution module - core evolutionary algorithm orchestration."""

from typing import TypeVar, Generic
from evolutionary_algorithm.specimen import SpecimenCollection, Specimen
from evolutionary_algorithm.genotype import Genotype
from evolutionary_algorithm.phenotype import Phenotype, PhenotypeGenerator
from evolutionary_algorithm.evaluation import Evaluator, Fitness
from evolutionary_algorithm.selection import Selector
from evolutionary_algorithm.mutation import CollectionMutator
from evolutionary_algorithm.recombination import CollectionRecombinator
from evolutionary_algorithm.cleanup import Cleanup


G = TypeVar("G", bound=Genotype)
P = TypeVar("P", bound=Phenotype)


class Evolver(Generic[G, P]):
    """Main evolutionary algorithm orchestrator.

    Manages the complete evolution cycle: evaluation, selection, and replenishment
    through mutation and recombination.

    This replaces the PHP Evolver class and factory pattern - in Python we can
    just instantiate this directly without needing a factory.
    """

    def __init__(
        self,
        phenotype_generator: PhenotypeGenerator[G, P],
        evaluator: Evaluator[P],
        selector: Selector[G, P],
        mutators: list[CollectionMutator[G, P]] | None = None,
        recombinators: list[CollectionRecombinator[G, P]] | None = None,
        cleanup: Cleanup[G, P] | None = None,
    ):
        """Initialize evolver with all necessary components.

        Args:
            phenotype_generator: Converts genotypes to phenotypes
            evaluator: Evaluates phenotype fitness
            selector: Selects specimens for next generation
            mutators: List of mutation strategies (optional)
            recombinators: List of recombination strategies (optional)
            cleanup: Cleanup strategy for removing duplicates (optional)
        """
        self.phenotype_generator = phenotype_generator
        self.evaluator = evaluator
        self.selector = selector
        self.mutators = mutators or []
        self.recombinators = recombinators or []
        self.cleanup = cleanup

    def evolve(self, population: SpecimenCollection[G, P]) -> SpecimenCollection[G, P]:
        """Execute one evolution cycle.

        Args:
            population: Current generation

        Returns:
            Next generation
        """
        # 1. Evaluate all specimens
        self._evaluate_population(population)

        # 2. Select survivors
        survivors = self.selector(population)

        # 3. Replenish population
        target_size = len(population)
        new_generation = self._replenish(survivors, target_size)

        return new_generation

    def apply_cleanup(self, population: SpecimenCollection[G, P]) -> None:
        """Apply cleanup strategy to population.

        Args:
            population: Population to clean (modified in-place)
        """
        if self.cleanup:
            self.cleanup(population)

    def _evaluate_population(self, population: SpecimenCollection[G, P]) -> None:
        """Evaluate all specimens in population.

        Only evaluates specimens that don't already have a fitness value.
        """
        for specimen in population:
            if specimen.fitness is None:
                # Generate phenotype if needed
                if specimen.phenotype is None:
                    specimen.phenotype = self.phenotype_generator(specimen.genotype)

                # Evaluate fitness
                specimen.fitness = self.evaluator(specimen.phenotype)

    def _replenish(
        self,
        survivors: SpecimenCollection[G, P],
        target_size: int,
    ) -> SpecimenCollection[G, P]:
        """Replenish population to target size using mutation and recombination.

        Args:
            survivors: Selected specimens
            target_size: Desired population size

        Returns:
            Replenished population
        """
        new_population = SpecimenCollection(list(survivors))
        needed = target_size - len(survivors)

        if needed <= 0:
            return new_population

        # Calculate how many offspring each strategy should produce
        total_strategies = len(self.mutators) + len(self.recombinators)

        if total_strategies == 0:
            # No strategies - just return survivors
            return new_population

        per_strategy = needed // total_strategies
        remainder = needed % total_strategies

        # Apply mutators
        for i, mutator in enumerate(self.mutators):
            count = per_strategy + (1 if i < remainder else 0)
            offspring = mutator(survivors, count)
            new_population.extend(offspring)

        # Apply recombinators
        for i, recombinator in enumerate(self.recombinators):
            count = per_strategy + (1 if (i + len(self.mutators)) < remainder else 0)
            offspring = recombinator(survivors, count)
            new_population.extend(offspring)

        return new_population


class Tournament(Generic[G, P]):
    """Tournament-based evolution manager.

    Runs multiple rounds of evolution with optional periodic cleanup.

    This replaces the PHP DefaultTournament class.
    """

    def __init__(
        self,
        evolver: Evolver[G, P],
        population: SpecimenCollection[G, P],
        rounds: int = 100,
        cleanup_interval: int | None = None,
    ):
        """Initialize tournament.

        Args:
            evolver: The evolver to use
            population: Initial population
            rounds: Number of evolution rounds to run
            cleanup_interval: Run cleanup every N rounds (None = no cleanup)
        """
        self.evolver = evolver
        self.population = population
        self.rounds = rounds
        self.cleanup_interval = cleanup_interval

    def run(self) -> SpecimenCollection[G, P]:
        """Run the tournament.

        Returns:
            Final population after all rounds
        """
        for round_num in range(self.rounds):
            # Evolve one generation
            self.population = self.evolver.evolve(self.population)

            # Periodic cleanup
            if (
                self.cleanup_interval
                and (round_num + 1) % self.cleanup_interval == 0
            ):
                self.evolver.apply_cleanup(self.population)

        return self.population

    @property
    def best_specimen(self) -> Specimen[G, P] | None:
        """Get the best specimen from current population."""
        if len(self.population) == 0:
            return None

        self.population.sort_by_fitness(reverse=True)
        return self.population[0]


__all__ = ["Evolver", "Tournament"]
