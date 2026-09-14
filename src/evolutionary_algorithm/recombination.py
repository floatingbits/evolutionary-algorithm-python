"""Recombination module for genetic crossover operations."""

import random
from typing import Generic, Protocol, TypeVar

from evolutionary_algorithm.genotype import Genotype, SymbolArrayGenotype
from evolutionary_algorithm.phenotype import Phenotype
from evolutionary_algorithm.specimen import Specimen, SpecimenCollection

G = TypeVar("G", bound=Genotype)
P = TypeVar("P", bound=Phenotype)
T = TypeVar("T")


class Recombinator(Protocol[G]):
    """Protocol for crossover/recombination operations.

    Allows both classes and functions to perform recombination.
    """

    def __call__(self, parent1: G, parent2: G) -> G:
        """Recombine two genotypes.

        Args:
            parent1: First parent genotype
            parent2: Second parent genotype

        Returns:
            Offspring genotype
        """
        ...


class SymbolArrayCrossoverRecombinator:
    """N-point crossover for array genotypes.

    Performs multi-point crossover, optionally considering parent fitness
    when selecting segments.
    """

    def __init__(
        self,
        crossover_points: int = 1,
        fitness_bias: bool = False,
    ):
        """Initialize recombinator.

        Args:
            crossover_points: Number of crossover points
            fitness_bias: If True, favor segments from fitter parent
        """
        if crossover_points < 1:
            raise ValueError(f"Need at least 1 crossover point, got {crossover_points}")

        self.crossover_points = crossover_points
        self.fitness_bias = fitness_bias

    def __call__(
        self,
        parent1: SymbolArrayGenotype[T],
        parent2: SymbolArrayGenotype[T],
    ) -> SymbolArrayGenotype[T]:
        """Perform crossover."""
        if len(parent1) != len(parent2):
            raise ValueError(
                f"Parents must have same length: {len(parent1)} != {len(parent2)}"
            )

        length = len(parent1)

        if length <= 1:
            return SymbolArrayGenotype(parent1.symbols)

        # Generate crossover points
        max_points = min(self.crossover_points, length - 1)
        points = sorted(random.sample(range(1, length), max_points))

        # Build offspring by alternating between parents
        symbols: list[T] = []
        current_parent = parent1
        last_point = 0

        for point in points + [length]:
            # Take segment from current parent
            symbols.extend(current_parent.symbols[last_point:point])

            # Switch parent for next segment
            current_parent = parent2 if current_parent is parent1 else parent1
            last_point = point

        return SymbolArrayGenotype(symbols)


class CollectionRecombinator(Generic[G, P]):
    """Applies recombination to a collection of specimens.

    This is a utility class for collection-level operations.
    """

    def __init__(self, recombinator: Recombinator[G]):
        """Initialize with a recombinator.

        Args:
            recombinator: The recombination function to apply
        """
        self.recombinator = recombinator

    def __call__(
        self,
        collection: SpecimenCollection[G, P],
        count: int,
    ) -> list[Specimen[G, P]]:
        """Generate offspring through recombination.

        Args:
            collection: Source population
            count: Number of offspring to generate

        Returns:
            List of new specimens with recombined genotypes
        """
        if len(collection) < 2:
            return []

        offspring: list[Specimen[G, P]] = []
        specimens = list(collection)

        for _ in range(count):
            # Pick two random parents
            parent1, parent2 = random.sample(specimens, 2)

            # Recombine genotypes
            child_genotype = self.recombinator(parent1.genotype, parent2.genotype)

            # Create new specimen
            offspring.append(Specimen(genotype=child_genotype))

        return offspring


__all__ = [
    "Recombinator",
    "SymbolArrayCrossoverRecombinator",
    "CollectionRecombinator",
]
