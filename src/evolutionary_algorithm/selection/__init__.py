"""Selection module for choosing specimens for next generation."""

from typing import Protocol, TypeVar
from evolutionary_algorithm.specimen import SpecimenCollection, Specimen
from evolutionary_algorithm.genotype import Genotype
from evolutionary_algorithm.phenotype import Phenotype


G = TypeVar("G", bound=Genotype)
P = TypeVar("P", bound=Phenotype)


class Selector(Protocol[G, P]):
    """Protocol for selection strategies.

    Allows both classes and functions to act as selectors.
    """

    def __call__(self, collection: SpecimenCollection[G, P]) -> SpecimenCollection[G, P]:
        """Select specimens for the next generation.

        Args:
            collection: Current population

        Returns:
            Selected specimens
        """
        ...


class SimpleSelector:
    """Survival rate-based selection with optional duplicate removal.

    This is a concrete implementation that's commonly used.
    """

    def __init__(self, survival_rate: float, remove_duplicates: bool = True):
        """Initialize selector.

        Args:
            survival_rate: Fraction of population to keep (0.0 to 1.0)
            remove_duplicates: Whether to remove duplicate genotypes
        """
        if not 0.0 <= survival_rate <= 1.0:
            raise ValueError(f"Survival rate must be between 0 and 1, got {survival_rate}")

        self.survival_rate = survival_rate
        self.remove_duplicates = remove_duplicates

    def __call__(
        self, collection: SpecimenCollection[G, P]
    ) -> SpecimenCollection[G, P]:
        """Select top performers from the population."""
        # Sort by fitness (best first)
        collection.sort_by_fitness(reverse=True)

        # Calculate how many to keep
        keep_count = max(1, int(len(collection) * self.survival_rate))

        # Keep top specimens
        selected = SpecimenCollection(list(collection)[:keep_count])

        # Optionally remove duplicates
        if self.remove_duplicates:
            selected.remove_duplicates()

        return selected


__all__ = ["Selector", "SimpleSelector"]
