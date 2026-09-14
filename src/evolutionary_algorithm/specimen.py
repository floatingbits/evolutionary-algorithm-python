"""Specimen module for individual solutions in the population."""

from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Protocol, TypeVar

from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.genotype import Genotype
from evolutionary_algorithm.phenotype import Phenotype

G = TypeVar("G", bound=Genotype)
P = TypeVar("P", bound=Phenotype)


@dataclass
class Specimen(Generic[G, P]):
    """Individual solution combining genotype, phenotype, and fitness.

    Uses dataclass for clean, pythonic syntax instead of PHP's verbose
    getter/setter pattern.
    """

    genotype: G
    phenotype: P | None = None
    fitness: Fitness | None = None

    def __repr__(self) -> str:
        """String representation."""
        fitness_str = f"{self.fitness.value:.4f}" if self.fitness else "None"
        return f"Specimen(fitness={fitness_str})"


class SpecimenCollection(Generic[G, P]):
    """Collection of specimens with population management capabilities.

    This is a pythonic collection that implements iteration and standard
    container protocols instead of PHP's explicit Iterator interface.
    """

    def __init__(self, specimens: list[Specimen[G, P]] | None = None):
        """Initialize collection.

        Args:
            specimens: Initial list of specimens (optional)
        """
        self._specimens = specimens.copy() if specimens else []

    def add(self, specimen: Specimen[G, P]) -> None:
        """Add a specimen to the collection."""
        self._specimens.append(specimen)

    def extend(self, specimens: list[Specimen[G, P]]) -> None:
        """Add multiple specimens."""
        self._specimens.extend(specimens)

    def remove_duplicates(self) -> None:
        """Remove specimens with duplicate genotypes.

        Uses set-based deduplication for efficiency.
        """
        seen_genotypes: set[G] = set()
        unique_specimens: list[Specimen[G, P]] = []

        for specimen in self._specimens:
            if specimen.genotype not in seen_genotypes:
                seen_genotypes.add(specimen.genotype)
                unique_specimens.append(specimen)

        self._specimens = unique_specimens

    def sort_by_fitness(self, reverse: bool = True) -> None:
        """Sort specimens by fitness.

        Args:
            reverse: If True, sort descending (best first). Default True.
        """
        self._specimens.sort(
            key=lambda s: s.fitness.value if s.fitness else float("-inf"),
            reverse=reverse,
        )

    def get_best_fitness(self) -> Fitness | None:
        """Get the best fitness value in the collection."""
        if not self._specimens:
            return None

        best = max(
            (s.fitness for s in self._specimens if s.fitness is not None),
            default=None,
        )
        return best

    def __len__(self) -> int:
        """Get collection size."""
        return len(self._specimens)

    def __iter__(self) -> Iterator[Specimen[G, P]]:
        """Iterate over specimens."""
        return iter(self._specimens)

    def __getitem__(self, index: int) -> Specimen[G, P]:
        """Get specimen by index."""
        return self._specimens[index]

    def __repr__(self) -> str:
        """String representation."""
        return f"SpecimenCollection(size={len(self)}, best={self.get_best_fitness()})"


class SpecimenGenerator(Protocol[G, P]):
    """Protocol for generating initial populations.

    Allows both classes and functions to act as generators.
    """

    def __call__(self, count: int) -> SpecimenCollection[G, P]:
        """Generate a collection of specimens.

        Args:
            count: Number of specimens to generate

        Returns:
            Collection of generated specimens
        """
        ...


# Type alias for function-based generators
SpecimenGeneratorFunc = Callable[[int], SpecimenCollection[G, P]]


__all__ = [
    "Specimen",
    "SpecimenCollection",
    "SpecimenGenerator",
    "SpecimenGeneratorFunc",
]
