"""Mutation module for genetic variation."""

from typing import Protocol, TypeVar, Callable, Generic
import random
from evolutionary_algorithm.genotype import Genotype, SymbolArrayGenotype
from evolutionary_algorithm.specimen import Specimen, SpecimenCollection
from evolutionary_algorithm.phenotype import Phenotype


G = TypeVar("G", bound=Genotype)
P = TypeVar("P", bound=Phenotype)
T = TypeVar("T")


class Mutator(Protocol[G]):
    """Protocol for mutation operations.

    Allows both classes and functions to perform mutations.
    """

    def __call__(self, genotype: G) -> G:
        """Mutate a genotype.

        Args:
            genotype: The genotype to mutate

        Returns:
            Mutated genotype (should be a new instance)
        """
        ...


class SimpleSymbolArrayMutator:
    """Probabilistic symbol replacement mutation for array genotypes.

    Replaces each symbol with a random alternative based on mutation rate.
    """

    def __init__(
        self,
        mutation_rate: float,
        symbol_generator: Callable[[], T],
    ):
        """Initialize mutator.

        Args:
            mutation_rate: Probability of mutating each position (0.0 to 1.0)
            symbol_generator: Function that generates random symbols
        """
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"Mutation rate must be between 0 and 1, got {mutation_rate}")

        self.mutation_rate = mutation_rate
        self.symbol_generator = symbol_generator

    def __call__(self, genotype: SymbolArrayGenotype[T]) -> SymbolArrayGenotype[T]:
        """Apply mutation to genotype."""
        symbols = genotype.symbols

        for i in range(len(symbols)):
            if random.random() < self.mutation_rate:
                symbols[i] = self.symbol_generator()

        return SymbolArrayGenotype(symbols)


class SwapSymbolArrayMutator:
    """Position-swapping mutation for array genotypes.

    Randomly swaps two positions in the array.
    """

    def __call__(self, genotype: SymbolArrayGenotype[T]) -> SymbolArrayGenotype[T]:
        """Apply swap mutation."""
        symbols = genotype.symbols

        if len(symbols) < 2:
            return SymbolArrayGenotype(symbols)

        # Pick two random positions
        i, j = random.sample(range(len(symbols)), 2)

        # Swap them
        symbols[i], symbols[j] = symbols[j], symbols[i]

        return SymbolArrayGenotype(symbols)


class CollectionMutator(Generic[G, P]):
    """Applies mutations to a collection of specimens.

    This is a utility class that wraps a mutator for collection-level operations.
    """

    def __init__(self, mutator: Mutator[G]):
        """Initialize with a mutator.

        Args:
            mutator: The mutation function to apply
        """
        self.mutator = mutator

    def __call__(
        self,
        collection: SpecimenCollection[G, P],
        count: int,
    ) -> list[Specimen[G, P]]:
        """Generate mutated specimens.

        Args:
            collection: Source population
            count: Number of mutants to generate

        Returns:
            List of new specimens with mutated genotypes
        """
        if len(collection) == 0:
            return []

        mutants: list[Specimen[G, P]] = []

        for _ in range(count):
            # Pick random specimen
            parent = random.choice(list(collection))

            # Mutate genotype
            mutated_genotype = self.mutator(parent.genotype)

            # Create new specimen (phenotype and fitness will be computed later)
            mutants.append(Specimen(genotype=mutated_genotype))

        return mutants


__all__ = [
    "Mutator",
    "SimpleSymbolArrayMutator",
    "SwapSymbolArrayMutator",
    "CollectionMutator",
]
