"""Phenotype module for observable characteristics."""

from typing import Protocol, TypeVar, Callable
from evolutionary_algorithm.genotype import Genotype


class Phenotype(Protocol):
    """Protocol for phenotype representations.

    Phenotypes represent the "real-world interpretation" of genotypes
    that can be evaluated for fitness.
    """

    pass


G = TypeVar("G", bound=Genotype)
P = TypeVar("P", bound=Phenotype)


class PhenotypeGenerator(Protocol[G, P]):
    """Protocol for converting genotypes to phenotypes.

    In Python, we use a callable protocol instead of an interface,
    allowing both classes and functions to be used as generators.
    """

    def __call__(self, genotype: G) -> P:
        """Generate phenotype from genotype.

        Args:
            genotype: The genotype to convert

        Returns:
            The resulting phenotype
        """
        ...


# Type alias for simple function-based generators
PhenotypeGeneratorFunc = Callable[[G], P]


__all__ = ["Phenotype", "PhenotypeGenerator", "PhenotypeGeneratorFunc"]
