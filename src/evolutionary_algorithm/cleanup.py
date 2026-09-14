"""Cleanup module for population maintenance."""

from typing import Protocol, TypeVar

from evolutionary_algorithm.genotype import Genotype
from evolutionary_algorithm.phenotype import Phenotype
from evolutionary_algorithm.specimen import SpecimenCollection

G = TypeVar("G", bound=Genotype)
P = TypeVar("P", bound=Phenotype)


class Cleanup(Protocol[G, P]):
    """Protocol for cleanup strategies.

    Allows both classes and functions to perform cleanup.
    """

    def __call__(self, collection: SpecimenCollection[G, P]) -> None:
        """Clean up a specimen collection in-place.

        Args:
            collection: The collection to clean
        """
        ...


def remove_duplicates(collection: SpecimenCollection[G, P]) -> None:
    """Remove specimens with duplicate genotypes.

    This is a simple function that can be used directly as a cleanup strategy.

    Args:
        collection: The collection to clean (modified in-place)
    """
    collection.remove_duplicates()


__all__ = ["Cleanup", "remove_duplicates"]
