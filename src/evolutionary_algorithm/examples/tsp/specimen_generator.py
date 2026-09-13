"""Specimen generator for the TSP example.

A specimen consists of a genotype that is a permutation of city ids. The
generator creates a collection of random permutations.
"""

from __future__ import annotations

import random
from typing import List

from evolutionary_algorithm.genotype import SymbolArrayGenotype
from evolutionary_algorithm.specimen import Specimen, SpecimenCollection
from evolutionary_algorithm.examples.tsp.phenotype import TSPPhenotype


def create_specimen_generator(num_cities: int):
    """Return a function that creates an initial population of random tours.

    Args:
        num_cities: Number of cities (size of the permutation).

    Returns:
        A callable ``generate(count)`` that returns a ``SpecimenCollection``.
    """

    def generate(count: int) -> SpecimenCollection[SymbolArrayGenotype[int], 'TSPPhenotype']:
        specimens: List[Specimen[SymbolArrayGenotype[int], None]] = []
        base = list(range(num_cities))
        for _ in range(count):
            perm = base.copy()
            random.shuffle(perm)
            genotype = SymbolArrayGenotype(perm)
            specimens.append(Specimen(genotype=genotype))
        return SpecimenCollection(specimens)

    return generate


__all__ = ["create_specimen_generator"]
