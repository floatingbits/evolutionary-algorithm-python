"""Specimen generator for the 0/1-Knapsack example.

Each specimen receives a bit-string genotype. To keep the initial population
feasible, every genotype is built by taking items in random order and packing
each one while its weight still fits into the remaining capacity — this is
essentially a randomised greedy heuristic, so initial solutions are already
feasible, diverse and of decent quality.
"""

from __future__ import annotations

import random
from typing import Callable, List

from evolutionary_algorithm.examples.knapsack.phenotype import KnapsackPhenotype
from evolutionary_algorithm.examples.knapsack.problem import Item
from evolutionary_algorithm.genotype import SymbolArrayGenotype
from evolutionary_algorithm.specimen import Specimen, SpecimenCollection


def create_specimen_generator(
    items: List[Item], capacity: int
) -> Callable[[int], SpecimenCollection]:
    """Return a function that creates an initial population of feasible packings.

    Args:
        items: The items of the problem instance.
        capacity: The strict weight limit of the knapsack.

    Returns:
        A callable ``generate(count)`` that returns a ``SpecimenCollection``.
    """

    def generate(count: int) -> SpecimenCollection[SymbolArrayGenotype[int], KnapsackPhenotype]:
        specimens: List[Specimen[SymbolArrayGenotype[int], None]] = []
        n = len(items)
        for _ in range(count):
            # Random packing order, greedily fill while capacity lasts
            order = list(range(n))
            random.shuffle(order)
            symbols = [0] * n
            weight = 0
            for idx in order:
                w = items[idx].weight
                if weight + w <= capacity:
                    symbols[idx] = 1
                    weight += w
            genotype = SymbolArrayGenotype(symbols)
            specimens.append(Specimen(genotype=genotype))
        return SpecimenCollection(specimens)

    return generate


__all__ = ["create_specimen_generator"]
