"""Recombination operators for 0/1-Knapsack genotypes.

Uniform recombination is a classic choice for bit-string encodings: each gene
(fully packed or not) is independently taken from either parent. Because
combining two feasible parents can produce an overweight child, the result is
passed through the ``KnapsackRepair`` repair operator to keep the hard weight
constraint intact.
"""

from __future__ import annotations

import random

from evolutionary_algorithm.examples.knapsack.problem import Item
from evolutionary_algorithm.examples.knapsack.repair import KnapsackRepair
from evolutionary_algorithm.genotype import SymbolArrayGenotype


class UniformCrossoverRecombinator:
    """Uniform crossover with feasibility repair.

    For every position, the offspring inherits the bit from parent 1 or
    parent 2 with 50/50 chance, then the genotype is repaired to obey the
    strict weight limit.
    """

    def __init__(self, items: list[Item], capacity: int):
        self.repair = KnapsackRepair(items, capacity)

    def __call__(
        self,
        parent1: SymbolArrayGenotype[int],
        parent2: SymbolArrayGenotype[int],
    ) -> SymbolArrayGenotype[int]:
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length for uniform crossover")

        symbols = [
            p1 if random.random() < 0.5 else p2
            for p1, p2 in zip(parent1, parent2)
        ]
        return self.repair(SymbolArrayGenotype(symbols))


__all__ = ["UniformCrossoverRecombinator"]
