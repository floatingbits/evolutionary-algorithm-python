"""Permutation‑preserving recombination operators for TSP.

The generic ``SymbolArrayCrossoverRecombinator`` does not guarantee a valid
permutation (duplicate symbols may appear). For TSP we implement Order Crossover
(OX), which produces offspring that are valid permutations of the city ids.
"""

from __future__ import annotations

import random
from typing import List

from evolutionary_algorithm.genotype import SymbolArrayGenotype


class OrderCrossoverRecombinator:
    """Order Crossover (OX) for permutations.

    Given two parent permutations, OX copies a random slice from the first
    parent and preserves the relative order of the remaining symbols from the
    second parent. The result is always a valid permutation.
    """

    def __call__(self, parent1: SymbolArrayGenotype[int], parent2: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        size = len(parent1)
        if size != len(parent2):
            raise ValueError("Parents must have the same length for OX recombination")
        if size < 2:
            return SymbolArrayGenotype(parent1.symbols)

        # Choose a random crossover segment
        a, b = sorted(random.sample(range(size), 2))
        # Slice from parent1
        slice_segment = parent1.symbols[a : b + 1]

        # Build child by filling remaining positions with order from parent2
        child: List[int] = [None] * size
        # Insert the slice
        child[a : b + 1] = slice_segment

        # Fill the rest respecting parent2 order, skipping already used symbols
        p2_symbols = [sym for sym in parent2.symbols if sym not in slice_segment]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_symbols[idx]
                idx += 1

        return SymbolArrayGenotype(child)


__all__ = ["OrderCrossoverRecombinator"]
