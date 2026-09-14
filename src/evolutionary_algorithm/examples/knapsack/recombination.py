"""Recombination operators for 0/1-Knapsack genotypes.

Uniform recombination is a classic choice for bit-string encodings: each gene
(included or not) is independently taken from either parent. Combining two
parents can produce an overweight child; the ``RepairPolicy`` decides whether
it is strictly repaired or merely clamped to the configured overweight
allowance (see the slippable-repair concept).
"""

from __future__ import annotations

import random

from evolutionary_algorithm.examples.knapsack.repair import RepairPolicy
from evolutionary_algorithm.genotype import SymbolArrayGenotype


class UniformCrossoverRecombinator:
    """Uniform crossover with policy-controlled feasibility repair.

    For every position, the offspring inherits the bit from parent1 or
    parent2 with 50/50 chance, then the genotype goes through the repair
    policy.
    """

    def __init__(self, policy: RepairPolicy):
        self.policy = policy

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
        return self.policy(SymbolArrayGenotype(symbols))


__all__ = ["UniformCrossoverRecombinator"]
