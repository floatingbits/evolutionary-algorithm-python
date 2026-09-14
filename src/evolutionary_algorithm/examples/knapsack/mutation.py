"""Mutation operators specific to 0/1-Knapsack genotypes.

A knapsack genotype is a plain bit-string, so flipping bits never breaks the
representation. How much the weight constraint is enforced is governed by
the ``RepairPolicy`` the operator receives: with probability
``repair_rate`` the offspring is strictly repaired; otherwise it only gets
the relaxed repair (overweight clamped to the configured allowance) and may
stay slightly invalid — the "creative" children the selector's minority
lane can pick up (see ``FeasibilityAwareSelector``).
"""

from __future__ import annotations

import random

from evolutionary_algorithm.examples.knapsack.repair import (
    RepairPolicy,
)
from evolutionary_algorithm.genotype import SymbolArrayGenotype


class FlipItemMutator:
    """Bit-flip mutation for the knapsack bit-string.

    Selects one random item and toggles its inclusion/exclusion, then lets
    the repair policy decide how strictly the capacity is enforced.
    """

    def __init__(self, policy: RepairPolicy, probability: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self.policy = policy

    def __call__(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        if random.random() > self.probability:
            return genotype  # no mutation

        symbols = genotype.symbols
        n = len(symbols)
        i = random.randrange(n)
        symbols[i] = 1 - symbols[i]
        return self.policy(SymbolArrayGenotype(symbols))


class SwapInOutMutator:
    """Exchange mutation: trade a packed item for unpacked capacity.

    Removes one randomly chosen selected item and then fills the freed
    capacity with randomly chosen excluded items that still fit. Explores
    the classic knapsack neighbourhood (swap one in, one out); the repair
    policy decides how strictly the result stays valid.
    """

    def __init__(self, policy: RepairPolicy, probability: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self.policy = policy

    def __call__(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        if random.random() > self.probability:
            return genotype  # no mutation

        symbols = genotype.symbols
        selected = [i for i, bit in enumerate(symbols) if bit == 1]
        excluded = [i for i, bit in enumerate(symbols) if bit == 0]

        if selected:
            # Remove one random selected item to free capacity
            i = random.choice(selected)
            symbols[i] = 0

        # Try to add a few random excluded items (most will not fit)
        random.shuffle(excluded)
        for j in excluded[:3]:
            symbols[j] = 1

        return self.policy(SymbolArrayGenotype(symbols))


class GreedyFillMutator:
    """Greedy capacity-fill mutation.

    Keeps the genotype as is, then repeatedly adds random excluded items
    that fit into the remaining capacity. This biases the population toward
    "fully packed" solutions — important because an under-filled knapsack
    wastes capacity and undervalues the phenotype.
    """

    def __init__(self, policy: RepairPolicy, probability: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self.policy = policy
        self.repair = policy  # kept for symmetry with the other mutators
        item_by_id = policy.relaxed.item_by_id
        self.item_by_id = item_by_id
        self.capacity = policy.relaxed.capacity

    def __call__(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        if random.random() > self.probability:
            return genotype  # no mutation

        symbols = genotype.symbols
        weight = sum(
            self.item_by_id[i].weight for i, bit in enumerate(symbols) if bit == 1
        )
        excluded = [i for i, bit in enumerate(symbols) if bit == 0]
        random.shuffle(excluded)

        for j in excluded:
            w = self.item_by_id[j].weight
            if weight + w <= self.capacity:
                symbols[j] = 1
                weight += w

        return self.policy(SymbolArrayGenotype(symbols))


__all__ = ["FlipItemMutator", "SwapInOutMutator", "GreedyFillMutator"]
