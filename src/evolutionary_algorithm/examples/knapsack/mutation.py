"""Mutation operators specific to 0/1-Knapsack genotypes.

A knapsack genotype is a plain bit-string, so flipping bits never breaks the
representation. What it *can* break is the hard weight constraint — every
mutator in this module therefore passes its result through the
``KnapsackRepair`` repair operator, guaranteeing feasibility of every
offspring.
"""

from __future__ import annotations

import random

from evolutionary_algorithm.examples.knapsack.problem import Item
from evolutionary_algorithm.examples.knapsack.repair import KnapsackRepair
from evolutionary_algorithm.genotype import SymbolArrayGenotype


class FlipItemMutator:
    """Bit-flip mutation for the knapsack bit-string.

    Selects one random item and toggles its inclusion/exclusion, then repairs
    the genotype so the capacity is never exceeded.
    """

    def __init__(self, items: list[Item], capacity: int, probability: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self.repair = KnapsackRepair(items, capacity)

    def __call__(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        if random.random() > self.probability:
            return genotype  # no mutation

        symbols = genotype.symbols
        n = len(symbols)
        i = random.randrange(n)
        symbols[i] = 1 - symbols[i]
        return self.repair(SymbolArrayGenotype(symbols))


class SwapInOutMutator:
    """Exchange mutation: trade a packed item for unpacked capacity.

    Removes one randomly chosen selected item and then fills the freed
    capacity with randomly chosen excluded items that still fit. This
    explores the classic knapsack neighbourhood (swap one in, one out) while
    the final repair pass guarantees hard feasibility.
    """

    def __init__(self, items: list[Item], capacity: int, probability: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self.repair = KnapsackRepair(items, capacity)

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

        # Try to add a few random excluded items (most will not fit);
        # the repair pass fixes any residual overweight.
        random.shuffle(excluded)
        for j in excluded[:3]:
            symbols[j] = 1

        return self.repair(SymbolArrayGenotype(symbols))


class GreedyFillMutator:
    """Greedy capacity-fill mutation.

    Keeps the genotype as is, then repeatedly adds random excluded items that
    fit into the remaining capacity. This biases the population toward
    "fully packed" solutions — important because an under-filled knapsack
    wastes capacity and undervalues the phenotype.
    """

    def __init__(self, items: list[Item], capacity: int, probability: float = 1.0):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability
        self.capacity = capacity
        self.repair = KnapsackRepair(items, capacity)
        self.item_by_id = {item.id: item for item in items}

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

        return SymbolArrayGenotype(symbols)


__all__ = ["FlipItemMutator", "SwapInOutMutator", "GreedyFillMutator"]
