"""Feasibility repair for 0/1-Knapsack genotypes.

The knapsack capacity is a hard constraint: the weight limit must never be
exceeded. Rather than relying on penalised fitness only, every genetic
operator in this example repairs its output, so every individual in the
population is always feasible.

The repair removes overweight by un-selecting items with the worst
value-to-weight ratio first — the least valuable use of the scarce capacity.
"""

from __future__ import annotations

from evolutionary_algorithm.examples.knapsack.problem import Item
from evolutionary_algorithm.genotype import SymbolArrayGenotype


class KnapsackRepair:
    """Turns any bit-string genotype into a feasible one.

    Arguably the simplest strategy: while the selection is overweight, drop
    the currently selected item with the lowest value/weight ratio. Items are
    drawn in random order among ties so the operator stays stochastic.
    """

    def __init__(self, items: list[Item], capacity: int):
        """Create a repair operator for the given problem instance.

        Args:
            items: The items of the problem instance.
            capacity: The strict weight limit of the knapsack.
        """
        self.items = items
        self.capacity = capacity
        self.item_by_id = {item.id: item for item in items}

    def repair(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        """Return a repaired copy of ``genotype`` that respects the capacity."""
        symbols = genotype.symbols
        weight = sum(
            self.item_by_id[i].weight
            for i, bit in enumerate(symbols)
            if bit == 1
        )

        if weight > self.capacity:
            # Selected items sorted by worst value-per-weight first
            selected = [
                i for i, bit in enumerate(symbols) if bit == 1
            ]
            selected.sort(
                key=lambda i: (
                    self.item_by_id[i].value / self.item_by_id[i].weight
                )
            )
            for i in selected:
                if weight <= self.capacity:
                    break
                symbols[i] = 0
                weight -= self.item_by_id[i].weight

        return SymbolArrayGenotype(symbols)

    __call__ = repair


__all__ = ["KnapsackRepair"]
