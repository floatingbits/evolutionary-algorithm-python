"""Feasibility repair for 0/1-Knapsack genotypes.

The knapsack capacity can be enforced strictly, or *mostly*: variation
operations may leave their offspring slightly overweight. Two knobs control
this (see :class:`RepairPolicy`):

* **Repair rate** — the share of operations that apply the strict repair.
  The remaining operations only apply the *relaxed* repair, which clamps
  the overweight to a configurable extra allowance instead of resolving it
  entirely.
* **Allowed overweight** — the extra weight (as a fraction of the capacity)
  that an unrepaired offspring may carry.

The repair itself un-selects items with the worst value-to-weight ratio
first — the least valuable use of the scarce capacity.
"""

from __future__ import annotations

import random

from evolutionary_algorithm.examples.knapsack.problem import Item
from evolutionary_algorithm.genotype import SymbolArrayGenotype


class KnapsackRepair:
    """Turns any bit-string genotype into one that respects a weight limit.

    While the selection is overweight, drop the currently selected item with
    the lowest value/weight ratio.
    """

    def __init__(self, items: list[Item], capacity: int, extra_overweight: int = 0):
        """Create a repair operator for the given problem instance.

        Args:
            items: The items of the problem instance.
            capacity: The strict weight limit of the knapsack.
            extra_overweight: Additional weight allowed beyond the capacity
                (0 = strict repair; the default). Used by the relaxed
                variant of the repair policy.
        """
        self.items = items
        self.capacity = capacity
        self.extra_overweight = extra_overweight
        self.limit = capacity + extra_overweight
        self.item_by_id = {item.id: item for item in items}

    def repair(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        """Return a repaired copy of ``genotype`` that respects the ``limit``."""
        symbols = genotype.symbols
        weight = sum(
            self.item_by_id[i].weight
            for i, bit in enumerate(symbols)
            if bit == 1
        )

        if weight > self.limit:
            # Selected items sorted by worst value-per-weight first
            selected = [i for i, bit in enumerate(symbols) if bit == 1]
            selected.sort(
                key=lambda i: self.item_by_id[i].value / self.item_by_id[i].weight
            )
            for i in selected:
                if weight <= self.limit:
                    break
                symbols[i] = 0
                weight -= self.item_by_id[i].weight

        return SymbolArrayGenotype(symbols)

    __call__ = repair


class RepairPolicy:
    """Decides per operation how (and how strictly) offspring are repaired.

    With probability ``repair_rate`` the offspring undergoes the **strict**
    repair (weight at or below capacity, i.e. fully valid); otherwise the
    **relaxed** repair is applied, which merely clamps overweight to
    ``max_overweight`` (a fraction of the capacity) — enough to keep
    "creative" children a defined elbow away, but not fully valid.
    """

    def __init__(
        self,
        items: list[Item],
        capacity: int,
        repair_rate: float = 1.0,
        max_overweight: float = 0.0,
    ):
        """Create a repair policy.

        Args:
            items: The items of the problem instance.
            capacity: The strict weight limit of the knapsack.
            repair_rate: Share of operations that repair strictly
                (1.0 = always, the classic behaviour; 0.4 = 60% of
                operations may leave their child overstuffed).
            max_overweight: Extra weight allowed beyond the capacity for
                unrepaired offspring, as a fraction of the capacity
                (0.05 = 5% overweight budget).
        """
        if not 0.0 <= repair_rate <= 1.0:
            raise ValueError("repair_rate must be between 0 and 1")
        if max_overweight < 0.0:
            raise ValueError("max_overweight must be non-negative")

        self.repair_rate = repair_rate
        self.max_overweight = max_overweight
        self.strict = KnapsackRepair(items, capacity)
        self.relaxed = KnapsackRepair(
            items, capacity, extra_overweight=int(round(capacity * max_overweight))
        )

    def apply(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        """Repair ``genotype`` according to the policy and the global stream."""
        if random.random() < self.repair_rate:
            return self.strict(genotype)
        return self.relaxed(genotype)

    __call__ = apply

    def strict_apply(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        """Always apply the strict repair (e.g. for the specimen generator)."""
        return self.strict(genotype)


__all__ = ["KnapsackRepair", "RepairPolicy"]
