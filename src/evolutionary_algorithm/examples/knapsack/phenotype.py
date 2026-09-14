"""Phenotype for the 0/1-Knapsack problem.

The phenotype holds the selected item ids, the resulting total weight and
total value, and whether the selection is feasible. The fitness evaluator
rewards total value while strictly respecting the weight limit.
"""

from __future__ import annotations

from dataclasses import dataclass

from evolutionary_algorithm.examples.knapsack.problem import Item
from evolutionary_algorithm.genotype import SymbolArrayGenotype


@dataclass(frozen=True)
class KnapsackPhenotype:
    """Phenotype representing one packing choice.

    Attributes
    ----------
    selected: List[int]
        Ids of the items packed into the knapsack.
    total_weight: int
        Sum of the weights of the selected items.
    total_value: int
        Sum of the values of the selected items.
    feasible: bool
        ``True`` if ``total_weight`` does not exceed the capacity.
    """

    selected: list[int]
    total_weight: int
    total_value: int
    feasible: bool

    def __repr__(self) -> str:  # pragma: no cover – debugging convenience
        return (
            f"KnapsackPhenotype(value={self.total_value}, "
            f"weight={self.total_weight}, feasible={self.feasible}, "
            f"items={self.selected})"
        )


def create_phenotype_generator(items: list[Item], capacity: int):
    """Return a function that converts a genotype into a ``KnapsackPhenotype``.

    The genotype is a ``SymbolArrayGenotype[int]`` of 0/1 symbols where position
    ``i`` marks whether ``items[i]`` is packed.
    """

    item_by_id = {item.id: item for item in items}

    def generate(genotype: SymbolArrayGenotype[int]) -> KnapsackPhenotype:
        selected = [idx for idx, bit in enumerate(genotype.symbols) if bit == 1]
        total_weight = sum(item_by_id[i].weight for i in selected)
        total_value = sum(item_by_id[i].value for i in selected)
        return KnapsackPhenotype(
            selected=selected,
            total_weight=total_weight,
            total_value=total_value,
            feasible=total_weight <= capacity,
        )

    return generate


__all__ = ["KnapsackPhenotype", "create_phenotype_generator"]
