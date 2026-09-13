"""Phenotype for the Traveling Salesman Problem.

The phenotype holds the total tour distance and (optionally) the visit order.
The fitness evaluator will negate the distance because the evolutionary
framework maximises fitness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from evolutionary_algorithm.genotype import SymbolArrayGenotype
from evolutionary_algorithm.examples.tsp.problem import City


@dataclass(frozen=True)
class TSPPhenotype:
    """Phenotype representing a tour.

    Attributes
    ----------
    total_distance: float
        The length of the tour (including the return edge).
    order: List[int]
        The permutation of city ids representing the visitation order.
    """

    total_distance: float
    order: List[int]

    def __repr__(self) -> str:  # pragma: no cover – debugging convenience
        return f"TSPPhenotype(distance={self.total_distance:.2f}, order={self.order})"


def create_phenotype_generator(cities: List[City]):
    """Return a function that converts a genotype into a ``TSPPhenotype``.

    The genotype is a ``SymbolArrayGenotype[int]`` where ``symbols`` is a
    permutation of city ids. The generator computes the Euclidean tour length
    by summing the distances between consecutive cities and finally adding the
    distance from the last back to the first city.
    """

    # Map id → City for O(1) lookup
    city_by_id = {city.id: city for city in cities}

    def generate(genotype: SymbolArrayGenotype[int]) -> TSPPhenotype:
        order = genotype.symbols
        if not order:
            return TSPPhenotype(total_distance=0.0, order=[])

        # Compute total distance by walking the permutation circularly
        distance = 0.0
        for i in range(len(order)):
            a = city_by_id[order[i]]
            b = city_by_id[order[(i + 1) % len(order)]]  # wrap‑around
            distance += a.distance_to(b)

        return TSPPhenotype(total_distance=distance, order=order)

    return generate


__all__ = ["TSPPhenotype", "create_phenotype_generator"]
