"""Solver configuration for the Traveling Salesman Problem (TSP).

This mirrors the job‑assignment solver but uses a swap‑mutation that preserves
the permutation property of the genotype.
"""

from __future__ import annotations

from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.selection import SimpleSelector
from evolutionary_algorithm.mutation import (
    SwapSymbolArrayMutator,
    CollectionMutator,
)
from evolutionary_algorithm.examples.tsp.mutation import TwoOptMutator
from evolutionary_algorithm.examples.tsp.recombination import OrderCrossoverRecombinator

from evolutionary_algorithm.recombination import CollectionRecombinator
from evolutionary_algorithm.cleanup import remove_duplicates

from typing import List

from evolutionary_algorithm.examples.tsp.problem import City, create_example_problem
from evolutionary_algorithm.examples.tsp.phenotype import create_phenotype_generator
from evolutionary_algorithm.examples.tsp.evaluator import evaluate_tsp
from evolutionary_algorithm.examples.tsp.specimen_generator import (
    create_specimen_generator,
)


def create_tsp_solver(cities: List[City]) -> Evolver:
    """Create an ``Evolver`` configured for the TSP.

    Args:
        cities: List of ``City`` objects defining the problem instance.

    Returns:
        A ready‑to‑run ``Evolver`` instance.
    """
    num_cities = len(cities)

    # Phenotype generator – converts a permutation into a distance
    phenotype_gen = create_phenotype_generator(cities)

    # Evaluator – negative distance for maximisation
    evaluator = evaluate_tsp

    # Selector – keep 30% of the population, remove duplicates
    selector = SimpleSelector(survival_rate=0.3, remove_duplicates=True)

    # Mutator – swap two cities (preserves permutation) and 2‑opt mutation
    swap_mutator = CollectionMutator(SwapSymbolArrayMutator())
    two_opt_mutator = CollectionMutator(TwoOptMutator())

    # Recombination – Order Crossover (OX) for permutations
    recombinator = CollectionRecombinator(OrderCrossoverRecombinator())


    evolver = Evolver(
        phenotype_generator=phenotype_gen,
        evaluator=evaluator,
        selector=selector,
        mutators=[swap_mutator, two_opt_mutator],
        recombinators=[recombinator],
        cleanup=remove_duplicates,
    )

    return evolver


def solve_example_problem(seed: int | None = None) -> tuple[Evolver, callable, List[City]]:
    """Convenience function that builds the whole TSP pipeline.

    Returns a tuple ``(evolver, specimen_generator, cities)``.
    """
    # If no seed is provided, generate a random one for variability
    if seed is None:
        import random as _rand
        seed = _rand.randint(0, 1_000_000)
    cities = create_example_problem(seed=seed)
    evolver = create_tsp_solver(cities)
    specimen_gen = create_specimen_generator(len(cities))
    return evolver, specimen_gen, cities


__all__ = ["create_tsp_solver", "solve_example_problem"]
