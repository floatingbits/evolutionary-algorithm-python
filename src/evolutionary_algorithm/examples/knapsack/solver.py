"""Solver configuration for the 0/1-Knapsack problem.

This mirrors the TSP solver but is tailored to the bit-string encoding: a
uniform-crossover recombinator plus three knapsack-specific mutators, all of
which repair their output so the strict weight limit is never exceeded.
"""

from __future__ import annotations

from typing import List, Tuple

from evolutionary_algorithm.cleanup import remove_duplicates
from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.examples.knapsack.evaluator import evaluate_knapsack
from evolutionary_algorithm.examples.knapsack.mutation import (
    FlipItemMutator,
    GreedyFillMutator,
    SwapInOutMutator,
)
from evolutionary_algorithm.examples.knapsack.phenotype import create_phenotype_generator
from evolutionary_algorithm.examples.knapsack.problem import Item, create_example_problem
from evolutionary_algorithm.examples.knapsack.recombination import (
    UniformCrossoverRecombinator,
)
from evolutionary_algorithm.examples.knapsack.specimen_generator import (
    create_specimen_generator,
)
from evolutionary_algorithm.mutation import CollectionMutator
from evolutionary_algorithm.recombination import CollectionRecombinator
from evolutionary_algorithm.selection import SimpleSelector


def create_knapsack_solver(items: List[Item], capacity: int) -> Evolver:
    """Create an ``Evolver`` configured for the knapsack.

    Args:
        items: List of items defining the problem instance.
        capacity: The strict weight limit of the knapsack.

    Returns:
        A ready-to-run ``Evolver`` instance.
    """
    # Phenotype generator – converts a bit-string into weight/value/feasibility
    phenotype_gen = create_phenotype_generator(items, capacity)

    # Evaluator – total value; infeasible solutions are heavily penalised
    evaluator = evaluate_knapsack

    # Selector – keep 30% of the population, remove duplicates
    selector = SimpleSelector(survival_rate=0.3, remove_duplicates=True)

    # Mutators – three knapsack-specific neighbourhood moves
    mutators = [
        CollectionMutator(FlipItemMutator(items, capacity)),
        CollectionMutator(SwapInOutMutator(items, capacity)),
        CollectionMutator(GreedyFillMutator(items, capacity)),
    ]

    # Recombination – uniform crossover with feasibility repair
    recombinator = CollectionRecombinator(UniformCrossoverRecombinator(items, capacity))

    evolver = Evolver(
        phenotype_generator=phenotype_gen,
        evaluator=evaluator,
        selector=selector,
        mutators=mutators,
        recombinators=[recombinator],
        cleanup=remove_duplicates,
    )

    return evolver


def solve_example_problem(seed: int | None = None) -> Tuple[Evolver, callable, List[Item], int]:
    """Convenience function that builds the whole knapsack pipeline.

    Returns a tuple ``(evolver, specimen_generator, items, capacity)``.
    """
    items, capacity = create_example_problem(seed=seed)
    evolver = create_knapsack_solver(items, capacity)
    specimen_gen = create_specimen_generator(items, capacity)
    return evolver, specimen_gen, items, capacity


__all__ = ["create_knapsack_solver", "solve_example_problem"]
