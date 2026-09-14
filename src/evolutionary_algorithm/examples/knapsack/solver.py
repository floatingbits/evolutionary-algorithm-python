"""Solver configuration for the 0/1-Knapsack problem.

Default configuration mirrors a strict solver: every operation repairs its
offspring strictly, all survivors are selected on the packing-value scale —
the constraint is always fulfilled.

Configurable *slippable* operation is optional: with ``repair_rate < 1``
some operations skip the strict repair (their children keep an overweight
of up to ``max_overweight`` × capacity), and the ``FeasibilityAwareSelector``
then reserves ``1 - feasible_quota`` of the survivor slots for the best of
the rejected specimens under the mild secondary scale of
``KnapsackFitness`` — the protected lane for promising-but-invalid children.
"""

from __future__ import annotations

from collections.abc import Callable

from evolutionary_algorithm.cleanup import remove_duplicates
from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.examples.knapsack.evaluator import (
    create_dual_evaluator,
    evaluate_knapsack,
)
from evolutionary_algorithm.examples.knapsack.mutation import (
    FlipItemMutator,
    GreedyFillMutator,
    SwapInOutMutator,
)
from evolutionary_algorithm.examples.knapsack.phenotype import create_phenotype_generator
from evolutionary_algorithm.examples.knapsack.problem import (
    Item,
    create_example_problem,
)
from evolutionary_algorithm.examples.knapsack.recombination import (
    UniformCrossoverRecombinator,
)
from evolutionary_algorithm.examples.knapsack.repair import RepairPolicy
from evolutionary_algorithm.examples.knapsack.specimen_generator import (
    create_specimen_generator,
)
from evolutionary_algorithm.mutation import CollectionMutator
from evolutionary_algorithm.recombination import CollectionRecombinator
from evolutionary_algorithm.selection import FeasibilityAwareSelector, SimpleSelector


def create_knapsack_solver(
    items: list[Item],
    capacity: int,
    *,
    repair_rate: float = 1.0,
    max_overweight: float = 0.0,
    feasible_quota: float = 0.9,
    secondary_overweight_penalty: float = 0.25,
    with_greedy_fill: bool = True,
) -> Evolver:
    """Create an ``Evolver`` configured for the knapsack.

    Strict mode (default, ``repair_rate=1.0``) guarantees validity with the
    single-scale evaluator and plain ``SimpleSelector``.

    Slipped mode (``repair_rate < 1`` and/or ``max_overweight > 0``) lets a
    configurable share of operations produce slightly overweight offspring
    (bounded by ``max_overweight`` as a fraction of the capacity): the
    evaluator then supplies a composite ``KnapsackFitness`` and the selector
    fills ``1 - feasible_quota`` of the survivor slots from the rejected
    specimens ranked by the mild secondary scale.

    Args:
        items: List of items defining the problem instance.
        capacity: The strict weight limit of the knapsack.
        repair_rate: Share of operations that apply the strict repair.
        max_overweight: Extra weight (fraction of capacity) allowed beyond
            the limit for offspring that skip the strict repair.
        feasible_quota: Share of survivor slots filled with the *main*
            selection (top solutions on the primary scale).
        secondary_overweight_penalty: Weight units the secondary scale
            deducts per unit of overweight (0 = fully blind to overweight).
        with_greedy_fill: Include the greedy capacity-fill mutator.
    """
    # Phenotype generator – converts a bit-string into weight/value/feasibility
    phenotype_gen = create_phenotype_generator(items, capacity)

    policy = RepairPolicy(
        items=items,
        capacity=capacity,
        repair_rate=repair_rate,
        max_overweight=max_overweight,
    )

    if repair_rate >= 1.0:
        # Strict mode: single-scale evaluator, plain selector
        evaluator = evaluate_knapsack
        selector = SimpleSelector(survival_rate=0.3, remove_duplicates=True)
    else:
        # Slipped mode: dual-scale evaluator with the secondary selection lane
        value_bound = float(sum(item.value for item in items))
        evaluator = create_dual_evaluator(
            capacity=capacity,
            value_bound=value_bound,
            secondary_overweight_penalty=secondary_overweight_penalty,
        )
        selector = FeasibilityAwareSelector(
            survival_rate=0.3,
            feasible_quota=feasible_quota,
            secondary_key=lambda fitness: fitness.secondary,
            remove_duplicates=True,
        )

    # Mutators – three knapsack-specific neighbourhood moves
    mutator_ops = [FlipItemMutator(policy), SwapInOutMutator(policy)]
    if with_greedy_fill:
        mutator_ops.append(GreedyFillMutator(policy))
    mutators = [CollectionMutator(m) for m in mutator_ops]

    # Recombination – uniform crossover with policy-controlled repair
    recombinator = CollectionRecombinator(UniformCrossoverRecombinator(policy))

    evolver = Evolver(
        phenotype_generator=phenotype_gen,
        evaluator=evaluator,
        selector=selector,
        mutators=mutators,
        recombinators=[recombinator],
        cleanup=remove_duplicates,
    )

    return evolver


def solve_example_problem(seed: int = 2024) -> tuple[Evolver, Callable, list[Item], int]:
    """Convenience function that builds the whole knapsack pipeline.

    Returns a tuple ``(evolver, specimen_generator, items, capacity)``.
    """
    items, capacity = create_example_problem(seed=seed)
    evolver = create_knapsack_solver(items, capacity)
    specimen_gen = create_specimen_generator(items, capacity)
    return evolver, specimen_gen, items, capacity


__all__ = ["create_knapsack_solver", "solve_example_problem"]
