"""Evaluators for the 0/1-Knapsack problem.

The framework maximises fitness and sorts by ``fitness.value``, so:

* :func:`evaluate_knapsack` — the simple, strictly-valid-only evaluator
  (requires an always-repaired population).
* :func:`create_dual_evaluator` — the composite evaluator that pairs with a
  slipped-repair population: it produces a
  :class:`~evolutionary_algorithm.examples.knapsack.fitness.KnapsackFitness`
  whose primary value is feasibility-dominant (any valid solution beats any
  invalid one on the main scale) and whose secondary value barely cares
  about overweight — the ordering used by the selector's minority lane.
"""

from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.examples.knapsack.fitness import KnapsackFitness
from evolutionary_algorithm.examples.knapsack.phenotype import KnapsackPhenotype


def evaluate_knapsack(phenotype: KnapsackPhenotype) -> Fitness:
    """Return fitness for a knapsack phenotype (strict, single scale).

    Feasible solutions get their total value. Infeasible ones (which the
    operators should already have repaired) get a large negative penalty,
    proportional to their overweight, so they can never be preferred to any
    feasible solution.
    """
    if not phenotype.feasible:
        # Penalty keeps invalid solutions below zero regardless of their value;
        # worse on the main scale than any feasible solution (value >= 0).
        return Fitness(-1_000_000.0 - float(phenotype.total_weight))
    return Fitness(float(phenotype.total_value))


def create_dual_evaluator(
    capacity: int,
    value_bound: float,
    secondary_overweight_penalty: float = 0.25,
) -> "callable":
    """Build a composite knapsack evaluator producing ``KnapsackFitness``.

    Args:
        capacity: The strict weight limit (to detect overweight).
        value_bound: An upper bound of the achievable value (e.g. the total
            value of all items). Used to keep invalid solutions strictly
            below every valid one on the primary scale.
        secondary_overweight_penalty: Weight units deducted from the
            secondary scale per unit of overweight (0 = fully blind to
            overweight).

    Returns:
        An evaluator callable ``phenotype -> KnapsackFitness``.
    """

    def evaluate(phenotype: KnapsackPhenotype) -> KnapsackFitness:
        overweight = max(0, phenotype.total_weight - capacity)
        if overweight > 0:
            # Primary: feasibility-dominant scale. Subtracting (bound +
            # overweight + 1) pushes the value strictly below zero — below
            # every feasible solution — and grades by overweight.
            main = float(phenotype.total_value) - (value_bound + overweight + 1.0)
            # Secondary: value with only a slight overweight penalty
            secondary = phenotype.total_value - secondary_overweight_penalty * overweight
        else:
            main = float(phenotype.total_value)
            secondary = main
        return KnapsackFitness(value=main, secondary=secondary)

    return evaluate


__all__ = ["evaluate_knapsack", "create_dual_evaluator", "Fitness"]
