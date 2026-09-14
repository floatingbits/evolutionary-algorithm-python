"""Evaluator for the 0/1-Knapsack problem.

The framework maximises fitness, so we reward the total value of the packed
items. Because all variation operators in this example repair infeasible
offspring, the evaluator should only ever see feasible phenotypes; an
overweight solution is heavily penalised as a defensive fallback.
"""

from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.examples.knapsack.phenotype import KnapsackPhenotype


def evaluate_knapsack(phenotype: KnapsackPhenotype) -> Fitness:
    """Return fitness for a knapsack phenotype.

    Feasible solutions get their total value. Infeasible ones (which the
    operators should already have repaired) get a large negative penalty,
    proportional to their overweight, so they can never be preferred to any
    feasible solution.
    """
    if not phenotype.feasible:
        return Fitness(-1_000_000.0 - float(phenotype.total_weight))
    return Fitness(float(phenotype.total_value))


__all__ = ["evaluate_knapsack"]
