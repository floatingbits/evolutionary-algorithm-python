"""Evaluator for the Traveling Salesman Problem.

The objective is to minimise the total tour distance. The evolutionary
framework maximises fitness, so we return the negative distance as a
``Fitness`` value.
"""

from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.examples.tsp.phenotype import TSPPhenotype


def evaluate_tsp(phenotype: TSPPhenotype) -> Fitness:
    """Return fitness for a TSP phenotype.

    The lower the tour length, the higher the fitness (negative distance).
    """
    return Fitness(-phenotype.total_distance)


__all__ = ["evaluate_tsp"]
