"""Evaluator for job assignment problem."""

from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.examples.job_assignment.phenotype import JobAssignmentPhenotype


def evaluate_job_assignment(phenotype: JobAssignmentPhenotype) -> Fitness:
    """Evaluate fitness of a job assignment.

    Goal: Minimize the maximum completion time (makespan).
    We negate the max time so higher fitness = better solution.

    Args:
        phenotype: The job assignment phenotype

    Returns:
        Fitness value (higher is better)
    """
    # Lower max time is better, so negate for fitness maximization
    return Fitness(-phenotype.max_time)


__all__ = ["evaluate_job_assignment"]
