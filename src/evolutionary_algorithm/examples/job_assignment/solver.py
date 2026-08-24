"""Complete solver setup for job assignment problem."""

from evolutionary_algorithm.evolution import Evolver
from evolutionary_algorithm.selection import SimpleSelector
from evolutionary_algorithm.mutation import SimpleSymbolArrayMutator, CollectionMutator
from evolutionary_algorithm.recombination import (
    SymbolArrayCrossoverRecombinator,
    CollectionRecombinator,
)
from evolutionary_algorithm.cleanup import remove_duplicates
from evolutionary_algorithm.randomizer import random_int
from evolutionary_algorithm.examples.job_assignment.problem import Job, create_example_problem
from evolutionary_algorithm.examples.job_assignment.phenotype import create_phenotype_generator
from evolutionary_algorithm.examples.job_assignment.evaluator import evaluate_job_assignment
from evolutionary_algorithm.examples.job_assignment.specimen_generator import (
    create_specimen_generator,
)


def create_job_assignment_solver(jobs: list[Job], num_machines: int) -> Evolver:
    """Create a configured evolver for the job assignment problem.

    This replicates the PHP EvolverFactory configuration with pythonic simplicity.

    Args:
        jobs: List of jobs to assign
        num_machines: Number of machines available

    Returns:
        Configured evolver ready to use
    """
    # Phenotype generator
    phenotype_gen = create_phenotype_generator(jobs, num_machines)

    # Evaluator
    evaluator = evaluate_job_assignment

    # Selector: 30% survival rate
    selector = SimpleSelector(survival_rate=0.3, remove_duplicates=True)

    # Mutators
    # Conservative swap mutation
    conservative_mutator = CollectionMutator(
        SimpleSymbolArrayMutator(
            mutation_rate=0.05,  # 5% mutation rate
            symbol_generator=lambda: random_int(0, num_machines - 1),
        )
    )

    # Creative swap mutation
    creative_mutator = CollectionMutator(
        SimpleSymbolArrayMutator(
            mutation_rate=0.10,  # 10% mutation rate
            symbol_generator=lambda: random_int(0, num_machines - 1),
        )
    )

    # Recombinator: 4-point crossover
    recombinator = CollectionRecombinator(
        SymbolArrayCrossoverRecombinator(crossover_points=4)
    )

    # Create evolver
    evolver = Evolver(
        phenotype_generator=phenotype_gen,
        evaluator=evaluator,
        selector=selector,
        mutators=[conservative_mutator, creative_mutator],
        recombinators=[recombinator],
        cleanup=remove_duplicates,
    )

    return evolver


def solve_example_problem():
    """Solve the example job assignment problem.

    This is a convenience function that sets up and returns everything needed.

    Returns:
        Tuple of (evolver, specimen_generator, jobs, num_machines)
    """
    jobs = create_example_problem()
    num_machines = 5

    evolver = create_job_assignment_solver(jobs, num_machines)
    specimen_gen = create_specimen_generator(jobs, num_machines)

    return evolver, specimen_gen, jobs, num_machines


__all__ = ["create_job_assignment_solver", "solve_example_problem"]
