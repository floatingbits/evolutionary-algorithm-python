"""Specimen generator for job assignment problem."""

from evolutionary_algorithm.specimen import Specimen, SpecimenCollection
from evolutionary_algorithm.genotype import SymbolArrayGenotype
from evolutionary_algorithm.randomizer import random_int
from evolutionary_algorithm.examples.job_assignment.problem import Job
from evolutionary_algorithm.examples.job_assignment.phenotype import JobAssignmentPhenotype


def create_specimen_generator(jobs: list[Job], num_machines: int):
    """Create a specimen generator for the job assignment problem.

    Args:
        jobs: List of jobs
        num_machines: Number of machines

    Returns:
        A function that generates initial populations
    """

    def generate(
        count: int,
    ) -> SpecimenCollection[SymbolArrayGenotype[int], JobAssignmentPhenotype]:
        """Generate initial population with random machine assignments.

        Args:
            count: Number of specimens to generate

        Returns:
            Collection of specimens with random genotypes
        """
        specimens: list[Specimen[SymbolArrayGenotype[int], JobAssignmentPhenotype]] = []

        for _ in range(count):
            # Randomly assign each job to a machine
            assignments = [random_int(0, num_machines - 1) for _ in jobs]

            # Create genotype
            genotype = SymbolArrayGenotype(assignments)

            # Create specimen (phenotype and fitness computed during evolution)
            specimen = Specimen(genotype=genotype)
            specimens.append(specimen)

        return SpecimenCollection(specimens)

    return generate


__all__ = ["create_specimen_generator"]
