"""Phenotype for job assignment problem."""

from dataclasses import dataclass

from evolutionary_algorithm.examples.job_assignment.problem import Job
from evolutionary_algorithm.genotype import SymbolArrayGenotype


@dataclass(frozen=True)
class JobAssignmentPhenotype:
    """Phenotype representing total processing time per machine.

    This is the "real-world interpretation" of the genotype that can be
    evaluated for fitness.
    """

    machine_times: list[float]  # Total time for each machine

    @property
    def max_time(self) -> float:
        """Get the maximum time across all machines (bottleneck)."""
        return max(self.machine_times) if self.machine_times else 0.0

    def __repr__(self) -> str:
        """String representation."""
        times_str = ", ".join(f"{t:.1f}" for t in self.machine_times)
        return f"JobAssignmentPhenotype(max={self.max_time:.1f}, times=[{times_str}])"


def create_phenotype_generator(jobs: list[Job], num_machines: int):
    """Create a phenotype generator for the job assignment problem.

    Args:
        jobs: List of jobs with processing times
        num_machines: Number of machines

    Returns:
        A function that converts genotypes to phenotypes
    """

    def generate(genotype: SymbolArrayGenotype[int]) -> JobAssignmentPhenotype:
        """Convert genotype (machine assignments) to phenotype (machine times).

        Args:
            genotype: Array where genotype[i] = machine assigned to job i

        Returns:
            Phenotype with total time per machine
        """
        # Initialize machine times
        machine_times = [0.0] * num_machines

        # Sum up processing times for each machine
        for job_idx, machine_id in enumerate(genotype.symbols):
            job = jobs[job_idx]
            machine_times[machine_id] += job.time_on_machine(machine_id)

        return JobAssignmentPhenotype(machine_times)

    return generate


__all__ = ["JobAssignmentPhenotype", "create_phenotype_generator"]
