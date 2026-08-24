#!/usr/bin/env python3
"""Example: Job Assignment to Machines using Evolutionary Algorithm.

This example demonstrates how to use the evolutionary algorithm framework
to solve a job assignment optimization problem.

Problem: Assign 33 jobs to 5 machines to minimize the maximum completion time.
"""

from evolutionary_algorithm.evolution import Tournament
from evolutionary_algorithm.examples.job_assignment.solver import solve_example_problem


def main():
    """Run the job assignment example."""
    print("=" * 70)
    print("Job Assignment Optimization using Evolutionary Algorithm")
    print("=" * 70)
    print()

    # Setup the problem
    print("Setting up problem...")
    evolver, specimen_gen, jobs, num_machines = solve_example_problem()

    print(f"Jobs: {len(jobs)}")
    print(f"Machines: {num_machines}")
    print()

    # Generate initial population
    print("Generating initial population...")
    population_size = 50
    initial_population = specimen_gen(population_size)
    print(f"Population size: {population_size}")
    print()

    # Create tournament
    print("Configuring tournament...")
    rounds = 50
    iterations = 100
    cleanup_interval = 49

    print(f"Iterations: {iterations}")
    print(f"Rounds per iteration: {rounds}")
    print(f"Cleanup interval: every {cleanup_interval} rounds")
    print()

    # Run evolution
    print("Starting evolution...")
    print("-" * 70)

    tournament = Tournament(
        evolver=evolver,
        population=initial_population,
        rounds=rounds,
        cleanup_interval=cleanup_interval,
    )

    for iteration in range(1, iterations + 1):
        # Run one iteration
        final_population = tournament.run()

        # Reset tournament for next iteration
        tournament.population = final_population

        # Print progress every 10 iterations
        if iteration % 10 == 0:
            best = tournament.best_specimen
            if best and best.fitness:
                makespan = -best.fitness.value  # Negate back to original
                print(f"Iteration {iteration:3d}: Best makespan = {makespan:.1f}")

    print("-" * 70)
    print()

    # Show final results
    print("Evolution complete!")
    print()
    print("Top 7 Solutions:")
    print("=" * 70)

    final_population = tournament.population
    final_population.sort_by_fitness(reverse=True)

    for rank, specimen in enumerate(list(final_population)[:7], start=1):
        if specimen.fitness and specimen.phenotype:
            makespan = -specimen.fitness.value  # Negate back to original
            machine_times = specimen.phenotype.machine_times

            print(f"\n#{rank} - Makespan: {makespan:.1f}")
            print("    Machine times:", end="")
            for machine_id, time in enumerate(machine_times):
                print(f" M{machine_id}={time:.1f}", end="")
            print()

            # Show which jobs are assigned where
            assignments = specimen.genotype.symbols
            for machine_id in range(num_machines):
                job_ids = [
                    job_idx + 1
                    for job_idx, assigned_machine in enumerate(assignments)
                    if assigned_machine == machine_id
                ]
                if job_ids:
                    print(f"    Machine {machine_id}: jobs {job_ids}")

    print()
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
