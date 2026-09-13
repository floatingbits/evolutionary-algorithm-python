import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
"""Example script for solving the Traveling Salesman Problem (TSP).

The script mirrors the job‑assignment example and demonstrates how to use the
evolutionary algorithm framework for a classic combinatorial optimisation
problem.
"""

from evolutionary_algorithm.evolution import Tournament
from evolutionary_algorithm.examples.tsp.solver import solve_example_problem


def main() -> None:
    """Run the TSP example.

    The function sets up the problem, creates an initial population, runs a
    tournament for a number of iterations and finally prints the best tour
    found.
    """
    print("=" * 70)
    print("Traveling Salesman Problem – Evolutionary Algorithm Example")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------------
    # Problem setup
    # ---------------------------------------------------------------------
    print("Setting up problem…")
    evolver, specimen_gen, cities = solve_example_problem()
    num_cities = len(cities)
    print(f"Number of cities: {num_cities}")
    print()

    # ---------------------------------------------------------------------
    # Initial population
    # ---------------------------------------------------------------------
    print("Generating initial population…")
    population_size = 60
    initial_population = specimen_gen(population_size)
    print(f"Population size: {population_size}")
    print()

    # Evaluate initial population for observability
    for specimen in initial_population:
        specimen.phenotype = evolver.phenotype_generator(specimen.genotype)
        specimen.fitness = evolver.evaluator(specimen.phenotype)

    print("Initial population (distance and order):")
    for idx, specimen in enumerate(initial_population, start=1):
        distance = -specimen.fitness.value if specimen.fitness else float('nan')
        order = specimen.phenotype.order if specimen.phenotype else []
        print(f"  {idx:2d}: Distance={distance:.2f}, Order={order}")
    print()

    # ---------------------------------------------------------------------
    # Tournament configuration
    # ---------------------------------------------------------------------
    rounds = 30
    iterations = 80
    cleanup_interval = 49  # same as job‑assignment example – periodic duplicate removal

    print("Configuring tournament…")
    print(f"Iterations: {iterations}")
    print(f"Rounds per iteration: {rounds}")
    print(f"Cleanup interval: every {cleanup_interval} rounds")
    print()

    # ---------------------------------------------------------------------
    # Evolution loop
    # ---------------------------------------------------------------------
    print("Starting evolution…")
    print("-" * 70)

    tournament = Tournament(
        evolver=evolver,
        population=initial_population,
        rounds=rounds,
        cleanup_interval=cleanup_interval,
    )

    for iteration in range(1, iterations + 1):
        final_pop = tournament.run()
        tournament.population = final_pop

        # Print best distance after each iteration
        best = tournament.best_specimen
        if best and best.fitness:
            distance = -best.fitness.value
            print(f"Iteration {iteration:3d}: Best distance = {distance:.2f}")

    print("-" * 70)
    print()

    # Capture final population for reporting
    final_population = tournament.population

    # ---------------------------------------------------------------------
    # Result reporting
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Full final population details
    # ---------------------------------------------------------------------
    print("\nFinal population (sorted by distance):")
    final_population.sort_by_fitness(reverse=True)
    for idx, specimen in enumerate(final_population, start=1):
        if specimen.fitness and specimen.phenotype:
            distance = -specimen.fitness.value
            order = specimen.phenotype.order
            print(f"  {idx:2d}: Distance={distance:.2f}, Order={order}")
    print()

    # Show top 5 solutions for quick reference
    print("Top 5 solutions:")
    print("=" * 70)
    for rank, specimen in enumerate(list(final_population)[:5], start=1):
        if specimen.fitness and specimen.phenotype:
            distance = -specimen.fitness.value
            order = specimen.phenotype.order
            print(f"\n#{rank} – Distance: {distance:.2f}")
            print("   Tour order:", " -> ".join(str(city_id) for city_id in order))
    print()
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
