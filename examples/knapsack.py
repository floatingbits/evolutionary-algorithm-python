import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
"""Example script for solving the 0/1-Knapsack problem.

The script mirrors the TSP example and demonstrates how to use the
evolutionary algorithm framework for the classic subset-selection problem:
fill a knapsack with a subset of items to maximise total value without ever
exceeding the strict weight limit (capacity).

The runtime output prints – after every iteration – how the best total value
evolves relative to the (dynamically computed) optimal value of the instance,
so the progress of the optimisation can be witnessed directly.
"""

from evolutionary_algorithm.evolution import Tournament
from evolutionary_algorithm.examples.knapsack.solver import solve_example_problem
from evolutionary_algorithm.examples.knapsack.problem import compute_dp_optimum


def main() -> None:
    """Run the 0/1-Knapsack example."""
    print("=" * 70)
    print("0/1-Knapsack Problem – Evolutionary Algorithm Example")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------------
    # Problem setup
    # ---------------------------------------------------------------------
    print("Setting up problem…")
    evolver, specimen_gen, items, capacity = solve_example_problem()
    num_items = len(items)
    total_weight = sum(item.weight for item in items)
    total_value = sum(item.value for item in items)

    print(f"Number of items: {num_items}")
    print(f"Capacity: {capacity}")
    print(f"Total weight of all items: {total_weight}")
    print(f"Total value of all items: {total_value}")
    print()

    # ---------------------------------------------------------------------
    # Reference optimum (dynamic programming over weight)
    # ---------------------------------------------------------------------
    optimum = compute_dp_optimum(items, capacity)
    print(f"Optimal value (exact, via dynamic programming): {optimum}")
    print()

    # ---------------------------------------------------------------------
    # Initial population
    # ---------------------------------------------------------------------
    print("Generating initial population…")
    population_size = 80
    initial_population = specimen_gen(population_size)
    print(f"Population size: {population_size}")
    print()

    # Evaluate initial population for observability
    for specimen in initial_population:
        specimen.phenotype = evolver.phenotype_generator(specimen.genotype)
        specimen.fitness = evolver.evaluator(specimen.phenotype)

    print("Initial population (value, weight and packing):")
    for idx, specimen in enumerate(initial_population, start=1):
        phenotype = specimen.phenotype
        print(
            f"  {idx:2d}: Value={phenotype.total_value}, "
            f"Weight={phenotype.total_weight}/{capacity}, "
            f"Items={phenotype.selected}"
        )
    print()

    # ---------------------------------------------------------------------
    # Tournament configuration
    # ---------------------------------------------------------------------
    rounds = 25
    iterations = 40
    cleanup_interval = 37  # periodic duplicate removal

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
    print(f"{'Iteration':>9} | {'Best value':>10} | {'Weight used':>11} | "
          f"{'vs. optimum':>11} | Gap")
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

        # Print best value after each iteration – witness the evolution!
        best = tournament.best_specimen
        if best and best.fitness and best.phenotype:
            value = best.phenotype.total_value
            weight = best.phenotype.total_weight
            ratio = value / optimum if optimum else 0.0
            gap = optimum - value
            print(
                f"{iteration:9d} | {value:10d} | {weight:6d}/{capacity:<4d} | "
                f"{ratio:10.1%} | {gap}"
            )

    print("-" * 70)
    print()

    # Capture final population for reporting
    final_population = tournament.population

    # ---------------------------------------------------------------------
    # Result reporting
    # ---------------------------------------------------------------------
    print("\nFinal population (sorted by value):")
    final_population.sort_by_fitness(reverse=True)
    for idx, specimen in enumerate(final_population, start=1):
        if specimen.fitness and specimen.phenotype:
            phenotype = specimen.phenotype
            print(
                f"  {idx:2d}: Value={phenotype.total_value}, "
                f"Weight={phenotype.total_weight}/{capacity}, "
                f"Items={phenotype.selected}"
            )
    print()

    # Show top 3 solutions for quick reference
    print("Top 3 solutions:")
    print("=" * 70)
    for rank, specimen in enumerate(list(final_population)[:3], start=1):
        if specimen.fitness and specimen.phenotype:
            phenotype = specimen.phenotype
            print(f"\n#{rank} – Value: {phenotype.total_value} "
                  f"(Weight {phenotype.total_weight}/{capacity})")
            print("   Packed item ids:", ", ".join(map(str, phenotype.selected)))
    print()

    best_value = max(
        (s.phenotype.total_value for s in final_population if s.phenotype),
        default=0,
    )
    print(f"Optimal value was {optimum}; best evolved value is {best_value} "
          f"({best_value / optimum:.1%} of the optimum)." if optimum else "")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
