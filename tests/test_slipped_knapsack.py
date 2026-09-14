"""Tests for slippable repair: KnapsackFitness, RepairPolicy, FeasibilityAwareSelector."""

from evolutionary_algorithm.examples.knapsack.evaluator import create_dual_evaluator
from evolutionary_algorithm.examples.knapsack.fitness import KnapsackFitness
from evolutionary_algorithm.examples.knapsack.problem import create_example_problem
from evolutionary_algorithm.examples.knapsack.repair import RepairPolicy
from evolutionary_algorithm.examples.knapsack.solver import create_knapsack_solver
from evolutionary_algorithm.examples.knapsack.specimen_generator import (
    create_specimen_generator,
)
from evolutionary_algorithm.genotype import SymbolArrayGenotype
from evolutionary_algorithm.selection import FeasibilityAwareSelector
from evolutionary_algorithm.specimen import Specimen, SpecimenCollection

ITEMS, CAPACITY = create_example_problem(seed=99)


def test_knapsack_fitness_primary_only_ordering():
    """Ordering of KnapsackFitness follows the primary value exclusively."""
    f_good_secondary = KnapsackFitness(value=100, secondary=2_000)
    f_top = KnapsackFitness(value=200, secondary=1_000)
    assert f_top > f_good_secondary       # primary decides, not the secondary
    assert f_top.value == 200


def test_dual_evaluator_validity_dominance():
    """Any valid solution outranks any invalid one on the main scale."""

    class _P:
        def __init__(self, weight, value):
            self.total_weight = weight
            self.total_value = value
            self.feasible = weight <= 1000

    evaluator = create_dual_evaluator(capacity=1000, value_bound=100_000.0)

    valid_med = _P(900, 100)            # mediocre valid solution
    invalid_star = _P(1200, 100_000)    # wildly valuable, grossly overweight
    invalid_slight = _P(1010, 90_000)   # slightly overweight, very valuable

    assert evaluator(valid_med).value > evaluator(invalid_star).value
    # Overweight is graded on the primary scale: less overweight ranks better
    assert evaluator(invalid_star).value > evaluator(invalid_slight).value
    # The secondary scale still makes the near-valid solution attractive
    assert evaluator(invalid_slight).secondary > evaluator(valid_med).secondary


def test_repair_policy_strict_and_relaxed_bounds():
    import random

    items = ITEMS[:60]
    rng = random.Random(1)

    strict_policy = RepairPolicy(items, CAPACITY, repair_rate=1.0)
    relaxed_policy = RepairPolicy(items, CAPACITY, repair_rate=0.0, max_overweight=0.05)
    allowance = int(round(CAPACITY * 0.05))

    for _ in range(30):
        symbols = [rng.randint(0, 1) for _ in range(len(items))]

        strict = strict_policy(SymbolArrayGenotype(symbols))
        weight = sum(items[i].weight for i, b in enumerate(strict.symbols) if b)
        assert weight <= CAPACITY    # strict repair always fully valid

        relaxed = relaxed_policy(SymbolArrayGenotype(symbols))
        weight = sum(items[i].weight for i, b in enumerate(relaxed.symbols) if b)
        assert weight <= CAPACITY + allowance


def test_feasibility_aware_selector_splits_lanes():
    """90% of survivors come from the main scale, 10% from the rejected's best secondary."""
    fitnesses = [
        KnapsackFitness(value=1000, secondary=1000),
        KnapsackFitness(value=900, secondary=900),
        KnapsackFitness(value=0, secondary=500),     # invalid but creative
        KnapsackFitness(value=-10, secondary=50),    # invalid and weak
        KnapsackFitness(value=800, secondary=800),
        KnapsackFitness(value=700, secondary=700),
        KnapsackFitness(value=-20, secondary=400),   # invalid, second-best creative
        KnapsackFitness(value=600, secondary=600),
    ]
    specimens = [
        Specimen(genotype=SymbolArrayGenotype([i])) for i in range(len(fitnesses))
    ]
    for specimen, fitness in zip(specimens, fitnesses):
        specimen.fitness = fitness

    selector = FeasibilityAwareSelector(
        survival_rate=0.5,           # keep 4 specimens
        feasible_quota=0.75,         # main lane: 3, secondary lane: 1
        secondary_key=lambda f: f.secondary,
        remove_duplicates=False,
    )
    selected = selector(SpecimenCollection(specimens))

    by_value = sorted(selected, key=lambda s: s.fitness.value, reverse=True)
    # Main lane: the top of the primary scale (1000, 900, 800)
    assert [s.fitness.value for s in by_value[:3]] == [1000, 900, 800]
    # The one lane slot goes to the best *secondary* among the rejected:
    # here the 4th-best valid specimen (secondary=700) beats the creative
    # invalid one (secondary=500) — the lane only excludes main selections.
    assert len(selected) == 4
    assert {s.fitness.value for s in selected} == {1000, 900, 800, 700}


def test_strict_mode_is_backward_compatible():
    """Default solver configuration keeps the constraint strictly fulfilled."""
    import random as _random

    _random.seed(3)
    items, capacity = create_example_problem(seed=5)
    evolver = create_knapsack_solver(items, capacity)  # all defaults = strict mode
    gen = create_specimen_generator(items, capacity)
    population = gen(10)

    import random

    from evolutionary_algorithm.evolution import Tournament

    random.seed(3)
    tournament = Tournament(evolver=evolver, population=population, rounds=3)
    final = tournament.run()

    for specimen in final:
        phenotype = evolver.phenotype_generator(specimen.genotype)
        assert phenotype.total_weight <= capacity       # always valid
