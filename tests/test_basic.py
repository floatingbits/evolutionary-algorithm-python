"""Basic tests for the evolutionary algorithm framework."""

import pytest

from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.genotype import SymbolArrayGenotype
from evolutionary_algorithm.mutation import CollectionMutator, SimpleSymbolArrayMutator
from evolutionary_algorithm.recombination import (
    CollectionRecombinator,
    SymbolArrayCrossoverRecombinator,
)
from evolutionary_algorithm.selection import SimpleSelector
from evolutionary_algorithm.specimen import Specimen, SpecimenCollection


def test_genotype_creation():
    """Test genotype creation and basic operations."""
    genotype = SymbolArrayGenotype([1, 2, 3, 4, 5])
    assert len(genotype) == 5
    assert genotype[0] == 1
    assert genotype[4] == 5


def test_genotype_equality():
    """Test genotype equality comparison."""
    g1 = SymbolArrayGenotype([1, 2, 3])
    g2 = SymbolArrayGenotype([1, 2, 3])
    g3 = SymbolArrayGenotype([1, 2, 4])

    assert g1.equals(g2)
    assert g1 == g2
    assert not g1.equals(g3)
    assert g1 != g3


def test_fitness_ordering():
    """Test fitness comparison."""
    f1 = Fitness(10.0)
    f2 = Fitness(20.0)
    f3 = Fitness(10.0)

    assert f2 > f1
    assert f1 < f2
    assert f1 == f3


def test_specimen_collection():
    """Test specimen collection operations."""
    genotypes = [SymbolArrayGenotype([i]) for i in range(5)]
    specimens = [
        Specimen(genotype=g, fitness=Fitness(float(i)))
        for i, g in enumerate(genotypes)
    ]

    collection = SpecimenCollection(specimens)
    assert len(collection) == 5

    # Test sorting
    collection.sort_by_fitness(reverse=True)
    assert collection[0].fitness.value == 4.0
    assert collection[4].fitness.value == 0.0

    # Test best fitness
    assert collection.get_best_fitness() == Fitness(4.0)


def test_duplicate_removal():
    """Test duplicate genotype removal."""
    g1 = SymbolArrayGenotype([1, 2, 3])
    g2 = SymbolArrayGenotype([1, 2, 3])  # Duplicate
    g3 = SymbolArrayGenotype([4, 5, 6])

    specimens = [
        Specimen(genotype=g1, fitness=Fitness(1.0)),
        Specimen(genotype=g2, fitness=Fitness(2.0)),
        Specimen(genotype=g3, fitness=Fitness(3.0)),
    ]

    collection = SpecimenCollection(specimens)
    assert len(collection) == 3

    collection.remove_duplicates()
    assert len(collection) == 2


def test_simple_selector():
    """Test simple selection strategy."""
    genotypes = [SymbolArrayGenotype([i]) for i in range(10)]
    specimens = [
        Specimen(genotype=g, fitness=Fitness(float(i)))
        for i, g in enumerate(genotypes)
    ]

    collection = SpecimenCollection(specimens)
    selector = SimpleSelector(survival_rate=0.3)

    selected = selector(collection)
    assert len(selected) == 3

    # Check that top 3 were selected
    selected.sort_by_fitness(reverse=True)
    assert selected[0].fitness.value == 9.0
    assert selected[1].fitness.value == 8.0
    assert selected[2].fitness.value == 7.0


def test_mutation():
    """Test mutation operation."""
    genotype = SymbolArrayGenotype([0, 0, 0, 0, 0])

    mutator = SimpleSymbolArrayMutator(
        mutation_rate=1.0,  # Always mutate
        symbol_generator=lambda: 1,
    )

    mutated = mutator(genotype)
    assert all(s == 1 for s in mutated.symbols)
    assert genotype != mutated  # Original unchanged


def test_recombination():
    """Test crossover recombination."""
    parent1 = SymbolArrayGenotype([1, 1, 1, 1, 1])
    parent2 = SymbolArrayGenotype([2, 2, 2, 2, 2])

    recombinator = SymbolArrayCrossoverRecombinator(crossover_points=2)
    child = recombinator(parent1, parent2)

    # Child should have a mix of 1s and 2s
    assert len(child) == 5
    assert 1 in child.symbols
    assert 2 in child.symbols


def test_collection_mutator():
    """Test collection-level mutation."""
    genotypes = [SymbolArrayGenotype([0, 0, 0]) for _ in range(5)]
    specimens = [Specimen(genotype=g, fitness=Fitness(1.0)) for g in genotypes]
    collection = SpecimenCollection(specimens)

    mutator = SimpleSymbolArrayMutator(
        mutation_rate=1.0,
        symbol_generator=lambda: 1,
    )
    collection_mutator = CollectionMutator(mutator)

    offspring = collection_mutator(collection, count=3)
    assert len(offspring) == 3
    assert all(s.genotype != genotypes[0] for s in offspring)


def test_collection_recombinator():
    """Test collection-level recombination."""
    genotypes = [
        SymbolArrayGenotype([1, 1, 1]),
        SymbolArrayGenotype([2, 2, 2]),
    ]
    specimens = [Specimen(genotype=g, fitness=Fitness(1.0)) for g in genotypes]
    collection = SpecimenCollection(specimens)

    recombinator = SymbolArrayCrossoverRecombinator(crossover_points=1)
    collection_recombinator = CollectionRecombinator(recombinator)

    offspring = collection_recombinator(collection, count=5)
    assert len(offspring) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
