"""Selection module for choosing specimens for next generation."""

from collections.abc import Callable
from typing import Protocol, TypeVar

from evolutionary_algorithm.evaluation import Fitness
from evolutionary_algorithm.genotype import Genotype
from evolutionary_algorithm.phenotype import Phenotype
from evolutionary_algorithm.specimen import Specimen, SpecimenCollection

G = TypeVar("G", bound=Genotype)
P = TypeVar("P", bound=Phenotype)


class Selector(Protocol[G, P]):
    """Protocol for selection strategies.

    Allows both classes and functions to act as selectors.
    """

    def __call__(self, collection: SpecimenCollection[G, P]) -> SpecimenCollection[G, P]:
        """Select specimens for the next generation.

        Args:
            collection: Current population

        Returns:
            Selected specimens
        """
        ...


class SimpleSelector:
    """Survival rate-based selection with optional duplicate removal.

    This is a concrete implementation that's commonly used.
    """

    def __init__(self, survival_rate: float, remove_duplicates: bool = True):
        """Initialize selector.

        Args:
            survival_rate: Fraction of population to keep (0.0 to 1.0)
            remove_duplicates: Whether to remove duplicate genotypes
        """
        if not 0.0 <= survival_rate <= 1.0:
            raise ValueError(f"Survival rate must be between 0 and 1, got {survival_rate}")

        self.survival_rate = survival_rate
        self.remove_duplicates = remove_duplicates

    def __call__(
        self, collection: SpecimenCollection[G, P]
    ) -> SpecimenCollection[G, P]:
        """Select top performers from the population."""
        # Sort by fitness (best first)
        collection.sort_by_fitness(reverse=True)

        # Calculate how many to keep
        keep_count = max(1, int(len(collection) * self.survival_rate))

        # Keep top specimens
        selected = SpecimenCollection(list(collection)[:keep_count])

        # Optionally remove duplicates
        if self.remove_duplicates:
            selected.remove_duplicates()

        return selected


class FeasibilityAwareSelector:
    """Selection with a protected minority lane for secondary-scale winners.

    Division of labour between two orderings of the fitness:

    * the **main ordering** (``specimen.fitness.value``) — used to fill
      ``feasible_quota`` of the survivor slots; solutions that the main
      scale heavily punishes (e.g. constraint violations) end up here only
      if they are genuinely good on the main scale;
    * the **secondary ordering** (caller-supplied ``secondary_key``) — an
      alternative, usually much milder scale (e.g. value with no or only
      slight constraint punishment). The *remaining* survivors are chosen
      exclusively among the specimens that the main selection already
      rejected, ranked by this key. This gives promising-but-not-yet-valid
      solutions a protected lane, while weak scatter is kept out.

    With a single ordering (``secondary_key`` = main value) or with all
    solutions valid, the behaviour is equivalent to ``SimpleSelector``.
    """

    def __init__(
        self,
        survival_rate: float,
        feasible_quota: float,
        secondary_key: Callable[[Fitness], float],
        remove_duplicates: bool = True,
    ):
        """Create the selector.

        Args:
            survival_rate: Fraction of the population to keep (0.0 to 1.0).
            feasible_quota: Share of survivor slots reserved for the main
                selection (0.0 to 1.0). The rest goes to the secondary lane.
            secondary_key: Extracts the alternative ordering value from a
                fitness object (e.g. a second dimension of a composite
                Fitness value class).
            remove_duplicates: Whether to remove duplicate genotypes.
        """
        if not 0.0 <= survival_rate <= 1.0:
            raise ValueError(f"Survival rate must be between 0 and 1, got {survival_rate}")
        if not 0.0 <= feasible_quota <= 1.0:
            raise ValueError(f"Feasible quota must be between 0 and 1, got {feasible_quota}")

        self.survival_rate = survival_rate
        self.feasible_quota = feasible_quota
        self.secondary_key = secondary_key
        self.remove_duplicates = remove_duplicates

    def __call__(
        self, collection: SpecimenCollection[G, P]
    ) -> SpecimenCollection[G, P]:
        """Select survivors by main ordering plus a secondary-lane minority."""
        keep_count = max(1, int(len(collection) * self.survival_rate))
        main_count = int(keep_count * self.feasible_quota)
        lane_count = keep_count - main_count

        # Main selection: top specimens by the primary ordering
        collection.sort_by_fitness(reverse=True)
        selected: list[Specimen[G, P]] = list(collection)[:main_count]
        rejected = list(collection)[main_count:]

        # Secondary lane: best of the *rejected*, by the secondary key
        rejected.sort(key=lambda s: self.secondary_key(s.fitness), reverse=True)
        selected.extend(rejected[:lane_count])

        selected_collection = SpecimenCollection(selected)

        # Optionally remove duplicates
        if self.remove_duplicates:
            selected_collection.remove_duplicates()

        return selected_collection


__all__ = ["Selector", "SimpleSelector", "FeasibilityAwareSelector"]
