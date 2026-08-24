"""Evaluation module for fitness assessment."""

from dataclasses import dataclass
from typing import Protocol, TypeVar
from evolutionary_algorithm.phenotype import Phenotype


@dataclass(frozen=True, order=True)
class Fitness:
    """Fitness value with numeric main fitness.

    Using dataclass with frozen=True and order=True gives us:
    - Immutability
    - Automatic comparison operators
    - Hash support
    - Clean syntax

    This replaces the PHP SimpleFitness class.
    """

    value: float

    def __repr__(self) -> str:
        """String representation."""
        return f"Fitness({self.value:.4f})"


P = TypeVar("P", bound=Phenotype)


class Evaluator(Protocol[P]):
    """Protocol for fitness evaluation.

    In Python, we use Protocol to allow both classes and callables
    to act as evaluators without forcing inheritance.
    """

    def __call__(self, phenotype: P) -> Fitness:
        """Evaluate a phenotype and return its fitness.

        Args:
            phenotype: The phenotype to evaluate

        Returns:
            The fitness value
        """
        ...


__all__ = ["Fitness", "Evaluator"]
