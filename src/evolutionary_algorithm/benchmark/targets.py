"""Target definitions for benchmark milestones.

A *target* is a fitness value a solver should reach; the benchmark answers
"how many evolution rounds did solver X need to get there?". Because the
right reference differs per problem and per intent, targets are resolved
through a flexible strategy (``TargetStrategy``):

* :class:`AbsoluteValue` — an arbitrary absolute value that matters to
  the user, independent of any other solver.
* :class:`FractionOfKnownOptimum` — a fraction of the *per problem* ground
  truth, when one is known.
* :class:`FractionOfBestAchieved` — a fraction of the best value any solver
  achieved on this problem within the same benchmark. Useful when no
  theoretical optimum is computable (e.g. the TSP): comparison stays
  relative between the configurations.

All strategies are resolved against two context values, so user code never
has to thread context manually: the scenario's ground truth and the best
value achieved across all runs of the benchmark.
"""

from __future__ import annotations

from typing import Protocol


class TargetStrategy(Protocol):
    """Protocol for target strategies.

    A strategy resolves itself to a concrete fitness value ("threshold")
    given the benchmark context, or returns ``None`` if it cannot be
    resolved (e.g. a fraction of a ground truth that was never provided).
    """

    label: str

    def resolve(
        self,
        ground_truth: float | None,
        best_achieved: float | None,
    ) -> float | None:
        """Return the concrete fitness threshold of this target."""
        ...


class AbsoluteValue:
    """An arbitrary absolute fitness value that matters to the user.

    The value is fixed and per problem — no normalisation happens.
    """

    def __init__(self, value: float, label: str | None = None):
        self.value = value
        self.label = label if label is not None else f"abs={value:g}"

    def resolve(
        self,
        ground_truth: float | None,
        best_achieved: float | None,
    ) -> float | None:
        return float(self.value)


class FractionOfKnownOptimum:
    """Fraction of the problem's *known optimum*.

    The reference comes from the benchmark scenario's ``ground_truth``
    field — a fact of the problem, not of the solvers being compared.
    """

    def __init__(self, fraction: float):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction
        self.label = f"{fraction:.0%} of optimum"

    def resolve(
        self,
        ground_truth: float | None,
        best_achieved: float | None,
    ) -> float | None:
        if ground_truth is None:
            return None
        return ground_truth * self.fraction


class FractionOfBestAchieved:
    """A fraction of the best value any solver achieved on this problem.

    The reference is the best fitness across *all configurations in the
    same benchmark run* — a fair relative yardstick when no theoretical
    optimum is available.
    """

    def __init__(self, fraction: float):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction
        self.label = f"{fraction:.0%} of best achieved"

    def resolve(
        self,
        ground_truth: float | None,
        best_achieved: float | None,
    ) -> float | None:
        if best_achieved is None:
            return None
        return best_achieved * self.fraction


__all__ = [
    "TargetStrategy",
    "AbsoluteValue",
    "FractionOfKnownOptimum",
    "FractionOfBestAchieved",
]
