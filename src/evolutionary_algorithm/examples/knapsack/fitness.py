"""Composite Fitness value class for the 0/1-Knapsack example.

A knapsack solution carries two scales:

* ``value`` (main/primary) — the fitness the framework everywhere sees and
  sorts by. Feasibility-dominant: valid solutions keep their packing value,
  overweight solutions are pushed onto a strictly lower, overweight-graded
  scale, so *any* valid solution ranks above *any* invalid one, and better
  (= less overweight) invalid solutions rank above worse ones.
* ``secondary`` (mild scale) — the packing value with only a slight
  punishment for overweight (configurable, may even be none). The
  ``FeasibilityAwareSelector`` uses this scale for its protected minority
  lane, so a solution that is slightly overweight but one item-swap away
  from being excellent remains attractive while naive/low-value material
  stays out.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class KnapsackFitness:
    """Composite fitness: feasibility-dominant primary + mild secondary.

    Ordering follows the primary ``value`` only (``secondary`` is excluded
    from comparisons so the main scale stays the framework's sorting metric
    and ``secondary`` is reserved for the selection lane).

    Attributes
    ----------
    value: float
        Primary scale: packing value for valid solutions, dominant penalty
        (proportional to the overweight) for invalid ones.
    secondary: float
        Mild scale: packing value with (at most) a slight overweight
        punishment. Used to rank the protected minority lane.
    """

    value: float
    secondary: float = field(default=0.0, compare=False)

    def __repr__(self) -> str:  # pragma: no cover – debugging convenience
        return f"KnapsackFitness(main={self.value:.2f}, secondary={self.secondary:.2f})"


__all__ = ["KnapsackFitness"]
