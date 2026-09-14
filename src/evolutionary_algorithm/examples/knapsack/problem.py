"""0/1-Knapsack problem definition and example data.

The 0/1-Knapsack problem is defined by a set of items, each with a weight and
a value, plus a strict capacity limit. The goal is to choose a subset of items
that maximises total value while the total weight must never exceed the
capacity.

This module also exposes a dynamic-programming ``compute_dp_optimum`` helper so
the example script can report how far the evolved solution is from the true
optimum.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Item:
    """An item that can be packed into the knapsack.

    Attributes
    ----------
    id: int
        Unique identifier (used as the index of the genotype bit).
    weight: int
        Weight the item contributes to the knapsack.
    value: int
        Value the item contributes to the objective.
    """

    id: int
    weight: int
    value: int

    def __repr__(self) -> str:  # pragma: no cover – simple debugging aid
        return f"Item(id={self.id}, weight={self.weight}, value={self.value})"


def compute_dp_optimum(items: List[Item], capacity: int) -> int:
    """Compute the optimal value of the 0/1-Knapsack instance by dynamic programming.

    Runs in O(len(items) * capacity) time and O(capacity) memory, which is fine
    for the example sizes used here.

    Args:
        items: List of items with integer weights and values.
        capacity: The strict weight limit.

    Returns:
        The maximum total value achievable without exceeding the capacity.
    """
    # dp[w] = best value achievable with weight budget of exactly w (monotone)
    dp = [0] * (capacity + 1)
    for item in items:
        w = item.weight
        if w <= capacity:
            # Iterate backwards so each item is used at most once
            for weight in range(capacity, w - 1, -1):
                candidate = dp[weight - w] + item.value
                if candidate > dp[weight]:
                    dp[weight] = candidate
    return dp[capacity]


def _make_fixed_problem() -> tuple[List[Item], int]:
    """Create a deterministic knapsack instance.

    The instance uses *strongly correlated* data (``value ≈ weight + small
    noise``), which is a classic hard configuration for subset-selection
    heuristics: value-per-weight ratios are almost uniform, so greedy sorting
    cannot land on the optimum and evolution has to make real decisions.
    The capacity is a tight fraction (~25%) of the total weight so the
    selection is highly constrained.
    """
    rng = random.Random(7)
    num_items = 100
    weights = [rng.randint(5, 94) for _ in range(num_items)]
    values = [weights[i] + rng.randint(0, 9) for i in range(num_items)]
    items = [Item(id=i, weight=weights[i], value=values[i]) for i in range(num_items)]
    total_weight = sum(weights)
    capacity = max(1, round(total_weight * 0.25))  # tight but non-trivial limit
    return items, capacity


def create_example_problem(seed: int | None = None) -> tuple[List[Item], int]:
    """Return ``(items, capacity)`` for the example knapsack.

    Args:
        seed: Optional seed. When ``None`` a fixed instance is returned;
            when given, a random strongly-correlated instance is generated
            from that seed.
    """
    if seed is not None:
        rng = random.Random(seed)
        num_items = rng.randint(60, 120)
        weights = [rng.randint(5, 94) for _ in range(num_items)]
        values = [weights[i] + rng.randint(0, 9) for i in range(num_items)]
        items = [Item(id=i, weight=weights[i], value=values[i]) for i in range(num_items)]
        total_weight = sum(weights)
        capacity = max(1, round(total_weight * 0.25))
        return items, capacity
    else:
        return _make_fixed_problem()


__all__ = ["Item", "create_example_problem", "compute_dp_optimum"]
