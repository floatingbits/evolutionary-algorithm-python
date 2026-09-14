"""0/1-Knapsack problem definition and example data.

The 0/1-Knapsack problem is defined by a set of items, each with a weight and
a value, plus a strict capacity limit. The goal is to choose a subset of items
that maximises total value while the total weight must never exceed the
capacity.

This module generates a deliberately *hard* instance family:

* **Size classes with bulky items.** The heaviest items each take up
  25–50% of the container capacity themselves, while many small filler
  items exist. Filling the container exactly therefore requires
  conflicting decisions: taking a bulky item blocks several complementary
  combinations.
* **Strongly correlated values.** ``value = weight + small noise``. The
  value-per-weight ratios are almost uniform, so greedy sorting cannot land
  on the optimum and the achievable value depends on the exact fill level of
  the container — a subset-sum problem in disguise.
* **Scale.** With the default 1000 items the search space is 2^1000, far
  beyond naive search, and the exact optimum is still computable via
  dynamic programming to allow exact gap reporting.

Empirically, a simple evolutionary configuration plateaus around 94–95% of
the dynamic-programming optimum even after a full budget of iterations —
a gap that visibly rewards more sophisticated variation operators.

This module also exposes a dynamic-programming ``compute_dp_optimum`` helper
so scripts can report how far the evolved solution is from the true optimum.

Every instance is generated from an explicit integer ``seed``, so the
example is fully reproducible when the same seed is used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

#: Seed used by :func:`create_example_problem` when none is provided.
DEFAULT_SEED = 2024


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


def compute_dp_optimum(items: list[Item], capacity: int) -> int:
    """Compute the optimal value of the 0/1-Knapsack instance by dynamic programming.

    Runs in O(len(items) * capacity) time and O(capacity) memory, which is
    fine for the example sizes used here (1000 items * ~6000 capacity).

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


def _generate_instance(
    rng: random.Random,
    num_items: int = 1000,
    max_weight: int = 3000,
    capacity_scale: float = 2.0,
    value_noise: int = 9,
) -> tuple[list[Item], int]:
    """Generate a hard instance from the given (local) random stream.

    Args:
        rng: The random stream to draw from (kept local so instance generation
            does not disturb the global stream used by the operators).
        num_items: Number of items (default 1000 — the search space is 2^1000).
        max_weight: Upper bound for item weights.
        capacity_scale: Capacity as a fraction of ``max_weight``. 2.0 gives a
            tight but non-trivial fit: even the heaviest class items fit alone,
            and exact fill of the container is far from obvious.
        value_noise: Size of the small random noise added to the strongly
            correlated values.

    Returns:
        ``(items, capacity)``.
    """
    weights: list[int] = []
    values: list[int] = []

    for i in range(num_items):
        r = rng.random()
        if r < 0.20:
            # Bulky items: half to the full container's nominal size budget
            w = rng.randint(int(max_weight * 0.5), max_weight)
        elif r < 0.50:
            # Medium items
            w = rng.randint(int(max_weight * 0.2), int(max_weight * 0.5))
        else:
            # Many small filler items
            w = rng.randint(20, int(max_weight * 0.2))

        # Strongly correlated value: near-uniform ratios forbid greedy shortcuts
        v = w + rng.randint(0, value_noise)

        weights.append(w)
        values.append(v)

    capacity = max(1, round(max_weight * capacity_scale))
    items = [Item(id=i, weight=weights[i], value=values[i]) for i in range(num_items)]
    return items, capacity


def create_example_problem(seed: int = DEFAULT_SEED) -> tuple[list[Item], int]:
    """Return ``(items, capacity)`` for the example knapsack.

    The same seed always produces the same instance, so benchmark runs are
    reproducible.

    Args:
        seed: Integer seed (default ``DEFAULT_SEED``).
    """
    rng = random.Random(seed)
    return _generate_instance(rng)


__all__ = ["Item", "create_example_problem", "compute_dp_optimum", "DEFAULT_SEED"]
