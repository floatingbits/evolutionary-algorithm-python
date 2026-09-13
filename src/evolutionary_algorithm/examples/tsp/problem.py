"""Traveling Salesman Problem (TSP) definition and example data.

The TSP is defined by a set of cities, each with 2‑D coordinates. The goal is to
find a permutation of the cities that minimizes the total tour length (including
the return from the last city to the first).

The framework expects a simple immutable data class and a function that returns
an example instance that can be used by the example script.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class City:
    """A city with a unique id and Cartesian coordinates.

    The id is used by the genotype (a permutation of ids). The ``distance_to``
    method returns Euclidean distance to another city.
    """

    id: int
    x: float
    y: float

    def distance_to(self, other: "City") -> float:
        """Euclidean distance between this city and ``other``."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:  # pragma: no cover – simple debugging aid
        return f"City(id={self.id}, x={self.x:.1f}, y={self.y:.1f})"


def _make_fixed_cities() -> List[City]:
    """Create a deterministic set of cities.

    The coordinates are hard‑coded so the example is reproducible without
    needing a random seed. The set contains 10 cities arranged roughly in a
    circle, which makes the optimal tour easy to reason about while still
    providing a non‑trivial search space.
    """
    # Pre‑computed coordinates for 10 cities (radius ~50, centre (50, 50))
    radius = 45.0
    centre = (50.0, 50.0)
    cities: List[City] = []
    for i in range(10):
        angle = 2 * math.pi * i / 10
        x = centre[0] + radius * math.cos(angle)
        y = centre[1] + radius * math.sin(angle)
        cities.append(City(id=i, x=x, y=y))
    return cities


def create_example_problem(seed: int | None = None) -> List[City]:
    """Return a list of ``City`` objects for the example TSP.

    The optional ``seed`` makes the output deterministic when random data is
    desired. If ``seed`` is ``None`` the fixed 10‑city circle is returned.
    """
    if seed is not None:
        random.seed(seed)
        # Generate a random number of cities between 8 and 15
        num = random.randint(80, 150)
        cities = []
        for i in range(num):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            cities.append(City(id=i, x=x, y=y))
        return cities
    else:
        return _make_fixed_cities()


__all__ = ["City", "create_example_problem"]
