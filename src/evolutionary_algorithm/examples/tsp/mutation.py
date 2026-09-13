"""Mutation operators specific to permutation‑based problems like TSP.

The existing ``SwapSymbolArrayMutator`` only swaps two positions. For TSP a
more effective large‑scale move is 2‑opt: select two indices and reverse the
segment between them. This preserves the permutation constraint while
producing a substantial rearrangement of the tour.
"""

from __future__ import annotations

import random
from typing import Callable

from evolutionary_algorithm.mutation import Mutator
from evolutionary_algorithm.genotype import SymbolArrayGenotype


class TwoOptMutator:
    """2‑opt mutation for a permutation genotype.

    The mutation selects two distinct positions ``i < j`` and reverses the
    sub‑list ``symbols[i:j+1]``. This maintains a valid permutation (no
    duplicates, all cities present) and provides a larger neighbourhood than a
    simple swap.
    """

    def __init__(self, probability: float = 1.0):
        """Create the mutator.

        Args:
            probability: Probability of applying the 2‑opt move to a given
                genotype (default ``1.0`` – always apply). The surrounding
                ``CollectionMutator`` can repeat the mutator to generate many
                offspring, so a probability < 1.0 can be used to control the
                mutation rate.
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        self.probability = probability

    def __call__(self, genotype: SymbolArrayGenotype[int]) -> SymbolArrayGenotype[int]:
        """Apply a 2‑opt operation to ``genotype``.

        The operation is performed in‑place on a copy of the symbol list and
        returns a new ``SymbolArrayGenotype`` instance.
        """
        if random.random() > self.probability:
            return genotype  # no mutation

        symbols = genotype.symbols  # copy from property
        n = len(symbols)
        if n < 4:
            # Too few cities to meaningfully reverse a segment
            return genotype

        i, j = sorted(random.sample(range(n), 2))
        # Reverse the sub‑segment inclusive of both ends
        symbols[i : j + 1] = list(reversed(symbols[i : j + 1]))
        return SymbolArrayGenotype(symbols)


__all__ = ["TwoOptMutator"]
