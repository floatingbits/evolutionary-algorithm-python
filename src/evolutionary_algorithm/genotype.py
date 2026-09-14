"""Genotype module for genetic representations."""

from typing import Generic, Protocol, TypeVar


class Genotype(Protocol):
    """Protocol for genetic representations.

    In Python, we use Protocol instead of a marker interface for duck typing.
    This allows any class with an equals method to act as a Genotype.
    """

    def equals(self, other: "Genotype") -> bool:
        """Check if this genotype is equal to another.

        Args:
            other: Another genotype to compare with

        Returns:
            True if genotypes are equal, False otherwise
        """
        ...


T = TypeVar("T")


class SymbolArrayGenotype(Generic[T]):
    """Array-based genotype using symbols of type T.

    This is a concrete implementation using generics for flexibility.
    """

    def __init__(self, symbols: list[T]):
        """Initialize with a list of symbols.

        Args:
            symbols: List of genetic symbols
        """
        self._symbols = symbols.copy()

    @property
    def symbols(self) -> list[T]:
        """Get a copy of the symbols."""
        return self._symbols.copy()

    def __len__(self) -> int:
        """Get the length of the genotype."""
        return len(self._symbols)

    def __getitem__(self, index: int) -> T:
        """Get symbol at specific position."""
        return self._symbols[index]

    def __setitem__(self, index: int, value: T) -> None:
        """Set symbol at specific position."""
        self._symbols[index] = value

    def equals(self, other: "Genotype") -> bool:
        """Check equality with another genotype."""
        if not isinstance(other, SymbolArrayGenotype):
            return False
        return self._symbols == other._symbols

    def __eq__(self, other: object) -> bool:
        """Python equality operator."""
        if not isinstance(other, SymbolArrayGenotype):
            return False
        return self.equals(other)

    def __hash__(self) -> int:
        """Make genotype hashable for use in sets/dicts."""
        return hash(tuple(self._symbols))

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"SymbolArrayGenotype({self._symbols})"


__all__ = ["Genotype", "SymbolArrayGenotype"]
