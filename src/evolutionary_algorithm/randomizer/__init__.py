"""Randomizer module for random value generation.

Python's built-in random module is sufficient for most cases, so we simplify
the PHP randomizer hierarchy significantly.
"""

import random
import string


def random_int(min_val: int, max_val: int) -> int:
    """Generate random integer in range [min_val, max_val].

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Random integer
    """
    return random.randint(min_val, max_val)


def random_float(min_val: float, max_val: float) -> float:
    """Generate random float in range [min_val, max_val].

    Args:
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Random float
    """
    return random.uniform(min_val, max_val)


def random_bool(probability: float = 0.5) -> bool:
    """Generate random boolean with given probability.

    Args:
        probability: Probability of returning True (default 0.5)

    Returns:
        Random boolean
    """
    return random.random() < probability


def random_string(length: int, charset: str = string.ascii_letters) -> str:
    """Generate random string from charset.

    Args:
        length: Length of string to generate
        charset: Character set to choose from (default: a-z, A-Z)

    Returns:
        Random string
    """
    return "".join(random.choice(charset) for _ in range(length))


def random_choice(items: list):
    """Choose random item from list.

    Args:
        items: List to choose from

    Returns:
        Random item
    """
    return random.choice(items)


__all__ = [
    "random_int",
    "random_float",
    "random_bool",
    "random_string",
    "random_choice",
]
