"""Random helper utilities.

Everything should flow through a local random.Random instance so that generated
systems are reproducible from a seed.
"""

from __future__ import annotations

import random
from typing import Iterable, TypeVar

T = TypeVar("T")


def create_rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def weighted_choice(rng: random.Random, choices: Iterable[tuple[T, float]]) -> T:
    """Return a random item from (item, weight) pairs."""
    items = list(choices)
    total = sum(weight for _, weight in items)
    if total <= 0:
        raise ValueError("weighted_choice requires total weight > 0")

    roll = rng.uniform(0, total)
    running = 0.0
    for item, weight in items:
        running += weight
        if roll <= running:
            return item
    return items[-1][0]
