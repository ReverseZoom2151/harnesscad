"""Command-distribution imbalance and rare-operation coverage reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CommandBalance:
    counts: dict[str, int]
    frequencies: dict[str, float]
    inverse_weights: dict[str, float]
    sqrt_weights: dict[str, float]
    rare_commands: tuple[str, ...]
    rare_coverage: float


def command_balance(
    sequences: Iterable[Sequence[str]],
    *,
    vocabulary: Iterable[str] = (),
    rare_frequency: float = 0.01,
) -> CommandBalance:
    if not 0 <= rare_frequency <= 1:
        raise ValueError("rare_frequency must be in [0, 1]")
    sequence_list = tuple(tuple(sequence) for sequence in sequences)
    counts = Counter(command for sequence in sequence_list for command in sequence)
    names = sorted(set(vocabulary) | set(counts))
    total = sum(counts.values())
    frequencies = {
        name: counts[name] / total if total else 0.0 for name in names
    }
    nonzero = [value for value in counts.values() if value]
    maximum = max(nonzero, default=1)
    inverse = {
        name: maximum / counts[name] if counts[name] else 0.0 for name in names
    }
    sqrt_weights = {name: sqrt(value) for name, value in inverse.items()}
    rare = tuple(name for name in names if frequencies[name] <= rare_frequency)
    represented_rare = sum(1 for name in rare if counts[name])
    coverage = represented_rare / len(rare) if rare else 1.0
    return CommandBalance(
        dict(sorted(counts.items())),
        frequencies,
        inverse,
        sqrt_weights,
        rare,
        coverage,
    )
