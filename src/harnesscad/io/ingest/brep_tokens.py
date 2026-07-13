"""Canonical encodings for cyclic directed B-rep loops."""

from __future__ import annotations

from typing import Hashable, Sequence


def canonical_cycle(items: Sequence[Hashable], *, orientation_semantic: bool = True):
    if not items:
        raise ValueError("cycle cannot be empty")
    values = tuple(items)
    rotations = [values[i:] + values[:i] for i in range(len(values))]
    if not orientation_semantic:
        reverse = tuple(reversed(values))
        rotations += [reverse[i:] + reverse[:i] for i in range(len(reverse))]
    return min(rotations, key=lambda value: tuple(map(repr, value)))


def wrap_pad(items: Sequence[Hashable], width: int = 1):
    if not items or width < 0:
        raise ValueError("items must be non-empty and width non-negative")
    values = tuple(items)
    width = min(width, len(values))
    return values[-width:] + values + values[:width] if width else values


def loop_tokens(items: Sequence[Hashable], *, orientation_semantic: bool = True,
                padding: int = 1):
    return wrap_pad(canonical_cycle(items, orientation_semantic=orientation_semantic),
                    padding)
