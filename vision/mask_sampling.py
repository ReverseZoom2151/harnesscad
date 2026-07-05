"""Seeded point prompts sampled from binary foreground masks."""

from __future__ import annotations

import random


def sample_mask(mask, count=25, seed=0):
    if count < 0:
        raise ValueError("count must be non-negative")
    pixels = [(x, y) for y, row in enumerate(mask)
              for x, value in enumerate(row) if bool(value)]
    if count and not pixels:
        raise ValueError("foreground mask is empty")
    rng = random.Random(seed)
    if count <= len(pixels):
        return tuple(sorted(rng.sample(pixels, count)))
    return tuple(pixels[index % len(pixels)] for index in range(count))
