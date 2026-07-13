"""Typed, seeded domain-randomization scene manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random


@dataclass(frozen=True)
class RandomAxis:
    name: str
    low: float
    high: float

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError("invalid axis range")


def draw_scene(axes, *, seed, identity=""):
    rng = random.Random(seed)
    values = {axis.name: rng.uniform(axis.low, axis.high) for axis in sorted(axes,
                                                                            key=lambda a: a.name)}
    payload = {"seed": seed, "identity": identity, "values": values}
    payload["digest"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def independence_audit(scenes):
    """Flag axes whose value is constant within identity but differs across identities."""
    identities = {}
    for scene in scenes:
        identities.setdefault(scene["identity"], []).append(scene["values"])
    suspicious = []
    axes = sorted({key for values in identities.values() for row in values for key in row})
    for axis in axes:
        per_identity = [{row.get(axis) for row in rows} for rows in identities.values()]
        if len(per_identity) > 1 and all(len(values) == 1 for values in per_identity) \
                and len({next(iter(values)) for values in per_identity}) > 1:
            suspicious.append(axis)
    return tuple(suspicious)
