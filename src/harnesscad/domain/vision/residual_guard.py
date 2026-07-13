"""Finite, shape-compatible and magnitude-bounded residual corrections."""

from __future__ import annotations

import math


def guard_residual(base, residual, *, scale: float = 0.1, bound: float = 1.0,
                   policy: str = "reject"):
    if len(base) != len(residual):
        return None, ("shape-mismatch",)
    if scale < 0 or bound < 0 or policy not in {"reject", "clip"}:
        raise ValueError("invalid residual policy")
    if any(not math.isfinite(float(value)) for value in residual):
        return None, ("non-finite-residual",)
    excessive = any(abs(float(value)) > bound for value in residual)
    if excessive and policy == "reject":
        return None, ("residual-bound",)
    safe = [max(-bound, min(bound, float(value))) for value in residual]
    return tuple(float(a) + scale * b for a, b in zip(base, safe)), (
        ("residual-clipped",) if excessive else ())
