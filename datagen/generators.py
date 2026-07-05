"""Seeded parametric part generators (HARNESS_BLUEPRINT.md sec.21 data engine).

The #1 named risk is data: there is no "GitHub for CAD". The bootstrap answer is
**synthetic parametric generation + solver-in-the-loop for ground truth** — cheaply
manufacture (NL brief -> CISP ops) pairs, then let the HarnessSession verifier decide
which are real (see datagen/pipeline.py).

This module is the *generation* half: pure, deterministic functions that sample
realistic dimensions and emit a CISP op stream. Each generator returns the triple
``(brief_text, ops, params)`` so the pair (NL description -> ops) is directly a
training / eval example, and ``params`` records exactly what was drawn (for the
verifiers-as-labor decomposition and for auditing the synthetic distribution).

Determinism: every draw goes through a :class:`ParametricSampler` wrapping
``random.Random(seed)``. Same seed -> same parts, byte for byte. (``random`` with a
fixed seed is deterministic; we never touch wall-clock or os entropy.)

The plate/bracket op-templates are reused from ``memory.skills`` via a lazy import
(the Voyager skill library already ships execution-verified templates); a self-
contained fallback keeps datagen usable even if that module is absent.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

from cisp.ops import (
    Op, NewSketch, AddRectangle, AddCircle, Constrain, Extrude, Boolean,
)

# A generator: (rng) -> (brief, ops, params)
Generator = Callable[["ParametricSampler"], Tuple[str, List[Op], dict]]


class ParametricSampler:
    """Reproducible parameter draws for the generators.

    Wraps ``random.Random(seed)`` and exposes the few draw shapes the generators
    need. Two samplers built from the same seed yield identical sequences, which
    is what makes a whole synthetic dataset reproducible.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def uniform(self, lo: float, hi: float) -> float:
        return self.rng.uniform(lo, hi)

    def randint(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def choice(self, seq):
        return self.rng.choice(list(seq))

    def dim(self, lo: float, hi: float, step: float = 0.5) -> float:
        """A quantised dimension in ``[lo, hi]`` — realistic round-ish millimetres
        rather than 17.3841... A step keeps parts looking manufacturable."""
        raw = self.rng.uniform(lo, hi)
        snapped = round(raw / step) * step
        snapped = min(max(snapped, lo), hi)
        return round(snapped, 2)


# --- op-template reuse (memory.skills, lazily) -----------------------------
def _skill_templates():
    """Return (plate_ops, bracket_ops), preferring the verified skill library."""
    try:
        from memory.skills import plate_ops, bracket_ops
        return plate_ops, bracket_ops
    except Exception:  # pragma: no cover - fallback path
        return _fallback_plate_ops, _fallback_bracket_ops


def _fallback_plate_ops(w: float = 10.0, h: float = 10.0,
                        thickness: float = 2.0) -> List[Op]:
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h),
        Constrain(kind="horizontal", a="e1"),
        Constrain(kind="vertical", a="e1"),
        Constrain(kind="distance", a="e1", value=w),
        Constrain(kind="distance", a="e1", value=h),
        Extrude(sketch="sk1", distance=thickness),
    ]


def _fallback_bracket_ops(w: float = 20.0, h: float = 20.0,
                         thickness: float = 3.0, hole_r: float = 3.0) -> List[Op]:
    ops = _fallback_plate_ops(w=w, h=h, thickness=thickness)
    ops += [
        NewSketch(plane="XY"),
        AddCircle(sketch="sk2", cx=w / 2.0, cy=h / 2.0, r=hole_r),
        Constrain(kind="distance", a="e2", value=w / 2.0),
        Constrain(kind="distance", a="e2", value=h / 2.0),
        Constrain(kind="radius", a="e2", value=hole_r),
        Extrude(sketch="sk2", distance=thickness),
        Boolean(kind="cut", target="f1", tool="f2"),
    ]
    return ops


# --- generators ------------------------------------------------------------
def gen_plate(rng: ParametricSampler) -> Tuple[str, List[Op], dict]:
    """A flat, fully-constrained rectangular plate."""
    plate_ops, _ = _skill_templates()
    w = rng.dim(20.0, 200.0)
    h = rng.dim(20.0, 200.0)
    t = rng.dim(1.0, 20.0)
    ops = list(plate_ops(w=w, h=h, thickness=t))
    brief = (f"A flat rectangular plate {w} mm wide, {h} mm deep and "
             f"{t} mm thick.")
    params = {"generator": "plate", "w": w, "h": h, "thickness": t}
    return brief, ops, params


def gen_bracket(rng: ParametricSampler) -> Tuple[str, List[Op], dict]:
    """A mounting bracket: a plate with a single central through-hole."""
    _, bracket_ops = _skill_templates()
    w = rng.dim(30.0, 200.0)
    h = rng.dim(30.0, 200.0)
    t = rng.dim(2.0, 20.0)
    # Keep the hole comfortably inside the material: diameter < shorter side.
    hole_r = round(rng.uniform(2.0, min(w, h) * 0.3), 1)
    ops = list(bracket_ops(w=w, h=h, thickness=t, hole_r=hole_r))
    brief = (f"A mounting bracket {w} x {h} mm and {t} mm thick with a central "
             f"through-hole of radius {hole_r} mm.")
    params = {"generator": "bracket", "w": w, "h": h, "thickness": t,
              "hole_r": hole_r}
    return brief, ops, params


def gen_plate_with_holes(rng: ParametricSampler) -> Tuple[str, List[Op], dict]:
    """A plate with a row of ``n`` (1-4) through-holes cut into it.

    Ids follow StubBackend's scheme: plate is sk1/e1/f1; hole ``i`` (0-based) is
    sketch ``sk{2+i}`` and circle entity ``e{2+i}``. Each hole is a fully-
    constrained circle, extruded and boolean-cut from the running body.
    """
    plate_ops, _ = _skill_templates()
    w = rng.dim(40.0, 200.0)
    h = rng.dim(30.0, 120.0)
    t = rng.dim(2.0, 15.0)
    n = rng.randint(1, 4)
    hole_r = round(rng.uniform(1.5, min(h * 0.2, 6.0)), 1)
    margin = hole_r + 2.0

    ops: List[Op] = list(plate_ops(w=w, h=h, thickness=t))  # sk1, e1, f1
    body = "f1"
    ent = 1   # rectangle is e1
    feat = 1  # plate extrude is f1
    holes = []
    cy = round(h / 2.0, 2)
    span = w - 2.0 * margin
    for i in range(n):
        frac = 0.5 if n == 1 else i / (n - 1)
        cx = round(margin + span * frac, 2)
        sid = f"sk{2 + i}"
        ent += 1
        eid = f"e{ent}"
        ops.append(NewSketch(plane="XY"))
        ops.append(AddCircle(sketch=sid, cx=cx, cy=cy, r=hole_r))
        ops.append(Constrain(kind="distance", a=eid, value=cx))
        ops.append(Constrain(kind="distance", a=eid, value=cy))
        ops.append(Constrain(kind="radius", a=eid, value=hole_r))
        ops.append(Extrude(sketch=sid, distance=t))
        feat += 1
        tool = f"f{feat}"
        ops.append(Boolean(kind="cut", target=body, tool=tool))
        feat += 1
        body = f"f{feat}"
        holes.append({"cx": cx, "cy": cy})

    plural = "hole" if n == 1 else "holes"
    brief = (f"A {w} x {h} mm plate {t} mm thick with {n} through-{plural} of "
             f"radius {hole_r} mm.")
    params = {"generator": "plate_with_holes", "w": w, "h": h, "thickness": t,
              "n_holes": n, "hole_r": hole_r, "holes": holes}
    return brief, ops, params


# The default synthetic mix. generate_dataset cycles through these round-robin.
DEFAULT_GENERATORS: List[Generator] = [gen_plate, gen_bracket, gen_plate_with_holes]
