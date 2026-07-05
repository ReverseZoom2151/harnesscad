"""Constraint-Preserving Transformations (CPTs) for CAD sketch augmentation.

DAVINCI (Karadeniz et al., 2024, Sec. 4) introduces Constraint-Preserving
Transformations: augmentations that change a CAD sketch's *parameterization* while
leaving its constraint graph intact. The abstract defines a CPT as a **random
permutation of the parametric primitives of a CAD sketch that preserves its
constraints**. The paper's headline realisation drives FreeCAD's constraint solver
to perturb a subreference inside a bounding box and let the change cascade -- that
solver-in-the-loop variant is external/proprietary (needs FreeCAD).

Two of the paper's augmentations are, however, fully deterministic and
solver-free, and are implemented here:

* **Primitive permutation (the abstract's exact definition).** Relabelling the
  primitive slots and re-indexing every constraint's endpoints yields a sketch that
  is *identical as a constraint graph* but has a different token ordering -- exactly
  the invariance a set-based model like DAVINCI should be robust to. This is
  constraint-preserving by construction, and :func:`constraints_preserved` proves it
  by checking the two constraint sets are equal up to the relabelling.

* **Rotated sketches (Sec. 5.2 baseline).** Rotating the sketch by a multiple of 90
  degrees on the quantised grid, while **dropping orientation-dependent constraints**
  (horizontal / vertical) that a 90/270-degree turn would invalidate, and keeping
  them under a 180-degree turn.

Randomness is seeded via ``random.Random(seed)`` so every augmentation is
reproducible. Nothing else in the repo performs constraint-index remapping under a
primitive permutation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Constraints whose validity depends on absolute sketch orientation (Sec. 5.2).
ORIENTATION_DEPENDENT = frozenset({"horizontal", "vertical"})


@dataclass(frozen=True)
class Sketch:
    """A minimal CAD sketch: primitive slots plus constraints over them.

    ``primitives`` is an ordered sequence; each item is opaque to this module except
    that a ``dict`` with a ``"coords"`` key (a tuple of ``(x, y)`` points in
    ``[0, 1]``) can be geometrically rotated. ``constraints`` is a sequence of
    ``(kind, i, si, j, sj)`` tuples, where ``i``/``j`` are primitive indices.
    """

    primitives: tuple
    constraints: tuple


def _normalise(sketch) -> Sketch:
    if isinstance(sketch, Sketch):
        return sketch
    prims, cons = sketch
    return Sketch(primitives=tuple(prims), constraints=tuple(cons))


def apply_permutation(sketch, perm) -> Sketch:
    """Relabel primitive slots by ``perm`` and re-index constraint endpoints.

    ``perm`` is a sequence where ``perm[old_index] = new_index`` (a bijection over
    ``range(n)``). The returned sketch places each primitive at its new index and
    rewrites every constraint's ``i``/``j`` to the new indices, preserving the graph.
    """
    s = _normalise(sketch)
    n = len(s.primitives)
    perm = tuple(perm)
    if sorted(perm) != list(range(n)):
        raise ValueError("perm must be a bijection over range(n)")
    new_prims = [None] * n
    for old, new in enumerate(perm):
        new_prims[new] = s.primitives[old]
    new_cons = tuple(
        (kind, perm[i], si, perm[j], sj) for (kind, i, si, j, sj) in s.constraints)
    return Sketch(primitives=tuple(new_prims), constraints=new_cons)


def random_permutation(sketch, seed: int) -> Sketch:
    """A CPT: apply a seeded random primitive permutation (constraint-preserving)."""
    s = _normalise(sketch)
    rng = random.Random(seed)
    perm = list(range(len(s.primitives)))
    rng.shuffle(perm)
    return apply_permutation(s, perm)


def _canonical_constraints(constraints) -> frozenset:
    """Orientation-invariant, permutation-invariant view of a constraint set."""
    out = set()
    for (kind, i, si, j, sj) in constraints:
        a, b = (i, si), (j, sj)
        lo, hi = (a, b) if a <= b else (b, a)
        out.add((kind, lo, hi))
    return frozenset(out)


def constraints_preserved(original, transformed, perm) -> bool:
    """True if ``transformed`` has exactly ``original``'s constraints under ``perm``.

    Applies ``perm`` to the original constraint indices and checks the two undirected
    constraint sets are equal -- a machine-checkable proof that the permutation is
    constraint-preserving.
    """
    o = _normalise(original)
    t = _normalise(transformed)
    remapped = tuple(
        (kind, perm[i], si, perm[j], sj) for (kind, i, si, j, sj) in o.constraints)
    return _canonical_constraints(remapped) == _canonical_constraints(t.constraints)


def _rotate_point(x: float, y: float, quarter_turns: int) -> tuple:
    """Rotate a point in the unit square by ``quarter_turns * 90`` degrees.

    Rotation is about the square centre ``(0.5, 0.5)`` so coordinates stay in
    ``[0, 1]`` on the quantised grid.
    """
    cx, cy = 0.5, 0.5
    dx, dy = x - cx, y - cy
    q = quarter_turns % 4
    for _ in range(q):
        dx, dy = -dy, dx      # 90-degree counter-clockwise
    return (cx + dx, cy + dy)


def rotate_sketch(sketch, quarter_turns: int) -> Sketch:
    """Rotate primitive coordinates by 90*``quarter_turns`` deg; drop invalidated
    orientation constraints.

    A 90- or 270-degree turn swaps the horizontal and vertical axes, so
    orientation-dependent constraints (``horizontal``/``vertical``) no longer hold
    and are removed (Sec. 5.2). A 0- or 180-degree turn keeps them. Primitives that
    expose a ``"coords"`` tuple of ``(x, y)`` points get those points rotated; other
    primitives pass through untouched.
    """
    s = _normalise(sketch)
    q = quarter_turns % 4
    new_prims = []
    for p in s.primitives:
        if isinstance(p, dict) and "coords" in p:
            rotated = tuple(_rotate_point(x, y, q) for (x, y) in p["coords"])
            new_prims.append({**p, "coords": rotated})
        else:
            new_prims.append(p)
    if q in (1, 3):
        new_cons = tuple(
            c for c in s.constraints if c[0] not in ORIENTATION_DEPENDENT)
    else:
        new_cons = s.constraints
    return Sketch(primitives=tuple(new_prims), constraints=new_cons)
