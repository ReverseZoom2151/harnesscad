"""Controllability & code-consistency metrics for a hierarchical neural code
representation of CAD models.

The headline claims being tested are *controllability* (edits at any hierarchy level
propagate sensibly) and *instance-agnostic design patterns* (data mapped to the same
code share high-level patterns while ignoring instance detail). This module turns
those claims into deterministic, learning-free metrics over the code tree of
:mod:`generation.hnc_code_control`:

* :func:`code_fix_preservation` -- given a :class:`ControlMask` and the before/after
  code trees, the fraction of *fixed* code nodes that survived unchanged. The
  design-preserving edit ideal requires this to be 1.0.

* :func:`edit_locality` -- for a single-level edit, checks that only the edited level
  (and, for structural edits, its descendant scope) changed. Encodes the
  hierarchy: loop codes -> shape geometry, profile codes -> 2D loop dimension /
  positioning, solid codes -> extrusion height / 3D combination.

* :func:`instance_agnostic_consistency` -- a silhouette-style score over
  (feature, code) pairs: items sharing a code should be closer to each other than to
  items with other codes. Higher = cleaner instance-agnostic patterns.

* codebook-diversity summaries: :func:`uniqueness_rate` (fraction of generated code
  trees that appear once) and :func:`novelty_rate` (fraction absent from a training
  reference set).

Distinct from other controllability metric modules that check *flat text-token*
preservation for one concrete model; here we operate on the abstract *neural-code
tree* and its three-level controls. Pure stdlib, deterministic.
"""

from __future__ import annotations

from harnesscad.agents.generation.code_tree_control import (
    LEVELS,
    LOOP,
    PROFILE,
    SOLID,
    CodeNode,
    CodeTree,
    ControlMask,
    serialize,
)

Vector = tuple[float, ...]


# --- controllability over the code tree ------------------------------------
def code_fix_preservation(mask: ControlMask, after: CodeTree) -> float:
    """Fraction of the mask's *fixed* code nodes that are unchanged in ``after``.

    The ``mask`` carries the *before* serialization; we compare its fixed code nodes
    against the same positions in ``serialize(after)``. Returns 1.0 when every frozen
    code survived (the design-preserving ideal). ``<SEP>`` markers are ignored.
    """
    after_els = serialize(after)
    before_els = mask.elements
    if len(after_els) != len(before_els):
        # A structural change means fixed nodes could not all be preserved.
        # Compare by position up to the shorter length; missing positions count as
        # not preserved.
        pass
    fixed_idx = [i for i in mask.fixed_positions()
                 if isinstance(before_els[i], CodeNode)]
    if not fixed_idx:
        return 1.0
    kept = 0
    for i in fixed_idx:
        if i < len(after_els) and after_els[i] == before_els[i]:
            kept += 1
    return kept / len(fixed_idx)


# Which levels are allowed to change for an edit rooted at a given level.
# An edit at a level may propagate to itself and to descendant levels; coarser
# (ancestor) levels must stay fixed for the edit to be *local*.
_DESCENDANTS = {
    SOLID: {SOLID, PROFILE, LOOP},
    PROFILE: {PROFILE, LOOP},
    LOOP: {LOOP},
}


def changed_levels(before: CodeTree, after: CodeTree) -> frozenset[str]:
    """The set of levels whose codes differ between two trees (by serialized order).

    A structural difference (different number of nodes at a level) also counts that
    level as changed.
    """
    b = serialize(before)
    a = serialize(after)
    changed: set[str] = set()
    # Bucket code nodes by level, preserving order.
    for level in LEVELS:
        bl = [e.code for e in b if isinstance(e, CodeNode) and e.level == level]
        al = [e.code for e in a if isinstance(e, CodeNode) and e.level == level]
        if bl != al:
            changed.add(level)
    return frozenset(changed)


def edit_locality(before: CodeTree, after: CodeTree, edit_level: str) -> bool:
    """True when the changes stay within ``edit_level`` and its descendant scope.

    A loop edit may only change loop codes; a profile edit may change profile and
    loop codes; a solid edit may touch anything. Any change to a *coarser* level than
    ``edit_level`` violates locality (the edit leaked upward).
    """
    if edit_level not in LEVELS:
        raise ValueError(f"unknown level {edit_level!r}")
    return changed_levels(before, after) <= _DESCENDANTS[edit_level]


# --- instance-agnostic design-pattern consistency ---------------------------
def _euclid(a: Vector, b: Vector) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b)) ** 0.5


def instance_agnostic_consistency(items: list[tuple[Vector, int]]) -> float:
    """Silhouette-style score over (feature, code) pairs; higher = cleaner patterns.

    For each item, ``a`` = mean distance to same-code items and ``b`` = min over other
    codes of the mean distance to that code's items; silhouette ``(b - a)/max(a, b)``.
    The mean over all items lies in ``[-1, 1]``. Items whose code is a singleton
    contribute ``a = 0``. Returns 0.0 when fewer than two distinct codes are present.
    """
    if len(items) < 2:
        return 0.0
    codes = sorted({c for _v, c in items})
    if len(codes) < 2:
        return 0.0
    by_code: dict[int, list[Vector]] = {c: [] for c in codes}
    for v, c in items:
        by_code[c].append(v)

    total = 0.0
    n = 0
    for v, c in items:
        same = by_code[c]
        if len(same) > 1:
            a = sum(_euclid(v, o) for o in same if o is not v) / (len(same) - 1)
        else:
            a = 0.0
        b = min(
            sum(_euclid(v, o) for o in by_code[oc]) / len(by_code[oc])
            for oc in codes if oc != c
        )
        denom = max(a, b)
        total += 0.0 if denom == 0 else (b - a) / denom
        n += 1
    return total / n


# --- code-tree diversity summaries ------------------------------------------
def _canonical(tree: CodeTree) -> tuple[object, ...]:
    return tuple(
        e if e == "<SEP>" else (e.level, e.code) for e in serialize(tree)
    )


def uniqueness_rate(generated: list[CodeTree]) -> float:
    """Fraction of generated code trees that appear exactly once in the set (Unique)."""
    if not generated:
        return 0.0
    keys = [_canonical(t) for t in generated]
    counts: dict[tuple[object, ...], int] = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return sum(1 for k in keys if counts[k] == 1) / len(keys)


def novelty_rate(generated: list[CodeTree], reference: list[CodeTree]) -> float:
    """Fraction of generated code trees absent from the reference set (Novel)."""
    if not generated:
        return 0.0
    ref = {_canonical(t) for t in reference}
    return sum(1 for t in generated if _canonical(t) not in ref) / len(generated)
