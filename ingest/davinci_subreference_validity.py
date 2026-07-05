"""DAVINCI valid-subreference combination set S (constraint-graph consistency).

In DAVINCI (Karadeniz et al., 2024, Sec. 3 / 3.3) a constraint ``c`` is an
undirected edge between two primitive *subreferences* ``(i, si)`` and ``(j, sj)``,
where a subreference ``s in [1..4]`` selects the primitive's **start, end, middle
point, or entire** geometry. Not every subreference makes sense for every
primitive type, so the paper defines the *valid combination set*

    S = { (i, si, j, sj) | si in V(t1_i), sj in V(t1_j) }

with per-type allowed subreference sets (Sec. 3.3):

    V(arc)    = {1, 2, 3, 4}     (start, end, middle, entire)
    V(circle) = {2, 3}
    V(line)   = {1, 2, 4}        (start, end, entire -- a line has no middle)
    V(point)  = {4}              (entire only)

During both training loss and inference, DAVINCI restricts constraint prediction
to pairs in ``S``; topologically invalid pairs (e.g. ``<circle.start - line.end>``)
are removed. That filtering is a pure, deterministic combinatorial rule -- this
module implements it as (a) a per-type subreference validator, (b) an enumerator of
the canonical (permutation-invariant) candidate pairs in ``S`` for a set of
primitives, and (c) a checker that filters an inferred constraint set down to the
topologically valid ones, reporting the rejects.

This is complementary to ``ingest.cadvlm_sketch_validity`` (which checks token
ranges and reference existence) -- that module has no notion of subreferences or of
the per-type subreference legality rule.
"""

from __future__ import annotations

from dataclasses import dataclass

# Subreference labels in [1..4].
START, END, MIDDLE, ENTIRE = 1, 2, 3, 4
SUBREF_NAMES = {START: "start", END: "end", MIDDLE: "middle", ENTIRE: "entire"}

# V(t): allowed subreferences per primitive type (Sec. 3.3).
VALID_SUBREFS = {
    "arc": frozenset({START, END, MIDDLE, ENTIRE}),
    "circle": frozenset({END, MIDDLE}),
    "line": frozenset({START, END, ENTIRE}),
    "point": frozenset({ENTIRE}),
    "none": frozenset(),
}


def valid_subreferences(ptype: str) -> frozenset:
    """Allowed subreference labels ``V(ptype)`` for a primitive type."""
    if ptype not in VALID_SUBREFS:
        raise KeyError(f"unknown primitive type: {ptype!r}")
    return VALID_SUBREFS[ptype]


def is_valid_subreference(ptype: str, subref: int) -> bool:
    """True if ``subref`` is a legal subreference for a primitive of ``ptype``."""
    return subref in valid_subreferences(ptype)


def is_valid_pair(type_i: str, si: int, type_j: str, sj: int) -> bool:
    """True if ``(type_i.si, type_j.sj)`` is a member of the combination set S."""
    return is_valid_subreference(type_i, si) and is_valid_subreference(type_j, sj)


def canonical_pair(i: int, si: int, j: int, sj: int) -> tuple:
    """Permutation-invariant key for an undirected subreference pair.

    Because DAVINCI's constraints are undirected (Sec. 3.2, the constraint head is
    made permutation-invariant), ``(i,si,j,sj)`` and ``(j,sj,i,si)`` denote the same
    edge; this orders the two endpoints so both map to one key.
    """
    a, b = (i, si), (j, sj)
    return (a, b) if a <= b else (b, a)


def enumerate_candidates(primitive_types, *, include_self=True) -> frozenset:
    """All canonical valid subreference pairs ``S`` over a list of primitives.

    ``primitive_types`` is an ordered sequence of type names (one per primitive
    slot). Returns the set of permutation-invariant ``((i,si),(j,sj))`` keys that
    are members of ``S``. ``none`` slots contribute nothing. Self-pairs (a
    constraint on a single primitive, e.g. a horizontal line) are included when
    ``include_self`` is set, but only across two *distinct* legal subreferences of
    that primitive (an edge from a subreference to itself is excluded).
    """
    types = tuple(primitive_types)
    candidates = set()
    for i, ti in enumerate(types):
        for j, tj in enumerate(types):
            if j < i:
                continue
            if not include_self and i == j:
                continue
            for si in valid_subreferences(ti):
                for sj in valid_subreferences(tj):
                    if i == j and si == sj:
                        continue
                    candidates.add(canonical_pair(i, si, j, sj))
    return frozenset(candidates)


@dataclass(frozen=True)
class ConstraintFilterResult:
    """Outcome of :func:`filter_constraints`."""

    valid: tuple      # constraints whose endpoints are members of S
    invalid: tuple    # (constraint, reason) for rejected ones

    @property
    def all_valid(self) -> bool:
        return not self.invalid


def filter_constraints(constraints, primitive_types) -> ConstraintFilterResult:
    """Split inferred constraints into topologically valid vs invalid.

    Each constraint is a 4-tuple ``(i, si, j, sj)`` of primitive index + subreference
    for both endpoints. ``primitive_types`` gives the type name of each primitive
    slot. A constraint is *valid* iff both indices exist, are not ``none`` slots, and
    both subreferences are members of the per-type set ``V`` (i.e. the pair is in S).
    """
    types = tuple(primitive_types)
    n = len(types)
    keep, drop = [], []
    for c in constraints:
        i, si, j, sj = c
        reason = None
        if not (0 <= i < n and 0 <= j < n):
            reason = "index-out-of-range"
        elif types[i] == "none" or types[j] == "none":
            reason = "references-empty-slot"
        elif not is_valid_subreference(types[i], si):
            reason = f"invalid-subref:{types[i]}.{SUBREF_NAMES.get(si, si)}"
        elif not is_valid_subreference(types[j], sj):
            reason = f"invalid-subref:{types[j]}.{SUBREF_NAMES.get(sj, sj)}"
        elif i == j and si == sj:
            reason = "degenerate-self-edge"
        if reason is None:
            keep.append(tuple(c))
        else:
            drop.append((tuple(c), reason))
    return ConstraintFilterResult(valid=tuple(keep), invalid=tuple(drop))
