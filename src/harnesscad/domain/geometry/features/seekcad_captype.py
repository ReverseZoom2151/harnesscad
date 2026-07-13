"""CapType reference mechanism for SSR refinement targeting.

Deterministic re-implementation of the CapType reference system from "Seek-CAD"
(Li et al., ICLR 2026), Section 4(2) and Appendix A.2.

Refinement features (fillet, chamfer, shell) must reference specific
topological primitives produced *during* modeling, which are not retained in
the parametric design history.  CapType establishes an explicit link between a
primitive ``a`` of the 2D sketch and a resulting primitive ``b`` of the swept
geometry via a mapping

    phi(a, C) -> b,   a in A,   b in B,   C in {START, END, SWEPT}       (Eq. 10)

where, for a sketch-based operation f applied to sketch primitive a:

  * START  -> the primitive at the *start cap* of the operation (the original
    sketch primitive, e.g. the bottom face/edge of an extrusion);
  * END    -> the primitive at the *end cap* (the translated/rotated copy,
    e.g. the top face/edge);
  * SWEPT  -> the primitive generated *along the trajectory* of the operation
    (the side wall swept out by a as f is applied).

This module models A (sketch primitives) -> B (swept primitives) purely
symbolically: given the tagged primitives of a sketch, it materialises the
START/END/SWEPT primitives an operation produces and resolves ``phi(a, C)`` to
a stable reference id.  The paper notes CapType cannot address primitives with
no sketch antecedent (e.g. solid-intersection edges); this module rejects such
references explicitly (App. A.2, Limitations).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# CapType categories C (Eq. 10).
START, END, SWEPT = "START", "END", "SWEPT"
CAP_TYPES = (START, END, SWEPT)


class CapTypeError(KeyError):
    """Raised when a (primitive, cap-type) pair cannot be resolved."""


@dataclass(frozen=True)
class SweptPrimitive:
    """A resulting primitive b in B produced by applying a feature to a sketch
    primitive a, tagged with its CapType category."""

    reference_id: str
    source_tag: str
    cap_type: str


class CapTypeResolver:
    """Materialise B from a tagged sketch A and resolve phi(a, C) -> b.

    A sketch is supplied as an ordered list of primitive tags (the ``pointTag``
    / ``curveTag`` labels of Listing 1).  ``feature_id`` names the sketch-based
    feature (extrude/revolve) so reference ids are unique per operation.
    """

    def __init__(self, sketch_tags: List[str], feature_id: str):
        if not sketch_tags:
            raise ValueError("sketch must define at least one tagged primitive")
        if len(set(sketch_tags)) != len(sketch_tags):
            raise ValueError("sketch primitive tags must be unique")
        if not feature_id:
            raise ValueError("feature_id must be non-empty")
        self.sketch_tags = list(sketch_tags)
        self.feature_id = feature_id
        # Build B: every sketch primitive yields a START, END and SWEPT result.
        self._table: Dict[Tuple[str, str], SweptPrimitive] = {}
        for tag in self.sketch_tags:
            for cap in CAP_TYPES:
                ref = "%s/%s/%s" % (feature_id, cap, tag)
                self._table[(tag, cap)] = SweptPrimitive(ref, tag, cap)

    def sketch_primitives(self) -> Tuple[str, ...]:
        """The primitive set A."""
        return tuple(self.sketch_tags)

    def resolve(self, a: str, cap_type: str) -> SweptPrimitive:
        """phi(a, C) -> b.  Raises :class:`CapTypeError` for a primitive with no
        sketch antecedent or an unknown cap type."""
        if cap_type not in CAP_TYPES:
            raise CapTypeError("unknown cap type %r" % (cap_type,))
        try:
            return self._table[(a, cap_type)]
        except KeyError:
            raise CapTypeError(
                "primitive %r has no sketch antecedent; CapType cannot "
                "reference it (e.g. intersection-generated edges)" % (a,)
            )

    def resolve_id(self, a: str, cap_type: str) -> str:
        """Convenience: the stable reference id of ``phi(a, C)``."""
        return self.resolve(a, cap_type).reference_id

    def swept_set(self) -> Tuple[SweptPrimitive, ...]:
        """The resulting primitive set B, ordered by (sketch order, cap order)."""
        out: List[SweptPrimitive] = []
        for tag in self.sketch_tags:
            for cap in CAP_TYPES:
                out.append(self._table[(tag, cap)])
        return tuple(out)


def build_refinement_entities(
    resolver: CapTypeResolver, refs: List[Tuple[str, str]]
) -> List[str]:
    """Resolve a list of (sketch_primitive, cap_type) pairs into the entity
    reference-id list a Refinement carries.

    Deterministically preserves input order and de-duplicates while keeping the
    first occurrence.  Every reference must resolve or :class:`CapTypeError` is
    raised (mirroring the paper's rule to exclude un-referenceable primitives
    from refinement commands, App. A.2).
    """
    seen: List[str] = []
    for a, cap in refs:
        rid = resolver.resolve_id(a, cap)
        if rid not in seen:
            seen.append(rid)
    return seen
