"""SSR (Sketch, Sketch-based feature, Refinements) triple design paradigm.

Deterministic re-implementation of the CAD representation introduced in
"Seek-CAD: A Self-refined Generative Modeling for 3D Parametric CAD Using Local
Inference via DeepSeek" (Li et al., ICLR 2026), Section 4.

The paper replaces the conventional Sketch-Extrude (SE) paradigm with the SSR
paradigm, where each modeling step is a triple

    S = (s, f, <r_1, r_2, ..., r_k> or empty)                            (Eq. 8)

with ``s`` a 2D sketch feature, ``f`` a sketch-based feature (``extrude`` or
``revolve``) drawn from a fixed set F, and ``<r_1..r_k>`` an ordered list of
zero or more refinement features (``fillet``, ``chamfer``, ``shell``) from a
fixed set R.  A complete model composes n triples joined by boolean operations

    M = <S_1, op_1, S_2, op_2, ..., S_n>                                 (Eq. 9)

with ``op_i`` in {Union, Cut, Intersect}.

This module provides the pure-Python data model (no geometry kernel): triple
construction with validation, boolean-op sequencing, a DeepCAD-compatible JSON
serialisation, and the *command count* used by the paper (Sec 5.2(5)) to bucket
model complexity.  The learned generation of these triples is external; the
representation itself is fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# Sketch-based feature set F (Sec 4).
SKETCH_FEATURES = ("extrude", "revolve")
# Refinement feature set R (Sec 4).
REFINEMENT_FEATURES = ("fillet", "chamfer", "shell")
# Boolean operators op_i (Eq. 9).
BOOLEAN_OPS = ("Union", "Cut", "Intersect")


@dataclass(frozen=True)
class Refinement:
    """A single refinement feature r_i in R applied to referenced entities.

    ``entities`` is a list of CapType references (see :mod:`geometry.seekcad_captype`);
    kept opaque here as strings so the representation has no kernel dependency.
    """

    kind: str
    param: float
    entities: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in REFINEMENT_FEATURES:
            raise ValueError(
                "refinement kind must be one of %r, got %r"
                % (REFINEMENT_FEATURES, self.kind)
            )
        if self.param <= 0:
            raise ValueError("refinement param must be positive")


@dataclass
class SSRTriple:
    """One SSR triple S = (s, f, <r_1..r_k>) (Eq. 8).

    ``sketch_curves`` is the number of primitive curves in the 2D sketch s
    (lines/arcs/splines/circles); the identity of the curves is irrelevant to
    the paradigm's command-count and is abstracted to a count here.
    ``feature`` is f in F; ``refinements`` is the ordered <r_1..r_k>, possibly
    empty (k >= 0).
    """

    sketch_curves: int
    feature: str
    refinements: List[Refinement] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sketch_curves < 1:
            raise ValueError("a sketch must contain at least one curve")
        if self.feature not in SKETCH_FEATURES:
            raise ValueError(
                "feature must be one of %r, got %r"
                % (SKETCH_FEATURES, self.feature)
            )
        self.refinements = list(self.refinements)

    def command_count(self) -> int:
        """Number of CAD commands contributed by this triple.

        One sketch command, one command per sketch curve, one sketch-based
        feature command, and one command per refinement feature.
        """
        return 1 + self.sketch_curves + 1 + len(self.refinements)

    def to_dict(self) -> Dict:
        return {
            "sketch": {"curves": self.sketch_curves},
            "feature": self.feature,
            "refinements": [
                {"kind": r.kind, "param": r.param, "entities": list(r.entities)}
                for r in self.refinements
            ],
        }


class SSRModel:
    """A complete SSR model M = <S_1, op_1, ..., S_n> (Eq. 9).

    Constructed from a non-empty list of :class:`SSRTriple` and a list of
    boolean operators of length ``len(triples) - 1`` (the operator joining each
    adjacent pair of geometry units).  The first triple has no preceding
    operator.
    """

    def __init__(self, triples: Sequence[SSRTriple], ops: Sequence[str]):
        triples = list(triples)
        ops = list(ops)
        if not triples:
            raise ValueError("an SSR model needs at least one triple")
        if len(ops) != len(triples) - 1:
            raise ValueError(
                "expected %d boolean ops for %d triples, got %d"
                % (len(triples) - 1, len(triples), len(ops))
            )
        for op in ops:
            if op not in BOOLEAN_OPS:
                raise ValueError(
                    "boolean op must be one of %r, got %r" % (BOOLEAN_OPS, op)
                )
        self.triples = triples
        self.ops = ops

    def __len__(self) -> int:
        return len(self.triples)

    def command_count(self) -> int:
        """Total CAD-command count over the whole model (Sec 5.2(5) metric).

        Sum of each triple's commands plus one command per boolean operation.
        """
        return sum(t.command_count() for t in self.triples) + len(self.ops)

    def complexity_band(self) -> str:
        """Bucket the model by command count: Low [0,30], Medium [31,70],
        High [71, inf) (Sec 5.2(5), Table 5)."""
        n = self.command_count()
        if n <= 30:
            return "Low"
        if n <= 70:
            return "Medium"
        return "High"

    def refinement_kinds(self) -> Tuple[str, ...]:
        """The distinct refinement kinds used, in first-appearance order."""
        seen: List[str] = []
        for t in self.triples:
            for r in t.refinements:
                if r.kind not in seen:
                    seen.append(r.kind)
        return tuple(seen)

    def to_json_dict(self) -> Dict:
        """DeepCAD-compatible dict: an ordered sequence of geometry units and
        the operators joining them (Sec 4, App. A.2)."""
        sequence: List[Dict] = [{"unit": self.triples[0].to_dict()}]
        for op, triple in zip(self.ops, self.triples[1:]):
            sequence.append({"op": op, "unit": triple.to_dict()})
        return {
            "paradigm": "SSR",
            "n_units": len(self.triples),
            "sequence": sequence,
        }
