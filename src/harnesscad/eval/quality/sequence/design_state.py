"""Design-state extraction and an editability score for a CAD program.

**Arko-T** (Wang et al., 2026) reframes text-to-CAD as *text-to-design*: the output is
not merely runnable code but an editable *design state* (their Sec. 3, Eq. 2)::

    z = (F, Phi, C, H, A)

where ``F`` is the feature vocabulary used (holes, ribs, fillets, shells, patterns),
``Phi`` the named/adjustable parameters (radii, thicknesses, spacings), ``C`` the
constraints/relations (symmetry, spacing), ``H`` the construction history (the ordered
sketch -> extrude -> secondary-features -> finishing operations), and ``A`` the
attachments (references binding features to faces/edges/planes). The paper argues a
program can compile yet be a poor *design* -- "A syntactically valid CAD script can
produce an empty body ... or a shape that bears no relation to the requested features"
-- so a design must be scored on whether it preserves this structure, not only on
whether it runs.

This module extracts the five-tuple from an abstract op-stream (a sequence of op dicts,
matching the CISP op protocol's ``OP`` tag + fields) and turns it into a deterministic
**editability score**: does the program surface named parameters, use higher-level
features beyond bare sketch/extrude, declare constraints, and carry face/edge
attachments? Arko-T's own "design-state code normalization" (Sec. 4.3) also enforces a
canonical construction order (sketch -> extrude -> secondary -> finishing);
:func:`construction_order_score` measures adherence to it. It is an *analyser*
(a number/structure), not a verifier gate. Stdlib only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    "FEATURE_OPS",
    "PRIMITIVE_OPS",
    "CONSTRAINT_OPS",
    "STAGE_ORDER",
    "DesignState",
    "extract_design_state",
    "editability_score",
    "construction_order_score",
]

# Higher-level features (Arko-T F beyond bare sketch/extrude).
FEATURE_OPS: frozenset[str] = frozenset({
    "hole", "fillet", "chamfer", "shell", "rib", "pattern", "mirror",
    "revolve", "loft", "sweep", "draft", "groove", "boss", "pocket",
})
# The bare primitives every DeepCAD-style sequence already has.
PRIMITIVE_OPS: frozenset[str] = frozenset({"sketch", "extrude", "cut", "line", "circle", "arc"})
# Ops that declare relations / constraints (Arko-T C).
CONSTRAINT_OPS: frozenset[str] = frozenset({
    "coincident", "parallel", "perpendicular", "tangent", "symmetric",
    "equal", "distance", "angle", "concentric", "constraint",
})

# Arko-T canonical construction order (Sec. 4.3): stage index per op family.
STAGE_ORDER: Mapping[str, int] = {
    "sketch": 0, "line": 0, "circle": 0, "arc": 0,
    "extrude": 1, "revolve": 1, "loft": 1, "sweep": 1, "cut": 1, "pocket": 1,
    "hole": 2, "rib": 2, "boss": 2, "groove": 2, "pattern": 2, "mirror": 2, "shell": 2,
    "fillet": 3, "chamfer": 3, "draft": 3,  # finishing
}


def _op_tag(op) -> str:
    if isinstance(op, Mapping):
        tag = op.get("OP") or op.get("op") or op.get("type") or ""
    else:
        tag = getattr(op, "OP", None) or getattr(op, "op", None) or ""
    return str(tag).lower()


def _op_params(op) -> Mapping:
    if isinstance(op, Mapping):
        p = op.get("params") or op.get("parameters")
        if isinstance(p, Mapping):
            return p
        # named-parameter fields: anything that looks scalar and named.
        return {k: v for k, v in op.items() if k not in ("OP", "op", "type", "refs", "params")}
    return getattr(op, "params", {}) or {}


def _op_refs(op) -> tuple:
    if isinstance(op, Mapping):
        r = op.get("refs") or op.get("attachments") or op.get("on")
    else:
        r = getattr(op, "refs", None)
    if r is None:
        return ()
    if isinstance(r, (list, tuple)):
        return tuple(r)
    return (r,)


@dataclass(frozen=True)
class DesignState:
    """Arko-T's z = (F, Phi, C, H, A) extracted from an op stream."""

    features: frozenset[str]            # F
    parameters: frozenset[str]          # Phi (named parameter keys)
    constraints: tuple[str, ...]        # C
    history: tuple[str, ...]            # H (ordered op tags)
    attachments: tuple[object, ...]     # A (face/edge references)


def extract_design_state(ops: Iterable) -> DesignState:
    """Extract the five design-state components from a CISP-style op stream."""
    features: set[str] = set()
    parameters: set[str] = set()
    constraints: list[str] = []
    history: list[str] = []
    attachments: list[object] = []
    for op in ops:
        tag = _op_tag(op)
        history.append(tag)
        if tag in FEATURE_OPS:
            features.add(tag)
        if tag in CONSTRAINT_OPS:
            constraints.append(tag)
        for key, val in _op_params(op).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                parameters.add(str(key))
        attachments.extend(_op_refs(op))
    return DesignState(
        features=frozenset(features),
        parameters=frozenset(parameters),
        constraints=tuple(constraints),
        history=tuple(history),
        attachments=tuple(attachments),
    )


def editability_score(state: DesignState) -> float:
    """A 0..1 editability score: does the program behave as a *design*, not a shape?

    Four equally-weighted signals from Arko-T's argument that an editable design must
    carry named parameters, real features, constraints, and attachments:

    * named parameters present (Phi non-empty),
    * higher-level features used (F non-empty -- not just sketch/extrude),
    * constraints/relations declared (C non-empty),
    * face/edge attachments present (A non-empty).
    """
    signals = (
        1.0 if state.parameters else 0.0,
        1.0 if state.features else 0.0,
        1.0 if state.constraints else 0.0,
        1.0 if state.attachments else 0.0,
    )
    return sum(signals) / len(signals)


def construction_order_score(state: DesignState) -> float:
    """Adherence to Arko-T's canonical stage order (Sec. 4.3), in 0..1.

    Maps each op to its stage (sketch=0, primary=1, secondary=2, finishing=3) and
    measures the fraction of consecutive op pairs that are non-decreasing in stage. A
    program that sketches, extrudes, adds holes, then fillets scores 1.0; one that
    fillets before extruding is penalised. Ops with no stage are skipped.
    """
    stages = [STAGE_ORDER[t] for t in state.history if t in STAGE_ORDER]
    if len(stages) < 2:
        return 1.0
    ordered = sum(1 for a, b in zip(stages, stages[1:]) if a <= b)
    return ordered / (len(stages) - 1)
