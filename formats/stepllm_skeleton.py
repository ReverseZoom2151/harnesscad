"""Deterministic STEP-skeleton builder (the scaffold an LLM would fill in).

STEP-LLM maps a natural-language caption to a full STEP file with an external
LLM. The language model is out of scope, but everything *around* it is
deterministic: assembling well-numbered entity instances, wiring their
cross-references, and wrapping them in a valid ISO 10303-21 header/footer. This
module provides that scaffold.

:class:`StepBuilder` auto-assigns sequential entity ids, wires references, and
emits a complete file with a standard header, so callers never hand-manage ``#N``
identifiers. On top of it, :func:`skeleton_from_keywords` performs a tiny,
deterministic caption -> geometry mapping (the kind of grounded "skeleton" the
paper's SFT model learns to produce): recognised primitive words yield a minimal
valid entity anchored at an origin placement. This is intentionally rule-based
(no learned model) - it produces a *syntactically valid, reference-consistent*
starting point that downstream steps (or the LLM) elaborate.

Depends only on :mod:`formats.stepllm_parser` and the geometric constructors in
:mod:`formats.stepllm_schema`.
"""

from __future__ import annotations

from formats.stepllm_parser import Ref, StepFile, Typed, serialize
from formats.stepllm_schema import make


def default_header(schema: str = "AUTOMOTIVE_DESIGN",
                   name: str = "") -> list:
    """A minimal, well-formed part-21 HEADER (FILE_DESCRIPTION/NAME/SCHEMA)."""

    return [
        Typed("FILE_DESCRIPTION", ([""], "2;1")),
        Typed("FILE_NAME", (name, "", [""], [""], "", "", "")),
        Typed("FILE_SCHEMA", ([schema],)),
    ]


class StepBuilder:
    """Assemble a STEP DATA section with auto-numbered, wired entities."""

    def __init__(self, schema: str = "AUTOMOTIVE_DESIGN", name: str = "") -> None:
        self._step = StepFile(header=default_header(schema, name))
        self._next = 1

    def add(self, keyword: str, *params) -> Ref:
        """Add an entity, validating its arity via the schema; return its Ref."""

        ent = make(keyword, self._next, *params)
        self._step.add(ent)
        self._next += 1
        return Ref(ent.id)

    # convenience geometric constructors --------------------------------------

    def point(self, x=0.0, y=0.0, z=0.0) -> Ref:
        from formats.stepllm_schema import _real
        return self.add("CARTESIAN_POINT", "", [_real(x), _real(y), _real(z)])

    def direction(self, dx, dy, dz) -> Ref:
        from formats.stepllm_schema import _real
        return self.add("DIRECTION", "", [_real(dx), _real(dy), _real(dz)])

    def placement(self, location: Ref, axis: Ref, ref_dir: Ref) -> Ref:
        return self.add("AXIS2_PLACEMENT_3D", "", location, axis, ref_dir)

    def origin_placement(self) -> Ref:
        """A canonical placement at the origin with +Z axis and +X ref dir."""

        loc = self.point(0.0, 0.0, 0.0)
        axis = self.direction(0.0, 0.0, 1.0)
        ref = self.direction(1.0, 0.0, 0.0)
        return self.placement(loc, axis, ref)

    def build(self) -> StepFile:
        return self._step

    def to_text(self) -> str:
        return serialize(self._step)


# --- caption -> skeleton (rule-based, deterministic) -------------------------

# Recognised primitive keywords -> the entity they anchor at an origin placement.
PRIMITIVE_KEYWORDS: dict = {
    "plane": "PLANE",
    "flat": "PLANE",
    "circle": "CIRCLE",
    "circular": "CIRCLE",
    "round": "CIRCLE",
    "disc": "CIRCLE",
    "cylinder": "CYLINDRICAL_SURFACE",
    "cylindrical": "CYLINDRICAL_SURFACE",
    "cone": "CONICAL_SURFACE",
    "conical": "CONICAL_SURFACE",
    "sphere": "SPHERICAL_SURFACE",
    "spherical": "SPHERICAL_SURFACE",
}


def detect_primitives(caption: str) -> list:
    """Return the distinct entity types implied by a caption's keywords.

    Order follows first appearance in the caption so the mapping is stable.
    """

    words = "".join(c.lower() if (c.isalnum() or c.isspace()) else " "
                    for c in caption).split()
    found: list = []
    for w in words:
        ent = PRIMITIVE_KEYWORDS.get(w)
        if ent is not None and ent not in found:
            found.append(ent)
    return found


def _anchor(builder: StepBuilder, ent_type: str, radius: float = 1.0) -> Ref:
    placement = builder.origin_placement()
    if ent_type == "PLANE":
        return builder.add("PLANE", "", placement)
    if ent_type == "CIRCLE":
        from formats.stepllm_schema import _real
        return builder.add("CIRCLE", "", placement, _real(radius))
    if ent_type == "CYLINDRICAL_SURFACE":
        from formats.stepllm_schema import _real
        return builder.add("CYLINDRICAL_SURFACE", "", placement, _real(radius))
    if ent_type == "CONICAL_SURFACE":
        from formats.stepllm_schema import _real
        return builder.add("CONICAL_SURFACE", "", placement, _real(radius),
                           _real(0.5))
    if ent_type == "SPHERICAL_SURFACE":
        from formats.stepllm_schema import _real
        return builder.add("SPHERICAL_SURFACE", "", placement, _real(radius))
    raise ValueError(f"unsupported primitive {ent_type!r}")


def skeleton_from_keywords(caption: str, radius: float = 1.0) -> StepFile:
    """Build a valid STEP skeleton for the primitives named in ``caption``.

    Unrecognised captions yield an empty-but-valid DATA section (header/footer
    only). This is a deterministic grounding scaffold, not a geometric solver.
    """

    builder = StepBuilder()
    for ent_type in detect_primitives(caption):
        _anchor(builder, ent_type, radius)
    return builder.build()
