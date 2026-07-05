"""Parametric part generators — the retrieval-not-modelling half of HarnessCAD.

The blueprint's mechanism/standard-parts idea: rather than model a gear or a
flange from scratch every time, *retrieve* a known, parametric part from a
library and *instantiate* it with concrete parameters. Each part is a
:class:`ModelCard` — a named, functionally-tagged op-template with a parameter
schema (types + valid ranges), so instantiation is (a) discoverable by function
and (b) range-checked before it ever touches geometry.

A ModelCard is the same execution-verified idea as :class:`memory.skills.Skill`:
its ``build(**params)`` expands to a ``list[cisp.ops.Op]`` — the exact op stream
the agent would emit by hand — so it is trivially verifiable (run the ops through
a HarnessSession; admit only if they build; see :mod:`library.catalog`). This
module deliberately *reuses* that machinery: :meth:`ModelCard.to_skill` bridges a
card into a ``Skill`` so the Voyager gate in ``SkillLibrary.add_verified`` can be
reused verbatim, and ``bracket`` composes ``memory.skills.bracket_ops`` directly.

All parts build against StubBackend id conventions (first NewSketch -> 'sk1',
first primitive -> 'e1', first feature -> 'f1'). Holes and circular patterns pass
an empty reference where a solid/feature is required: the stub treats an empty
``face_or_sketch``/``feature`` as "the current solid", which keeps the templates
robust against feature-id drift while still exercising real op semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from cisp.ops import (
    Op, NewSketch, AddCircle, Constrain, Extrude, Hole, CircularPattern,
)
from memory.skills import Skill, bracket_ops

# build(**params) -> list[Op]
Builder = Callable[..., List[Op]]


# ---------------------------------------------------------------------------
# Model Card
# ---------------------------------------------------------------------------
@dataclass
class ModelCard:
    """A retrievable, parametric standard part.

    - ``function_tags``: what the part *is for* (e.g. "flange", "mounting",
      "bearing", "shaft") — the index :meth:`library.catalog.PartCatalog.find`
      retrieves by.
    - ``param_schema``: name -> {"type", "default", "min", "max", "doc"}. ``min``/
      ``max`` bound a valid range that :meth:`validate` enforces before build.
    - ``build(**params)``: expands to a CISP op list (the verifiable artefact).
    - ``notes``: honest limitations (e.g. a gear *blank* has no involute teeth).
    """

    name: str
    function_tags: List[str]
    param_schema: Dict[str, dict]
    build: Builder
    description: str = ""
    notes: str = ""
    verified: bool = False

    def defaults(self) -> Dict[str, Any]:
        return {k: spec["default"] for k, spec in self.param_schema.items()
                if "default" in spec}

    def validate(self, **params: Any) -> Dict[str, Any]:
        """Merge params over defaults and enforce the schema.

        Raises ``ValueError`` on an unknown parameter or a value outside its
        declared [min, max] range. Returns the fully-resolved parameter dict.
        """
        merged = self.defaults()
        for key, value in params.items():
            if key not in self.param_schema:
                raise ValueError(
                    f"{self.name}: unknown parameter '{key}' "
                    f"(known: {sorted(self.param_schema)})")
            merged[key] = value
        for key, spec in self.param_schema.items():
            if key not in merged:
                raise ValueError(f"{self.name}: missing parameter '{key}'")
            value = merged[key]
            if spec.get("type") == "int" and not _is_int(value):
                raise ValueError(
                    f"{self.name}: parameter '{key}' must be an integer "
                    f"(got {value!r})")
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and value < lo:
                raise ValueError(
                    f"{self.name}: parameter '{key}'={value} below minimum {lo}")
            if hi is not None and value > hi:
                raise ValueError(
                    f"{self.name}: parameter '{key}'={value} above maximum {hi}")
        return merged

    def instantiate(self, **params: Any) -> List[Op]:
        """Range-validate then build — the safe path to an op stream."""
        return list(self.build(**self.validate(**params)))

    def to_skill(self) -> Skill:
        """Bridge to :class:`memory.skills.Skill` so the Voyager execution gate
        (``SkillLibrary.add_verified``) can be reused unchanged."""
        return Skill(
            name=self.name,
            description=f"{self.description} [tags: {', '.join(self.function_tags)}]",
            template=self.build,
            params=self.param_schema,
            sample_params=self.defaults(),
            verified=self.verified,
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Part builders (each returns a verifiable CISP op list)
# ---------------------------------------------------------------------------
def flange_ops(diameter: float = 60.0, bolt_circle: float = 45.0,
               n_holes: int = 4, thickness: float = 6.0,
               bore: float = 20.0, bolt_hole: float = 6.6) -> List[Op]:
    """A round mounting flange: a disc (diameter x thickness) with a central
    bore and a ``n_holes`` bolt circle at ``bolt_circle`` PCD.

    sk1/e1 -> disc solid f1; central bore f2; one bolt hole f3 replicated into a
    circular pattern f4."""
    return [
        NewSketch(plane="XY"),                                    # sk1
        AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=diameter / 2.0),  # e1
        Constrain(kind="radius", a="e1", value=diameter / 2.0),
        Extrude(sketch="sk1", distance=thickness),                # f1 (solid)
        Hole(face_or_sketch="", x=0.0, y=0.0, diameter=bore, through=True),  # f2 bore
        Hole(face_or_sketch="", x=bolt_circle / 2.0, y=0.0,
             diameter=bolt_hole, through=True),                   # f3 one bolt hole
        CircularPattern(feature="", count=n_holes, angle=360.0),  # f4 bolt circle
    ]


def bracket_part_ops(width: float = 40.0, height: float = 30.0,
                     thickness: float = 4.0, hole_r: float = 3.5) -> List[Op]:
    """An L-less flat mounting bracket: a plate with a central through-hole.

    Composed directly from ``memory.skills.bracket_ops`` — the library reuses the
    verified skill rather than re-deriving the geometry."""
    return bracket_ops(w=width, h=height, thickness=thickness, hole_r=hole_r)


def spur_gear_blank_ops(module: float = 2.0, teeth: int = 20,
                        thickness: float = 8.0, bore: float = 8.0) -> List[Op]:
    """A spur-gear BLANK: a plain cylinder at the pitch diameter (module x teeth)
    with a central shaft bore.

    LIMITATION: this is a blank only. No involute tooth profile is generated —
    the addendum/dedendum, pressure angle and involute flanks that make it an
    actual gear are NOT modelled. It stands in for the gear body so downstream
    assembly/bore/keyway ops have something to reference."""
    pitch_diameter = module * teeth
    return [
        NewSketch(plane="XY"),                                       # sk1
        AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=pitch_diameter / 2.0),  # e1
        Constrain(kind="radius", a="e1", value=pitch_diameter / 2.0),
        Extrude(sketch="sk1", distance=thickness),                   # f1 (solid)
        Hole(face_or_sketch="", x=0.0, y=0.0, diameter=bore, through=True),  # f2 bore
    ]


def shaft_ops(diameter: float = 10.0, length: float = 50.0) -> List[Op]:
    """A plain cylindrical shaft: a circle of ``diameter`` extruded to ``length``."""
    return [
        NewSketch(plane="XY"),                                    # sk1
        AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=diameter / 2.0),  # e1
        Constrain(kind="radius", a="e1", value=diameter / 2.0),
        Extrude(sketch="sk1", distance=length),                   # f1 (solid)
    ]


def bearing_seat_ops(bore: float = 8.0, outer_diameter: float = 22.0,
                     width: float = 7.0, wall: float = 3.0) -> List[Op]:
    """A housing seat for a rolling-element bearing: a cylindrical body with a
    blind counterbore pocket (sized to the bearing OD, depth = bearing width) and
    a through shaft bore.

    ``outer_diameter``/``width`` are the bearing's OD/width; ``wall`` is the
    radial material around the seat. Builds body f1, seat pocket f2, shaft bore f3."""
    body_r = outer_diameter / 2.0 + wall
    return [
        NewSketch(plane="XY"),                                    # sk1
        AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=body_r),        # e1
        Constrain(kind="radius", a="e1", value=body_r),
        Extrude(sketch="sk1", distance=width + wall),             # f1 (solid)
        Hole(face_or_sketch="", x=0.0, y=0.0, diameter=outer_diameter,
             through=False, depth=width, kind="counterbore"),     # f2 seat pocket
        Hole(face_or_sketch="", x=0.0, y=0.0, diameter=bore, through=True),  # f3 shaft bore
    ]


# ---------------------------------------------------------------------------
# Model Cards (schema + valid ranges + function tags)
# ---------------------------------------------------------------------------
def flange_card() -> ModelCard:
    return ModelCard(
        name="flange",
        function_tags=["flange", "mounting", "bolt-circle", "coupling", "plate", "disc"],
        description="Round mounting flange: a bored disc with a bolt circle.",
        build=flange_ops,
        param_schema={
            "diameter":    {"type": "float", "default": 60.0, "min": 5.0,  "max": 1000.0, "doc": "outer disc diameter (mm)"},
            "bolt_circle": {"type": "float", "default": 45.0, "min": 2.0,  "max": 990.0,  "doc": "bolt-circle (pitch) diameter (mm)"},
            "n_holes":     {"type": "int",   "default": 4,    "min": 2,    "max": 24,     "doc": "number of bolt holes"},
            "thickness":   {"type": "float", "default": 6.0,  "min": 0.5,  "max": 200.0,  "doc": "flange thickness (mm)"},
            "bore":        {"type": "float", "default": 20.0, "min": 1.0,  "max": 900.0,  "doc": "central bore diameter (mm)"},
            "bolt_hole":   {"type": "float", "default": 6.6,  "min": 0.5,  "max": 100.0,  "doc": "bolt clearance-hole diameter (mm)"},
        },
    )


def bracket_card() -> ModelCard:
    return ModelCard(
        name="bracket",
        function_tags=["bracket", "mounting", "plate", "fastener-mount", "support"],
        description="Flat mounting bracket: a plate with a central through-hole.",
        build=bracket_part_ops,
        param_schema={
            "width":     {"type": "float", "default": 40.0, "min": 5.0,  "max": 1000.0, "doc": "plate width x (mm)"},
            "height":    {"type": "float", "default": 30.0, "min": 5.0,  "max": 1000.0, "doc": "plate height y (mm)"},
            "thickness": {"type": "float", "default": 4.0,  "min": 0.5,  "max": 200.0,  "doc": "plate thickness (mm)"},
            "hole_r":    {"type": "float", "default": 3.5,  "min": 0.5,  "max": 100.0,  "doc": "central hole radius (mm)"},
        },
    )


def spur_gear_blank_card() -> ModelCard:
    return ModelCard(
        name="spur_gear_blank",
        function_tags=["gear", "spur-gear", "transmission", "blank", "wheel"],
        description="Spur-gear BLANK (pitch-diameter cylinder + bore); no teeth.",
        build=spur_gear_blank_ops,
        notes="BLANK ONLY: a plain cylinder at the pitch diameter with a bore. "
              "Involute tooth geometry (profile, pressure angle, addendum/"
              "dedendum) is NOT generated.",
        param_schema={
            "module":    {"type": "float", "default": 2.0, "min": 0.2, "max": 50.0,   "doc": "gear module m (mm/tooth)"},
            "teeth":     {"type": "int",   "default": 20,  "min": 6,   "max": 400,    "doc": "number of teeth z"},
            "thickness": {"type": "float", "default": 8.0, "min": 0.5, "max": 300.0,  "doc": "face width / thickness (mm)"},
            "bore":      {"type": "float", "default": 8.0, "min": 1.0, "max": 500.0,  "doc": "central shaft bore diameter (mm)"},
        },
    )


def shaft_card() -> ModelCard:
    return ModelCard(
        name="shaft",
        function_tags=["shaft", "axle", "pin", "rod", "transmission"],
        description="Plain cylindrical shaft (diameter x length).",
        build=shaft_ops,
        param_schema={
            "diameter": {"type": "float", "default": 10.0, "min": 0.5, "max": 500.0,  "doc": "shaft diameter (mm)"},
            "length":   {"type": "float", "default": 50.0, "min": 1.0, "max": 5000.0, "doc": "shaft length (mm)"},
        },
    )


def bearing_seat_card() -> ModelCard:
    return ModelCard(
        name="bearing_seat",
        function_tags=["bearing", "bearing-seat", "housing", "bore", "mount"],
        description="Housing seat for a rolling-element bearing (pocket + shaft bore).",
        build=bearing_seat_ops,
        param_schema={
            "bore":           {"type": "float", "default": 8.0,  "min": 1.0, "max": 500.0, "doc": "shaft through-bore diameter (mm)"},
            "outer_diameter": {"type": "float", "default": 22.0, "min": 3.0, "max": 700.0, "doc": "bearing outer diameter / seat pocket (mm)"},
            "width":          {"type": "float", "default": 7.0,  "min": 1.0, "max": 200.0, "doc": "bearing width / pocket depth (mm)"},
            "wall":           {"type": "float", "default": 3.0,  "min": 0.5, "max": 100.0, "doc": "radial wall around the seat (mm)"},
        },
    )


def default_cards() -> List[ModelCard]:
    """The parts shipped pre-registered in a default catalog."""
    return [
        flange_card(),
        bracket_card(),
        spur_gear_blank_card(),
        shaft_card(),
        bearing_seat_card(),
    ]
