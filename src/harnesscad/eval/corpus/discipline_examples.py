"""OpenCAD discipline examples: a fidelity record with an explicit truth boundary.

OpenCAD-Examples contributes five scripts across hardware, software, firmware,
and device disciplines. All five are retained as source-faithful records and as
inputs to the deterministic no-LLM script generator. Only three are safe to turn
into corpus briefs:

* the HMI panel, whose closed form now adds back the exact lens shared by its
  overlapping 9 mm and 3 mm holes;
* the programmer fixture; and
* the cable grommet.

The mounting bracket is RETIRED from scoring because its top-edge-only fillet has
no stated closed-form volume. The PCB carrier is RETIRED because its requested
0.4 mm reinforcement lowers to no CISP operation, so its stream does not build
the part described by its text. Retired records remain visible, but cannot become
a benchmark reference by accident.

``corroboration_briefs`` exposes only the verified trio as analytic DEV briefs;
``consensus.corroborate_discipline_examples`` is the real consumer. They are
corroboration inputs, not a new dev split or a substitute for held-out data.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import (
    AddRectangle,
    Boolean,
    Extrude,
    Hole,
    Primitive,
)
from harnesscad.domain.programs.fluent_builder import Part, Sketch
from harnesscad.eval.corpus.spec import Brief, Source, Split

__all__ = [
    "DisciplineExample",
    "DISCIPLINES",
    "TRUSTED",
    "RETIRED",
    "all_examples",
    "trusted_examples",
    "corroboration_briefs",
    "example_by_id",
    "verify_example",
    "generate_example_script",
    "main",
]

PI = math.pi
DISCIPLINES = ("hardware", "software", "firmware", "device")
TRUSTED = "trusted"
RETIRED = "retired"

# Relative tolerance for the closed-form cross-check (pure arithmetic on both
# sides, so this only absorbs float rounding).
_REL_TOL = 1e-9


@dataclass(frozen=True)
class DisciplineExample:
    """One OpenCAD record, with scoring eligibility stated rather than implied."""

    example_id: str
    discipline: str  # hardware | software | firmware | device
    title: str
    brief: str
    build: Callable[[], Part] = field(compare=False)
    expected_area_mm2: Optional[float] = None
    expected_volume_mm3: Optional[float] = None
    hole_count: int = 0
    depth: float = 0.0
    bbox_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    genus: Optional[int] = None
    truth_status: str = TRUSTED
    retirement_reason: str = ""


# --------------------------------------------------------------------------- #
# the five example builders (dimensions verbatim from the OpenCAD scripts)
# --------------------------------------------------------------------------- #
def _build_mounting_bracket() -> Part:
    profile = (
        Sketch(name="Bracket Profile")
        .rect(80, 30)
        .circle(3, center=(8, 8), subtract=True)
        .circle(3, center=(72, 8), subtract=True)
        .circle(3, center=(8, 22), subtract=True)
        .circle(3, center=(72, 22), subtract=True)
        .circle(5, center=(40, 15), subtract=True)
    )
    return Part(name="Mounting Bracket").extrude(
        profile, depth=4, name="Bracket Body"
    ).fillet(edges="top", radius=0.75, name="Bracket Edge Relief")


def _build_pcb_carrier() -> Part:
    profile = (
        Sketch(name="PCB Carrier Profile")
        .rect(90, 60)
        .circle(2.2, center=(8, 8), subtract=True)
        .circle(2.2, center=(82, 8), subtract=True)
        .circle(2.2, center=(8, 52), subtract=True)
        .circle(2.2, center=(82, 52), subtract=True)
        .circle(7, center=(45, 30), subtract=True)
    )
    return Part(name="PCB Carrier").extrude(
        profile, depth=3, name="Carrier Plate"
    ).offset(0.4, name="Carrier Reinforcement")


def _build_hmi_panel() -> Part:
    profile = (
        Sketch(name="HMI Panel Profile")
        .rect(120, 70)
        .circle(6, center=(20, 18), subtract=True)
        .circle(6, center=(40, 18), subtract=True)
        .circle(6, center=(60, 18), subtract=True)
        .circle(6, center=(80, 18), subtract=True)
        .circle(6, center=(100, 18), subtract=True)
        .circle(9, center=(104, 50), subtract=True)
        .circle(3, center=(12, 58), subtract=True)
        .circle(3, center=(108, 58), subtract=True)
    )
    return Part(name="HMI Panel").extrude(profile, depth=3, name="Panel Blank")


def _build_programmer_fixture() -> Part:
    profile = (
        Sketch(name="Programmer Fixture Profile")
        .rect(70, 40)
        .circle(2.5, center=(10, 10), subtract=True)
        .circle(2.5, center=(60, 10), subtract=True)
        .circle(2.5, center=(10, 30), subtract=True)
        .circle(2.5, center=(60, 30), subtract=True)
        .circle(4, center=(25, 20), subtract=True)
        .circle(4, center=(35, 20), subtract=True)
        .circle(4, center=(45, 20), subtract=True)
    )
    return Part(name="Programmer Fixture").extrude(
        profile, depth=5, name="Fixture Plate"
    )


def _build_cable_grommet() -> Part:
    outer = Part(name="Outer Grommet").cylinder(14, 10, name="Outer Cylinder")
    inner = Part(name="Inner Clearance").cylinder(8, 10, name="Inner Cylinder")
    return outer.cut(inner, name="Cable Passage")


def _plate_expectation(
    w: float, h: float, radii: Tuple[float, ...], depth: float
) -> Tuple[float, float]:
    area = w * h - sum(PI * r * r for r in radii)
    return area, area * depth


def _circle_lens_area(r1: float, r2: float, distance: float) -> float:
    """Exact intersection area of two disks, with the disjoint/contained cases."""
    if distance >= r1 + r2:
        return 0.0
    if distance <= abs(r1 - r2):
        return PI * min(r1, r2) ** 2
    a1 = math.acos((distance * distance + r1 * r1 - r2 * r2) / (2.0 * distance * r1))
    a2 = math.acos((distance * distance + r2 * r2 - r1 * r1) / (2.0 * distance * r2))
    radical = (-distance + r1 + r2) * (distance + r1 - r2)
    radical *= (distance - r1 + r2) * (distance + r1 + r2)
    return r1 * r1 * a1 + r2 * r2 * a2 - 0.5 * math.sqrt(max(0.0, radical))


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #
def all_examples() -> List[DisciplineExample]:
    """All five source-faithful examples, including explicitly retired records."""
    carrier_area, carrier_vol = _plate_expectation(
        90, 60, (2.2, 2.2, 2.2, 2.2, 7), 3
    )
    panel_area, panel_vol = _plate_expectation(
        120, 70, (6, 6, 6, 6, 6, 9, 3, 3), 3
    )
    # The encoder (r=9 at 104,50) overlaps the indicator (r=3 at 108,58).
    # Subtracting every disk independently removes their shared lens twice.
    panel_lens = _circle_lens_area(9.0, 3.0, math.hypot(4.0, 8.0))
    panel_area += panel_lens
    panel_vol += panel_lens * 3.0
    fixture_area, fixture_vol = _plate_expectation(
        70, 40, (2.5, 2.5, 2.5, 2.5, 4, 4, 4), 5
    )
    grommet_area = PI * (14.0 * 14.0 - 8.0 * 8.0)
    grommet_vol = grommet_area * 10.0

    return [
        DisciplineExample(
            example_id="opencad-hardware-mounting-bracket",
            discipline="hardware",
            title="Mounting bracket",
            brief="Mounting bracket with corner fasteners and a cable "
            "pass-through: an 80 mm by 30 mm plate, 4 mm thick, with four "
            "3 mm-radius fastener holes at the corners and one 5 mm-radius "
            "centre hole, top edges filleted 0.75 mm.",
            build=_build_mounting_bracket,
            expected_area_mm2=None,
            expected_volume_mm3=None,
            hole_count=5,
            depth=4.0,
            bbox_mm=(80.0, 30.0, 4.0),
            truth_status=RETIRED,
            retirement_reason=(
                "top-edge-only 0.75 mm fillet has no stated closed-form volume; "
                "the former pre-fillet number was not the built part's volume"
            ),
        ),
        DisciplineExample(
            example_id="opencad-hardware-pcb-carrier",
            discipline="hardware",
            title="PCB carrier",
            brief="Carrier plate for a controller or sensor PCB: a 90 mm by "
            "60 mm plate, 3 mm thick, with four 2.2 mm-radius mounting holes "
            "at the corners and one 7 mm-radius clearance opening in the "
            "centre, with a 0.4 mm reinforcement offset.",
            build=_build_pcb_carrier,
            expected_area_mm2=carrier_area,
            expected_volume_mm3=None,
            hole_count=5,
            depth=3.0,
            bbox_mm=(90.0, 60.0, 3.0),
            truth_status=RETIRED,
            retirement_reason=(
                "the requested 0.4 mm reinforcement lowers to no CISP operation, "
                "so the reference stream omits a requested feature"
            ),
        ),
        DisciplineExample(
            example_id="opencad-software-hmi-panel",
            discipline="software",
            title="HMI panel",
            brief="Front panel for a software-driven operator interface: a "
            "120 mm by 70 mm plate, 3 mm thick, with five 6 mm-radius button "
            "holes in a row, one 9 mm-radius encoder hole, and two "
            "3 mm-radius indicator holes.",
            build=_build_hmi_panel,
            expected_area_mm2=panel_area,
            expected_volume_mm3=panel_vol,
            hole_count=8,
            depth=3.0,
            bbox_mm=(120.0, 70.0, 3.0),
            genus=7,
        ),
        DisciplineExample(
            example_id="opencad-firmware-programmer-fixture",
            discipline="firmware",
            title="Programmer fixture",
            brief="Fixture plate for firmware flashing or debug access: a "
            "70 mm by 40 mm plate, 5 mm thick, with four 2.5 mm-radius "
            "alignment-pin holes at the corners and three 4 mm-radius "
            "programming-header openings in a row.",
            build=_build_programmer_fixture,
            expected_area_mm2=fixture_area,
            expected_volume_mm3=fixture_vol,
            hole_count=7,
            depth=5.0,
            bbox_mm=(70.0, 40.0, 5.0),
            genus=7,
        ),
        DisciplineExample(
            example_id="opencad-device-cable-grommet",
            discipline="device",
            title="Cable grommet",
            brief="Cable-management grommet built from concentric cylindrical "
            "primitives: a 14 mm-radius, 10 mm-tall outer cylinder with an "
            "8 mm-radius, 10 mm-tall inner clearance cut through it.",
            build=_build_cable_grommet,
            expected_area_mm2=grommet_area,
            expected_volume_mm3=grommet_vol,
            hole_count=1,
            depth=10.0,
            bbox_mm=(28.0, 28.0, 10.0),
            genus=1,
        ),
    ]


def trusted_examples() -> List[DisciplineExample]:
    """Records whose prompt, stream, and independent closed form all agree."""
    return [example for example in all_examples() if example.truth_status == TRUSTED]


def corroboration_briefs(split: str = Split.DEV) -> Tuple[Brief, ...]:
    """The verified OpenCAD trio as analytic inputs for differential corroboration.

    This does not alter ``dev.BRIEFS``: these are a named, opt-in corroboration
    set. The OpenCAD file supplies dimensions; the volume is independently
    derived by geometry, including the panel's disk-union correction.
    """
    briefs = []
    for example in trusted_examples():
        assert example.expected_volume_mm3 is not None
        briefs.append(
            Brief(
                id="discipline_" + example.example_id,
                split=split,
                source=Source.ANALYTIC,
                citation=(
                    "OpenCAD-Examples %s; volume independently derived from the "
                    "stated primitive geometry" % example.example_id
                ),
                text=example.brief,
                reference=tuple(example.build().ops()),
                volume=example.expected_volume_mm3,
                bbox=example.bbox_mm,
                genus=example.genus,
                note="OpenCAD discipline record used only for differential corroboration.",
            )
        )
    return tuple(briefs)


def example_by_id(example_id: str) -> DisciplineExample:
    """Look up one example; raises KeyError with the known ids on a miss."""
    for example in all_examples():
        if example.example_id == example_id:
            return example
    known = ", ".join(e.example_id for e in all_examples())
    raise KeyError("unknown example id %r (known: %s)" % (example_id, known))


# --------------------------------------------------------------------------- #
# the grader the answer key must pass
# --------------------------------------------------------------------------- #
def verify_example(example: DisciplineExample) -> Tuple[bool, float, str]:
    """Cross-check a *trusted* record's closed form against its lowered stream.

    Retired records never green-light: their missing truth is a finding, not a
    zero or a tolerance to tune. For planar through-holes this handles pairwise
    disk overlap exactly and refuses a triple-overlap configuration rather than
    pretending pairwise inclusion-exclusion is enough.
    """
    if example.truth_status != TRUSTED or example.expected_volume_mm3 is None:
        return False, 0.0, "retired: " + example.retirement_reason

    part = example.build()
    ops = part.ops()

    volume = 0.0
    rect_area = 0.0
    last_depth = 0.0
    holes: List[Tuple[float, float, float]] = []
    body_volumes: List[float] = []  # per body, in emission order (f1, f2, ...)
    for op in ops:
        if isinstance(op, AddRectangle):
            rect_area += op.w * op.h
        elif isinstance(op, Extrude):
            body = rect_area * op.distance
            body_volumes.append(body)
            volume += body
            rect_area = 0.0
            last_depth = op.distance
            holes = []
        elif isinstance(op, Hole):
            if not op.through:
                return (False, volume, "blind holes are outside this grader")
            r = op.diameter / 2.0
            volume -= PI * r * r * last_depth
            overlaps = []
            for x, y, prior_r in holes:
                distance = math.hypot(op.x - x, op.y - y)
                if distance < r + prior_r:
                    overlaps.append((prior_r, distance))
            if len(overlaps) > 1:
                return (False, volume, "triple-or-higher hole overlap is not supported")
            if overlaps:
                prior_r, distance = overlaps[0]
                volume += _circle_lens_area(prior_r, r, distance) * last_depth
            holes.append((op.x, op.y, r))
        elif isinstance(op, Primitive):
            if op.shape != "cylinder":
                return (False, volume, "non-cylinder primitive %r" % op.shape)
            body = PI * op.r * op.r * op.h
            body_volumes.append(body)
            volume += body
        elif isinstance(op, Boolean):
            if op.kind != "cut":
                return (False, volume, "non-cut boolean %r" % op.kind)
            # tool is "f<n>": subtract that body twice over (it was added once
            # when built, and the cut removes it from the target).
            try:
                tool_index = int(op.tool.lstrip("f")) - 1
            except ValueError:
                return (False, volume, "unparseable tool ref %r" % op.tool)
            if tool_index < 0 or tool_index >= len(body_volumes):
                return (False, volume, "tool ref %r out of range" % op.tool)
            volume -= 2.0 * body_volumes[tool_index]
        elif type(op).__name__ == "Fillet":
            return False, volume, "fillet has no closed-form volume in this corpus"

    expected = example.expected_volume_mm3
    ok = math.isclose(volume, expected, rel_tol=_REL_TOL)
    detail = "recomputed %.6f mm3 vs expected %.6f mm3" % (volume, expected)
    return ok, volume, detail


# --------------------------------------------------------------------------- #
# the deterministic no-LLM code generator (OpenCAD agents/ example)
# --------------------------------------------------------------------------- #
_SCRIPT_TEMPLATES: Dict[str, str] = {
    "bracket": '''\
"""Hardware example: mounting bracket with corner fasteners and a cable pass-through."""

from harnesscad.domain.programs.fluent_builder import Part, Sketch


bracket_profile = (
    Sketch(name="Bracket Profile")
    .rect(80, 30)
    .circle(3, center=(8, 8), subtract=True)
    .circle(3, center=(72, 8), subtract=True)
    .circle(3, center=(8, 22), subtract=True)
    .circle(3, center=(72, 22), subtract=True)
    .circle(5, center=(40, 15), subtract=True)
)

part = Part(name="Mounting Bracket").extrude(bracket_profile, depth=4, name="Bracket Body").fillet(
    edges="top",
    radius=0.75,
    name="Bracket Edge Relief",
)
''',
    "carrier": '''\
"""Hardware example: PCB carrier plate with mounting holes and a clearance opening."""

from harnesscad.domain.programs.fluent_builder import Part, Sketch


carrier_profile = (
    Sketch(name="PCB Carrier Profile")
    .rect(90, 60)
    .circle(2.2, center=(8, 8), subtract=True)
    .circle(2.2, center=(82, 8), subtract=True)
    .circle(2.2, center=(8, 52), subtract=True)
    .circle(2.2, center=(82, 52), subtract=True)
    .circle(7, center=(45, 30), subtract=True)
)

part = Part(name="PCB Carrier").extrude(carrier_profile, depth=3, name="Carrier Plate").offset(
    0.4,
    name="Carrier Reinforcement",
)
''',
    "panel": '''\
"""Software example: operator panel for a display, buttons, and encoder access."""

from harnesscad.domain.programs.fluent_builder import Part, Sketch


panel_profile = (
    Sketch(name="HMI Panel Profile")
    .rect(120, 70)
    .circle(6, center=(20, 18), subtract=True)
    .circle(6, center=(40, 18), subtract=True)
    .circle(6, center=(60, 18), subtract=True)
    .circle(6, center=(80, 18), subtract=True)
    .circle(6, center=(100, 18), subtract=True)
    .circle(9, center=(104, 50), subtract=True)
    .circle(3, center=(12, 58), subtract=True)
    .circle(3, center=(108, 58), subtract=True)
)

part = Part(name="HMI Panel").extrude(panel_profile, depth=3, name="Panel Blank")
''',
    "fixture": '''\
"""Firmware example: fixture plate for programming headers and alignment pins."""

from harnesscad.domain.programs.fluent_builder import Part, Sketch


fixture_profile = (
    Sketch(name="Programmer Fixture Profile")
    .rect(70, 40)
    .circle(2.5, center=(10, 10), subtract=True)
    .circle(2.5, center=(60, 10), subtract=True)
    .circle(2.5, center=(10, 30), subtract=True)
    .circle(2.5, center=(60, 30), subtract=True)
    .circle(4, center=(25, 20), subtract=True)
    .circle(4, center=(35, 20), subtract=True)
    .circle(4, center=(45, 20), subtract=True)
)

part = Part(name="Programmer Fixture").extrude(fixture_profile, depth=5, name="Fixture Plate")
''',
    "grommet": '''\
"""Full-device example: cable grommet built from concentric cylindrical primitives."""

from harnesscad.domain.programs.fluent_builder import Part


outer = Part(name="Outer Grommet").cylinder(14, 10, name="Outer Cylinder")
inner = Part(name="Inner Clearance").cylinder(8, 10, name="Inner Cylinder")

part = outer.cut(inner, name="Cable Passage")
''',
}

# Keyword -> template key, checked in a fixed order (first hit wins) so the
# generator is deterministic for any brief text.
_KEYWORD_ORDER: Tuple[Tuple[str, str], ...] = (
    ("bracket", "bracket"),
    ("carrier", "carrier"),
    ("pcb", "carrier"),
    ("panel", "panel"),
    ("hmi", "panel"),
    ("fixture", "fixture"),
    ("programmer", "fixture"),
    ("grommet", "grommet"),
    ("cable", "grommet"),
)


def generate_example_script(brief: str) -> str:
    """Deterministic, no-LLM code generation (OpenCAD's built-in generator).

    Keyword-matches the brief against the five example families and returns
    the corresponding fluent-builder Python source text. This is the ported
    fallback the OpenCAD agents/ example uses when no provider is configured:
    same request surface, zero model calls, byte-identical output for the
    same brief. Raises ValueError when no family keyword matches.
    """
    text = brief.lower()
    for keyword, key in _KEYWORD_ORDER:
        if keyword in text:
            return _SCRIPT_TEMPLATES[key]
    raise ValueError(
        "no example family matched the brief; known keywords: "
        + ", ".join(sorted({k for k, _ in _KEYWORD_ORDER}))
    )


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #
def _selfcheck() -> int:
    examples = all_examples()
    assert len(examples) == 5
    assert {e.discipline for e in examples} == set(DISCIPLINES)

    header = "%-38s %-9s %-8s %5s %6s %14s  %s" % (
        "example_id", "disc", "truth", "holes", "depth", "volume mm3", "verify"
    )
    print(header)
    print("-" * len(header))
    for example in examples:
        ok, recomputed, detail = verify_example(example)
        print(
            "%-38s %-9s %-8s %5d %6g %14s  %s"
            % (
                example.example_id,
                example.discipline,
                example.truth_status,
                example.hole_count,
                example.depth,
                ("%.4f" % example.expected_volume_mm3)
                if example.expected_volume_mm3 is not None else "--",
                "OK" if ok else detail,
            )
        )
        if example.truth_status == TRUSTED:
            assert ok, "%s: %s" % (example.example_id, detail)
        else:
            assert not ok and detail.startswith("retired:"), detail
        # hole_count must match the recorded ops too.
        part = example.build()
        ops = part.ops()
        holes = sum(1 for op in ops if isinstance(op, Hole))
        cuts = sum(
            1 for op in ops if isinstance(op, Boolean) and op.kind == "cut"
        )
        assert holes + cuts == example.hole_count, (
            "%s: %d holes + %d cuts != declared %d"
            % (example.example_id, holes, cuts, example.hole_count)
        )

    # Lookup and the deterministic generator.
    assert (
        example_by_id("opencad-device-cable-grommet").discipline == "device"
    )
    for phrase, keyword in (
        ("a mounting bracket with corner fasteners", "bracket"),
        ("carrier plate for a controller PCB", "carrier"),
        ("operator HMI panel", "panel"),
        ("firmware programmer fixture", "fixture"),
        ("a cable grommet", "grommet"),
    ):
        script = generate_example_script(phrase)
        assert "fluent_builder import" in script, keyword
        # The generated source must itself be compilable.
        compile(script, "<generated:%s>" % keyword, "exec")
    try:
        generate_example_script("a completely unrelated widget")
    except ValueError:
        pass
    else:
        raise AssertionError("unmatched brief should raise ValueError")

    trusted = trusted_examples()
    briefs = corroboration_briefs()
    assert len(trusted) == len(briefs) == 3
    assert {e.example_id for e in trusted} == {
        "opencad-software-hmi-panel",
        "opencad-firmware-programmer-fixture",
        "opencad-device-cable-grommet",
    }
    print("SELFCHECK OK: 3 independently stated records are corroboration-ready;")
    print("the bracket and PCB carrier remain source-faithful but explicitly retired")
    print("from scoring because their reference stream cannot support their text.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenCAD discipline-tagged example corpus with "
        "closed-form self-grading."
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="verify all five examples against their own closed-form "
        "expectations and print a table.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
