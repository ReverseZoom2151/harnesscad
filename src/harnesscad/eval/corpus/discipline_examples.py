"""OpenCAD's discipline-tagged examples. A PORT RECORD -- NOT A BENCHMARK SPLIT.

Source: OpenCAD-Examples (resources/cad_repos/OpenCAD-Examples-main). That repo
organises its example scripts by DISCIPLINE -- hardware (mounting bracket, PCB
carrier), software (HMI panel), firmware (programmer fixture), device (cable
grommet) -- plus an agents/ example that generates example-style code
deterministically when no LLM is configured. What is ported here is all five
example parts, at their exact dimensions, PLUS that deterministic no-LLM
code-generation mode (:func:`generate_example_script`).

WHY THIS MODULE IS IMPORTED BY NOTHING, ON PURPOSE
--------------------------------------------------
It is an orphan in the module graph and that is the HONEST state, not a wiring
debt. It was written to look like a third brief source next to
:mod:`harnesscad.eval.corpus.analytic` and :mod:`harnesscad.eval.corpus.standards`,
and it cannot be one. Three reasons, in order of how much they cost to learn:

1.  **It would break the dev/heldout invariant for nothing.**
    :mod:`harnesscad.eval.corpus.dev` states it plainly: every brief there is a
    call into ``analytic`` or ``standards``, so the DIFFERENCE between dev and
    heldout is a set of NUMBERS -- not a second hand-written file that can drift
    into a different opinion about what correct means. This module is that second
    hand-written file. And the three examples it COULD express exactly (panel,
    fixture, grommet) are already ``analytic.plate_with_holes`` and
    ``analytic.tube`` with different constants -- so the trade on offer is: spend
    the invariant, receive a ``discipline`` tag that :class:`~.spec.Brief` has no
    field for and no grader reads.

2.  **Two of the five expectations are NOT the volume of the part the ops build.**
    Measured on cadquery (exact OCCT B-rep, volume tolerance 1e-9), against the
    numbers stored below:

      * ``opencad-hardware-mounting-bracket`` -- 8793.5270 built vs 8833.4514
        stored, 4.5e-3 relative, 4.5 MILLION times the tolerance. The part carries
        ``Fillet(('>Z',), 0.75)`` and the closed form here excludes it. There is no
        cheat available: ``analytic.filleted_plate`` gets an exact answer from
        Steiner's formula only for an ALL-edge fillet on a solid box, and this is a
        top-edge-only fillet on a five-hole plate. That volume is not in closed
        form, so this example cannot state one.
      * ``opencad-software-hmi-panel`` -- 22610.8806 built vs 22570.4869 stored,
        1.8e-3 relative. THE STORED NUMBER IS WRONG. Its 9 mm hole at (104, 50) and
        its 3 mm hole at (108, 58) are 8.944 mm apart and OVERLAP; ``w*h - sum(pi*r^2)``
        subtracts the shared lens twice. The lens is 13.4645 mm2, times the 3 mm
        depth is 40.3936 mm3 -- and the built-minus-stored gap is 40.3937 mm3.
        The defect is left in place deliberately: it is evidence for point 3, and
        deciding whether the port keeps OpenCAD's overlapping geometry or restates
        the formula is a call for a human, not a silent edit.

    ``opencad-hardware-pcb-carrier`` measures exact (1.2e-16) but only because
    ``Part.offset`` lowers to a NAMED NO-OP (CISP has no 2D face-offset op), so the
    stream does not build the 0.4 mm reinforcement its own brief text asks for.
    Three of five examples are therefore not a part that matches its prompt, its
    stored volume, or both -- and ``run.py``'s step 2 is the reference self-test:
    a corpus whose reference solution fails its own grader is measuring the
    engine's bugs and billing them to the model.

3.  **Its grader shares the answer key's blind spot.** :func:`verify_example` does
    not measure geometry. It re-derives ``w*h - sum(pi*r^2)`` from the recorded ops
    and compares that to a stored number computed by ``w*h - sum(pi*r^2)``. So the
    panel's 40 mm3 error passes the check, loudly and green, because ONE HAND WROTE
    BOTH SIDES -- the exact failure :mod:`harnesscad.eval.corpus` exists to remove
    ("Fleet and corpus shared the blind spot, because one hand wrote both"). Only an
    independent kernel found it. A ``--selfcheck`` pass here means "the arithmetic
    is self-consistent", and it must not be read as "the parts are right".

Rooting it in :data:`harnesscad.registry.ROOTS` for having ``main()`` and
``--selfcheck`` would be the same dishonesty in a different costume: every module in
this campaign has those by convention, and ROOTS means "imported by nothing because
it IS the entry point". This is not an entry point. It is a record.

WHAT IT ACTUALLY IS, AND WHAT IT IS STILL GOOD FOR
--------------------------------------------------
A fidelity record of the OpenCAD port: the five example dimensions, verbatim, and
proof that :mod:`harnesscad.domain.programs.fluent_builder` lowers them to op
streams whose arithmetic matches OpenCAD's own numbers. It is also the ONLY caller
of that builder anywhere in the tree -- the fluent surface's sole exercise.

An honest route exists and is not yet earned. Points 2 and 3 describe a real
instrument: closed form vs. an independent B-rep kernel is what
:mod:`harnesscad.eval.corpus.consensus` does for a living, and it is how the panel
bug surfaced. Wire this module there -- as a corroboration input, never as a brief
source -- once the bracket has a defensible volume and the panel's formula handles
overlapping holes. Until then it is imported by nothing, and saying so in this
docstring is the accurate answer rather than the tidy one.

Stdlib only, deterministic, no clock, no randomness.
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

__all__ = [
    "DisciplineExample",
    "DISCIPLINES",
    "all_examples",
    "example_by_id",
    "verify_example",
    "generate_example_script",
    "main",
]

PI = math.pi
DISCIPLINES = ("hardware", "software", "firmware", "device")

# Relative tolerance for the closed-form cross-check (pure arithmetic on both
# sides, so this only absorbs float rounding).
_REL_TOL = 1e-9


@dataclass(frozen=True)
class DisciplineExample:
    """One OpenCAD example: discipline tag, brief, builder, closed-form truth."""

    example_id: str
    discipline: str  # hardware | software | firmware | device
    title: str
    brief: str
    build: Callable[[], Part] = field(compare=False)
    expected_area_mm2: float = 0.0   # plate profile area (annulus for grommet)
    expected_volume_mm3: float = 0.0
    hole_count: int = 0
    depth: float = 0.0


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


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #
def all_examples() -> List[DisciplineExample]:
    """The five OpenCAD examples, discipline-tagged, with closed-form truth."""
    bracket_area, bracket_vol = _plate_expectation(80, 30, (3, 3, 3, 3, 5), 4)
    carrier_area, carrier_vol = _plate_expectation(
        90, 60, (2.2, 2.2, 2.2, 2.2, 7), 3
    )
    panel_area, panel_vol = _plate_expectation(
        120, 70, (6, 6, 6, 6, 6, 9, 3, 3), 3
    )
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
            expected_area_mm2=bracket_area,
            expected_volume_mm3=bracket_vol,
            hole_count=5,
            depth=4.0,
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
            expected_volume_mm3=carrier_vol,
            hole_count=5,
            depth=3.0,
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
        ),
    ]


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
    """Recompute the closed-form volume FROM THE RECORDED OPS and compare.

    NOT AN INDEPENDENT CHECK, and it must not be quoted as one. This re-derives
    ``w*h - sum(pi*r^2)`` and compares it to a stored number that was computed by
    ``w*h - sum(pi*r^2)``: both sides share every assumption, so it catches a
    drifted dimension or a broken lowering and CANNOT catch a wrong formula. It
    reports the HMI panel green while that panel's stored volume is 40.3936 mm3
    wrong (overlapping holes, lens subtracted twice -- see the module docstring),
    and it ignores Fillet entirely, which is why the bracket's stored number is not
    the bracket's volume. An exact B-rep kernel found both; this function found
    neither.

    Walks the example's lowered op stream: each Extrude closes a profile whose
    area is the sum of its AddRectangle areas; each through Hole subtracts a
    ``pi*(d/2)^2 * depth`` cylinder from the last extrude -- ASSUMING HOLES DO NOT
    OVERLAP, which in this corpus is false; each cylinder Primitive contributes
    ``pi*r^2*h``; a Boolean cut subtracts the tool body's volume. Returns
    ``(ok, recomputed_volume, detail)``.
    """
    part = example.build()
    ops = part.ops()

    volume = 0.0
    rect_area = 0.0
    last_depth = 0.0
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
        elif isinstance(op, Hole):
            if not op.through:
                return (False, volume, "blind holes are outside this grader")
            r = op.diameter / 2.0
            volume -= PI * r * r * last_depth
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
        # Fillet contributes a rounding correction the closed form here
        # deliberately excludes (OpenCAD's expectations are pre-fillet, and
        # the stored expectation matches); everything else is a no-op note.

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

    header = "%-38s %-9s %5s %6s %14s %14s  %s" % (
        "example_id", "disc", "holes", "depth", "area mm2", "volume mm3", "verify"
    )
    print(header)
    print("-" * len(header))
    failures = 0
    for example in examples:
        ok, recomputed, detail = verify_example(example)
        if not ok:
            failures += 1
        print(
            "%-38s %-9s %5d %6g %14.4f %14.4f  %s"
            % (
                example.example_id,
                example.discipline,
                example.hole_count,
                example.depth,
                example.expected_area_mm2,
                example.expected_volume_mm3,
                "OK" if ok else "FAIL (%s)" % detail,
            )
        )
        # The answer key must pass its own grader.
        assert ok, "%s: %s" % (example.example_id, detail)
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

    assert failures == 0
    print("SELFCHECK OK: the arithmetic is SELF-CONSISTENT -- all 5 examples")
    print("agree with a stored number derived from the same formula. This is NOT")
    print("evidence the parts are right: the bracket's number excludes its fillet")
    print("and the HMI panel's is 40.3936 mm3 wrong (overlapping holes, lens")
    print("subtracted twice). Both are invisible from here by construction; an")
    print("exact B-rep kernel found them. See the module docstring.")
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
