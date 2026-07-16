"""Executable seed skills ported from text-to-cad's CAD skill pack.

Source: earthtojake/text-to-cad (resources/cad_repos/text-to-cad-main),
``skills/cad/references/cad-brief.md`` (the worked mounting-plate and
enclosure briefs) and ``skills/cad/SKILL.md`` (the default assumptions those
briefs rely on). The pack ships this knowledge as prose recipes; here the
same constructions are op-template *expanders* -- executable CISP op streams
-- so they can pass the Voyager gate
(:meth:`~harnesscad.agents.memory.skills.SkillLibrary.add_verified`) and
enter the library as VERIFIED skills, not trusted text.

This is the promotion half of the skill-pack story
(:mod:`harnesscad.agents.memory.skillpack` is the import half): an imported
pack skill is registered UNVERIFIED, and when an expander below matches it,
the caller promotes it through ``add_verified`` -- execution decides, never
provenance.

Dimensional knowledge comes from the ported defaults
(:mod:`harnesscad.domain.standards.cad_defaults`): metric normal clearance
holes (M3/M4/M5 -> 3.4/4.5/5.5 mm) and the 2.0-3.0 mm enclosure wall.

Stdlib-only, deterministic, absolute imports. ``--selfcheck`` runs both seed
skills through a real ``HarnessSession(StubBackend())`` and requires the
Voyager gate to admit them.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable, Dict, List, Optional

from harnesscad.agents.memory.skills import Expander, Skill, SkillLibrary
from harnesscad.core.cisp.ops import (
    AddCircle, AddRectangle, Boolean, Constrain, Extrude, NewSketch, Op,
)
from harnesscad.domain.standards.cad_defaults import (
    clearance_hole_radius, default_wall_thickness,
)

__all__ = [
    "mounting_plate_ops",
    "enclosure_base_ops",
    "mounting_plate_skill",
    "enclosure_base_skill",
    "seed_expanders",
    "add_seed_skills",
    "main",
]


def mounting_plate_ops(w: float = 100.0, h: float = 60.0,
                       thickness: float = 6.0, screw: str = "M4",
                       hole_inset: float = 10.0) -> List[Op]:
    """The cad-brief.md worked example: a mounting plate with four metric
    clearance holes inset from the corners.

    The hole radius is looked up from the ported normal-clearance table, so
    ``screw="M4"`` yields the source skill's 4.5 mm holes. Ids follow the
    stub scheme: plate sketch 'sk1'/'e1'/feature 'f1'; the four hole circles
    share sketch 'sk2' ('e2'..'e5'), extrude to 'f2', boolean-cut from 'f1'.
    """
    if hole_inset <= 0 or hole_inset * 2 >= min(w, h):
        raise ValueError("hole_inset must be positive and fit inside the plate")
    hole_r = clearance_hole_radius(screw)
    ops: List[Op] = [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h),
        Constrain(kind="horizontal", a="e1"),
        Constrain(kind="vertical", a="e1"),
        Constrain(kind="distance", a="e1", value=w),
        Constrain(kind="distance", a="e1", value=h),
        Extrude(sketch="sk1", distance=thickness),          # f1
        NewSketch(plane="XY"),                              # sk2
    ]
    centers = [
        (hole_inset, hole_inset),
        (w - hole_inset, hole_inset),
        (hole_inset, h - hole_inset),
        (w - hole_inset, h - hole_inset),
    ]
    for i, (cx, cy) in enumerate(centers):
        eid = f"e{2 + i}"
        ops.append(AddCircle(sketch="sk2", cx=cx, cy=cy, r=hole_r))
        ops.append(Constrain(kind="distance", a=eid, value=cx))
        ops.append(Constrain(kind="distance", a=eid, value=cy))
        ops.append(Constrain(kind="radius", a=eid, value=hole_r))
    ops.append(Extrude(sketch="sk2", distance=thickness))   # f2
    ops.append(Boolean(kind="cut", target="f1", tool="f2"))
    return ops


def enclosure_base_ops(w: float = 120.0, h: float = 80.0,
                       height: float = 32.0,
                       wall: Optional[float] = None) -> List[Op]:
    """The cad-brief.md enclosure brief's base: a hollow lower shell.

    The cavity is an inner block inset by the wall on all sides and cut down
    to leave a floor of the same wall thickness. ``wall`` defaults to the
    ported 2.0-3.0 mm range midpoint (2.5 mm).
    """
    t = default_wall_thickness() if wall is None else wall
    if t <= 0 or 2 * t >= min(w, h) or t >= height:
        raise ValueError("wall must be positive and leave a cavity")
    cavity_w = w - 2 * t
    cavity_h = h - 2 * t
    cavity_depth = height - t
    return [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h),
        Constrain(kind="horizontal", a="e1"),
        Constrain(kind="vertical", a="e1"),
        Constrain(kind="distance", a="e1", value=w),
        Constrain(kind="distance", a="e1", value=h),
        Extrude(sketch="sk1", distance=height),             # f1
        NewSketch(plane="XY"),                              # sk2
        AddRectangle(sketch="sk2", x=t, y=t, w=cavity_w, h=cavity_h),
        Constrain(kind="horizontal", a="e2"),
        Constrain(kind="vertical", a="e2"),
        Constrain(kind="distance", a="e2", value=cavity_w),
        Constrain(kind="distance", a="e2", value=cavity_h),
        Extrude(sketch="sk2", distance=cavity_depth),       # f2
        Boolean(kind="cut", target="f1", tool="f2"),
    ]


def mounting_plate_skill() -> Skill:
    return Skill(
        name="mounting-plate",
        description="A rectangular mounting plate with four metric normal "
                    "clearance holes inset from the corners; hole diameter "
                    "from the ported M3/M4/M5 clearance table "
                    "(text-to-cad cad-brief example).",
        template=mounting_plate_ops,
        params={
            "w": {"type": "float", "default": 100.0, "doc": "plate width (x)"},
            "h": {"type": "float", "default": 60.0, "doc": "plate height (y)"},
            "thickness": {"type": "float", "default": 6.0, "doc": "extrude depth"},
            "screw": {"type": "str", "default": "M4",
                      "doc": "metric screw size for the clearance holes"},
            "hole_inset": {"type": "float", "default": 10.0,
                           "doc": "hole center inset from each corner"},
        },
        sample_params={"w": 100.0, "h": 60.0, "thickness": 6.0,
                       "screw": "M4", "hole_inset": 10.0},
    )


def enclosure_base_skill() -> Skill:
    return Skill(
        name="enclosure-base",
        description="A hollow enclosure base shell: outer block minus an "
                    "inner cavity inset by the wall thickness, floor kept "
                    "(text-to-cad enclosure brief; 2.0-3.0 mm default wall).",
        template=enclosure_base_ops,
        params={
            "w": {"type": "float", "default": 120.0, "doc": "footprint width (x)"},
            "h": {"type": "float", "default": 80.0, "doc": "footprint depth (y)"},
            "height": {"type": "float", "default": 32.0, "doc": "base height (z)"},
            "wall": {"type": "float", "default": 2.5,
                     "doc": "wall thickness (ported 2.0-3.0 mm range midpoint)"},
        },
        sample_params={"w": 120.0, "h": 80.0, "height": 32.0, "wall": 2.5},
    )


def seed_expanders() -> Dict[str, Expander]:
    """name -> expander, for SkillLibrary.load re-attachment and for
    promoting matching imported pack skills."""
    return {
        "mounting-plate": mounting_plate_ops,
        "enclosure-base": enclosure_base_ops,
    }


def add_seed_skills(library: SkillLibrary,
                    session_factory: Callable[[], Any]) -> List[str]:
    """Run every seed through the Voyager gate; returns the names ADMITTED.

    A seed that fails to build is simply not added -- same monotonic-trust
    contract as :meth:`SkillLibrary.add_verified`.
    """
    admitted: List[str] = []
    for skill in (mounting_plate_skill(), enclosure_base_skill()):
        if library.add_verified(skill, session_factory):
            admitted.append(skill.name)
    return admitted


def _stub_session_factory() -> Callable[[], Any]:
    from harnesscad.core.loop import HarnessSession
    from harnesscad.io.backends.stub import StubBackend
    return lambda: HarnessSession(StubBackend())


def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    ops = mounting_plate_ops()
    holes = [op for op in ops if isinstance(op, AddCircle)]
    check(len(holes) == 4, "four corner holes")
    check(all(abs(h.r - 2.25) < 1e-9 for h in holes),
          "M4 normal clearance radius 2.25 mm flows from the ported table")
    m3 = [op for op in mounting_plate_ops(screw="M3")
          if isinstance(op, AddCircle)]
    check(all(abs(h.r - 1.7) < 1e-9 for h in m3), "M3 clearance radius 1.7 mm")
    try:
        mounting_plate_ops(w=10.0, h=10.0, hole_inset=5.0)
        check(False, "holes that do not fit must raise")
    except ValueError:
        pass

    enc = enclosure_base_ops()
    rects = [op for op in enc if isinstance(op, AddRectangle)]
    check(len(rects) == 2 and abs(rects[1].w - 115.0) < 1e-9,
          "cavity inset by the 2.5 mm default wall on both sides")
    try:
        enclosure_base_ops(w=4.0, h=4.0, height=10.0)
        check(False, "a wall that consumes the part must raise")
    except ValueError:
        pass

    lib = SkillLibrary()
    admitted = add_seed_skills(lib, _stub_session_factory())
    check(admitted == ["mounting-plate", "enclosure-base"],
          f"both seeds pass the Voyager gate: {admitted!r}")
    check(all(lib.get(n).verified for n in admitted),
          "admitted seeds are verified")
    check([s.name for s in lib.find("plate with clearance holes", k=1)]
          == ["mounting-plate"], "retrieval finds the plate seed")

    for message in failures:
        print("selfcheck FAIL: " + message)
    print("selfcheck: %s" % ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="skillpack_seeds",
        description="executable seed skills ported from text-to-cad's CAD "
                    "skill pack (verified through the Voyager gate)")
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the built-in self-test and exit")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    lib = SkillLibrary()
    admitted = add_seed_skills(lib, _stub_session_factory())
    for name in admitted:
        print(f"verified: {name} -- {lib.get(name).description}")
    rejected = [n for n in ("mounting-plate", "enclosure-base")
                if n not in admitted]
    for name in rejected:
        print(f"rejected: {name} (failed the execution gate)")
    return 0 if not rejected else 1


if __name__ == "__main__":
    sys.exit(main())
