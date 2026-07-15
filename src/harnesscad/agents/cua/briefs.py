"""briefs — natural-language design briefs with MACHINE-CHECKABLE targets.

The agent is handed the ``brief`` string only. It is NEVER handed the numbers in
:class:`Target`; those are the acceptance oracle the grader uses AFTER the part
is built and measured, exactly as a human reviewer would check a delivered part
against a spec. A brief is "solved" when the geometry the GUI actually produced
(read back through the real kernel) satisfies its target within tolerance AND the
part is a valid, closed solid.

Every brief here is deliberately inside the parameter-dialog subset the GUI can
drive coordinate-free (a rectangle sketched at the origin and padded IS FreeCAD's
additive Box primitive). Briefs that would need a viewport pick are marked
``needs_pick=True`` and exist to exercise the honest-refusal path, not to be
solved by the box recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Target:
    """The acceptance oracle for one brief. Checked against GUI-measured geometry.

    ``volume`` and ``bbox`` are the exact expected values; ``vol_tol`` /
    ``len_tol`` are absolute tolerances (mm^3 / mm). ``solids`` pins the expected
    solid count so a plan that builds two disjoint lumps where one was asked for
    is not scored as a solve.
    """

    volume: float
    bbox: Tuple[float, float, float]
    solids: int = 1
    vol_tol: float = 1e-4
    len_tol: float = 1e-6

    def satisfied(self, metrics: dict) -> Tuple[bool, List[str]]:
        """(did the measured part meet spec, list of human-readable misses)."""
        misses: List[str] = []
        vol = metrics.get("volume")
        if vol is None:
            return False, ["no volume measured (no solid built)"]
        if abs(float(vol) - self.volume) > self.vol_tol:
            misses.append("volume %.6f != target %.6f (tol %.1e)"
                          % (vol, self.volume, self.vol_tol))
        bbox = metrics.get("bbox") or []
        if len(bbox) != 3:
            misses.append("no bounding box measured")
        else:
            # A box is symmetric under axis permutation, so compare the SORTED
            # extents: "40 x 10 x 10" and "10 x 40 x 10" are the same block, and
            # which axis the agent chose to call length is not part of the spec.
            got = sorted(float(v) for v in bbox)
            want = sorted(self.bbox)
            for g, w in zip(got, want):
                if abs(g - w) > self.len_tol:
                    misses.append("bbox extents %s != target %s (tol %.1e)"
                                  % (["%.4f" % v for v in got],
                                     ["%.4f" % v for v in want], self.len_tol))
                    break
        solids = metrics.get("solids")
        if solids is not None and int(solids) != self.solids:
            misses.append("%d solids, target %d" % (int(solids), self.solids))
        return (not misses), misses


@dataclass(frozen=True)
class Brief:
    id: str
    text: str
    target: Target
    needs_pick: bool = False
    note: str = ""


#: The corpus. Boxes and fused boxes — the subset the GUI drives coordinate-free
#: — plus two pick-only briefs that MUST be refused honestly by the box recipe.
BRIEFS: Tuple[Brief, ...] = (
    Brief(
        id="block_30x20x10",
        text=("Design a solid rectangular block 30 mm long, 20 mm wide and "
              "10 mm tall. Sketch the rectangle at the origin on the XY plane "
              "and pad it to height."),
        target=Target(volume=6000.0, bbox=(30.0, 20.0, 10.0)),
        note="the canonical box; GUI has driven this to volume 6000.0000000000",
    ),
    Brief(
        id="plate_50x40x3",
        text=("Model a thin base plate: 50 mm by 40 mm and only 3 mm thick. "
              "Rectangle at the origin on XY, extruded 3 mm."),
        target=Target(volume=6000.0, bbox=(50.0, 40.0, 3.0)),
        note="a thin plate; same volume as the block, very different shape",
    ),
    Brief(
        id="bar_fractional_37.5",
        text=("Make a small bar 37.5 mm by 12.5 mm in plan, 6.25 mm high. "
              "Sketch at the origin on the XY plane and pad it."),
        target=Target(volume=2929.6875, bbox=(37.5, 12.5, 6.25)),
        note="THE LOCALE TRAP: 37.5 naively typed into a comma-locale FreeCAD "
             "becomes 375 and the volume is 10x out",
    ),
    Brief(
        id="cube_15",
        text="A 15 mm cube, sketched at the origin on XY and padded.",
        target=Target(volume=3375.0, bbox=(15.0, 15.0, 15.0)),
        note="a cube; three equal extents",
    ),
    Brief(
        id="post_10x10x40",
        text=("A square post 10 mm by 10 mm in cross-section and 40 mm tall, "
              "sketched at the origin on XY and extruded upward."),
        target=Target(volume=4000.0, bbox=(10.0, 10.0, 40.0)),
        note="tall thin post; extrude distance dominates",
    ),
    # --- pick-only: exist to be REFUSED honestly, never solved by the box path -
    Brief(
        id="filleted_block",
        text=("Take a 30x20x10 mm block and round its four vertical edges with a "
              "3 mm fillet."),
        target=Target(volume=-1.0, bbox=(0.0, 0.0, 0.0)),
        needs_pick=True,
        note="fillet needs an EDGE pick; the box recipe must refuse it",
    ),
)


def by_id(brief_id: str) -> Brief:
    for b in BRIEFS:
        if b.id == brief_id:
            return b
    raise KeyError("no brief %r (have: %s)"
                   % (brief_id, ", ".join(b.id for b in BRIEFS)))


def buildable() -> List[Brief]:
    """The briefs the coordinate-free GUI subset can actually build."""
    return [b for b in BRIEFS if not b.needs_pick]
