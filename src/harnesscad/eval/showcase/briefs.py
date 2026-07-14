"""The brief set: ten plain-English parts a real engineer would ask for.

Each brief is written the way a request arrives in a shop -- prose, millimetres,
no CAD vocabulary -- but is unambiguous enough to be *graded*: every one carries
the analytic volume of the part it describes, plus the op families the part
cannot be built without. That turns "did the model do it?" from an opinion into
a measurement:

    solved      -- the harness reached a verified solid (no ERROR diagnostics)
    on-brief    -- that solid's measured volume matches the analytic volume, and
                   the required features are actually present in the op stream

A model can be `solved` and not `on-brief`: it built *a* part, just not the one
that was asked for. That distinction is the honest half of this eval and is
reported separately -- a bracket with no holes in it verifies perfectly.

Volumes are exact analytic values (mm^3). The frep backend measures volume from
a marching-cubes mesh of a signed distance field, so its number carries a
sampling error of a few percent; `volume_tol` is set per brief to accommodate
that discretisation, NOT to excuse a wrong part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Brief", "BRIEFS", "brief_by_id", "brief_ids", "grade_geometry"]


@dataclass(frozen=True)
class Brief:
    """One gradeable design request.

    Attributes
    ----------
    id:
        Stable slug; used in file names (``assets/showcase/<id>-<model>.png``).
    text:
        The prose handed to the model. Nothing else about the part is told to it
        -- no op hints, no vocabulary priming beyond the standing system prompt.
    tier:
        1 (single extrude) .. 5 (needs real feature synthesis). Difficulty is
        graduated so a failure can be located, not just counted.
    volume_mm3:
        Analytic volume of the part the brief describes.
    volume_tol:
        Relative tolerance for the measured-vs-analytic comparison. Wider on
        briefs whose geometry is curved or filleted (bigger marching-cubes
        sampling error), never wide enough to admit a different part.
    requires:
        Op-tag alternatives the part cannot be built without: a list of groups,
        each group satisfied when ANY of its tags appears in the op stream.
        ``[["hole", "boolean"]]`` reads "the model must cut the holes somehow".
    rationale:
        What the brief is actually testing.
    """

    id: str
    text: str
    tier: int
    volume_mm3: float
    volume_tol: float = 0.12
    requires: Sequence[Sequence[str]] = field(default_factory=tuple)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "tier": self.tier,
            "volume_mm3": round(self.volume_mm3, 3),
            "volume_tol": self.volume_tol,
            "requires": [list(g) for g in self.requires],
            "rationale": self.rationale,
        }


# --- analytic volumes ------------------------------------------------------
# Written as expressions so the number in the table is checkable by eye.
_PLATE = 60.0 * 40.0 * 6.0                                    # 14400
_DISC = math.pi * 25.0 ** 2 * 5.0                             # 9817.48
_WASHER = math.pi * (15.0 ** 2 - 6.0 ** 2) * 3.0              # 1781.28
_BRACKET = 40.0 * 24.0 * 8.0 - 2.0 * math.pi * 3.0 ** 2 * 8.0  # 7227.61
_FLANGE = math.pi * 30.0 ** 2 * 8.0 - 6.0 * math.pi * 3.0 ** 2 * 8.0  # 21262.30
_STANDOFF = math.pi * (6.0 ** 2 - 3.0 ** 2) * 25.0            # 2120.58
_SHELLED = 50.0 * 30.0 * 20.0 - 46.0 * 26.0 * 18.0            # 8472
_FILLET_CUBE = (
    30.0 ** 3
    - math.pi * 5.0 ** 2 * 30.0                    # 10 mm through hole
    - 4.0 * (2.0 ** 2 - math.pi * 2.0 ** 2 / 4.0) * 30.0  # four r2 vertical fillets
)                                                             # 24540.8
_POCKET = 60.0 * 60.0 * 15.0 - 30.0 * 30.0 * 5.0              # 49500
_GEAR = math.pi * 21.0 ** 2 * 8.0 * 0.86  # ~ addendum-circle cylinder, 86% fill


BRIEFS: Tuple[Brief, ...] = (
    Brief(
        id="plate",
        text=("A rectangular mounting plate, 60 mm wide by 40 mm deep and 6 mm "
              "thick. No holes, no fillets -- just the plate."),
        tier=1,
        volume_mm3=_PLATE,
        volume_tol=0.05,
        requires=(("extrude",),),
        rationale="The floor: can the model sketch a rectangle, constrain it and extrude it?",
    ),
    Brief(
        id="disc",
        text=("A round disc 50 mm in diameter and 5 mm thick, like a blank "
              "cover plate."),
        tier=1,
        volume_mm3=_DISC,
        volume_tol=0.08,
        requires=(("extrude",),),
        rationale="Same as the plate but with a circle: does 'diameter' become radius 25?",
    ),
    Brief(
        id="washer",
        text=("A flat washer: 30 mm outside diameter, 12 mm inside diameter, "
              "3 mm thick."),
        tier=2,
        volume_mm3=_WASHER,
        volume_tol=0.10,
        requires=(("hole", "boolean"),),
        rationale="The first subtraction. Tests diameter->radius twice and a through cut.",
    ),
    Brief(
        id="bracket",
        text=("A mounting bracket: a 40 mm by 24 mm plate, 8 mm thick, with two "
              "6 mm diameter holes drilled through it, 20 mm apart and centred "
              "on the plate."),
        tier=2,
        volume_mm3=_BRACKET,
        volume_tol=0.08,
        requires=(("hole", "boolean"),),
        rationale="The canonical text-to-CAD ask: plate plus a located hole pattern.",
    ),
    Brief(
        id="flange",
        text=("A 60 mm diameter flange, 8 mm thick, with six M6 clearance holes "
              "(6.5 mm diameter) equally spaced on a 45 mm pitch circle."),
        tier=3,
        volume_mm3=_FLANGE,
        volume_tol=0.10,
        requires=(("hole", "boolean"), ("circular_pattern", "hole", "boolean")),
        rationale="Bolt circle: the model must place six holes on a PCD (or pattern one).",
    ),
    Brief(
        id="standoff",
        text=("A cylindrical standoff 25 mm tall, 12 mm outside diameter, with a "
              "6 mm hole bored all the way through the middle."),
        tier=2,
        volume_mm3=_STANDOFF,
        volume_tol=0.10,
        requires=(("hole", "boolean"),),
        rationale="A tube. Trivial for a human; a surprising number of models widen the bore.",
    ),
    Brief(
        id="shelled_box",
        text=("An open-top box: outside dimensions 50 mm by 30 mm by 20 mm tall, "
              "hollowed out to a uniform 2 mm wall thickness, open on the top face."),
        tier=3,
        volume_mm3=_SHELLED,
        volume_tol=0.12,
        requires=(("shell", "boolean"),),
        rationale="Shelling. The kernel preflight will reject a wall thicker than the cavity.",
    ),
    Brief(
        id="filleted_cube",
        text=("A 30 mm cube with a 10 mm diameter hole through it from top to "
              "bottom, and a 2 mm radius fillet on each of the four vertical "
              "edges."),
        tier=4,
        volume_mm3=_FILLET_CUBE,
        volume_tol=0.12,
        requires=(("hole", "boolean"), ("fillet",)),
        rationale="Fillet + hole. The fillet radius is where models overshoot and get blocked.",
    ),
    Brief(
        id="pocket_block",
        text=("A 60 mm by 60 mm block, 15 mm tall, with a square pocket 30 mm by "
              "30 mm and 5 mm deep machined into the centre of the top face."),
        tier=4,
        volume_mm3=_POCKET,
        volume_tol=0.10,
        requires=(("boolean", "hole"),),
        rationale="A blind subtraction: needs a second body and a cut, not a through hole.",
    ),
    Brief(
        id="spur_gear",
        text=("A 20 tooth spur gear, module 2, with an 8 mm face width and a "
              "10 mm bore through the centre."),
        tier=5,
        volume_mm3=_GEAR,
        volume_tol=0.30,
        requires=(("boolean", "circular_pattern", "hole"),),
        rationale=("The ceiling. Involute teeth are not in the op set; a model must "
                   "synthesise them from primitives or admit defeat."),
    ),
)


def brief_ids() -> List[str]:
    return [b.id for b in BRIEFS]


def brief_by_id(brief_id: str) -> Brief:
    for b in BRIEFS:
        if b.id == brief_id:
            return b
    raise KeyError(f"unknown brief {brief_id!r}; known: {', '.join(brief_ids())}")


def _requirements_met(brief: Brief, op_tags: Sequence[str]) -> Tuple[bool, List[str]]:
    """Which required op-groups the op stream does NOT satisfy."""
    present = set(op_tags)
    missing: List[str] = []
    for group in brief.requires:
        if not present.intersection(group):
            missing.append("|".join(group))
    return (not missing), missing


def grade_geometry(brief: Brief, volume: Optional[float],
                   op_tags: Sequence[str]) -> Dict[str, object]:
    """Is the verified solid the part that was ASKED for?

    Returns a dict carrying the volume comparison and the feature-presence check.
    `on_brief` is True only when both pass. A part that verifies but has no holes
    in it fails here, loudly, and that is the point.
    """
    meets, missing = _requirements_met(brief, op_tags)
    if volume is None or volume <= 0.0:
        rel = None
        vol_ok = False
    else:
        rel = abs(volume - brief.volume_mm3) / brief.volume_mm3
        vol_ok = rel <= brief.volume_tol
    reasons: List[str] = []
    if not vol_ok:
        if volume is None or volume <= 0.0:
            reasons.append("no measurable volume")
        else:
            reasons.append(
                "volume %.1f mm^3 is %.1f%% off the briefed %.1f mm^3 (tol %.0f%%)"
                % (volume, 100.0 * (rel or 0.0), brief.volume_mm3,
                   100.0 * brief.volume_tol))
    if not meets:
        reasons.append("op stream never uses: " + ", ".join(missing))
    return {
        "on_brief": bool(vol_ok and meets),
        "volume_mm3": None if volume is None else round(float(volume), 3),
        "expected_mm3": round(brief.volume_mm3, 3),
        "volume_rel_error": None if rel is None else round(rel, 4),
        "volume_ok": vol_ok,
        "features_ok": meets,
        "missing_features": missing,
        "reasons": reasons,
    }
