"""Rule-based (logic-rule / hint) machining-feature recogniser
(Khan et al., "Leveraging Vision-Language Models for Manufacturing Feature
Recognition in CAD Designs", Sec. 2.1: rule-based AFR uses "expert-defined rules
to recognize features based on their geometric and topological properties").

The paper itself replaces rules with a VLM (external, skipped). This module
implements the DETERMINISTIC baseline the paper contrasts against: classic
logic-rule AFR that names machining features from a lightweight boundary-
representation abstraction (surface types + loop topology + concavity hints).

It deliberately does NOT depend on an OCCT/CadQuery kernel. A part is described
by a small explicit model of faces; each :class:`Face` carries:

  * ``surface`` -- one of "plane", "cylinder", "cone", "torus", "sphere";
  * ``concave`` -- True when the face bounds removed material (a hint: it faces
    "into" the solid, e.g. the wall of a pocket or bore);
  * ``convex``  -- True for a blend that adds/rounds material (fillet vs chamfer);
  * ``inner_loops`` -- number of interior loops (openings) on the face;
  * ``on_boundary`` -- True when the face touches the stock outer boundary;
  * ``capped``  -- for a bore: True if a planar floor closes it (blind) else
    through;
  * ``walls``   -- for a floor face: number of surrounding wall faces;
  * ``open_sides`` -- for a floor/depression: how many sides open to the
    boundary (0 => enclosed pocket, >=1 => slot/step);
  * ``radius`` / ``half_angle`` -- optional geometry used for chamfer/fillet.

Detection rules (each deterministic, in priority order) map onto the canonical
leaf labels of :mod:`fabrication.mfgfeat_taxonomy`:

    cylinder + concave                        -> hole (blind if capped else through)
    planar floor, enclosed, walls>=3          -> pocket
    planar floor, open_sides==1               -> slot
    planar floor, open_sides>=2               -> step
    small planar bevel between two faces      -> chamfer
    small convex cylinder/torus blend         -> fillet
    convex cylinder/boss on boundary          -> boss
    concave planar dish (freeform)            -> depression

Output is a deterministic list of :class:`Detection` records (feature label +
the involved face ids + a subtype hint), which downstream code
(:mod:`fabrication.mfgfeat_attributes`) turns into dimensioned attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Lightweight B-rep abstraction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Face:
    id: str
    surface: str = "plane"          # plane|cylinder|cone|torus|sphere
    concave: bool = False           # bounds removed material (hint)
    convex: bool = False            # additive/rounding blend
    inner_loops: int = 0
    on_boundary: bool = False
    capped: bool = False            # bore closed by a floor (blind)
    walls: int = 0                  # surrounding wall count for a floor
    open_sides: int = 0             # sides of a floor/depression open to boundary
    radius: Optional[float] = None
    half_angle: Optional[float] = None  # degrees, for cone/chamfer
    entry_cone: bool = False        # countersink hint on a bore
    entry_counterbore: bool = False  # counterbore hint on a bore
    threaded: bool = False          # thread hint on a bore

    def __post_init__(self):
        valid = {"plane", "cylinder", "cone", "torus", "sphere"}
        if self.surface not in valid:
            raise ValueError("unknown surface type: %r" % (self.surface,))
        if self.concave and self.convex:
            raise ValueError("face cannot be both concave and convex")


@dataclass(frozen=True)
class Detection:
    feature: str                    # canonical leaf label
    face_ids: Tuple[str, ...]
    subtype: Optional[str] = None
    attrs: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Per-face rules
# --------------------------------------------------------------------------- #
def _hole_subtype(face):
    if face.threaded:
        return "threaded"
    if face.entry_counterbore:
        return "counterbore"
    if face.entry_cone:
        return "countersink"
    if face.surface == "cone":
        return "tapered"
    return "blind" if face.capped else "through"


def _detect_face(face):
    """Return a Detection for a single face, or None if no rule fires."""
    s = face.surface

    # Hole: a concave cylinder or (tapered) cone bore.
    if face.concave and s in ("cylinder", "cone"):
        subtype = _hole_subtype(face)
        attrs = {}
        if face.radius is not None:
            attrs["diameter"] = 2.0 * face.radius
        return Detection("hole", (face.id,), subtype=subtype, attrs=attrs)

    # Planar floor of a depression / pocket / slot / step.
    if face.concave and s == "plane" and face.walls > 0:
        if face.open_sides <= 0:
            return Detection("pocket", (face.id,))
        if face.open_sides == 1:
            return Detection("slot", (face.id,))
        return Detection("step", (face.id,))

    # Chamfer: a small planar bevel (angled face, no loops, not a floor).
    if (s == "plane" and face.half_angle is not None
            and 0.0 < face.half_angle < 90.0 and face.walls == 0):
        return Detection("chamfer", (face.id,),
                         attrs={"angle": face.half_angle})

    # Fillet: a small convex rounding blend (cylinder or torus).
    if face.convex and s in ("cylinder", "torus"):
        attrs = {}
        if face.radius is not None:
            attrs["radius"] = face.radius
        return Detection("fillet", (face.id,), attrs=attrs)

    # Boss / protrusion: a convex cylinder standing proud on the boundary.
    if face.convex and s == "cylinder" and face.on_boundary:
        attrs = {}
        if face.radius is not None:
            attrs["diameter"] = 2.0 * face.radius
        return Detection("boss", (face.id,), attrs=attrs)

    # Freeform depression: a concave dish (sphere/torus floor, no walls).
    if face.concave and s in ("sphere", "torus") and face.walls == 0:
        return Detection("depression", (face.id,))

    return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def detect_features(faces):
    """Recognise machining features from an iterable of :class:`Face`.

    Returns a list of :class:`Detection` in deterministic order (input face
    order). Faces that match no rule are silently ignored (no feature).
    """
    out = []
    for face in faces:
        if not isinstance(face, Face):
            raise TypeError("expected Face, got %r" % (type(face).__name__,))
        det = _detect_face(face)
        if det is not None:
            out.append(det)
    return out


def feature_counts(faces):
    """Detect features and return a ``{leaf_label: count}`` dict.

    This is the deterministic AFR analogue of the VLM output the paper's metrics
    (:mod:`bench.mfgfeat_afr_metrics`) consume, so a rule-based prediction can be
    scored with FNA/FQA/HR/MAE exactly like a VLM prediction.
    """
    counts = {}
    for det in detect_features(faces):
        counts[det.feature] = counts.get(det.feature, 0) + 1
    return counts
