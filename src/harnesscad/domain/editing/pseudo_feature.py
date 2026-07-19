"""Pseudo-Feature integration.

The pseudo-feature approach lets a user make direct edits inside a history-based
parametric modeler: "User-specified direct edits will be added to the end of the
model's construction history as pseudo-features, and the original history remains
exactly as before." It is the approach adopted by most CAD vendors because it is
trivial to implement.

Its documented failure: a trailing pseudo-feature is anchored to
geometry produced by an *earlier* feature; changing that earlier feature's
parameter (the example: P10 33mm -> 68mm) invalidates the anchor and
"leads to a failed history regeneration." The "perfect solution" is *not*
to append the direct edit, but to transform it into an appropriate redefinition
of the relevant feature (there, modifying the 2D sketch from a rectangular slot
to a slanted slot).

This module implements that mechanic deterministically:

* :func:`append_pseudo_feature` appends a direct push-pull as a pseudo-feature,
  recording the anchor (the feature + face-geometry key it was based on) while
  leaving the original history untouched.
* :func:`regenerate` replays a construction history under a parameter edit and
  detects the anchor-invalidation failure, returning a structured
  :class:`RegenResult`.
* :func:`transform_to_feature_redefinition` implements the perfect solution:
  drop the trailing pseudo-feature and fold its intent into the anchor feature's
  parameters, yielding a tree that regenerates cleanly.

Stdlib-only, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from harnesscad.domain.editing.hybrid_model import (
    DirectBRep, FeatureTree, ParametricFeature, PushPullEdit,
)

#: parameter key under which a pseudo-feature stores its push-pull distance.
PSEUDO_PARAM = "move_distance"
#: parameter key recording the anchor feature's dimension at capture time.
ANCHOR_SNAPSHOT = "anchor_offset"


@dataclass
class RegenResult:
    """Outcome of replaying a history under a parameter edit.

    ``ok`` is False when a pseudo-feature's anchor was invalidated. ``broken``
    lists the ids of pseudo-features whose anchoring geometry changed, and
    ``reason`` is a human/agent-readable explanation.
    """

    ok: bool
    tree: FeatureTree
    broken: List[str] = field(default_factory=list)
    reason: str = ""


def append_pseudo_feature(tree: FeatureTree, brep: DirectBRep,
                          edit: PushPullEdit, fid: Optional[str] = None,
                          ) -> FeatureTree:
    """Append ``edit`` as a pseudo-feature at the end of ``tree``.

    The pseudo-feature references the direct-model face's ``origin`` feature and
    snapshots that feature's controlling geometry (the face offset) so a later
    regeneration can detect anchor drift. The original history is copied
    unchanged; only a trailing feature is added.
    """
    face = brep.faces[edit.face_name]
    anchor = face.origin
    if anchor is None:
        raise ValueError(f"face {edit.face_name!r} has no parametric anchor")
    new = tree.copy()
    pid = fid or f"pseudo{len(new.features)}"
    new.features.append(ParametricFeature(
        pid, "pseudo_move_face",
        params={PSEUDO_PARAM: float(edit.distance),
                ANCHOR_SNAPSHOT: float(face.offset)},
        refs=(anchor,), direct_edit=True))
    return new


def _anchor_offset(tree: FeatureTree, anchor_fid: str) -> float:
    """The controlling offset an anchor feature contributes.

    Deterministic surrogate for the anchor's carrying geometry: the sum of the
    anchor feature's numeric params (any change to the anchor thus perturbs it).
    """
    f = tree.get(anchor_fid)
    return float(sum(v for k, v in sorted(f.params.items())))


def regenerate(tree: FeatureTree, edit) -> RegenResult:
    """Replay ``tree`` under a :class:`ParameterEdit`, detecting anchor failure.

    Regeneration fails when the edited feature is the anchor of one or more
    trailing pseudo-features and the edit moves the anchor's controlling offset
    away from the snapshot the pseudo-feature was captured against.
    """
    from harnesscad.domain.editing.hybrid_model import ParameterEdit
    if not isinstance(edit, ParameterEdit):
        raise TypeError("regenerate expects a ParameterEdit")
    new = tree.copy()
    before = _anchor_offset(new, edit.target_fid)
    new.set_parameter(edit.target_fid, edit.param, edit.new_value)
    after = _anchor_offset(new, edit.target_fid)

    broken: List[str] = []
    if before != after:
        for f in new.features:
            if (f.ftype == "pseudo_move_face"
                    and edit.target_fid in f.refs):
                broken.append(f.fid)
    if broken:
        return RegenResult(
            ok=False, tree=tree,
            broken=broken,
            reason=(f"parameter '{edit.param}' of anchor '{edit.target_fid}' "
                    f"changed {before} -> {after}, invalidating pseudo-features "
                    f"{broken}"))
    return RegenResult(ok=True, tree=new)


def transform_to_feature_redefinition(tree: FeatureTree, pseudo_fid: str,
                                      ) -> FeatureTree:
    """The "perfect solution": fold a pseudo-feature into its anchor.

    Instead of a trailing direct-edit pseudo-feature, the push-pull intent is
    expressed as a redefinition of the relevant (anchor) feature: the move
    distance is added onto the anchor's first parameter and the pseudo-feature
    is removed. The result carries the same net geometry but has no fragile
    trailing anchor, so it regenerates under parameter changes.
    """
    new = tree.copy()
    pf = new.get(pseudo_fid)
    if pf.ftype != "pseudo_move_face":
        raise ValueError(f"{pseudo_fid!r} is not a pseudo-feature")
    if not pf.refs:
        raise ValueError(f"pseudo-feature {pseudo_fid!r} has no anchor")
    anchor = new.get(pf.refs[0])
    dist = pf.params[PSEUDO_PARAM]
    # Redefine the anchor's controlling dimension (its first param) by the move.
    key = sorted(anchor.params)[0]
    anchor.params[key] = anchor.params[key] + dist
    new.features = [f for f in new.features if f.fid != pseudo_fid]
    return new
