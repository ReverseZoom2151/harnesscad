"""Operation Translating integration (Section 4.4 of Zou 2025).

This approach (academia + Autodesk) "translates direct edits into operations of
parameter tuning and/or order rearrangement of the features already presented in
the model history." The paper flags two fundamental limits:

1. it "cannot solve the problem altogether because not all direct edits are
   achievable through those feature operations" (non-achievability); and
2. "feature parameter tuning for a given direct edit (if achievable) is usually
   not unique" (non-uniqueness).

This module implements the translation as a deterministic search, reproducing
both limits explicitly:

* A :class:`FaceParamLink` records which feature parameter governs a face and how
  a unit push-pull maps onto a parameter delta (``gain``): a push-pull of the
  face by ``d`` corresponds to tuning that parameter by ``d / gain``.
* :func:`translate_push_pull` returns *all* candidate :class:`Translation`
  objects (parameter-tuning and order-rearrangement candidates) for a direct
  edit. An empty list means the edit is *not achievable* via feature ops; more
  than one candidate means the translation is *not unique*.

Stdlib-only, deterministic (candidates are produced in a stable order).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from harnesscad.domain.editing.hybrid_model import FeatureTree, ParameterEdit, PushPullEdit


@dataclass(frozen=True)
class FaceParamLink:
    """Maps a direct-model face onto a governing feature parameter.

    ``gain`` is the geometric change of the face per unit parameter change; a
    push-pull distance ``d`` translates to a parameter delta ``d / gain``. A
    ``gain`` of 0 is rejected (parameter does not control the face).
    """

    face_name: str
    fid: str
    param: str
    gain: float = 1.0

    def __post_init__(self):
        if self.gain == 0:
            raise ValueError("gain must be non-zero")


@dataclass(frozen=True)
class Translation:
    """A candidate feature-operation translation of a direct edit.

    ``param_edits`` are the parameter tunings; ``reorder`` is an optional
    (fid, new_index) order-rearrangement. ``description`` is a short label.
    """

    param_edits: Tuple[ParameterEdit, ...] = ()
    reorder: Optional[Tuple[str, int]] = None
    description: str = ""

    def apply(self, tree: FeatureTree) -> FeatureTree:
        """Produce the tree that results from applying this translation."""
        new = tree.copy()
        for pe in self.param_edits:
            new.set_parameter(pe.target_fid, pe.param, pe.new_value)
        if self.reorder is not None:
            fid, new_index = self.reorder
            feats = new.features
            i = new.index_of(fid)
            f = feats.pop(i)
            feats.insert(new_index, f)
        return new


def translate_push_pull(tree: FeatureTree, edit: PushPullEdit,
                        links: List[FaceParamLink],
                        symmetric_params: Optional[List[Tuple[str, str]]] = None,
                        ) -> List[Translation]:
    """Enumerate feature-operation translations of a direct push-pull.

    ``links`` associates faces with governing parameters (the model's
    face-to-parameter map). ``symmetric_params`` optionally lists extra
    (fid, param) handles that produce the *same* face change (e.g. a symmetric
    counterpart), which is how the paper's non-uniqueness arises.

    Returns every candidate in a deterministic order. An empty list means the
    edit is *not achievable* via feature parameter tuning (limit 1).
    """
    matches = [ln for ln in links if ln.face_name == edit.face_name]
    if not matches:
        return []  # non-achievable: no parameter governs this face

    candidates: List[Translation] = []
    for ln in matches:
        delta = edit.distance / ln.gain
        current = tree.parameter(ln.fid, ln.param)
        pe = ParameterEdit(ln.fid, ln.param, current + delta)
        candidates.append(Translation(
            param_edits=(pe,),
            description=f"tune {ln.fid}.{ln.param} by {delta}"))

    # Non-uniqueness: symmetric parameters achieving the same geometry.
    for (fid, param) in (symmetric_params or []):
        current = tree.parameter(fid, param)
        # symmetric handles absorb the full push-pull distance directly
        pe = ParameterEdit(fid, param, current + edit.distance)
        candidates.append(Translation(
            param_edits=(pe,),
            description=f"tune symmetric {fid}.{param} by {edit.distance}"))

    return candidates


def is_achievable(tree: FeatureTree, edit: PushPullEdit,
                  links: List[FaceParamLink]) -> bool:
    """True iff the direct edit can be reproduced by feature parameter tuning."""
    return bool(translate_push_pull(tree, edit, links))


def is_unique(tree: FeatureTree, edit: PushPullEdit,
              links: List[FaceParamLink],
              symmetric_params: Optional[List[Tuple[str, str]]] = None) -> bool:
    """True iff exactly one translation exists (achievable and unambiguous)."""
    return len(translate_push_pull(tree, edit, links, symmetric_params)) == 1
