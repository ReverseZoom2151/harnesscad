"""The EDIT surface: a model already in a HarnessSession can be CHANGED.

The harness could build (planner -> ops -> verify) and ingest (tokens/mesh -> ops),
but it had no edit leg. ``domain/editing`` carried the whole apparatus -- parametric
edits, edit locality/diffing, plan-verify loops, refine loops, revision provenance,
history-free (direct-manipulation) editing -- and nothing called any of it.

This module is that dispatcher.

    parameters(state)          -> the editable numeric parameters of an op stream
    apply_edit(state, edit)    -> EditResult (the edit applied, or blocked)
    diff(before, after)        -> Diff (which ops/params moved, and by how much)
    run_strategy(name, ...)    -> a named edit LOOP (plan-verify / refine / beam ...)

EDITS ARE DISPATCHED BY *TARGET*, AND RIVAL LOOPS ARE NEVER BLENDED
-------------------------------------------------------------------
An "edit" is not one thing. A parametric edit (change extrude.distance and
regenerate) and a push-pull edit (drag a face and reconcile the feature tree) are
different operations on different representations, so every edit kind declares the
``target`` it edits:

    session      a HarnessSession / CISP op stream   (param, align, distribute)
    hybrid       a FeatureTree + DirectBRep          (push_pull -- Zou 2025)
    tokens       a token sequence                    (mask: locate-then-infill)
    design       an mrCAD sketch Design              (sketch)
    text         CAD script source text              (text)
    csg_program  a bidirectional CSG program         (code_view)

The three *search* loops that edit an op stream toward a target shape -- CADMorph's
plan-generate-verify (``plan_verify``), CADReasoner's render-compare-refine
(``refine``) and CADReasoner's geometry-guided beam (``geometry_beam``) -- are
RIVALS: three different algorithms for the same job. They are selected by name and
their results are never averaged. Same for the three Zou-2025 direct-edit
integration strategies (``operation_translation`` / ``pseudo_feature`` /
``synchronous_partition``): a push-pull edit must say which one it wants.

THE SHAPE PROXY
---------------
The edit loops need ``render(program) -> shape``. The default backend (StubBackend)
is deliberately not a geometry kernel, so this module supplies an ANALYTIC proxy
computed from the op parameters themselves: every extruded sketch becomes the
axis-aligned box of its 2D profile bounds times the extrude distance
(:func:`shape_of`, :func:`points_of`). It is exact for the box/cylinder-style
extrusions CISP builds and is honestly coarse for everything else -- it is a
parameter-to-shape map, not a tessellation. Callers with a real kernel pass their
own ``render=``/``metric=``.

Stdlib-only, absolute imports, deterministic (the only randomness is a caller-seeded
``random.Random``). The edited modules are never modified: adapters only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry
from harnesscad.core.cisp.ops import (
    AddCircle, AddLine, AddPoint, AddRectangle, Extrude, Op, SetParam,
    canonical_json, parse_op,
)

__all__ = [
    "EditError",
    "UnknownEdit",
    "UnknownStrategy",
    "Unsupported",
    "RivalBlend",
    "ShapeSignature",
    "ParamRef",
    "ModelState",
    "Edit",
    "EditKind",
    "EditResult",
    "Diff",
    "Strategy",
    "StrategyResult",
    "TARGETS",
    "edits",
    "edit_kind",
    "strategies",
    "strategy",
    "rivals",
    "unadapted",
    "shape_of",
    "points_of",
    "shape_distance",
    "point_distance",
    "ops_of",
    "parameters",
    "snapshot",
    "apply_edit",
    "diff",
    "run_strategy",
    "add_arguments",
    "run_cli",
    "main",
]

EDITING_PACKAGE = "editing"
_PKG = "harnesscad.domain.editing."

#: What an edit kind can be applied to. An edit never guesses its target.
TARGETS: Tuple[str, ...] = (
    "session", "hybrid", "tokens", "design", "text", "csg_program",
)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class EditError(ValueError):
    """Base class for every edit-surface failure."""


class UnknownEdit(EditError):
    """An edit kind outside the discovered table."""


class UnknownStrategy(EditError):
    """A strategy name outside the discovered table."""


class Unsupported(EditError):
    """This edit genuinely cannot be applied to this state (no fallback)."""


class RivalBlend(EditError):
    """Two rival strategies were asked for at once. They are never blended."""


# --------------------------------------------------------------------------- #
# The analytic shape proxy (the parameter-to-shape map F the loops need)
# --------------------------------------------------------------------------- #
Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ShapeSignature:
    """A deterministic, kernel-free shape summary of an op stream.

    ``extents`` is the axis-aligned bounding box size of the union of the extruded
    profiles; ``volume`` is the sum of their box volumes (an upper bound, not a
    kernel mass property); ``solids`` counts the extrusions. Coarse by construction
    and documented as such -- it exists so an edit loop can rank candidates without
    a geometry kernel.
    """

    extents: Vec3 = (0.0, 0.0, 0.0)
    volume: float = 0.0
    solids: int = 0

    def vector(self) -> Tuple[float, ...]:
        """The comparable feature vector (mm, mm, mm, mm)."""
        return (self.extents[0], self.extents[1], self.extents[2],
                self.volume ** (1.0 / 3.0) if self.volume > 0 else 0.0)

    def to_dict(self) -> dict:
        return {"extents": list(self.extents), "volume": self.volume,
                "solids": self.solids}


def _profile_bounds(ops: Sequence[Op]) -> Dict[str, Tuple[float, float, float, float]]:
    """sketch id -> (xmin, ymin, xmax, ymax) of its 2D entities, in op order."""
    sketches: List[str] = []
    bounds: Dict[str, List[float]] = {}
    for op in ops:
        if op.OP == "new_sketch":
            sid = "sk%d" % (len(sketches) + 1)
            sketches.append(sid)
            continue
        sid = getattr(op, "sketch", "")
        if not sid:
            continue
        pts: List[Tuple[float, float]] = []
        if isinstance(op, AddRectangle):
            pts = [(op.x - op.w / 2.0, op.y - op.h / 2.0),
                   (op.x + op.w / 2.0, op.y + op.h / 2.0)]
        elif isinstance(op, AddCircle):
            pts = [(op.cx - op.r, op.cy - op.r), (op.cx + op.r, op.cy + op.r)]
        elif isinstance(op, AddLine):
            pts = [(op.x1, op.y1), (op.x2, op.y2)]
        elif isinstance(op, AddPoint):
            pts = [(op.x, op.y)]
        if not pts:
            continue
        cur = bounds.get(sid)
        for (x, y) in pts:
            if cur is None:
                cur = [x, y, x, y]
            else:
                cur = [min(cur[0], x), min(cur[1], y), max(cur[2], x), max(cur[3], y)]
        bounds[sid] = cur
    return {k: (v[0], v[1], v[2], v[3]) for k, v in bounds.items()}


def _boxes(ops: Sequence[Op]) -> List[Tuple[Vec3, Vec3]]:
    """The (min, max) corners of every extruded profile. [] when there is no solid."""
    prof = _profile_bounds(ops)
    out: List[Tuple[Vec3, Vec3]] = []
    for op in ops:
        if not isinstance(op, Extrude):
            continue
        b = prof.get(op.sketch)
        if b is None:
            continue
        z0, z1 = (0.0, float(op.distance)) if op.distance >= 0 else (float(op.distance), 0.0)
        out.append(((b[0], b[1], z0), (b[2], b[3], z1)))
    return out


def shape_of(state: Any) -> ShapeSignature:
    """The analytic shape signature of a session / op stream. Never raises."""
    ops = ops_of(state)
    boxes = _boxes(ops)
    if not boxes:
        return ShapeSignature()
    lo = [min(b[0][i] for b in boxes) for i in range(3)]
    hi = [max(b[1][i] for b in boxes) for i in range(3)]
    volume = sum(abs(b[1][0] - b[0][0]) * abs(b[1][1] - b[0][1]) * abs(b[1][2] - b[0][2])
                 for b in boxes)
    return ShapeSignature(
        extents=(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]),
        volume=float(volume), solids=len(boxes))


def points_of(state: Any) -> List[Vec3]:
    """The corner point set of the extruded boxes (the input the point loops want)."""
    pts: List[Vec3] = []
    for lo, hi in _boxes(ops_of(state)):
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    pts.append((float(x), float(y), float(z)))
    return pts


def shape_distance(a: ShapeSignature, b: ShapeSignature) -> float:
    """L2 distance between two shape signatures (mm)."""
    va, vb = a.vector(), b.vector()
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)))


def point_distance(a: Sequence[Vec3], b: Sequence[Vec3]) -> Optional[float]:
    """Chamfer distance between two point sets, in absolute mm.

    Delegates to ``agents.generation.shape_metrics.chamfer`` -- the CADSmith
    absolute-millimetre metric -- rather than re-deriving one here. Returns None
    for an empty side (an invalid/degenerate candidate), which is exactly what the
    refine loop and the beam treat as "invalid".
    """
    from harnesscad.agents.generation import shape_metrics

    if not a or not b:
        return None
    return shape_metrics.chamfer(list(a), list(b))


# --------------------------------------------------------------------------- #
# Op-stream state
# --------------------------------------------------------------------------- #
def ops_of(state: Any) -> Tuple[Op, ...]:
    """The EFFECTIVE op stream behind ``state`` (session / ModelState / list of ops).

    ``SetParam`` is an edit *to* the stream, not a member of it: the op DAG records
    it, but the backend's op log (which ``SetParam.target`` indexes) does not. So we
    fold every SetParam into the log exactly as the backend does -- the result is
    the stream that actually built the current model.
    """
    from harnesscad.core.cisp.ops import edit_oplog

    if isinstance(state, ModelState):
        return state.ops
    raw: Sequence[Any]
    if hasattr(state, "opdag"):
        raw = list(state.opdag.ops())
    elif isinstance(state, (list, tuple)):
        raw = list(state)
    else:
        raise EditError("cannot read an op stream from %r" % type(state).__name__)
    out: List[Op] = []
    for op in raw:
        if isinstance(op, dict):
            op = parse_op(op)
        if isinstance(op, SetParam):
            edited, err = edit_oplog(out, op)
            if err is not None:
                raise EditError("the recorded SetParam does not fold cleanly: %s"
                                % err[1])
            out = list(edited)
            continue
        out.append(op)
    return tuple(out)


@dataclass(frozen=True)
class ParamRef:
    """One editable numeric parameter of one op. ``index`` is the SetParam target."""

    index: int
    op: str
    param: str
    value: float

    def to_dict(self) -> dict:
        return {"index": self.index, "op": self.op, "param": self.param,
                "value": self.value}


def parameters(state: Any) -> Tuple[ParamRef, ...]:
    """Every numeric parameter of every op, addressable by (index, param)."""
    out: List[ParamRef] = []
    for i, op in enumerate(ops_of(state)):
        for f in dataclasses.fields(op):
            value = getattr(op, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            out.append(ParamRef(i, op.OP, f.name, float(value)))
    return tuple(out)


@dataclass(frozen=True)
class ModelState:
    """An immutable snapshot of an editable model."""

    ops: Tuple[Op, ...] = ()
    digest: str = ""
    shape: ShapeSignature = field(default_factory=ShapeSignature)

    def to_dict(self) -> dict:
        return {"ops": [op.to_dict() for op in self.ops], "digest": self.digest,
                "shape": self.shape.to_dict()}


def _digest_of(ops: Sequence[Op]) -> str:
    import hashlib

    blob = "|".join(canonical_json(op) for op in ops)
    return hashlib.sha256(blob.encode()).hexdigest()


def snapshot(state: Any) -> ModelState:
    """Freeze a session / op stream into a comparable :class:`ModelState`."""
    if isinstance(state, ModelState):
        return state
    ops = ops_of(state)
    digest = state.digest() if hasattr(state, "digest") else _digest_of(ops)
    return ModelState(ops=ops, digest=digest, shape=shape_of(ops))


# --------------------------------------------------------------------------- #
# Edits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Edit:
    """One requested change. ``kind`` selects the adapter; nothing is inferred."""

    kind: str
    target: Optional[int] = None          # op index (session edits)
    param: str = ""
    value: Any = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "target": self.target, "param": self.param,
                "value": self.value, "payload": dict(self.payload)}


@dataclass(frozen=True)
class Diff:
    """What an edit did. ``changed`` is the authoritative "did anything move?"."""

    changed_params: Tuple[Tuple[int, str, Any, Any], ...] = ()   # (idx, param, old, new)
    added_ops: Tuple[int, ...] = ()
    removed_ops: Tuple[int, ...] = ()
    digest_before: str = ""
    digest_after: str = ""
    shape_delta: Tuple[float, ...] = ()
    distance: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.changed_params or self.added_ops or self.removed_ops
                    or self.digest_before != self.digest_after)

    def to_dict(self) -> dict:
        return {"changed": self.changed,
                "changed_params": [list(c) for c in self.changed_params],
                "added_ops": list(self.added_ops),
                "removed_ops": list(self.removed_ops),
                "digest_before": self.digest_before,
                "digest_after": self.digest_after,
                "shape_delta": list(self.shape_delta),
                "distance": self.distance}


@dataclass(frozen=True)
class EditResult:
    """The outcome of one :func:`apply_edit`. A blocked edit is not an exception."""

    ok: bool
    kind: str
    before: Any = None
    after: Any = None
    diff: Optional[Diff] = None
    diagnostics: Tuple[str, ...] = ()
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "kind": self.kind,
                "diff": self.diff.to_dict() if self.diff else None,
                "diagnostics": list(self.diagnostics),
                "detail": {k: v for k, v in self.detail.items()
                           if isinstance(v, (str, int, float, bool, list, dict))}}


def diff(before: Any, after: Any) -> Diff:
    """Diff two states. Op-level, parameter-granular, plus the shape delta.

    Positions that exist in both streams are compared field by field; the tail of
    the longer stream is reported as added/removed. This is the EDIT LOCALITY
    answer: exactly which parameters of which ops moved.
    """
    a, b = snapshot(before), snapshot(after)
    changed: List[Tuple[int, str, Any, Any]] = []
    common = min(len(a.ops), len(b.ops))
    for i in range(common):
        oa, ob = a.ops[i], b.ops[i]
        if oa.OP != ob.OP:
            changed.append((i, "op", oa.OP, ob.OP))
            continue
        for f in dataclasses.fields(oa):
            va, vb = getattr(oa, f.name), getattr(ob, f.name)
            if va != vb:
                changed.append((i, f.name, va, vb))
    added = tuple(range(common, len(b.ops)))
    removed = tuple(range(common, len(a.ops)))
    va, vb = a.shape.vector(), b.shape.vector()
    return Diff(changed_params=tuple(changed), added_ops=added, removed_ops=removed,
                digest_before=a.digest, digest_after=b.digest,
                shape_delta=tuple(y - x for x, y in zip(va, vb)),
                distance=shape_distance(a.shape, b.shape))


# --- the edit adapters ----------------------------------------------------- #
def _edit_param(state: Any, e: Edit) -> EditResult:
    """The parametric edit: rewrite op[i].param and deterministically regenerate."""
    if e.target is None or not e.param:
        raise EditError("a 'param' edit needs target=<op index> and param=<name>")
    before = snapshot(state)
    if not (0 <= int(e.target) < len(before.ops)):
        return EditResult(False, e.kind, before, before, Diff(
            digest_before=before.digest, digest_after=before.digest),
            ("bad-ref: op index %d out of range (0..%d)"
             % (int(e.target), len(before.ops) - 1),))
    op = SetParam(target=int(e.target), param=str(e.param), value=e.value)
    if hasattr(state, "apply_ops"):
        result = state.apply_ops([op])
        after = snapshot(state)
        diags = tuple("%s: %s" % (d.code, d.message) for d in result.diagnostics)
        return EditResult(bool(result.ok), e.kind, before, after,
                          diff(before, after), diags,
                          {"applied": result.applied, "digest": result.digest})
    # Pure path: edit the op log without a backend.
    from harnesscad.core.cisp.ops import edit_oplog

    new_log, err = edit_oplog(list(before.ops), op)
    if err is not None:
        return EditResult(False, e.kind, before, before,
                          Diff(digest_before=before.digest,
                               digest_after=before.digest),
                          ("%s: %s" % (err[0], err[1]),))
    after = ModelState(ops=tuple(new_log), digest=_digest_of(new_log),
                       shape=shape_of(new_log))
    return EditResult(True, e.kind, before, after, diff(before, after))


def _instance_indices(ops: Sequence[Op]) -> List[int]:
    return [i for i, op in enumerate(ops) if op.OP == "add_instance"]


def _bbox_of_instances(ops: Sequence[Op], size: float) -> List[Tuple[float, ...]]:
    """Axis-aligned boxes for the placed instances (a cube of ``size`` at each origin).

    ``layout_ops`` aligns/distributes by axis-aligned bbox; CISP instances carry a
    placement but the stub carries no per-instance geometry, so the caller supplies
    the (uniform) instance size. Nothing is invented: the *positions* are the ops'.
    """
    half = size / 2.0
    out = []
    for i in _instance_indices(ops):
        op = ops[i]
        out.append((op.x - half, op.y - half, op.x + half, op.y + half))
    return out


def _edit_layout(state: Any, e: Edit) -> EditResult:
    """Align / distribute the placed instances (autocad_layout_ops) -> SetParam edits."""
    from harnesscad.domain.editing import layout_ops as m

    ops = ops_of(state)
    idx = _instance_indices(ops)
    if len(idx) < 2:
        raise Unsupported("layout edits need at least two placed instances "
                          "(add_instance ops); this model has %d" % len(idx))
    size = float(e.payload.get("size", 1.0))
    boxes = _bbox_of_instances(ops, size)
    mode = str(e.payload.get("mode", "left"))
    if e.kind == "align":
        try:
            align_mode = m.Align(mode)
        except ValueError:
            raise EditError("unknown align mode %r (one of: %s)"
                            % (mode, ", ".join(a.value for a in m.Align))) from None
        deltas = m.align(boxes, align_mode)
    elif e.kind == "distribute":
        axis = str(e.payload.get("axis", "x"))
        spacing = e.payload.get("spacing")
        deltas = (m.distribute_centers(boxes, axis) if spacing is None
                  else m.distribute_gaps(boxes, float(spacing), axis))
    else:  # pragma: no cover - table-driven
        raise UnknownEdit(e.kind)

    before = snapshot(state)
    result: Optional[EditResult] = None
    for op_index, (dx, dy) in zip(idx, deltas):
        for param, delta in (("x", dx), ("y", dy)):
            if not delta:
                continue
            current = float(getattr(ops[op_index], param))
            result = _edit_param(state, Edit("param", op_index, param,
                                             current + float(delta)))
            if not result.ok:
                return dataclasses.replace(result, kind=e.kind)
            ops = ops_of(state)
    after = snapshot(state)
    return EditResult(True, e.kind, before, after, diff(before, after), (),
                      {"instances": len(idx), "mode": mode})


#: The three Zou-2025 strategies for integrating a DIRECT edit into a parametric
#: model. They are RIVALS: same push-pull edit, three incompatible answers.
PUSH_PULL_INTEGRATIONS: Tuple[str, ...] = (
    "operation_translation", "pseudo_feature", "synchronous_partition",
)


def _edit_push_pull(state: Any, e: Edit) -> EditResult:
    """A direct (push-pull) face edit reconciled into a parametric FeatureTree.

    ``state`` is a ``hybrid_consistency.HybridModel`` (a DirectBRep + constraints +
    an optional FeatureTree). ``payload['integration']`` MUST name one of
    :data:`PUSH_PULL_INTEGRATIONS` -- the three papers' strategies disagree by
    design (translate the edit into parameters / append a pseudo-feature / drop the
    face out of the history), so the caller chooses and nothing is blended.
    """
    from harnesscad.domain.editing import hybrid_consistency as hc
    from harnesscad.domain.editing import hybrid_model as hm
    from harnesscad.domain.editing import operation_translation as ot
    from harnesscad.domain.editing import pseudo_feature as pf
    from harnesscad.domain.editing import synchronous_partition as sp

    if not isinstance(state, hc.HybridModel):
        raise Unsupported("a 'push_pull' edit targets a hybrid_consistency.HybridModel "
                          "(a DirectBRep + FeatureTree), not %r"
                          % type(state).__name__)
    integration = str(e.payload.get("integration", ""))
    if integration not in PUSH_PULL_INTEGRATIONS:
        raise RivalBlend(
            "a push-pull edit must NAME its integration strategy "
            "(payload['integration'] one of: %s). They are rivals -- "
            "Zou (2025) sec. 4.1/4.3/4.4 give three different answers and this "
            "surface will not pick one for you."
            % ", ".join(PUSH_PULL_INTEGRATIONS))
    edit = hm.PushPullEdit(face_name=str(e.param or e.payload.get("face", "")),
                           distance=float(e.value))
    tree = state.tree
    if tree is None:
        raise Unsupported("this HybridModel carries no FeatureTree to reconcile into")

    before_faces = {n: f.offset for n, f in state.brep.faces.items()}
    detail: Dict[str, Any] = {"integration": integration}
    ok = True
    diags: List[str] = []

    if integration == "operation_translation":
        links = [ot.FaceParamLink(**dict(link)) if isinstance(link, dict) else link
                 for link in e.payload.get("links", ())]
        if not links:
            raise Unsupported("operation_translation needs face->parameter links "
                              "(payload['links']); it will not guess which parameter "
                              "a face is driven by")
        symmetric = [tuple(s) for s in e.payload.get("symmetric_params", ())]
        candidates = ot.translate_push_pull(tree, edit, links, symmetric or None)
        detail["candidates"] = [c.description for c in candidates]
        detail["achievable"] = ot.is_achievable(tree, edit, links)
        detail["unique"] = ot.is_unique(tree, edit, links, symmetric or None)
        if not candidates:
            ok = False
            diags.append("not-achievable: no parameter edit realises this face move")
            new_tree = tree
        elif len(candidates) > 1 and "choice" not in e.payload:
            # The paper's non-uniqueness: several parameter edits produce the same
            # face. Picking one silently is a guess, so the caller must choose.
            ok = False
            diags.append("ambiguous: %d parameter edits realise this face move (%s); "
                         "pass payload['choice'] to select one"
                         % (len(candidates),
                            "; ".join(c.description for c in candidates)))
            new_tree = tree
        else:
            translation = candidates[int(e.payload.get("choice", 0))]
            new_tree = translation.apply(tree)
            detail["description"] = translation.description
    elif integration == "pseudo_feature":
        new_tree = pf.append_pseudo_feature(tree, state.brep, edit)
        detail["pseudo_features"] = [f.fid for f in new_tree.features
                                     if f.ftype == "pseudo_move_face"]
        # A LATER parameter edit on the anchor is what breaks a pseudo-feature
        # (the paper's regeneration failure). If one is supplied, run it and report.
        then = e.payload.get("then")
        if then is not None:
            edit_then = then if isinstance(then, hm.ParameterEdit) \
                else hm.ParameterEdit(**dict(then))
            regen = pf.regenerate(new_tree, edit_then)
            ok = regen.ok
            new_tree = regen.tree
            if not ok:
                diags.append("regen-failed: %s" % regen.reason)
            detail["broken"] = list(regen.broken)
    else:  # synchronous_partition
        partition = sp.from_tree(tree)
        fid = str(e.payload.get("fid", ""))
        if not fid:
            raise Unsupported("synchronous_partition needs the feature id whose face "
                              "was dragged (payload['fid'])")
        partition.move_to_direct_edit(fid)
        detail["parametric_loss"] = partition.parametric_loss()
        detail["ordinary"] = list(partition.ordinary())
        detail["direct_edit"] = list(partition.direct_edit())
        new_tree = tree

    new_brep = state.brep.copy()
    new_brep.push_pull(edit.face_name, edit.distance)
    after = hc.HybridModel(brep=new_brep, constraints=list(state.constraints),
                           tree=new_tree)
    inconsistencies = hc.check_consistency(after)
    detail["inconsistencies"] = [i.detail for i in inconsistencies]
    moved = tuple((0, name, before_faces[name], f.offset)
                  for name, f in sorted(new_brep.faces.items())
                  if name in before_faces and before_faces[name] != f.offset)
    d = Diff(changed_params=moved,
             digest_before=_hash(state.brep.to_dict()),
             digest_after=_hash(new_brep.to_dict()))
    return EditResult(ok, e.kind, state, after, d, tuple(diags), detail)


def _hash(obj: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _edit_tokens(state: Any, e: Edit) -> EditResult:
    """Locate-then-infill: mask exactly what changed between two token sequences.

    ``state`` is the original token sequence; ``payload['edited']`` the edited one.
    ``partial_mask`` gives the FINE-GRAINED (per-component) mask and
    ``locate_infill`` the coarse token mask + the infill/immutability check --
    the two granularities the papers distinguish, both reported, neither blended.
    """
    from harnesscad.domain.editing import locate_infill as li
    from harnesscad.domain.editing import partial_mask as pm

    original = [str(t) for t in state]
    edited = [str(t) for t in e.payload.get("edited", ())]
    if not edited:
        raise EditError("a 'mask' edit needs payload['edited'] (the edited tokens)")
    coarse = li.locate_mask(original, edited)
    fine = pm.generate_mask(original, edited)
    filled = li.infill(coarse, iter(e.payload.get("replacements", ())))
    return EditResult(
        True, e.kind, tuple(original), tuple(edited),
        Diff(digest_before=_hash(original), digest_after=_hash(edited)),
        (),
        {"coarse_mask": list(coarse), "fine_mask": list(fine),
         "spans": pm.mask_span_count(fine),
         "context_preserved": li.context_preserved(coarse, edited),
         "infilled": list(filled)})


def _edit_design(state: Any, e: Edit) -> EditResult:
    """An mrCAD sketch edit: apply the instruction's actions, resolve degeneracies.

    ``state`` is a ``sketch_edit_schema.Design``; ``payload['actions']`` the parsed
    edit actions. Degenerate curves (a curve collapsed to a point by a MovePoint)
    are resolved by ``curve_degeneracy``, not silently kept.
    """
    from harnesscad.domain.editing import curve_degeneracy as cd
    from harnesscad.domain.editing import refinement_session as rs

    actions = list(e.payload.get("actions", ()))
    if not actions:
        raise EditError("a 'sketch' edit needs payload['actions']")
    after = rs.apply_actions(state, actions)
    after = cd.canonicalize_design(after)
    return EditResult(True, e.kind, state, after,
                      Diff(digest_before=_hash(_design_key(state)),
                           digest_after=_hash(_design_key(after))),
                      (), {"curves_before": len(state.curves),
                           "curves_after": len(after.curves)})


def _design_key(design: Any) -> Any:
    return [[list(p) for p in c.points] for c in design.curves]


def _edit_text(state: Any, e: Edit) -> EditResult:
    """A source-text edit on a CAD script (indent / unindent / toggle comment)."""
    from harnesscad.domain.editing import code_text_edit as m

    action = str(e.payload.get("action", ""))
    # detect_eol sniffs the RAW BYTES (CQ-editor's file-load convention), so the
    # source is encoded for the sniff and split on whatever it reports.
    eol = m.detect_eol(state.encode("utf-8")) if isinstance(state, str) else "\n"
    lines = state.split(eol) if isinstance(state, str) else list(state)
    fns = {"indent": m.indent_lines, "unindent": m.unindent_lines,
           "comment": m.toggle_comment_block}
    if action not in fns:
        raise EditError("unknown text action %r (one of: %s)"
                        % (action, ", ".join(sorted(fns))))
    out = fns[action](lines)
    after = eol.join(out) if isinstance(state, str) else out
    return EditResult(True, e.kind, state, after,
                      Diff(digest_before=_hash(lines), digest_after=_hash(list(out))),
                      (), {"action": action,
                           "gutter": m.line_number_gutter_digits(len(out))})


def _edit_code_view(state: Any, e: Edit) -> EditResult:
    """A 3D-view edit propagated BACKWARD into the CSG program that produced it.

    ``state`` is a ``programs.ast.bidirectional_csg`` program; ``payload['path']``
    is the source path of the node the user dragged in the view. The put is checked
    against its get (the round-trip law), so a propagation that does not reproduce
    the requested world delta is reported as failed rather than accepted.
    """
    from harnesscad.domain.editing import backward_propagation as bp
    from harnesscad.domain.editing import code_view_navigation as nav

    path = tuple(e.payload.get("path", ()))
    if not path:
        raise EditError("a 'code_view' edit needs payload['path'] (the source path)")
    action = str(e.payload.get("action", "translate"))
    delta = tuple(float(v) for v in (e.value or (0.0, 0.0, 0.0)))
    puts = {"translate": bp.put_translate, "rotate": bp.put_rotate,
            "scale": bp.put_scale}
    if action not in puts:
        raise EditError("unknown code_view action %r (one of: %s)"
                        % (action, ", ".join(sorted(puts))))
    result = puts[action](state, path, delta)
    holds = (bp.put_get_translate_holds(state, path, delta)
             if action == "translate" else None)
    return EditResult(
        holds is not False, e.kind, state, result.program,
        Diff(digest_before=_hash(repr(state)), digest_after=_hash(repr(result.program))),
        () if holds is not False else ("put-get law violated: the propagated edit "
                                       "does not reproduce the requested delta",),
        {"edited_path": list(result.edited_path), "reused": result.reused,
         "get_put_holds": bp.get_put_holds(state),
         "consistency": nav.consistency(state)})


@dataclass(frozen=True)
class EditKind:
    """One edit kind: what it edits, which modules realise it."""

    name: str
    target: str
    description: str
    modules: Tuple[str, ...]
    apply: Callable[[Any, Edit], EditResult]

    def to_dict(self) -> dict:
        return {"name": self.name, "target": self.target,
                "description": self.description, "modules": list(self.modules)}


_EDIT_TABLE: Tuple[Tuple[str, str, str, Tuple[str, ...], Callable], ...] = (
    ("param", "session",
     "Change one numeric parameter of one op and deterministically regenerate the "
     "whole model (the CISP SetParam primitive). The edit is blocked, not applied "
     "half-way, when the rebuilt stream does not re-apply cleanly.",
     (), _edit_param),
    ("align", "session",
     "Align the placed instances by their axis-aligned bbox (left/right/top/bottom/"
     "centre) and emit the parametric moves that realise it.",
     ("layout_ops",), _edit_layout),
    ("distribute", "session",
     "Distribute the placed instances evenly along an axis (equal centres, or equal "
     "gaps at a given spacing).",
     ("layout_ops",), _edit_layout),
    ("push_pull", "hybrid",
     "A DIRECT face drag reconciled into a parametric feature tree. Requires "
     "payload['integration'] naming one of the three rival Zou-2025 strategies.",
     ("hybrid_model", "hybrid_consistency", "operation_translation",
      "pseudo_feature", "synchronous_partition"), _edit_push_pull),
    ("mask", "tokens",
     "Locate-then-infill: the coarse token mask AND the fine-grained per-component "
     "mask of exactly what changed, plus the immutable-context check.",
     ("locate_infill", "partial_mask"), _edit_tokens),
    ("sketch", "design",
     "An mrCAD sketch edit: apply the instruction's actions to a Design and resolve "
     "the curves the edit degenerated.",
     ("sketch_edit_schema", "refinement_session", "curve_degeneracy"), _edit_design),
    ("text", "text",
     "A CAD-script source-text edit (indent / unindent / toggle comment block), "
     "EOL-preserving.",
     ("code_text_edit",), _edit_text),
    ("code_view", "csg_program",
     "A 3D-view edit propagated BACKWARD into the CSG program (put), checked against "
     "its get (the round-trip law) and against the code<->view traceability.",
     ("backward_propagation", "code_view_navigation"), _edit_code_view),
)


# --------------------------------------------------------------------------- #
# Strategies: the edit LOOPS
# --------------------------------------------------------------------------- #
#: The deterministic parameter ladder the stub infiller/refiner proposes from. No
#: model, no LLM: a fixed multiplicative neighbourhood, seeded selection.
LADDER: Tuple[float, ...] = (0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0)


def _numeric_fields(op: Op) -> List[str]:
    return [f.name for f in dataclasses.fields(op)
            if isinstance(getattr(op, f.name), (int, float))
            and not isinstance(getattr(op, f.name), bool)
            and getattr(op, f.name) != 0]


def _scaled(op: Op, param: str, factor: float) -> Op:
    return dataclasses.replace(op, **{param: type(getattr(op, param))(
        getattr(op, param) * factor)})


def _neighbours(ops: Sequence[Op], params: Sequence[Tuple[int, str]]
                ) -> List[Tuple[Op, ...]]:
    """Every one-parameter-one-rung move from ``ops``. Deterministic and finite."""
    out: List[Tuple[Op, ...]] = []
    for (i, param) in params:
        for factor in LADDER:
            cand = list(ops)
            cand[i] = _scaled(cand[i], param, factor)
            out.append(tuple(cand))
    return out


#: The op types the analytic shape proxy actually depends on. The edit LOOPS edit
#: toward a *shape*, so by default they only move parameters the shape can see --
#: scaling a constraint's value changes the model but not this proxy's shape, and a
#: search that spends its budget there is measuring nothing. Pass ``params=`` to
#: override.
_SHAPE_OPS = (AddRectangle, AddCircle, AddLine, AddPoint, Extrude)


def _editable(ops: Sequence[Op], only: Optional[Sequence[Tuple[int, str]]] = None,
              *, shape_only: bool = False) -> List[Tuple[int, str]]:
    if only is not None:
        return [(int(i), str(p)) for i, p in only]
    return [(i, f) for i, op in enumerate(ops)
            if not shape_only or isinstance(op, _SHAPE_OPS)
            for f in _numeric_fields(op)]


def _run_plan_verify(state: Any, target: Any, *, seed: int = 0,
                     n_candidates: int = 4, max_rounds: int = 6,
                     params: Optional[Sequence[Tuple[int, str]]] = None,
                     **kw: Any):
    """CADMorph: plan (mask the worst segments) -> generate -> verify, over ops.

    ``target`` is the desired shape: another op stream / session / ShapeSignature.
    The paper's two learned pieces enter as callables; here the renderer is the
    analytic shape proxy and the infiller is the deterministic parameter ladder, so
    the loop is reproducible from ``seed`` alone.
    """
    from harnesscad.domain.editing import edit_planning as ep
    from harnesscad.domain.editing.plan_verify_loop import CADMorphLoop

    base = ops_of(state)
    target_shape = target if isinstance(target, ShapeSignature) else shape_of(target)
    editable = _editable(base, params, shape_only=True)

    def render(sequence: Sequence[Any]) -> ShapeSignature:
        return shape_of([op for op in sequence if isinstance(op, Op)])

    def distance(a: ShapeSignature, b: ShapeSignature) -> float:
        return shape_distance(a, b)

    contribution = ep.leave_one_out_contribution(render, distance)

    def generate(masked: Sequence[Any], n: int, rng: random.Random) -> List[Tuple]:
        """Fill every MASK slot with a laddered variant of the op that was there."""
        out: List[Tuple] = []
        for _ in range(n):
            cand = list(masked)
            for i, token in enumerate(cand):
                if isinstance(token, Op) or i >= len(base):
                    continue
                op = base[i]
                fields = _numeric_fields(op)
                if not fields:
                    cand[i] = op
                    continue
                param = rng.choice(sorted(fields))
                cand[i] = _scaled(op, param, rng.choice(LADDER))
            out.append(tuple(cand))
        return out

    loop = CADMorphLoop(render, distance, contribution, generate,
                        n_candidates=int(n_candidates), max_rounds=int(max_rounds),
                        **kw)
    result = loop.run(base, target_shape, seed=int(seed))
    return {"ops": tuple(result.sequence), "distance": result.distance,
            "rounds": len(result.rounds), "converged": result.converged,
            "editable": editable,
            "start_distance": shape_distance(shape_of(base), target_shape),
            "result": result}


def _run_refine(state: Any, target: Any, *, max_steps: int = 5, seed: int = 0,
                params: Optional[Sequence[Tuple[int, str]]] = None, **kw: Any):
    """CADReasoner: render -> compare -> refine, keeping the best-so-far program.

    The learned refiner is replaced by a deterministic best-neighbour editor over
    the parameter ladder: it renders every one-move neighbour and takes the one
    closest to the target point set (Chamfer). No model, fully reproducible.
    """
    from harnesscad.domain.editing.refine_loop import run_edit_loop

    base = ops_of(state)
    target_points = points_of(target)
    if not target_points:
        raise Unsupported("the refine loop needs a target with at least one solid")
    editable = _editable(base, params, shape_only=True)

    def render(program: Sequence[Op]) -> Optional[List[Vec3]]:
        pts = points_of(list(program))
        return pts or None

    def metric(a: Sequence[Vec3], b: Sequence[Vec3]) -> Optional[float]:
        return point_distance(a, b)

    def editor(target_pts, prev_render, prev_program, encoding):
        best, best_d = tuple(prev_program), None
        for cand in _neighbours(tuple(prev_program), editable):
            d = point_distance(target_pts, points_of(list(cand)))
            if d is not None and (best_d is None or d < best_d):
                best, best_d = cand, d
        return best

    result = run_edit_loop(target_points, base, editor, render,
                           select_metric=metric, max_steps=int(max_steps),
                           seed=int(seed), **kw)
    return {"ops": tuple(result.best_program or base),
            "distance": result.best_select_score,
            "steps": len(result.steps), "converged": result.converged,
            "stopped_reason": result.stopped_reason,
            "start_distance": point_distance(target_points, points_of(base)),
            "result": result}


def _run_geometry_beam(state: Any, target: Any, *, n: int = 5, steps: int = 4,
                       seed: int = 0,
                       params: Optional[Sequence[Tuple[int, str]]] = None,
                       **kw: Any):
    """CADReasoner's stochastic geometry-guided BEAM (rival of ``refine``).

    Same job as ``refine`` -- edit the program toward the target point set -- but a
    width-N beam over seeded neighbours instead of a single best-first chain. The
    two are never averaged: they explore differently and report different budgets.
    """
    from harnesscad.domain.editing.geometry_beam import run_geometry_beam

    base = ops_of(state)
    target_points = points_of(target)
    if not target_points:
        raise Unsupported("the beam needs a target with at least one solid")
    editable = _editable(base, params, shape_only=True)
    rng = random.Random(int(seed))
    moves = _neighbours(base, editable)
    order = list(range(len(moves)))
    rng.shuffle(order)                      # the beam is stochastic BY DESIGN

    def seed_generator(_target, slot: int) -> Tuple[Op, ...]:
        if not moves:
            return base
        return moves[order[slot % len(order)]]

    def child_generator(_target, parent_program, _parent_render, slot: int):
        children = _neighbours(tuple(parent_program), editable)
        if not children:
            return tuple(parent_program)
        return children[rng.randrange(len(children))]

    def render(program) -> Optional[List[Vec3]]:
        pts = points_of(list(program))
        return pts or None

    def score(a, b) -> Optional[float]:
        return point_distance(a, b)

    result = run_geometry_beam(target_points, seed_generator, child_generator,
                               render, score, n=int(n), steps=int(steps), **kw)
    return {"ops": tuple(result.best_program) if result.best else base,
            "distance": result.best_score,
            "renders": result.total_renders, "invalid": result.total_invalid,
            "start_distance": point_distance(target_points, points_of(base)),
            "result": result}


def _run_history(state: Any, instructions: Sequence[Any] = (), **kw: Any):
    """Append-only edit provenance with rollback (revision deltas).

    Every edit is recorded as a revision (instruction, before-digest, after-digest,
    result), so an edit session can be replayed or rolled back to any revision.
    ``instructions`` is a sequence of :class:`Edit`.
    """
    from harnesscad.domain.editing.iterative_session import IterativeEditSession

    session = IterativeEditSession(snapshot(state))

    def editor(before: ModelState, instruction: Edit) -> ModelState:
        result = apply_edit(before, instruction)
        if not result.ok:
            raise EditError("revision blocked: %s" % "; ".join(result.diagnostics))
        return snapshot(result.after)

    for instruction in instructions:
        session.apply(instruction, editor)
    return {"session": session, "current": session.current,
            "revisions": len(session.revisions),
            "digests": [r.after_digest for r in session.revisions]}


def _run_sketch2cad(seed_faces: Sequence[Any] = (), records: Sequence[Any] = (),
                    **kw: Any):
    """Sketch2CAD's incremental modelling state machine (replay / undo / redo).

    A history of face-referencing operations replayed onto a seed face set. The
    state machine validates each record against the *live* faces, so an op that
    references a face a previous op consumed is rejected rather than applied.
    """
    from harnesscad.domain.editing import modeling_session as m

    session = m.replay(list(seed_faces), list(records))
    return {"session": session, "summary": session.summary(),
            "steps": session.step_count(),
            "signature": session.state_signature(),
            "history": m.serialize_history(session)}


def _run_consistency(model: Any, edit: Any = None, **kw: Any):
    """Reconcile the three information layers of a hybrid (parametric+direct) model.

    Checks the parametric / geometric / constraint layers against each other,
    propagates an edit into the others, and reports the design-intent drift between
    the constraints the model declares and the ones its geometry actually implies.
    """
    from harnesscad.domain.editing import hybrid_consistency as hc
    from harnesscad.domain.editing import hybrid_model as hm

    before = hc.check_consistency(model)
    detail: Dict[str, Any] = {"before": [i.detail for i in before]}
    if edit is not None:
        layer = hm.edit_layer(edit)
        if isinstance(edit, hm.ParameterEdit):
            hc.propagate_parametric_to_geometry(model, edit)
        elif isinstance(edit, hm.PushPullEdit):
            hc.propagate_direct_to_constraint(model, edit)
        else:
            raise Unsupported("cannot propagate a %r" % type(edit).__name__)
        detail["paradigm"] = hm.classify_edit(edit).value
        detail["layer"] = layer.value
    recognized = hc.recognize_constraints(model.brep)
    detail["after"] = [i.detail for i in hc.check_consistency(model)]
    detail["consistent"] = hc.is_consistent(model)
    detail["drift"] = hc.design_intent_drift(list(model.constraints), recognized)
    return detail


@dataclass(frozen=True)
class Strategy:
    """One named edit strategy (a loop, a session, a reconciliation)."""

    name: str
    target: str
    description: str
    modules: Tuple[str, ...]
    run: Callable[..., Any]
    family: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "target": self.target, "family": self.family,
                "description": self.description, "modules": list(self.modules)}


_STRATEGY_TABLE: Tuple[Tuple[str, str, str, Tuple[str, ...], Callable, str], ...] = (
    ("plan_verify", "session",
     "CADMorph (NeurIPS 2025): mask the segments that contribute most to the gap, "
     "infill N candidates, verify each against the target and keep the global best "
     "in a cross-round queue.",
     ("plan_verify_loop", "edit_planning", "candidate_verify", "locate_infill"),
     _run_plan_verify, "edit_search"),
    ("refine", "session",
     "CADReasoner: render -> compare -> refine, single best-so-far chain, driven by "
     "the geometry-discrepancy encoding.",
     ("refine_loop", "discrepancy_encoding"), _run_refine, "edit_search"),
    ("geometry_beam", "session",
     "CADReasoner's stochastic geometry-guided beam: width-N exploration of the "
     "edit neighbourhood, with an explicit render budget.",
     ("geometry_beam",), _run_geometry_beam, "edit_search"),
    ("history", "session",
     "Append-only revision provenance: every edit recorded with before/after "
     "digests, rollback to any revision.",
     ("iterative_session",), _run_history, ""),
    ("sketch2cad", "faces",
     "Sketch2CAD's incremental modelling state machine: replay a face-referencing "
     "op history with validation, undo and redo.",
     ("modeling_session",), _run_sketch2cad, ""),
    ("consistency", "hybrid",
     "Three-layer information-consistency reconciliation of a hybrid "
     "parametric+direct model, plus design-intent drift.",
     ("hybrid_consistency", "hybrid_model"), _run_consistency, ""),
)

#: Strategies that answer the SAME question differently. Selected by name, never
#: averaged: they are different algorithms with different budgets and guarantees.
_RIVALS: Dict[str, Tuple[str, ...]] = {
    "edit_search": ("plan_verify", "refine", "geometry_beam"),
    "push_pull_integration": PUSH_PULL_INTEGRATIONS,
}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
_EDITS: Optional[Dict[str, EditKind]] = None
_STRATS: Optional[Dict[str, Strategy]] = None
_UNADAPTED: Tuple[str, ...] = ()

#: Editing modules an adapter reaches only through another adapted module.
_INDIRECT: Tuple[str, ...] = ("brep", "hybrid_model", "sketch_edit_schema",
                              "latent_preserve")


def _build() -> Tuple[Dict[str, EditKind], Dict[str, Strategy]]:
    global _UNADAPTED
    entries = {e.dotted for e in capability_registry.find(package=EDITING_PACKAGE)}
    adapted = set()
    kinds: Dict[str, EditKind] = {}
    for name, target, description, mods, fn in _EDIT_TABLE:
        dotted = tuple(_PKG + m for m in mods)
        if any(d not in entries for d in dotted):
            continue
        adapted.update(dotted)
        kinds[name] = EditKind(name, target, description, dotted, fn)
    strats: Dict[str, Strategy] = {}
    for name, target, description, mods, fn, family in _STRATEGY_TABLE:
        dotted = tuple(_PKG + m for m in mods)
        if any(d not in entries for d in dotted):
            continue
        adapted.update(dotted)
        strats[name] = Strategy(name, target, description, dotted, fn, family)
    adapted.update(_PKG + m for m in _INDIRECT)
    _UNADAPTED = tuple(sorted(d for d in entries
                              if d not in adapted and not d.endswith(".registry")))
    return kinds, strats


def _all_edits() -> Dict[str, EditKind]:
    global _EDITS, _STRATS
    if _EDITS is None:
        _EDITS, _STRATS = _build()
    return _EDITS


def _all_strategies() -> Dict[str, Strategy]:
    global _EDITS, _STRATS
    if _STRATS is None:
        _EDITS, _STRATS = _build()
    return _STRATS


def edits() -> Tuple[str, ...]:
    """Every edit kind whose modules are actually in the tree."""
    return tuple(sorted(_all_edits()))


def edit_kind(name: str) -> EditKind:
    try:
        return _all_edits()[name]
    except KeyError:
        raise UnknownEdit("unknown edit kind %r (one of: %s)"
                          % (name, ", ".join(edits()))) from None


def strategies() -> Tuple[str, ...]:
    return tuple(sorted(_all_strategies()))


def strategy(name: str) -> Strategy:
    try:
        return _all_strategies()[name]
    except KeyError:
        raise UnknownStrategy("unknown edit strategy %r (one of: %s)"
                              % (name, ", ".join(strategies()))) from None


def rivals() -> Dict[str, Tuple[str, ...]]:
    """Families of strategies that answer the same question DIFFERENTLY."""
    return {k: tuple(v) for k, v in sorted(_RIVALS.items())}


def unadapted() -> Tuple[str, ...]:
    """Editing modules the index knows but no edit/strategy binds."""
    _all_edits()
    return _UNADAPTED


def modules() -> Tuple[str, ...]:
    """Every editing module this surface dispatches to."""
    seen = set()
    for k in _all_edits().values():
        seen.update(k.modules)
    for s in _all_strategies().values():
        seen.update(s.modules)
    seen.update(_PKG + m for m in _INDIRECT)
    return tuple(sorted(seen))


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #
def apply_edit(state: Any, edit: Edit) -> EditResult:
    """Apply ``edit`` to ``state``. A rejected edit is an EditResult, not a crash.

    Only a genuinely malformed request (an unknown kind, a missing mandatory field,
    an edit aimed at the wrong target) raises: a *blocked* edit -- one the backend
    rejects -- comes back as ``EditResult(ok=False)`` with the diagnostics, and the
    state is left exactly as it was (block-and-correct).
    """
    if isinstance(edit, dict):
        edit = Edit(**edit)
    kind = edit_kind(edit.kind)
    return kind.apply(state, edit)


@dataclass(frozen=True)
class StrategyResult:
    """The outcome of one strategy run. A raising strategy is captured here."""

    name: str
    ok: bool
    value: Any = None
    error: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "error": self.error}


def run_strategy(name: str, *args: Any, **kwargs: Any) -> StrategyResult:
    """Run a named strategy. A component that raises is CAPTURED, never fatal."""
    strat = strategy(name)
    try:
        return StrategyResult(name, True, strat.run(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - the whole point: never fatal
        return StrategyResult(name, False, None, "%s: %s" % (type(exc).__name__, exc))


# --------------------------------------------------------------------------- #
# CLI (wired into core.cli as `harnesscad edit`)
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list the edit kinds and the edit strategies")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival strategy families (never blended)")
    parser.add_argument("--unadapted", action="store_true",
                        help="list editing modules with no call site yet")
    parser.add_argument("--ops", default=None, metavar="OPS.JSON",
                        help="the op stream to edit (default: the built-in demo)")
    parser.add_argument("--params", action="store_true",
                        help="list the editable parameters of the op stream")
    parser.add_argument("--set", default=None, metavar="I:PARAM=VALUE",
                        help="apply one parametric edit and print the diff")
    parser.add_argument("--strategy", default=None,
                        help="run a named edit strategy toward --target")
    parser.add_argument("--target", default=None, metavar="OPS.JSON",
                        help="the target op stream the strategy edits toward")
    parser.add_argument("--seed", type=int, default=0, help="strategy seed")
    parser.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
    parser.add_argument("--json", action="store_true", help="emit the result as JSON")


def _load_ops(path: Optional[str]) -> List[dict]:
    if path is None:
        from harnesscad.core.cli import DEMO_OPS

        return [dict(op) for op in DEMO_OPS]
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise EditError("%r must contain a JSON array of ops" % path)
    return data


def _session(ops: Sequence[dict], backend: str):
    from harnesscad.io.surfaces.server import CISPServer

    server = CISPServer(backend=backend)
    result = server.applyOps([dict(op) for op in ops])
    if not result["ok"]:
        raise EditError("the base op stream does not apply cleanly: %s"
                        % json.dumps(result.get("diagnostics") or []))
    return server.session


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "rivals", False):
        for family, names in rivals().items():
            print("%s: %s" % (family, ", ".join(names)))
            print("    selected by name; NEVER averaged")
        return 0
    if getattr(args, "unadapted", False):
        for dotted in unadapted():
            print(dotted)
        print("-- %d editing modules without a call site" % len(unadapted()))
        return 0

    try:
        if getattr(args, "params", False) or getattr(args, "set", None) \
                or getattr(args, "strategy", None):
            session = _session(_load_ops(args.ops), args.backend)
        else:
            session = None

        if getattr(args, "params", False):
            for p in parameters(session):
                print("%3d  %-16s %-10s %s" % (p.index, p.op, p.param, p.value))
            return 0

        if getattr(args, "set", None):
            spec = args.set
            if ":" not in spec or "=" not in spec:
                print("error: --set wants I:PARAM=VALUE", file=sys.stderr)
                return 2
            idx, rest = spec.split(":", 1)
            param, value = rest.split("=", 1)
            try:
                parsed: Any = float(value)
            except ValueError:
                parsed = value
            result = apply_edit(session, Edit("param", int(idx), param, parsed))
            if args.json:
                print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
            else:
                print("ok:       %s" % result.ok)
                for d in result.diagnostics:
                    print("  %s" % d)
                if result.diff:
                    for (i, p, old, new) in result.diff.changed_params:
                        print("  op[%d].%s: %s -> %s" % (i, p, old, new))
                    print("shape:    %s" % json.dumps(
                        snapshot(result.after).shape.to_dict(), sort_keys=True))
            return 0 if result.ok else 1

        if getattr(args, "strategy", None):
            if not args.target:
                print("error: --strategy needs --target OPS.JSON", file=sys.stderr)
                return 2
            target = _session(_load_ops(args.target), args.backend)
            out = run_strategy(args.strategy, session, target, seed=args.seed)
            if not out.ok:
                print("error: %s" % out.error, file=sys.stderr)
                return 1
            value = out.value
            print("strategy: %s" % args.strategy)
            print("distance: %s -> %s" % (value.get("start_distance"),
                                          value.get("distance")))
            print("ops:      %s" % json.dumps(
                [op.to_dict() for op in value["ops"]], sort_keys=True))
            return 0
    except EditError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except OSError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    for name in edits():
        k = edit_kind(name)
        print("%-14s [%s]" % (name, k.target))
        print("    %s" % k.description)
    print()
    for name in strategies():
        s = strategy(name)
        tag = (" (rival family: %s)" % s.family) if s.family else ""
        print("%-14s [%s]%s" % (name, s.target, tag))
        print("    %s" % s.description)
    print()
    print("-- %d edit kinds / %d strategies / %d editing modules unbound"
          % (len(edits()), len(strategies()), len(unadapted())))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad edit",
        description="the edit surface: apply an edit to a model, diff it, or run "
                    "a named edit loop")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
