"""Valid-action masking for the CAD gym -- by PREFLIGHT, not by trial execution.

RLCAD (arXiv:2503.18549, Algorithm 1 "Valid Action Generation") observes that
invalid actions "interfere with the agent's learning process, making the network
difficult to converge", and therefore redefines the action space each step as
exactly the set of operations that are valid *now*. Their method for deciding
validity is TRIAL EXECUTION: hand every candidate op to the geometry kernel and
keep the ones it accepts. That is correct and expensive -- one kernel round trip
per candidate per step, and the round trip is the single most expensive thing in
the loop.

This harness already answers the same question WITHOUT the kernel: verifying an
op stream before the kernel runs it is the whole thesis of the repo (see
``core/contract.py``, the verifier fleet, and the ``preflight-*`` diagnostics).
So the valid-action set is computed here by PREFLIGHT VERIFICATION:

  1. **Structural tier** -- the symbolic plan linter
     (:class:`eval.verifiers.precheck.PrecheckCheck`) is walked ONCE over the
     ops already applied in the session; each candidate op is then judged
     against a cheap copy of that walked state. Cost per candidate is a dict
     copy plus one visit -- no backend, no kernel, no geometry.
  2. **Reference tier** -- typed references (sketch / entity / feature /
     instance ids) are checked against the ids the session actually holds, when
     the backend exposes them. This is what keeps the mask from being
     over-permissive about dangling refs.
  3. **Value tier** -- the per-op preconditions the plan linter does not model
     (the newer ops: primitive, split, thicken, hull, minkowski, transform,
     scale, pattern_transform, arcs/ellipses/polygons/splines), mirrored from
     the op semantics, plus the symbolic op-stream edit check for
     ``set_param`` (``ops.edit_oplog``).
  4. **Selector tier** -- edge/face selector strings are parsed against the
     selector grammar (:mod:`domain.geometry.topology.selector_dsl`), so a
     malformed selector ("top" for ">Z") is refused without asking the kernel.
  5. **Numeric tier** (optional, on by default) -- the shape-level kernel
     preflight (:mod:`eval.verifiers.kernel_preflight`): a fillet radius or a
     shell thickness the stock cannot carry (``RADIUS_TOO_LARGE`` /
     ``THICKNESS_TOO_LARGE``). It fires ONLY when the stock extents are
     knowable (measured from the backend, or inferred from the sketch bounds +
     extrude depth by the plan walk); when they are not, it stays silent, since
     a false rejection silently shrinks the agent's action space.

:func:`trial_verdict` is the RLCAD baseline kept here on purpose: it answers the
same question by deep-copying the session and actually executing the op, so the
two methods can be compared for AGREEMENT and for COST
(:func:`compare_masks`). Agreement is the property that matters -- speed is
worthless if the cheap mask is not the same mask.

Stdlib only; deterministic; never mutates the session it is asked about.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad.core.cisp import ops as _ops
from harnesscad.core.cisp.ops import CONSTRAINT_DOF, Op, parse_op
from harnesscad.core.diagnostics import Severity
from harnesscad.eval.verifiers.kernel_preflight import (
    BoundingBox, ShapeInfo, preflight_fillet, preflight_shell,
)
from harnesscad.domain.geometry.topology.selector_dsl import SelectorError
from harnesscad.domain.geometry.topology.selector_dsl import parse as parse_selector
from harnesscad.eval.verifiers.precheck import PrecheckRules

try:  # the walked symbolic state; a rename must degrade, never crash
    from harnesscad.eval.verifiers.precheck import _PlanState as _PrecheckPlanState
except ImportError:  # pragma: no cover - defensive
    _PrecheckPlanState = None  # type: ignore[assignment]

__all__ = [
    "MASK_RULES",
    "Verdict",
    "StateView",
    "state_view",
    "bind_action",
    "default_proposals",
    "op_verdict",
    "mask_verdicts",
    "action_mask",
    "valid_actions",
    "trial_verdict",
    "compare_masks",
    "MaskComparison",
]

#: Plan-linter rules for MASKING (not for grading). ``min_wall`` is switched off:
#: a 0.4 mm wall is thin to MANUFACTURE but perfectly buildable, and a mask that
#: refused it would delete a legal action from the agent's action space. DFM
#: limits belong in the reward, not in the mask. Callers who want them back pass
#: their own :class:`PrecheckRules`.
MASK_RULES = PrecheckRules(min_wall=0.0)

#: The primitive shapes every backend models (mirrors ``io.backends.stub``).
_PRIMITIVE_SHAPES = ("box", "sphere", "cylinder", "cone", "torus", "wedge")

#: Ops that cannot run before a solid exists. The plan linter covers the classic
#: ones (fillet/chamfer/shell/draft/mirror/pattern/hole-on-a-face); these are the
#: ops it does not model at all.
_REQUIRES_SOLID = (
    "split", "thicken", "hull", "minkowski", "transform", "scale",
    "pattern_transform",
)


# ===========================================================================
# Verdicts
# ===========================================================================
@dataclass(frozen=True)
class Verdict:
    """Why one candidate action is (in)valid in the current state."""

    valid: bool
    code: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {"valid": self.valid, "code": self.code, "reason": self.reason}


_OK = Verdict(True)


def _no(code: str, reason: str) -> Verdict:
    return Verdict(False, code, reason)


# ===========================================================================
# The state the mask is computed from
# ===========================================================================
@dataclass
class StateView:
    """Everything the mask needs, extracted ONCE per step (kernel-free).

    ``plan`` is the symbolic plan-linter state already walked over the applied
    ops; each candidate is judged against a copy of it. ``stock`` is the solid's
    bounding box when it is knowable (measured, or inferred from the plan walk),
    else ``None`` -- the numeric tier stays silent rather than guess.
    """

    ops: Tuple[Op, ...] = ()
    sketch_ids: Tuple[str, ...] = ()
    entity_ids: Tuple[str, ...] = ()
    feature_ids: Tuple[str, ...] = ()
    instance_ids: Tuple[str, ...] = ()
    solid_present: bool = False
    ids_are_live: bool = False
    stock: Optional[BoundingBox] = None
    plan: Any = None
    plan_diags: Tuple = ()

    def known_refs(self) -> set:
        refs = set(self.feature_ids) | set(self.instance_ids)
        if self.solid_present:
            refs |= {"solid", "body", "last"}
        return refs


def _live_ids(backend) -> Optional[Tuple[Tuple[str, ...], ...]]:
    """(sketches, entities, features, instances) when the backend exposes them."""
    sketches = getattr(backend, "sketches", None)
    entities = getattr(backend, "entities", None)
    features = getattr(backend, "features", None)
    instances = getattr(backend, "instances", None)
    if not isinstance(sketches, dict) or not isinstance(features, list):
        return None
    ent = tuple(entities) if isinstance(entities, dict) else ()
    inst = tuple(str(i.get("id", "")) for i in instances) if isinstance(instances, list) else ()
    feat = tuple(str(f.get("id", "")) for f in features if isinstance(f, dict) and "id" in f)
    return (tuple(sketches), ent, feat, inst)


def _measured_stock(backend) -> Optional[BoundingBox]:
    """The live bounding box, when the backend can measure one."""
    try:
        measure = backend.query("measure")
    except Exception:  # noqa: BLE001 - a mask must never crash the env
        return None
    if not isinstance(measure, Mapping):
        return None
    box = measure.get("bbox")
    if not (isinstance(box, (list, tuple)) and len(box) == 3):
        return None
    try:
        dx, dy, dz = (float(v) for v in box)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if min(dx, dy, dz) <= 0.0:
        return None
    return BoundingBox(0.0, 0.0, 0.0, dx, dy, dz)


def _planned_stock(plan) -> Optional[BoundingBox]:
    """The stock box inferred symbolically: sketch footprint x extrude depth."""
    if plan is None:
        return None
    planar = getattr(plan, "stock_planar", None)
    wall = getattr(plan, "wall", None)
    if not planar or wall is None or wall <= 0:
        return None
    minx, miny, maxx, maxy = (float(v) for v in planar)
    if maxx <= minx or maxy <= miny:
        return None
    return BoundingBox(minx, miny, 0.0, maxx, maxy, float(wall))


def state_view(session, rules: Optional[PrecheckRules] = None) -> StateView:
    """Extract the masking state from a live session. Never mutates it."""
    if session is None:
        return StateView()
    ops = tuple(session.opdag.ops())
    backend = getattr(session, "backend", None)
    view = StateView(ops=ops)

    live = _live_ids(backend) if backend is not None else None
    if live is not None:
        view.sketch_ids, view.entity_ids, view.feature_ids, view.instance_ids = live
        view.ids_are_live = True
    else:
        n = sum(1 for op in ops if isinstance(op, _ops.NewSketch))
        view.sketch_ids = tuple("sk%d" % (i + 1) for i in range(n))

    if backend is not None:
        try:
            summary = backend.query("summary")
        except Exception:  # noqa: BLE001
            summary = {}
        view.solid_present = bool(isinstance(summary, Mapping)
                                  and summary.get("solid_present"))

    if _PrecheckPlanState is not None:
        plan = _PrecheckPlanState(rules or MASK_RULES)
        view.plan_diags = tuple(plan.run(list(ops)))
        view.plan = plan
        if not view.ids_are_live:
            view.solid_present = view.solid_present or plan.have_solid
        elif view.solid_present:
            # LIVE STATE SUPERSEDES SYMBOLIC INFERENCE. The plan linter does not
            # model every op (a `primitive` makes a solid it never counts), so a
            # solid it cannot see would make the mask refuse every solid-consuming
            # op -- exactly the over-restriction masking must not introduce.
            plan.n_solids = max(plan.n_solids, len(view.feature_ids), 1)

    stock = _measured_stock(backend) if backend is not None else None
    view.stock = stock if stock is not None else _planned_stock(view.plan)
    return view


# ===========================================================================
# Candidate binding -- what "the fillet action" means in THIS state
# ===========================================================================
_SKETCH_FIELDS = ("sketch", "path")


def bind_action(name: str, view: StateView,
                args: Optional[Mapping[str, Any]] = None) -> Op:
    """Build the candidate op for tool *name* in *view*.

    Unsupplied fields take their op defaults, except typed sketch references,
    which bind to the sketch the agent most recently created -- the question the
    mask answers is "could this action succeed now?", and an action whose only
    sketch reference is a placeholder would answer a different question. Every
    field can be overridden through *args*.
    """
    payload: Dict[str, Any] = {"op": name}
    last_sketch = view.sketch_ids[-1] if view.sketch_ids else "sk1"
    cls = _ops._REGISTRY.get(name)
    fields = getattr(cls, "__dataclass_fields__", {}) if cls is not None else {}
    for fname in fields:
        if fname in _SKETCH_FIELDS:
            payload[fname] = last_sketch
        elif fname == "sketches" and name == "loft":
            payload[fname] = list(view.sketch_ids[-2:]) or [last_sketch]
        elif fname == "a" and name == "constrain":
            payload[fname] = view.entity_ids[-1] if view.entity_ids else "e1"
    payload.update({k: v for k, v in (args or {}).items() if k != "op"})
    return parse_op(payload)


def default_proposals(view: StateView,
                      names: Optional[Sequence[str]] = None) -> Dict[str, Op]:
    """One state-bound candidate op per tool name (the name-level action space)."""
    out: Dict[str, Op] = {}
    for name in (names if names is not None else sorted(_ops._REGISTRY)):
        try:
            out[name] = bind_action(name, view)
        except Exception:  # noqa: BLE001 - an unbindable op is simply not offered
            continue
    return out


# ===========================================================================
# The preflight tiers
# ===========================================================================
def _structural_verdict(view: StateView, op: Op) -> Verdict:
    """Tier 1: the symbolic plan linter, applied incrementally to *op*."""
    plan = view.plan
    if plan is None:
        return _OK
    trial = copy.deepcopy(plan)
    seen = len(trial.diags)
    trial._visit(len(view.ops), op)
    for diag in trial.diags[seen:]:
        if diag.severity is Severity.ERROR:
            return _no("preflight-%s" % diag.code, diag.message)
    return _OK


def _ref_verdict(view: StateView, op: Op) -> Verdict:
    """Tier 2: typed references must resolve against ids the session holds."""
    if not view.ids_are_live:
        return _OK
    sketches = set(view.sketch_ids)
    for fname in _SKETCH_FIELDS:
        ref = getattr(op, fname, None)
        if isinstance(ref, str) and ref and ref not in sketches:
            return _no("preflight-bad-ref", "unknown sketch '%s'" % ref)
    for ref in getattr(op, "sketches", ()) or ():
        if ref not in sketches:
            return _no("preflight-bad-ref", "unknown sketch '%s'" % ref)
    if isinstance(op, _ops.Constrain):
        entities = set(view.entity_ids)
        for ref in (op.a, op.b):
            if ref is not None and ref not in entities:
                return _no("preflight-bad-ref", "unknown entity '%s'" % ref)
    if isinstance(op, _ops.Hole):
        ref = str(op.face_or_sketch)
        if ref.startswith("sk") and ref not in sketches:
            return _no("preflight-bad-ref", "unknown sketch '%s'" % ref)
    features = set(view.feature_ids)
    for fname in ("feature", "feature_or_body"):
        ref = getattr(op, fname, None)
        if isinstance(ref, str) and ref and ref not in features:
            return _no("preflight-bad-ref", "unknown feature '%s'" % ref)
    if isinstance(op, _ops.Hull):
        for ref in (op.target, op.tool):
            if ref and ref not in features:
                return _no("preflight-bad-ref", "unknown hull ref '%s'" % ref)
    if isinstance(op, _ops.AddInstance):
        if op.part not in view.known_refs():
            return _no("preflight-bad-ref", "unknown part '%s'" % op.part)
    if isinstance(op, _ops.Mate):
        refs = set(view.instance_ids) | features
        for ref in (op.a, op.b):
            if ref and ref not in refs:
                return _no("preflight-bad-ref", "unknown mate ref '%s'" % ref)
    return _OK


def _value_verdict(view: StateView, op: Op) -> Verdict:
    """Tier 3: the per-op preconditions the plan linter does not model."""
    tag = getattr(op, "OP", "")
    if tag in _REQUIRES_SOLID and not view.solid_present:
        return _no("preflight-no-solid", "%s requires an existing solid" % tag)
    if isinstance(op, _ops.AddArc):
        if op.r <= 0:
            return _no("preflight-bad-value", "arc radius must be > 0")
        if float(op.start) == float(op.end):
            return _no("preflight-bad-value", "arc start and end angle must differ")
    elif isinstance(op, _ops.AddEllipse):
        if op.rx <= 0 or op.ry <= 0:
            return _no("preflight-bad-value", "ellipse rx and ry must be > 0")
    elif isinstance(op, _ops.AddPolygon):
        if len(op.points) < 6 or len(op.points) % 2 != 0:
            return _no("preflight-bad-value", "polygon needs >= 3 vertices")
    elif isinstance(op, _ops.AddSpline):
        if len(op.points) < 4 or len(op.points) % 2 != 0:
            return _no("preflight-bad-value", "spline needs >= 2 points")
    elif isinstance(op, _ops.Primitive):
        shape = str(op.shape).lower()
        if shape not in _PRIMITIVE_SHAPES:
            return _no("preflight-bad-value", "unknown primitive shape '%s'" % op.shape)
        if shape in ("box", "wedge") and (op.dx <= 0 or op.dy <= 0 or op.dz <= 0):
            return _no("preflight-bad-value", "%s dx, dy, dz must be > 0" % shape)
        if shape in ("sphere", "torus") and op.r <= 0:
            return _no("preflight-bad-value", "%s radius r must be > 0" % shape)
        if shape in ("cylinder", "cone") and (op.r <= 0 or op.h <= 0):
            return _no("preflight-bad-value", "%s r and h must be > 0" % shape)
    elif isinstance(op, _ops.Split):
        if op.keep not in ("positive", "negative", "both"):
            return _no("preflight-bad-value", "unknown split keep '%s'" % op.keep)
    elif isinstance(op, _ops.Thicken):
        if op.thickness == 0:
            return _no("preflight-bad-value", "thicken thickness must be non-zero")
    elif isinstance(op, _ops.Minkowski):
        if op.radius <= 0:
            return _no("preflight-bad-value", "minkowski radius must be > 0")
    elif isinstance(op, _ops.Scale):
        if op.sx <= 0 or op.sy <= 0 or op.sz <= 0:
            return _no("preflight-bad-value", "scale factors must be > 0")
    elif isinstance(op, _ops.PatternTransform):
        n = len(op.placements)
        if n < 6 or n % 6 != 0:
            return _no("preflight-bad-value",
                       "pattern_transform placements must be flat six-float tuples")
    elif isinstance(op, _ops.Hull):
        if not view.solid_present or not view.feature_ids:
            return _no("preflight-no-solid", "hull requires an existing solid")
    elif isinstance(op, _ops.Boolean):
        if op.kind not in ("union", "cut", "intersect"):
            return _no("preflight-bad-value", "unknown boolean kind '%s'" % op.kind)
    elif isinstance(op, _ops.Constrain):
        if op.kind not in CONSTRAINT_DOF:
            return _no("preflight-bad-value", "unknown constraint kind '%s'" % op.kind)
        if op.kind in ("distance", "radius") and op.value is None:
            return _no("preflight-bad-value",
                       "'%s' constraint requires a value" % op.kind)
    elif isinstance(op, _ops.Mirror):
        if not view.solid_present:
            return _no("preflight-no-solid", "mirror requires an existing solid")
    elif isinstance(op, _ops.SetParam):
        # The op-stream edit is decided symbolically by the same function the
        # backend uses (`ops.edit_oplog`) -- a bad target index or an unknown
        # parameter is caught here instead of by a replay through the kernel.
        oplog = [o for o in view.ops if not isinstance(o, _ops.SetParam)]
        try:
            _new_log, err = _ops.edit_oplog(oplog, op)
        except Exception:  # noqa: BLE001 - never crash the mask
            return _OK
        if err is not None:
            return _no("preflight-%s" % err[0], err[1])
    return _OK


#: Op fields holding selector-string tuples.
_SELECTOR_FIELDS = ("edges", "faces")

#: Refs that mean "the current body", never a selector.
_BODY_ALIASES = ("solid", "body", "last")

#: Datum-plane names a neutral plane accepts instead of a face selector.
_DATUM_PLANES = ("XY", "YX", "XZ", "ZX", "YZ", "ZY")


def _selector_verdict(view: StateView, op: Op) -> Verdict:
    """Tier 3b: selector strings must parse against the selector grammar.

    A malformed edge/face selector ("top" instead of ">Z") is a typed
    ``bad-value`` in every kernel backend, and the grammar
    (:mod:`domain.geometry.topology.selector_dsl`) decides it symbolically -- so
    the mask decides it too, instead of paying a kernel round trip to be told.
    """
    selectors: List[str] = []
    for fname in _SELECTOR_FIELDS:
        value = getattr(op, fname, None)
        if isinstance(value, (list, tuple)):
            selectors.extend(str(s) for s in value)
    if isinstance(op, _ops.Hole):
        ref = str(op.face_or_sketch)
        # "", a body alias and a sketch id all mean "the default drilling face";
        # anything else is a face selector.
        if ref and not ref.startswith("sk") and ref not in _BODY_ALIASES:
            selectors.append(ref)
    if isinstance(op, _ops.Draft) and op.neutral_plane:
        # The neutral plane is either a datum-plane NAME or a face selector.
        if str(op.neutral_plane).strip().upper() not in _DATUM_PLANES:
            selectors.append(str(op.neutral_plane))
    for sel in selectors:
        text = sel.strip()
        if not text:
            continue
        try:
            parse_selector(text)
        except SelectorError as exc:
            return _no("preflight-bad-selector",
                       "selector %r is malformed: %s" % (text, exc))
        except Exception:  # noqa: BLE001 - grammar must never crash the mask
            return _OK
    return _OK


def _numeric_verdict(view: StateView, op: Op) -> Verdict:
    """Tier 4: the shape-level kernel preflight (radius / thickness vs stock).

    Silent unless the stock extents are knowable: an unfounded rejection here
    would shrink the agent's action space with no evidence.
    """
    stock = view.stock
    if stock is None:
        return _OK
    shape = ShapeInfo(id="stock", bbox=stock, volume=max(stock.volume, 0.0),
                      manifold=True)
    if shape.volume <= 0.0:
        return _OK
    failure = None
    if isinstance(op, _ops.Fillet):
        failure = preflight_fillet(shape, op.radius)
    elif isinstance(op, _ops.Shell):
        failure = preflight_shell(shape, op.thickness)
    if failure is None:
        return _OK
    return _no("preflight-%s" % failure.code, failure.message)


def op_verdict(view: StateView, op: Op, *, numeric: bool = True) -> Verdict:
    """The preflight verdict for one candidate op in one state."""
    for tier in (_structural_verdict, _ref_verdict, _value_verdict,
                 _selector_verdict):
        verdict = tier(view, op)
        if not verdict.valid:
            return verdict
    if numeric:
        return _numeric_verdict(view, op)
    return _OK


# ===========================================================================
# The mask
# ===========================================================================
def _proposal_ops(session, proposals, names, rules) -> Tuple[StateView, Dict[str, Op]]:
    view = state_view(session, rules)
    if proposals is None:
        return view, default_proposals(view, names)
    bound: Dict[str, Op] = {}
    for name, spec in proposals.items():
        if isinstance(spec, Op):
            bound[name] = spec
        elif isinstance(spec, Mapping):
            bound[name] = bind_action(spec.get("op", name), view, spec)
        else:
            bound[name] = bind_action(name, view)
    return view, bound


def mask_verdicts(session, proposals: Optional[Mapping[str, Any]] = None, *,
                  names: Optional[Sequence[str]] = None,
                  numeric: bool = True,
                  rules: Optional[PrecheckRules] = None) -> Dict[str, Verdict]:
    """``{action name -> Verdict}`` for the current state (one state walk)."""
    view, bound = _proposal_ops(session, proposals, names, rules)
    return {name: op_verdict(view, op, numeric=numeric)
            for name, op in bound.items()}


def action_mask(session, proposals: Optional[Mapping[str, Any]] = None, *,
                names: Optional[Sequence[str]] = None,
                numeric: bool = True,
                rules: Optional[PrecheckRules] = None) -> Dict[str, bool]:
    """``{action name -> bool}`` -- the boolean mask RL libraries consume."""
    return {name: v.valid for name, v in mask_verdicts(
        session, proposals, names=names, numeric=numeric, rules=rules).items()}


def valid_actions(session, proposals: Optional[Mapping[str, Any]] = None, *,
                  names: Optional[Sequence[str]] = None,
                  numeric: bool = True,
                  rules: Optional[PrecheckRules] = None) -> List[str]:
    """The valid subset of the action space, sorted (RLCAD's action set)."""
    mask = action_mask(session, proposals, names=names, numeric=numeric, rules=rules)
    return sorted(name for name, ok in mask.items() if ok)


# ===========================================================================
# The RLCAD baseline: trial execution, kept for agreement / cost comparison
# ===========================================================================
def trial_verdict(session, op: Op) -> Verdict:
    """RLCAD Algorithm 1: execute the candidate and keep it if the kernel agrees.

    The session is deep-copied first, so the real one never sees the trial --
    which is exactly the expense being measured: a kernel apply + regenerate +
    verify per candidate per step.
    """
    trial = copy.deepcopy(session)
    result = trial.apply_ops([op])
    if result.ok:
        return _OK
    codes = [d.code for d in result.diagnostics
             if d.severity is Severity.ERROR] or ["rejected"]
    return _no("trial-%s" % codes[0], "; ".join(
        d.message for d in result.diagnostics if d.severity is Severity.ERROR))


@dataclass
class MaskComparison:
    """Agreement + cost of preflight masking against trial-execution masking."""

    agree: int = 0
    disagree: int = 0
    over_restrictive: List[Tuple[str, str]] = field(default_factory=list)
    over_permissive: List[Tuple[str, str]] = field(default_factory=list)
    kernel_calls_preflight: int = 0
    kernel_calls_trial: int = 0
    seconds_preflight: float = 0.0
    seconds_trial: float = 0.0

    @property
    def total(self) -> int:
        return self.agree + self.disagree

    @property
    def agreement(self) -> float:
        return 1.0 if not self.total else self.agree / float(self.total)

    def to_dict(self) -> dict:
        return {
            "candidates": self.total,
            "agree": self.agree,
            "disagree": self.disagree,
            "agreement": self.agreement,
            "over_restrictive": list(self.over_restrictive),
            "over_permissive": list(self.over_permissive),
            "kernel_calls_preflight": self.kernel_calls_preflight,
            "kernel_calls_trial": self.kernel_calls_trial,
            "seconds_preflight": self.seconds_preflight,
            "seconds_trial": self.seconds_trial,
            "speedup": (self.seconds_trial / self.seconds_preflight
                        if self.seconds_preflight > 0 else float("inf")),
        }


def compare_masks(session, proposals: Optional[Mapping[str, Any]] = None, *,
                  names: Optional[Sequence[str]] = None,
                  numeric: bool = True,
                  rules: Optional[PrecheckRules] = None,
                  report: Optional[MaskComparison] = None) -> MaskComparison:
    """Preflight mask vs trial-execution mask on one state: agree? how costly?

    ``over_restrictive`` is the finding that matters: an action the preflight
    refuses but the kernel would have accepted is an action the agent can no
    longer take.
    """
    import time

    out = report if report is not None else MaskComparison()
    t0 = time.perf_counter()
    view, bound = _proposal_ops(session, proposals, names, rules)
    preflight = {name: op_verdict(view, op, numeric=numeric)
                 for name, op in bound.items()}
    out.seconds_preflight += time.perf_counter() - t0

    t0 = time.perf_counter()
    trial = {name: trial_verdict(session, op) for name, op in bound.items()}
    out.seconds_trial += time.perf_counter() - t0
    out.kernel_calls_trial += len(bound)

    for name, verdict in preflight.items():
        other = trial[name]
        if verdict.valid == other.valid:
            out.agree += 1
            continue
        out.disagree += 1
        detail = (name, verdict.reason or other.reason)
        if other.valid:
            out.over_restrictive.append(detail)
        else:
            out.over_permissive.append(detail)
    return out
