"""The GENERATION surface: named strategies that drive a HarnessSession to a solid.

``agents/generation`` carried four different ways to turn an intent into geometry --
CADSmith's dual correction loops, CADCodeVerify's question/answer refinement, an
error-driven prompt-evolution loop, and the worldcraft/assembly authoring stack --
plus the retrieval (``agents/rag``), memory (``agents/memory``) and context
(``agents/context``) machinery that is supposed to feed them. Nothing called any of
it. This module is that dispatcher.

    planners()                          -> the planners a strategy can be driven by
    generate(name, brief, ...)          -> GenerationResult (a built, verified model)
    retrieve(query) / api_context(q)    -> the retrieval layer, over THIS repo
    assemble_context(brief, ...)        -> the token-budgeted context an agent sees

NO LLM IS REQUIRED
------------------
Every strategy takes a ``planner`` with the same interface as
:class:`agents.agent.planner.Planner` (``plan(brief, state_summary, diagnostics) ->
[op dict]``). The default is :class:`StubPlanner`: a deterministic, stdlib-only
planner that reads the dimensions out of the brief and emits a fully-constrained op
stream, and that responds to diagnostics/feedback by repairing the offending
parameter. It is a STUB, not a model: it exists so the loops are runnable and
testable without ``[llm]``, and the real Planner drops straight into its place.

The "code" the correction loops iterate on is the CISP op stream (its canonical JSON
form). That is not a stand-in for code -- it is the program the harness actually
executes, so "execute" means apply_ops on a real session and "traceback" means the
backend's diagnostics.

RIVALS
------
``dual_loop`` (CADSmith), ``verify_loop`` (CADCodeVerify) and ``prompt_evolution``
are three different published answers to the same question -- how do you correct a
generated CAD program? -- and are exposed by name, never blended. Same for the two
retrieval backends (``hybrid`` rank fusion vs ``sphere_knn`` cosine k-NN).

Stdlib-only, absolute imports, deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry
from harnesscad.core.cisp.ops import Op, canonical_json, parse_op

__all__ = [
    "GenerationError",
    "UnknownStrategy",
    "RivalBlend",
    "Unsupported",
    "StubPlanner",
    "Brief",
    "GenerationResult",
    "Strategy",
    "strategies",
    "strategy",
    "rivals",
    "unadapted",
    "generate",
    "parse_brief",
    "design_plan_for",
    "metrics_of",
    "capability_corpus",
    "retrieve",
    "api_cards",
    "api_context",
    "assemble_context",
    "remember",
    "add_arguments",
    "run_cli",
    "main",
]

GENERATION_PACKAGE = "generation"
_PKG = "harnesscad.agents.generation."


class GenerationError(ValueError):
    """Base class for every generation-surface failure."""


class UnknownStrategy(GenerationError):
    """A strategy name outside the discovered table."""


class RivalBlend(GenerationError):
    """Rival strategies/backends were asked to be combined. They never are."""


class Unsupported(GenerationError):
    """This strategy genuinely cannot run on this input (no fallback)."""


# --------------------------------------------------------------------------- #
# The brief -> plan -> ops path (deterministic, no model)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Brief:
    """What a deterministic reading of a natural-language brief can honestly recover."""

    text: str
    width: float = 20.0
    height: float = 10.0
    depth: float = 5.0
    profile: str = "rectangle"          # rectangle | circle
    radius: float = 0.0
    hole_diameter: float = 0.0

    def to_dict(self) -> dict:
        return {"text": self.text, "width": self.width, "height": self.height,
                "depth": self.depth, "profile": self.profile,
                "radius": self.radius, "hole_diameter": self.hole_diameter}


_DIMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)"
                      r"(?:\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?))?", re.I)
_HOLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\s+(?:diameter\s+)?hole", re.I)
_RADIUS_RE = re.compile(r"(?:radius|r)\s*(?:of\s*)?(\d+(?:\.\d+)?)", re.I)


def parse_brief(text: str) -> Brief:
    """Read the dimensions a brief states. Nothing is invented: unstated numbers
    fall back to the documented defaults (20 x 10 x 5 mm)."""
    b = Brief(text=text)
    m = _DIMS_RE.search(text)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        d = float(m.group(3)) if m.group(3) else b.depth
        b = Brief(text, w, h, d, b.profile, b.radius, b.hole_diameter)
    lower = text.lower()
    if "cylinder" in lower or "disc" in lower or "round" in lower:
        rm = _RADIUS_RE.search(text)
        radius = float(rm.group(1)) if rm else max(b.width, b.height) / 2.0
        b = Brief(text, radius * 2, radius * 2, b.depth, "circle", radius,
                  b.hole_diameter)
    hm = _HOLE_RE.search(text)
    if hm:
        b = Brief(b.text, b.width, b.height, b.depth, b.profile, b.radius,
                  float(hm.group(1)))
    return b


def _ops_for(b: Brief) -> List[dict]:
    """A fully-constrained op stream for a parsed brief."""
    ops: List[dict] = [{"op": "new_sketch", "plane": "XY"}]
    if b.profile == "circle":
        ops.append({"op": "add_circle", "sketch": "sk1", "cx": 0.0, "cy": 0.0,
                    "r": float(b.radius)})
        ops.append({"op": "constrain", "kind": "radius", "a": "e1",
                    "value": float(b.radius)})
        ops.append({"op": "constrain", "kind": "coincident", "a": "e1"})
    else:
        ops.append({"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0,
                    "w": float(b.width), "h": float(b.height)})
        for value in (b.width, b.height, b.width, b.height):
            ops.append({"op": "constrain", "kind": "distance", "a": "e1",
                        "value": float(value)})
    ops.append({"op": "extrude", "sketch": "sk1", "distance": float(b.depth)})
    if b.hole_diameter > 0:
        ops.append({"op": "hole", "face_or_sketch": "f1", "x": 0.0, "y": 0.0,
                    "diameter": float(b.hole_diameter), "through": True,
                    "kind": "simple"})
    return ops


class StubPlanner:
    """A deterministic planner: brief (+ diagnostics) -> CISP ops. No model, no network.

    Interface-compatible with :class:`agents.agent.planner.Planner`, so every
    strategy in this module runs unchanged against the real LLM planner. On a replan
    it REPAIRS: a diagnostic naming a non-positive parameter is corrected to the
    briefed value, and a discrepancy against the design plan re-scales the offending
    dimension. That is a rule, not a guess -- and it is why the correction loops can
    be exercised end to end without ``[llm]``.
    """

    def __init__(self, brief: Optional[str] = None) -> None:
        self.brief = brief

    def plan(self, brief: str, state_summary: Optional[dict] = None,
             diagnostics: Optional[Sequence[Any]] = None) -> List[dict]:
        parsed = parse_brief(brief)
        ops = _ops_for(parsed)
        if diagnostics:
            ops = repair(ops, brief, diagnostics)
        return ops


def repair(ops: Sequence[dict], brief: str, diagnostics: Sequence[Any]) -> List[dict]:
    """Deterministically repair an op stream against diagnostics / plan discrepancies.

    Handles exactly two things, and says so: (1) a non-positive dimension is reset to
    the value the brief states; (2) a bbox discrepancy against the design plan
    re-scales the dimension that is out of tolerance. Anything else is left alone --
    a repair that does not understand the diagnostic must not touch the model.
    """
    parsed = parse_brief(brief)
    out = [dict(op) for op in ops]
    text = " ".join(str(getattr(d, "message", d)) for d in diagnostics).lower()
    for op in out:
        if op.get("op") == "add_rectangle":
            if float(op.get("w", 0)) <= 0 or "w and h must be > 0" in text:
                op["w"] = float(parsed.width)
            if float(op.get("h", 0)) <= 0:
                op["h"] = float(parsed.height)
        elif op.get("op") == "add_circle" and float(op.get("r", 0)) <= 0:
            op["r"] = float(parsed.radius or parsed.width / 2.0)
        elif op.get("op") == "extrude" and float(op.get("distance", 0)) == 0:
            op["distance"] = float(parsed.depth)
    for axis, want in (("x", parsed.width), ("y", parsed.height), ("z", parsed.depth)):
        m = re.search(r"bbox_%s.*?actual[= ]([0-9.]+)" % axis, text)
        if not m:
            continue
        for op in out:
            if axis in ("x", "y") and op.get("op") == "add_rectangle":
                op["w" if axis == "x" else "h"] = float(want)
            elif axis == "z" and op.get("op") == "extrude":
                op["distance"] = float(want)
    return out


def design_plan_for(brief: str):
    """The brief -> a CADSmith :class:`design_plan.DesignPlan` (the target contract)."""
    from harnesscad.agents.generation import design_plan as dp

    b = parse_brief(brief)
    components = [dp.Component(name="body", description=b.text.strip() or "body",
                               z_range=(0.0, b.depth))]
    constraints = dp.GeometricConstraints(
        hole_count=1 if b.hole_diameter > 0 else 0,
        hole_diameters_mm=(b.hole_diameter,) if b.hole_diameter > 0 else ())
    return dp.DesignPlan(components=tuple(components),
                         target_bbox_mm=(b.width, b.height, b.depth),
                         constraints=constraints, notes=b.text.strip())


# --------------------------------------------------------------------------- #
# Execute + measure (the loops' executor / judge)
# --------------------------------------------------------------------------- #
def _as_ops(code: Any) -> List[dict]:
    if isinstance(code, str):
        data = json.loads(code)
    else:
        data = list(code)
    return [op.to_dict() if isinstance(op, Op) else dict(op) for op in data]


def _as_code(ops: Sequence[Any]) -> str:
    return json.dumps(_as_ops(ops), sort_keys=True, indent=1)


def _session(backend: str = "stub"):
    from harnesscad.io.surfaces.server import CISPServer

    return CISPServer(backend=backend)


def _apply(code: Any, backend: str = "stub"):
    """Execute an op stream on a fresh session. Returns (server, result dict)."""
    server = _session(backend)
    result = server.applyOps(_as_ops(code))
    return server, result


def metrics_of(session_or_ops: Any):
    """CADSmith :class:`kernel_metrics.KernelMetrics` for a built model.

    Volume / bbox / centre come from the analytic shape proxy
    (:mod:`domain.editing.registry`); the topology counts are the counts OF THAT
    PROXY (each extruded profile is one box: 6 faces, 12 edges, 8 vertices). They
    are honest about what they measure -- they are not OCCT's B-rep counts, and the
    proxy says so.
    """
    from harnesscad.agents.generation import kernel_metrics as km
    from harnesscad.domain.editing import registry as editing

    ops = editing.ops_of(session_or_ops)
    shape = editing.shape_of(ops)
    boxes = editing._boxes(ops)
    centre = (0.0, 0.0, 0.0)
    if boxes:
        lo = [min(b[0][i] for b in boxes) for i in range(3)]
        hi = [max(b[1][i] for b in boxes) for i in range(3)]
        centre = tuple((lo[i] + hi[i]) / 2.0 for i in range(3))
    return km.KernelMetrics(
        volume=shape.volume, bbox_mm=shape.extents, center_of_mass=centre,
        face_count=6 * shape.solids, edge_count=12 * shape.solids,
        vertex_count=8 * shape.solids, is_valid=shape.solids > 0)


def _exec_result(code: Any, backend: str = "stub"):
    """The CADSmith ``Executor`` seam: run the program, gate it on kernel validity."""
    from harnesscad.agents.generation import dual_loop as dl
    from harnesscad.agents.generation import kernel_metrics as km

    try:
        server, result = _apply(code, backend)
    except Exception as exc:  # noqa: BLE001 - a broken program is data, not a crash
        return dl.ExecResult(False, "%s: %s" % (type(exc).__name__, exc)), None
    if not result["ok"]:
        trace = "; ".join("%s: %s" % (d["code"], d["message"])
                          for d in result.get("diagnostics") or [])
        return dl.ExecResult(False, trace or "the op stream was rejected"), server
    metrics = metrics_of(server.session)
    gate = km.hard_kernel_gate(metrics)
    if not gate.passed:
        return dl.ExecResult(False, "kernel gate: %s" % gate.reason, metrics), server
    return dl.ExecResult(True, "", metrics), server


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class GenerationResult:
    """The outcome of one generation run."""

    name: str
    ok: bool
    ops: Tuple[dict, ...] = ()
    digest: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 0
    diagnostics: Tuple[str, ...] = ()
    detail: Dict[str, Any] = field(default_factory=dict)
    session: Any = None

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "ops": [dict(o) for o in self.ops],
                "digest": self.digest, "summary": dict(self.summary),
                "iterations": self.iterations,
                "diagnostics": list(self.diagnostics),
                "detail": json.loads(json.dumps(self.detail, default=str))}


def _finish(name: str, code: Any, iterations: int, detail: Dict[str, Any],
            backend: str = "stub") -> GenerationResult:
    server, result = _apply(code, backend)
    diags = tuple("%s: %s" % (d["code"], d["message"])
                  for d in (result.get("diagnostics") or []))
    return GenerationResult(
        name=name, ok=bool(result["ok"]), ops=tuple(_as_ops(code)),
        digest=result["digest"], summary=server.query("summary")["result"],
        iterations=iterations, diagnostics=diags, detail=detail,
        session=server.session)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def _g_direct(brief: str, *, planner: Any = None, backend: str = "stub",
              **kw: Any) -> GenerationResult:
    """Plan once, apply once. The baseline every correction loop is measured against."""
    planner = planner or StubPlanner()
    ops = planner.plan(brief, None, None)
    return _finish("direct", ops, 1, {"planner": type(planner).__name__}, backend)


def _g_dual_loop(brief: str, *, planner: Any = None, backend: str = "stub",
                 code: Any = None, max_outer: int = 4, max_inner: int = 3,
                 **kw: Any) -> GenerationResult:
    """CADSmith: an INNER execution-repair loop nested inside an OUTER judge loop.

    Inner: execute; on failure, retrieve the matching error-solution pattern (KB2)
    and repair. Outer: compare the kernel metrics against the design plan and, while
    a dimension is out of tolerance, feed the discrepancy back -- with an escalation
    policy that forbids re-proposing a fingerprint that already failed and escalates
    the strategy when an issue persists.

    KB2's patterns are OCCT/CadQuery-shaped (fillet, boolean, open wire, ...). A CISP
    backend diagnostic may therefore match NO pattern, and the trace says so
    (``pattern: null``) instead of pretending a retrieval hit -- the repair then runs
    on the diagnostic alone.
    """
    from harnesscad.agents.generation import difficulty_tiers as dt
    from harnesscad.agents.generation import dual_loop as dl
    from harnesscad.agents.generation import error_patterns as ep
    from harnesscad.agents.generation import escalation as es
    from harnesscad.agents.generation import kernel_metrics as km

    planner = planner or StubPlanner()
    plan = design_plan_for(brief)
    kb = ep.default_kb()
    policy = es.EscalationPolicy()
    start = _as_code(code if code is not None else planner.plan(brief, None, None))
    trace: List[Dict[str, Any]] = []

    def executor(source: str):
        result, _server = _exec_result(source, backend)
        return result

    def error_refiner(source: str, traceback: str, attempt: int = 0) -> str:
        hits = kb.retrieve(traceback, top_k=1)
        trace.append({"stage": "inner", "attempt": int(attempt),
                      "traceback": traceback,
                      "pattern": hits[0].id if hits else None})
        return _as_code(repair(_as_ops(source), brief, [traceback]))

    def validator(source: str, exec_result):
        metrics = exec_result.metrics
        if metrics is None:
            return dl.ValidationResult(False, "no metrics", "no-metrics")
        comparison = km.compare_to_plan(metrics, plan)
        if comparison.all_within_tol:
            return dl.ValidationResult(True, "", "")
        feedback = km.discrepancy_feedback(comparison)
        codes = ",".join(d.field for d in comparison.out_of_tol)
        return dl.ValidationResult(False, feedback, codes)

    def refiner(source: str, validation, iteration: int = 0) -> str:
        policy.record(validation.issue_code, source)
        directive = policy.directive(int(iteration) or len(trace) + 1)
        trace.append({"stage": "outer", "iteration": int(iteration),
                      "issue": validation.issue_code,
                      "strategy": directive.strategy.value
                      if hasattr(directive.strategy, "value")
                      else str(directive.strategy),
                      "escalated": directive.escalated()})
        return _as_code(repair(_as_ops(source), brief, [validation.feedback]))

    result = dl.run_dual_loop(start, executor, error_refiner, validator, refiner,
                              max_outer=int(max_outer),
                              max_inner_retries=int(max_inner))
    tier = dt.classify([op["op"] for op in _as_ops(result.final_code)])
    out = _finish("dual_loop", result.final_code, result.outer_count,
                  {"passed": result.passed, "stop": str(result.stop),
                   "tier": tier.tier, "tier_reason": tier.reason,
                   "plan": plan.to_dict(), "trace": trace}, backend)
    return out


def _g_verify_loop(brief: str, *, planner: Any = None, backend: str = "stub",
                   code: Any = None, max_refinements: int = 3, **kw: Any
                   ) -> GenerationResult:
    """CADCodeVerify: generate verification QUESTIONS, answer them, refine on the No's.

    The paper answers the questions with a VLM looking at renders. There is no VLM
    and no renderer here, so the questions are answered by MEASURING the built model
    (the design plan's bbox vs the kernel metrics). That is a different -- and
    stricter -- oracle than a VLM, and it is stated rather than pretended.
    """
    from harnesscad.agents.generation import feedback_taxonomy as ft
    from harnesscad.agents.generation import kernel_metrics as km
    from harnesscad.agents.generation import verify_loop as vl

    planner = planner or StubPlanner()
    plan = design_plan_for(brief)
    start = _as_code(code if code is not None else planner.plan(brief, None, None))
    answers_seen: List[str] = []

    def compile_fn(source: str) -> Tuple[bool, str]:
        result, _ = _exec_result(source, backend)
        return bool(result.ok), result.traceback

    def fix_fn(source: str, error: str) -> str:
        return _as_code(repair(_as_ops(source), brief, [error]))

    repaired = vl.repair_until_compiles(start, compile_fn, fix_fn,
                                        max_iters=max(1, int(max_refinements)))
    current = repaired["code"]

    def question_fn(_description: str) -> List[str]:
        return ["Is the %s extent %.3f mm?" % (axis, want)
                for axis, want in zip("XYZ", plan.target_bbox_mm)]

    def answer_fn(_description: str, questions: Sequence[str]) -> List[str]:
        result, _ = _exec_result(current, backend)
        if result.metrics is None:
            return [vl.UNCLEAR for _ in questions]
        comparison = km.compare_to_plan(result.metrics, plan)
        bad = {d.field for d in comparison.out_of_tol}
        out = []
        for i, _q in enumerate(questions):
            field_name = "bbox_%s" % "xyz"[i]
            label = vl.NO if field_name in bad else vl.YES
            out.append(label)
            answers_seen.append(label)
        return out

    def feedback_fn(unresolved) -> str:
        return "; ".join("%s -> %s" % (q, a) for q, a in unresolved)

    def refine_fn(source: str, _description: str, feedback: str) -> str:
        nonlocal current
        current = _as_code(repair(_as_ops(source), brief, [feedback]))
        return current

    result = vl.run_cadcodeverify(current, brief, question_fn=question_fn,
                                  answer_fn=answer_fn, feedback_fn=feedback_fn,
                                  refine_fn=refine_fn,
                                  max_refinements=int(max_refinements))
    taxonomy = ft.feedback_distribution(
        [ft.classify_feedback("dimension %s" % a) for a in answers_seen]) \
        if answers_seen else {}
    return _finish("verify_loop", result["code"],
                   repaired["repair_attempts"] + result["rounds"],
                   {"rounds": result["rounds"],
                    "repair_attempts": repaired["repair_attempts"],
                    "answers": answers_seen, "feedback_taxonomy": taxonomy},
                   backend)


def _g_prompt_evolution(brief: str, *, planner: Any = None, backend: str = "stub",
                        max_retries: int = 4, **kw: Any) -> GenerationResult:
    """Error-driven PROMPT evolution: the failing terminal log becomes a constraint.

    The other two loops edit the *program*; this one edits the *prompt* and
    regenerates, accumulating one hard constraint per distinct error. Same goal, a
    genuinely different mechanism -- so it is a rival, not a variant.
    """
    from harnesscad.agents.generation import prompt_evolution as pe

    planner = planner or StubPlanner()

    def generate_fn(prompt: str) -> str:
        # The accumulated constraints are part of the prompt the planner sees.
        return _as_code(planner.plan(prompt, None, None))

    def execute_fn(script: str) -> Tuple[str, str]:
        result, server = _exec_result(script, backend)
        if result.ok:
            return (json.dumps(server.query("summary")["result"], sort_keys=True), "")
        return ("", result.traceback)

    result = pe.evolve(brief, generate_fn, execute_fn, max_retries=int(max_retries))
    return _finish("prompt_evolution", result.script, result.iterations,
                   {"converged": result.converged,
                    "constraints": list(result.constraints),
                    "steps": [(s.iteration, s.ok) for s in result.steps]},
                   backend)


def _g_tiled(brief: str, *, planner: Any = None, backend: str = "stub",
             exemplars: Sequence[Any] = (), k_per_tile: int = 1, **kw: Any
             ) -> GenerationResult:
    """Decompose the spec into TILES, plan each tile, compose the fragments.

    The spec is split into ordered tiles; per-tile in-context exemplars are chosen by
    the submodular DST selector; each tile is planned independently and its op stream
    is emitted as a CadQuery fragment; the fragments are merged into one program
    (imports de-duplicated, colliding ``result`` symbols renamed and unioned).
    """
    from harnesscad.agents.context import exemplar_prompt as ex
    from harnesscad.agents.generation import tile_compose as tc
    from harnesscad.domain.programs import registry as programs
    from harnesscad.domain.spec import spec_decompose as sd

    planner = planner or StubPlanner()
    tiles = sd.ordered_tiles(sd.decompose_spec(brief))
    if not tiles:
        raise Unsupported("the brief decomposes into no tiles")
    pool = [ex.Exemplar(spec=e[0], code=e[1]) if isinstance(e, tuple) else e
            for e in exemplars]
    selected = (ex.select_per_tile(brief, pool, int(k_per_tile), len(pool))
                if pool else [])

    ops: List[dict] = []
    fragments: List[Any] = []
    n_sketch = n_entity = n_feature = 0
    for i, tile in enumerate(tiles):
        tile_ops = planner.plan(tile.text, None, None)
        sketch = ""
        entity = ""
        feature = ""
        for op in tile_ops:
            op = dict(op)
            tag = op["op"]
            if tag == "new_sketch":
                n_sketch += 1
                sketch = "sk%d" % n_sketch
            elif "sketch" in op:
                op["sketch"] = sketch
            if tag.startswith("add_"):
                n_entity += 1
                entity = "e%d" % n_entity
            if tag == "constrain":
                op["a"] = entity
            if tag in ("extrude", "hole", "revolve", "loft", "sweep"):
                n_feature += 1
                feature = "f%d" % n_feature
            if tag == "hole":
                op["face_or_sketch"] = feature
            ops.append(op)
        neutral = _neutral_ops(tile_ops)
        if neutral:
            fragments.append(tc.TileFragment(
                tile_id=i, code=programs.emit(neutral, "cadquery")))
    composed = tc.compose_fragments(fragments) if fragments else ""
    return _finish("tiled", ops, len(tiles),
                   {"tiles": [t.text for t in tiles],
                    "exemplars_selected": [s for s in selected],
                    "code": composed}, backend)


def _neutral_ops(ops: Sequence[dict]) -> List[dict]:
    """CISP ops -> the neutral op IR that ``domain.programs`` emitters accept."""
    out: List[dict] = []
    profiles: Dict[str, str] = {}
    n = 0
    for op in ops:
        tag = op.get("op")
        if tag == "add_rectangle":
            n += 1
            name = "p%d" % n
            profiles[op["sketch"]] = name
            out.append({"operation": "rectangle", "result": name,
                        "args": {"center": [op["x"], op["y"], 0.0],
                                 "width": op["w"], "height": op["h"]}})
        elif tag == "add_circle":
            n += 1
            name = "p%d" % n
            profiles[op["sketch"]] = name
            out.append({"operation": "circle", "result": name,
                        "args": {"center": [op["cx"], op["cy"], 0.0],
                                 "radius": op["r"]}})
        elif tag == "extrude":
            profile = profiles.get(op["sketch"])
            if profile is None:
                continue
            n += 1
            out.append({"operation": "extrude", "result": "result",
                        "args": {"profile": profile, "height": op["distance"]}})
    return out


def _g_worldcraft(brief: str, *, planner: Any = None, backend: str = "stub",
                  objects: Sequence[dict] = (), room: Any = None, seed: int = 0,
                  customizations: Optional[Dict[str, Any]] = None, **kw: Any
                  ) -> GenerationResult:
    """Author a SCENE: solve the layout, instantiate the assets, place the instances.

    A spatial-constraint solver places the objects (non-overlap, on-top-of, within
    the room, ...); the layout is instantiated into world-space transforms; the
    customization schema validates each object's scale/material; and the resulting
    placements become real ``add_instance`` ops on the session, so the scene is a
    model the harness can verify and export, not a JSON blob.
    """
    from harnesscad.agents.generation import layout_solver as ls
    from harnesscad.agents.generation import scene_customization as sc
    from harnesscad.agents.generation import scene_instantiation as si
    from harnesscad.domain.reconstruction.scene import layout_spec as spec

    planner = planner or StubPlanner()
    if not objects:
        raise Unsupported("worldcraft needs objects=[{id, category, half_extent, "
                          "position}, ...]; it will not invent a scene")
    bounds = tuple(room) if room else ((-50.0, -50.0, 0.0), (50.0, 50.0, 30.0))
    layout = spec.LayoutSpec(room_bounds=bounds)
    for o in objects:
        layout.add(spec.ObjectPlacement(
            object_id=str(o["id"]), category=str(o.get("category", "part")),
            half_extent=tuple(float(v) for v in o["half_extent"]),
            pose=spec.Pose.at(*[float(v) for v in o.get("position", (0, 0, 0))])))
    constraints: List[Any] = [ls.WithinRoom(obj=str(o["id"])) for o in objects]
    ids = [str(o["id"]) for o in objects]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            constraints.append(ls.NonOverlap(a=ids[i], b=ids[j]))
    solved = ls.solve_layout(layout, constraints, seed=int(seed), iterations=400)

    schema = sc.CustomizationSchema()
    applied: Dict[str, Any] = {}
    for oid, edit in (customizations or {}).items():
        placement = solved.layout.get(oid)
        custom = edit if isinstance(edit, sc.ObjectCustomization) else \
            sc.ObjectCustomization(**edit)
        applied[oid] = sc.apply_customization(placement, custom,
                                              schema=schema).to_dict()

    instances = si.instantiate_layout(solved.layout)
    ops = planner.plan(brief, None, None)
    body = [o for o in ops if o["op"] == "extrude"]
    if not body:
        raise Unsupported("the planner produced no solid to instance")
    for inst in instances:
        cx, cy, cz = inst.world_center
        ops.append({"op": "add_instance", "part": "solid",
                    "x": float(cx), "y": float(cy), "z": float(cz),
                    "rx": 0.0, "ry": 0.0,
                    "rz": float(math.degrees(inst.transform.yaw))})
    return _finish("worldcraft", ops, 1,
                   {"initial_cost": solved.initial_cost,
                    "final_cost": solved.final_cost,
                    "satisfied": solved.satisfied(constraints),
                    "instances": [i.object_id for i in instances],
                    "bounds": [list(v) for v in si.instances_bounds(instances)],
                    "assets": len(instances),
                    "customized": applied}, backend)


def _g_building(brief: str, *, planner: Any = None, backend: str = "stub",
                plates: Sequence[dict] = (), **kw: Any) -> GenerationResult:
    """Assemble a PLATE STACK building: closed Catmull-Rom outlines, stacked in z.

    Each plate outline becomes a closed polyline sketch and an extrusion of that
    plate's thickness; the plate's z position becomes an ``add_instance`` placement,
    because CISP's extrude has no z offset -- the assembly primitive is the honest
    way to stack them, not a fabricated one.
    """
    from harnesscad.agents.generation import plate_stack_assembly as ps

    if not plates:
        raise Unsupported("the building strategy needs plates=[{...}, ...]; it will "
                          "not invent a building")
    building = ps.assemble_building(list(plates))
    ops: List[dict] = []
    for n, plate in enumerate(building.plates, start=1):
        sketch = "sk%d" % n
        ops.append({"op": "new_sketch", "plane": "XY"})
        ring = plate.bottom_ring
        for i in range(len(ring)):
            a, b = ring[i], ring[(i + 1) % len(ring)]
            ops.append({"op": "add_line", "sketch": sketch,
                        "x1": float(a[0]), "y1": float(a[1]),
                        "x2": float(b[0]), "y2": float(b[1])})
        thickness = plate.z_top - plate.z_bottom
        ops.append({"op": "extrude", "sketch": sketch,
                    "distance": float(thickness) or 1.0})
    solids = [o for o in ops if o["op"] == "extrude"]
    for n, plate in enumerate(building.plates, start=1):
        ops.append({"op": "add_instance", "part": "f%d" % n, "x": 0.0, "y": 0.0,
                    "z": float(plate.z_bottom)})
    return _finish("building", ops, len(building.plates),
                   {"height": building.height,
                    "bbox": [list(v) for v in building.bbox()],
                    "plates": [p.name for p in building.plates],
                    "solids": len(solids)}, backend)


def _g_freecad_macro(brief: str, *, planner: Any = None, backend: str = "stub",
                     macro: Any = None, **kw: Any) -> GenerationResult:
    """Query2CAD's FreeCAD Part-macro representation, lowered onto CISP ops.

    The macro (boxes / cylinders + booleans) is validated by its own module, its
    difficulty is estimated, and each primitive is lowered into the sketch+extrude
    ops the harness executes. A macro primitive CISP cannot express is refused, not
    approximated.
    """
    from harnesscad.agents.generation import freecad_macro as fm

    if macro is None:
        b = parse_brief(brief)
        macro = fm.FreeCADMacro(primitives=[fm.Primitive(
            name="Body", kind="box",
            params={"length": b.width, "width": b.height, "height": b.depth})])
    macro.validate()
    ops: List[dict] = []
    for n, prim in enumerate(macro.primitives, start=1):
        sketch = "sk%d" % n
        ops.append({"op": "new_sketch", "plane": "XY"})
        px, py, _pz = prim.position
        if prim.kind == "box":
            w = float(prim.params["length"])
            h = float(prim.params["width"])
            d = float(prim.params["height"])
            ops.append({"op": "add_rectangle", "sketch": sketch, "x": float(px),
                        "y": float(py), "w": w, "h": h})
        elif prim.kind == "cylinder":
            r = float(prim.params["radius"])
            d = float(prim.params["height"])
            ops.append({"op": "add_circle", "sketch": sketch, "cx": float(px),
                        "cy": float(py), "r": r})
        else:
            raise Unsupported(
                "CISP has no op for the FreeCAD primitive %r (box and cylinder are "
                "lowered; the rest are refused rather than approximated)" % prim.kind)
        ops.append({"op": "extrude", "sketch": sketch, "distance": d})
    for boolean in macro.booleans:
        kind = {"cut": "cut", "fuse": "union", "common": "intersect"}.get(
            boolean.op, boolean.op)
        ops.append({"op": "boolean", "kind": kind, "target": boolean.left,
                    "tool": boolean.right})
    return _finish("freecad_macro", ops, 1,
                   {"difficulty": fm.estimate_difficulty(macro),
                    "operations": macro.operation_summary(),
                    "source": macro.to_source()}, backend)


@dataclass(frozen=True)
class Strategy:
    """One named generation strategy."""

    name: str
    description: str
    modules: Tuple[str, ...]
    run: Callable[..., GenerationResult]
    family: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "family": self.family,
                "description": self.description, "modules": list(self.modules)}


_TABLE: Tuple[Tuple[str, str, Tuple[str, ...], Callable, str], ...] = (
    ("direct",
     "Plan once, apply once. No correction; the baseline the loops are measured "
     "against.",
     (), _g_direct, ""),
    ("dual_loop",
     "CADSmith: an inner execution-repair loop (error-solution knowledge base) "
     "nested in an outer judge loop (kernel metrics vs the design plan), with an "
     "anti-oscillation escalation policy.",
     ("dual_loop", "error_patterns", "escalation", "kernel_metrics", "design_plan",
      "difficulty_tiers"), _g_dual_loop, "correction"),
    ("verify_loop",
     "CADCodeVerify: generate verification questions from the spec, answer them "
     "against the built model, refine on the unresolved ones.",
     ("verify_loop", "feedback_taxonomy", "kernel_metrics", "design_plan"),
     _g_verify_loop, "correction"),
    ("prompt_evolution",
     "Error-driven prompt evolution: each distinct failure becomes an accumulated "
     "constraint on the prompt, and the program is regenerated -- the PROMPT is "
     "edited, not the program.",
     ("prompt_evolution",), _g_prompt_evolution, "correction"),
    ("tiled",
     "Spec-tiling: decompose the brief into tiles, select per-tile in-context "
     "exemplars, plan each tile, and compose the emitted fragments into one program.",
     ("tile_compose",), _g_tiled, ""),
    ("worldcraft",
     "Scene authoring: solve a spatial-constraint layout, instantiate the assets, "
     "validate the customizations, and place the instances as real assembly ops.",
     ("layout_solver", "scene_instantiation", "scene_customization"),
     _g_worldcraft, ""),
    ("building",
     "Plate-stack building assembly: closed Catmull-Rom plate outlines stacked in z "
     "and lowered onto polyline sketches + extrusions + placements.",
     ("plate_stack_assembly",), _g_building, ""),
    ("freecad_macro",
     "Query2CAD's FreeCAD Part-macro representation (primitives + booleans), "
     "validated, difficulty-scored, and lowered onto CISP ops.",
     ("freecad_macro",), _g_freecad_macro, ""),
)

#: Three published answers to "how do you correct a generated CAD program?".
#: Different mechanisms, different budgets. Selected by name; never averaged.
_RIVALS: Dict[str, Tuple[str, ...]] = {
    "correction": ("dual_loop", "verify_loop", "prompt_evolution"),
    "retrieval": ("hybrid", "sphere_knn"),
}


# --------------------------------------------------------------------------- #
# RAG / MEMORY / CONTEXT -- the layer that feeds a planner
# --------------------------------------------------------------------------- #
def capability_corpus(limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """The retrieval corpus: THIS repo's capability index, as (source, text) docs.

    Retrieval over the capability registry is the honest corpus here -- ~1,100 real
    modules with real summaries and real public symbols. There is no external
    document store to pretend about.
    """
    docs: List[Tuple[str, str]] = []
    for e in capability_registry.index()[:limit]:
        text = "# %s\n\n%s\n\nsymbols: %s\ntags: %s\n" % (
            e.dotted, e.summary or e.name, ", ".join(e.symbols), ", ".join(e.tags))
        docs.append((e.dotted, text))
    return docs


_RETRIEVER: Any = None

#: The hashing dimension the KD-tree retriever runs at (dense rows, tractable tree).
_KNN_DIM = 128


def _dense(sparse: Dict[int, float]) -> List[float]:
    return [float(sparse.get(i, 0.0)) for i in range(_KNN_DIM)]


def retrieve(query: str, k: int = 5, *, backend: str = "hybrid",
             limit: Optional[int] = 400, notebook: Any = None) -> List[Tuple[str, float]]:
    """Retrieve capability modules for a query. TWO BACKENDS, NEVER BLENDED.

    ``hybrid``     BM25 + a hashed dense index, fused by reciprocal rank.
    ``sphere_knn`` cosine k-NN via a spherical projection + KD-tree over the same
                   hashed embeddings -- a pure dense retriever with no lexical leg.

    They disagree by construction (one has a lexical channel, the other does not).
    Pass ``notebook=`` an :class:`memory.error_notebook.ErrorNotebook` to re-rank the
    hybrid candidates AWAY from specifications that historically failed.
    """
    global _RETRIEVER
    if backend not in ("hybrid", "sphere_knn"):
        raise RivalBlend("unknown retrieval backend %r; the two are 'hybrid' and "
                         "'sphere_knn' and they are never merged" % backend)
    docs = capability_corpus(limit)
    if backend == "sphere_knn":
        from harnesscad.agents.rag import index as ix
        from harnesscad.agents.rag import sphere_knn as sk

        # The hashed embedder is sparse; a KD-tree needs dense, equal-length rows,
        # so the SAME embedder is used at a KD-tree-sized dimension and densified.
        embedder = ix.HashedEmbedder(dim=_KNN_DIM)
        dense = [_dense(embedder.embed(text)) for _src, text in docs]
        tree = sk.SphereKDTree.from_vectors(dense)
        hits = tree.query(_dense(embedder.embed(query)), k=int(k))
        return [(docs[i][0], float(score)) for i, score in hits]

    from harnesscad.agents.rag import retriever as rt

    _RETRIEVER = rt.build_from_docs([(text, src) for src, text in docs])
    hits = _RETRIEVER.retrieve(query, k=int(k))
    ranked = [(h.source, float(h.score)) for h in hits]
    if notebook is not None:
        from harnesscad.agents.rag import rerank as rr

        reranked = rr.rerank_parts(query, ranked, notebook)
        return [(c.answer[0], c.final_score) for c in reranked]
    return ranked


def api_cards() -> Tuple[Any, ...]:
    """The CISP op vocabulary as parsed API cards (the planner's tool reference)."""
    from harnesscad.agents.agent import system_prompt as sp
    from harnesscad.agents.generation import api_reference as ar

    lines = []
    for line in sp.op_vocabulary().splitlines():
        m = re.match(r'-\s*"([^"]+)":\s*\{(.*)\}\s*$', line.strip())
        if not m:
            continue
        tag, fields = m.group(1), m.group(2)
        names = [f.split(":")[0].strip() for f in fields.split(",") if f.strip()]
        lines.append("cisp.%s(%s)- the CISP %s operation"
                     % (tag, ", ".join(names), tag.replace("_", " ")))
    return ar.parse_reference("\n".join(lines))


def api_context(query: str, top_k: int = 5) -> str:
    """A bounded, deterministically ordered API context block for a planner prompt."""
    from harnesscad.agents.generation import api_reference as ar
    from harnesscad.agents.rag import api_knowledge as ak

    cards = api_cards()
    apis = [ak.API(name=c.qualname, signature="(%s)" % ", ".join(c.required),
                   returns="ApplyOpsResult", example=(c.description,))
            for c in cards]
    ak.validate(apis)                     # the knowledge base must be well-formed
    return ar.build_prompt_context(query, cards, top_k=int(top_k),
                                   header="Relevant CISP ops:")


def remember(store: Any, brief: str, result: GenerationResult, *, day: float = 0.0,
             saliences: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """Record a run in episodic memory and age the store (reinforced decay).

    The episode is the (brief, ops, outcome, digest) tuple the harness can replay;
    the salience of the recalled node is reinforced and the store swept, so a memory
    that is never recalled decays out instead of growing without bound.
    """
    from harnesscad.agents.memory import decay as dc

    store.add_episodic(brief, list(result.ops),
                       outcome="ok" if result.ok else "failed",
                       digest=result.digest, summary=result.name)
    sal = list(saliences or [dc.Salience(node_id=result.digest[:12])])
    sal = [dc.reinforce(s, day) if s.node_id == result.digest[:12] else s
           for s in sal]
    sweep = dc.decay_sweep(sal, day, keep_min=1)
    return {"episodes": len(store.recall_episodic(brief, k=5)),
            "retained": [n for n, _ in sweep.retained],
            "forgotten": [n for n, _ in sweep.forgotten],
            "saliences": sal}


def assemble_context(brief: str, *, budget: int = 4000, session: Any = None,
                     store: Any = None, k: int = 4, stage_dir: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Assemble the token-budgeted context an agent turn actually sees.

    Retrieval over the capability index gives the candidate memory nodes; the
    progressive-tier planner decides how deeply each one can be read inside the
    budget; the meta-tag renderer produces the terse memory map; the context manager
    counts every message against the window and evicts history when it must; and,
    when a ``stage_dir`` is given, the staging area writes the per-task context to
    disk so it is inspectable rather than implicit.
    """
    from harnesscad.agents.agent import system_prompt as sp
    from harnesscad.agents.context import manager as cm
    from harnesscad.agents.context import meta_tags as mt
    from harnesscad.agents.context import progressive_tiers as pt
    from harnesscad.agents.llm.base import Message

    hits = retrieve(brief, k=int(k))
    profiles = [pt.make_profile(dotted, summary=dotted.rsplit(".", 1)[-1],
                                detailed_summary=dotted, priority=int(1000 * score))
                for dotted, score in hits]
    plan = pt.plan_reads(profiles, budget // 4)
    rows = [mt.Row(node_id=r.node_id, summary=r.node_id.rsplit(".", 1)[-1],
                   tags=("retrieval", "active")) for r in plan.reads]
    rendered = mt.render_rows(rows) if rows else ""
    memory_block = ("\n".join(rendered) if isinstance(rendered, (list, tuple))
                    else str(rendered))

    manager = cm.ContextManager(budget=int(budget))
    tree = manager.feature_tree_summary(session.opdag) if session is not None else ""
    history = [Message(role="assistant", content=tree)] if tree else []
    assembled = manager.assemble(
        system=sp.build_system_prompt(), first_user=brief,
        history=history, memory=memory_block or None)

    staged = None
    if stage_dir:
        from harnesscad.agents.context import staging as st

        area = st.StagingArea(stage_dir)
        area.build(brief=brief, model_tree=tree)
        lines = ["- %s (%.4f)" % (d, s) for d, s in hits]
        area.write("03_DOCS/retrieval.md", "\n".join(lines))
        manifest = area.read_manifest()
        sel = manifest.get("manifest", manifest)
        sel["docs"] = list(sel.get("docs") or []) + ["03_DOCS/retrieval.md"]
        area.write_manifest(manifest)
        staged = area.render_for_turn()

    recalled = (store.recall_episodic(brief, k=int(k)) if store is not None else [])
    return {"messages": [(m.role, m.content) for m in assembled.messages],
            "report": assembled.report.to_dict(),
            "evicted": assembled.evicted, "read_plan": plan.to_dict(),
            "memory_map": memory_block, "retrieved": [d for d, _ in hits],
            "episodes": [e.brief for e in recalled], "staged": staged}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
_STRATS: Optional[Dict[str, Strategy]] = None
_UNADAPTED: Tuple[str, ...] = ()

#: Generation modules an adapter reaches only through another adapted module.
_INDIRECT: Tuple[str, ...] = ("shape_metrics", "api_reference", "code_tree_control")


def _build() -> Dict[str, Strategy]:
    global _UNADAPTED
    entries = {e.dotted for e in capability_registry.find(package=GENERATION_PACKAGE)}
    adapted = set(_PKG + m for m in _INDIRECT)
    out: Dict[str, Strategy] = {}
    for name, description, mods, fn, family in _TABLE:
        dotted = tuple(_PKG + m for m in mods)
        if any(d not in entries for d in dotted):
            continue
        adapted.update(dotted)
        out[name] = Strategy(name, description, dotted, fn, family)
    _UNADAPTED = tuple(sorted(d for d in entries
                              if d not in adapted and not d.endswith(".registry")))
    return out


def _all() -> Dict[str, Strategy]:
    global _STRATS
    if _STRATS is None:
        _STRATS = _build()
    return _STRATS


def strategies(family: Optional[str] = None) -> Tuple[str, ...]:
    return tuple(sorted(n for n, s in _all().items()
                        if family is None or s.family == family))


def strategy(name: str) -> Strategy:
    try:
        return _all()[name]
    except KeyError:
        raise UnknownStrategy("unknown generation strategy %r (one of: %s)"
                              % (name, ", ".join(strategies()))) from None


def rivals() -> Dict[str, Tuple[str, ...]]:
    return {k: tuple(v) for k, v in sorted(_RIVALS.items())}


def unadapted() -> Tuple[str, ...]:
    _all()
    return _UNADAPTED


def planners() -> Tuple[str, ...]:
    """The planners a strategy can be driven by. ``stub`` needs no model."""
    return ("stub", "llm")


def generate(name: str, brief: str, **kwargs: Any) -> GenerationResult:
    """Run a named generation strategy. A failing component is captured, not fatal."""
    strat = strategy(name)
    try:
        return strat.run(brief, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a failing strategy is data
        return GenerationResult(name=name, ok=False,
                                diagnostics=("%s: %s" % (type(exc).__name__, exc),))


# --------------------------------------------------------------------------- #
# CLI (wired into core.cli as `harnesscad generate`)
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("brief", nargs="?", default=None,
                        help="the natural-language design brief")
    parser.add_argument("--list", action="store_true",
                        help="list the generation strategies")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival families (never blended)")
    parser.add_argument("--unadapted", action="store_true",
                        help="list generation modules with no call site yet")
    parser.add_argument("--strategy", default="direct",
                        help="the generation strategy to run (default: direct)")
    parser.add_argument("--planner", default="stub", choices=["stub", "llm"],
                        help="stub = deterministic, no model (default)")
    parser.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
    parser.add_argument("--retrieve", default=None, metavar="QUERY",
                        help="retrieve capability modules for a query")
    parser.add_argument("--retrieval-backend", default="hybrid",
                        choices=["hybrid", "sphere_knn"], dest="retrieval_backend")
    parser.add_argument("--context", action="store_true",
                        help="print the assembled, token-budgeted context")
    parser.add_argument("--json", action="store_true")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "rivals", False):
        for family, names in rivals().items():
            print("%s: %s" % (family, ", ".join(names)))
            print("    different mechanisms for the same job; NEVER averaged")
        return 0
    if getattr(args, "unadapted", False):
        for dotted in unadapted():
            print(dotted)
        print("-- %d generation modules without a call site" % len(unadapted()))
        return 0
    if getattr(args, "retrieve", None):
        for dotted, score in retrieve(args.retrieve, backend=args.retrieval_backend):
            print("%8.4f  %s" % (score, dotted))
        return 0

    if not getattr(args, "brief", None):
        for name in strategies():
            s = strategy(name)
            tag = (" (rival family: %s)" % s.family) if s.family else ""
            print("%-18s%s" % (name, tag))
            print("    %s" % s.description)
        print()
        print("-- %d strategies / %d generation modules unbound"
              % (len(strategies()), len(unadapted())))
        return 0

    planner = None
    if args.planner == "llm":
        from harnesscad.agents.agent.planner import Planner
        from harnesscad.agents.llm.client import default_client

        planner = Planner(default_client())
    if getattr(args, "context", False):
        ctx = assemble_context(args.brief)
        print(json.dumps({k: v for k, v in ctx.items() if k != "messages"},
                         sort_keys=True, indent=2, default=str))
        return 0

    result = generate(args.strategy, args.brief, planner=planner,
                      backend=args.backend)
    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
        return 0 if result.ok else 1
    print("strategy:   %s" % result.name)
    print("ok:         %s" % result.ok)
    print("iterations: %d" % result.iterations)
    print("digest:     %s" % result.digest)
    print("summary:    %s" % json.dumps(result.summary, sort_keys=True))
    for d in result.diagnostics:
        print("  %s" % d)
    return 0 if result.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad generate",
        description="generation strategies that drive a session to a solid "
                    "(deterministic stub planner by default -- no LLM required)")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
