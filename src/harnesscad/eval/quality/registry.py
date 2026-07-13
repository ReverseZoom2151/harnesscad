"""Quality analyser registry -- the analysis surface over ``eval/quality``.

The ``eval/quality`` tree carries ~60 modules (physics, sequence, geometry,
graph, assembly, edit, sketch, perception, reward, report). Every one of them was
correct, tested and unreachable: nothing imported them.

They are NOT verifiers. A verifier *gates* -- it returns a Diagnostic and the
loop blocks or corrects (:mod:`harnesscad.eval.verifiers.registry`). A quality
module *analyses*: it measures mass, ranks complexity, exposes parameters, scores
an anomaly, builds an intent graph. The answer is a number or a structure, not a
verdict. This module makes that fleet dispatchable behind one protocol and gives
the harness a quality REPORT (``harnesscad report``).

Design (the same three rules as the verifier / bench / format registries)
------------------------------------------------------------------------
*   **Discovery, not a hardcoded list.** :func:`unadapted` derives its answer from
    the static AST index (:mod:`harnesscad.registry`, ``package="quality"``), so a
    quality module with no honest call site stays *visible* as an orphan instead
    of being papered over.
*   **Adapters only.** Each :class:`Analyser` adapts a module's real public API.
    The quality modules are never modified. An analyser declares what it needs via
    ``applies_to(state)`` and SKIPS when the state cannot answer -- it never
    fabricates the input (a fabricated input is a fabricated number).
*   **Nothing crashes the report.** An analyser that raises becomes an ``error``
    entry in the report; the rest of the fleet still runs.
*   **Rivals stay selectable and are NEVER averaged.** Three anomaly scorers
    (z-score, IQR, isolation-forest) and two reward functions (gated composite vs
    execution/CD reward) answer the same question with incompatible protocols.
    :data:`RIVAL_FAMILIES` records them; :func:`analyse` refuses to fold two
    members of one family into a single score, and the report stamps every number
    with the analyser and the module that produced it.

Typical use::

    from harnesscad.eval.quality import registry as quality

    state = quality.model_state(session.backend, session.opdag)
    report = quality.report(state)
    report.to_dict()["analyses"]["sequence.complexity_taxonomy"]

Stdlib-only, absolute imports, deterministic (analysers sorted by (kind, name),
no wall clock, no unseeded randomness).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from harnesscad import registry as capability_registry
from harnesscad.eval.verifiers.registry import ModelState

__all__ = [
    "KINDS",
    "ModelState",
    "model_state",
    "Analyser",
    "FunctionAnalyser",
    "AnalysisResult",
    "QualityReport",
    "RIVAL_FAMILIES",
    "RivalBlendError",
    "analysers",
    "analyser",
    "kinds",
    "rivals",
    "unadapted",
    "analyse",
    "report",
    "add_arguments",
    "run",
    "main",
]

QUALITY_PACKAGE = "quality"

#: Analyser families (the sub-packages of ``eval/quality``).
KINDS: Tuple[str, ...] = (
    "geometry", "sequence", "physics", "graph", "assembly", "edit", "sketch",
    "perception", "reward", "report",
)
_KIND_ORDER = {k: i for i, k in enumerate(KINDS)}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def model_state(backend: Any, opdag: Any,
                extras: Optional[Dict[str, dict]] = None) -> ModelState:
    """The verifier fleet's :class:`ModelState`, optionally pre-seeded.

    ``extras`` injects projections the backend cannot answer itself but the caller
    knows (``{"brief": {"text": ...}}``, ``{"reward": {...}}``, an anomaly
    reference corpus, ...). They are seeded into the same memoised query cache the
    backend answers into, so an analyser reads them exactly like a backend
    projection -- and an analyser whose input is absent still SKIPS rather than
    guessing.
    """
    state = ModelState(backend, opdag)
    for key, value in (extras or {}).items():
        if isinstance(value, dict):
            state._q[key] = value  # noqa: SLF001 - the documented seam of ModelState
    return state


# ---------------------------------------------------------------------------
# Protocol + adapter
# ---------------------------------------------------------------------------

class Analyser(Protocol):
    """What the dispatcher needs from anything it analyses with."""

    name: str
    kind: str
    dotted: str

    def applies_to(self, state: ModelState) -> bool: ...

    def analyse(self, state: ModelState) -> dict: ...


@dataclass(frozen=True)
class FunctionAnalyser:
    """Adapts a function-style quality module into the Analyser protocol."""

    name: str
    kind: str
    dotted: str
    _applies: Callable[[ModelState], bool]
    _run: Callable[[ModelState], dict]
    summary: str = ""

    def applies_to(self, state: ModelState) -> bool:
        return bool(self._applies(state))

    def analyse(self, state: ModelState) -> dict:
        out = self._run(state)
        return dict(out) if isinstance(out, dict) else {"value": out}

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "dotted": self.dotted,
                "summary": self.summary}


@dataclass(frozen=True)
class AnalysisResult:
    """One analyser's outcome. ``status`` is ok | skipped | error."""

    name: str
    kind: str
    dotted: str
    status: str
    value: dict = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "dotted": self.dotted,
                "status": self.status, "value": self.value, "error": self.error}


@dataclass
class QualityReport:
    """The analysis surface's answer for one model state."""

    results: List[AnalysisResult] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    prose: str = ""
    claims: Dict[str, Any] = field(default_factory=dict)

    def ok(self) -> List[AnalysisResult]:
        return [r for r in self.results if r.status == "ok"]

    def errors(self) -> List[AnalysisResult]:
        return [r for r in self.results if r.status == "error"]

    def skipped(self) -> List[AnalysisResult]:
        return [r for r in self.results if r.status == "skipped"]

    def value(self, name: str) -> Optional[dict]:
        for r in self.results:
            if r.name == name and r.status == "ok":
                return r.value
        return None

    def to_dict(self) -> dict:
        return {
            "analyses": {r.name: r.to_dict() for r in self.results},
            "counts": {"ok": len(self.ok()), "skipped": len(self.skipped()),
                       "error": len(self.errors())},
            "scores": dict(sorted(self.scores.items())),
            "prose": self.prose,
            "claims": self.claims,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, default=str)


# ---------------------------------------------------------------------------
# Shared, kernel-free projections of the op stream
# ---------------------------------------------------------------------------

def _op_dicts(state: ModelState) -> List[dict]:
    out = []
    for op in state.ops():
        try:
            out.append(op.to_dict())
        except Exception:  # noqa: BLE001 - a hostile op must not break the fleet
            continue
    return out


def _op_names(state: ModelState) -> List[str]:
    return [str(op.get("op", "")) for op in _op_dicts(state)]


def _corners(bb: Tuple[float, ...]) -> List[Tuple[float, float, float]]:
    x0, y0, z0, x1, y1, z1 = bb
    return [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


# ---------------------------------------------------------------------------
# Adapters (module imported inside the closure: an optional-dep failure becomes
# a caught `error` result, never an import-time explosion).
# ---------------------------------------------------------------------------

# --- sequence ---------------------------------------------------------------

def _a_sequence_complexity() -> FunctionAnalyser:
    """`sequence.complexity` -- the design's complexity level (1-5) from its structure."""

    def applies(state: ModelState) -> bool:
        return bool(_op_dicts(state))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.sequence.complexity import classify

        ops = _op_dicts(state)
        names = [o.get("op", "") for o in ops]
        sketches = names.count("new_sketch")
        curves = sum(1 for n in names if n.startswith("add_"))
        features = sum(1 for n in names
                       if n in ("extrude", "revolve", "fillet", "chamfer", "shell",
                                "hole", "loft", "sweep", "draft", "boolean"))
        return classify(components=max(sketches, 1), loops=sketches, curves=curves,
                        type_diversity=len(set(names)), feature_depth=features,
                        repeated=len(names) - len(set(names)))

    return FunctionAnalyser("sequence.complexity_taxonomy", "sequence",
                            "harnesscad.eval.quality.sequence.complexity", applies, run)


def _a_sequence_quantization_risk() -> FunctionAnalyser:
    """`sequence.quantization_risk` -- which dimensions the quantiser would destroy."""

    def applies(state: ModelState) -> bool:
        return bool(_op_dicts(state))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.sequence.quantization_risk import quantization_risks

        ops = _op_dicts(state)
        step = float(state.query("quantiser").get("step", 1.0) or 1.0)
        extrusions = [float(o["distance"]) for o in ops
                      if o.get("op") == "extrude" and "distance" in o]
        radii = [float(o[k]) for o in ops for k in ("r", "radius")
                 if k in o and isinstance(o[k], (int, float))]
        risks = quantization_risks(step=step,
                                   extrusion=extrusions[0] if extrusions else None,
                                   radii=tuple(radii))
        return {"step": step, "risks": risks}

    return FunctionAnalyser("sequence.quantization_risk", "sequence",
                            "harnesscad.eval.quality.sequence.quantization_risk",
                            applies, run)


def _a_sequence_macro_collapse() -> FunctionAnalyser:
    """`sequence.macro_collapse` -- repeated op runs folded into macros."""

    def applies(state: ModelState) -> bool:
        return bool(_op_names(state))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.sequence.macro_collapse import Component, collapse, stats

        components = [Component(ops=(name,)) for name in _op_names(state)]
        collapsed = collapse(components,
                             int(state.query("macro").get("threshold", 4) or 4))
        return stats(components, collapsed)

    return FunctionAnalyser("sequence.macro_collapse", "sequence",
                            "harnesscad.eval.quality.sequence.macro_collapse", applies, run)


def _a_sequence_primitive_pooling() -> FunctionAnalyser:
    """`sequence.primitive_pooling` -- the primitive spans in the op-token stream."""

    def applies(state: ModelState) -> bool:
        return bool(_op_names(state))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.sequence.primitive_pooling import spans

        found, meta = spans(_op_names(state))
        return {"n_spans": len(found),
                "spans": [{"kind": s.kind, "start": s.start, "end": s.end}
                          for s in found],
                "exact_coverage": meta["exact_coverage"],
                "overflow": meta["overflow"]}

    return FunctionAnalyser("sequence.primitive_pooling", "sequence",
                            "harnesscad.eval.quality.sequence.primitive_pooling",
                            applies, run)


def _a_sequence_code_normalize() -> FunctionAnalyser:
    """`sequence.code_normalize` -- canonical form of the model's CAD script.

    Needs a script (``backend.query('code')['source']`` or the caller's extras):
    the harness's op stream is not source code, so this SKIPS when there is none.
    """

    def applies(state: ModelState) -> bool:
        return bool(state.query("code").get("source"))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.sequence.code_normalize import normalize_cad_code

        source = str(state.query("code")["source"])
        normalized = normalize_cad_code(source)
        return {"lines_in": len(source.splitlines()),
                "lines_out": len(normalized.splitlines()),
                "normalized": normalized}

    return FunctionAnalyser("sequence.code_normalize", "sequence",
                            "harnesscad.eval.quality.sequence.code_normalize", applies, run)


def _a_sequence_confidence() -> FunctionAnalyser:
    """`sequence.confidence` -- per-command confidence + the correction context.

    Confidences come from a generator, not from geometry, so this SKIPS unless the
    caller supplies them (``extras={"confidence": {"commands": [...]}}``).
    """

    def applies(state: ModelState) -> bool:
        return bool(state.query("confidence").get("commands"))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.sequence.confidence import (
            CommandConfidence, assess_sequence_confidence,
        )

        cmds = [
            CommandConfidence(index=int(c.get("index", i)), command=str(c.get("command", "")),
                              type_confidence=float(c.get("type_confidence", 0.0)),
                              arguments=dict(c.get("arguments") or {}))
            for i, c in enumerate(state.query("confidence")["commands"])
        ]
        res = assess_sequence_confidence(cmds)
        return {"minimum": res.minimum, "mean": res.mean,
                "low_confidence": list(res.low_confidence),
                "correction_context": res.correction_context}

    return FunctionAnalyser("sequence.confidence", "sequence",
                            "harnesscad.eval.quality.sequence.confidence", applies, run)


# --- geometry ---------------------------------------------------------------

def _a_geometry_canonical_pose() -> FunctionAnalyser:
    """`geometry.canonical_pose` -- centred? axes ordered? seated on its reference face?

    Runs on the op-stream envelope's corners (the stub backend has no B-rep), which
    is exactly the resolution a pose check needs.
    """

    def applies(state: ModelState) -> bool:
        return state.envelope() is not None

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.geometry.canonical_pose import pose_report

        bb = state.envelope()
        return pose_report(_corners(bb)).to_dict()

    return FunctionAnalyser("geometry.canonical_pose", "geometry",
                            "harnesscad.eval.quality.geometry.canonical_pose", applies, run)


def _a_geometry_invariance() -> FunctionAnalyser:
    """`geometry.invariance` -- is the harness's own extent measurement translation-invariant?

    A real contract, evaluated on the model's own envelope: translate the corner
    cloud, re-measure its extents, and require them unchanged. It is the
    self-consistency check every downstream geometric number depends on.
    """

    def applies(state: ModelState) -> bool:
        return state.envelope() is not None

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.geometry.invariance import (
            ContractMetadata, InvarianceContract, PerturbationCase, translate_points,
        )

        pts = [(p[0], p[1]) for p in _corners(state.envelope())]

        def measure(points) -> float:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (max(xs) - min(xs)) + (max(ys) - min(ys))

        contract = InvarianceContract(
            ContractMetadata(name="envelope-extent", transformation="translation",
                             relation="invariant", observable="x+y extent",
                             description="the model's footprint extent must not move "
                                         "when the model does"),
            transform=lambda subject, offset: translate_points(subject, offset),
            measure=measure,
        )
        report_ = contract.evaluate(
            pts,
            [PerturbationCase(name="translate+10", parameter=(10.0, 10.0)),
             PerturbationCase(name="translate-3.5", parameter=(-3.5, 2.25))],
        )
        d = report_.to_dict()
        d["passed"] = all(r.passed for r in report_.results)
        return d

    return FunctionAnalyser("geometry.invariance", "geometry",
                            "harnesscad.eval.quality.geometry.invariance", applies, run)


def _a_geometry_mesh_stability() -> FunctionAnalyser:
    """`geometry.mesh_stability` -- normal consistency + bottom-face roughness.

    Needs a tessellation (``backend.query('mesh')``); SKIPS on a kernel-free backend.
    """

    def applies(state: ModelState) -> bool:
        mesh = state.query("mesh")
        return bool(mesh.get("vertices")) and bool(mesh.get("faces"))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.geometry.mesh_stability import mesh_stability_metrics

        mesh = state.query("mesh")
        m = mesh_stability_metrics(mesh["vertices"], mesh["faces"])
        return {"normal_inconsistency": m.normal_inconsistency,
                "adjacent_pairs": m.adjacent_pairs,
                "bottom_laplacian": m.bottom_laplacian,
                "bottom_roughness": m.bottom_roughness,
                "bottom_vertices": m.bottom_vertices,
                "diagnostics": list(m.diagnostics)}

    return FunctionAnalyser("geometry.mesh_stability", "geometry",
                            "harnesscad.eval.quality.geometry.mesh_stability", applies, run)


def _a_geometry_usability_standard() -> FunctionAnalyser:
    """`geometry.usability_standard` -- defect readiness, loop-size variability, poly budget.

    Needs a mesh (face count + loops); SKIPS without one.
    """

    def applies(state: ModelState) -> bool:
        return bool(state.query("mesh").get("faces"))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.geometry.usability_standard import (
            evaluate_model_usability, loop_sizes_from_loops,
        )

        mesh = state.query("mesh")
        faces = list(mesh["faces"])
        loops = mesh.get("loops") or faces
        report_ = evaluate_model_usability(
            dict(mesh.get("defects") or {}),
            loop_sizes_from_loops(loops),
            face_count=len(faces),
            category=str(mesh.get("category")) if mesh.get("category") else None,
        )
        return {"verdict": report_.verdict, "reasons": list(report_.reasons),
                "clean": report_.defects.clean,
                "variability": report_.topology.variability,
                "closed_loops": report_.topology.closed_loop_count}

    return FunctionAnalyser("geometry.usability_standard", "geometry",
                            "harnesscad.eval.quality.geometry.usability_standard",
                            applies, run)


def _anomaly_analyser(name: str, method: str) -> FunctionAnalyser:
    """`geometry.anomaly` -- RIVAL scorers over ONE fitted baseline.

    z-score, IQR and the isolation-forest disagree on the same part by design (a
    tail that the z-score calls 3.6 sigma the IQR may call ordinary). They are
    exposed by name and NEVER averaged. All three need a reference corpus of
    known-good feature vectors (``extras={"anomaly": {"reference": [...]}}``);
    without one they SKIP -- an anomaly score against no baseline is noise.
    """

    def applies(state: ModelState) -> bool:
        return bool(state.query("anomaly").get("reference"))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.geometry.anomaly import (
            AnomalyModel, IsolationLite, feature_vector,
        )

        reference = [dict(v) for v in state.query("anomaly")["reference"]]
        vector = feature_vector(state.backend)
        if method == "isolation":
            model = IsolationLite(seed=0).fit(reference)
            score = model.score(vector)
        else:
            model = AnomalyModel(method=method).fit(reference)
            score = model.score(vector)
        payload = score.to_dict() if hasattr(score, "to_dict") else {"score": float(score)}
        payload["method"] = method
        payload["n_reference"] = len(reference)
        payload["features"] = dict(sorted(vector.items()))
        return payload

    return FunctionAnalyser(name, "geometry",
                            "harnesscad.eval.quality.geometry.anomaly", applies, run)


# --- physics ----------------------------------------------------------------

def _a_physics_mass_properties() -> FunctionAnalyser:
    """`physics.mass_properties` -- volume, mass, centre of mass, AABB.

    Built from the boxes the harness genuinely knows: the placed part AABBs when
    the backend exposes an assembly, else the single op-stream envelope. Density
    comes from ``backend.query('material')`` (default 1.0 -- volume-equivalent).
    """

    def applies(state: ModelState) -> bool:
        return bool(state.part_bboxes()) or state.envelope() is not None

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.physics.mass_properties import Assembly, Box

        density = float(state.query("material").get("density", 1.0) or 1.0)
        boxes = [bb for _, bb in state.part_bboxes()]
        if not boxes:
            boxes = [state.envelope()]
        asm = Assembly()
        for (x0, y0, z0, x1, y1, z1) in boxes:
            asm.add(Box(cx=(x0 + x1) / 2.0, cy=(y0 + y1) / 2.0, cz=(z0 + z1) / 2.0,
                        w=abs(x1 - x0), h=abs(y1 - y0), d=abs(z1 - z0),
                        density=density))
        mp = asm.mass_properties()
        return {"n_boxes": len(boxes), "density": density,
                "total_volume": mp.total_volume, "total_mass": mp.total_mass,
                "center_of_mass": list(mp.center_of_mass),
                "aabb": [list(mp.aabb[0]), list(mp.aabb[1])]}

    return FunctionAnalyser("physics.mass_properties", "physics",
                            "harnesscad.eval.quality.physics.mass_properties", applies, run)


def _a_physics_beam_screening() -> FunctionAnalyser:
    """`physics.beam_screening` -- deflection of a simply-supported span.

    Needs a load case the op vocabulary does not carry (span, load, modulus):
    ``backend.query('beam')``. SKIPS otherwise -- a deflection against an invented
    load is a fabricated number.
    """

    def applies(state: ModelState) -> bool:
        q = state.query("beam")
        return "span_m" in q and "point_load_kg" in q

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.physics.beam_screening import (
            rectangular_section, simply_supported_deflection,
        )

        q = state.query("beam")
        section = rectangular_section(float(q.get("width", 0.02)),
                                      float(q.get("height", 0.04)))
        res = simply_supported_deflection(
            float(q["span_m"]), float(q["point_load_kg"]), section.Iy,
            float(q.get("beam_kg_per_m", 1.0)))
        return {"point_load_mm": res.point_load_mm, "self_weight_mm": res.self_weight_mm,
                "total_mm": res.total_mm, "limit_mm": res.limit_mm, "passed": res.passed,
                "section": {"area": section.area, "Iy": section.Iy, "Iz": section.Iz}}

    return FunctionAnalyser("physics.beam_screening", "physics",
                            "harnesscad.eval.quality.physics.beam_screening", applies, run)


# --- graph ------------------------------------------------------------------

def _a_graph_intent_graph() -> FunctionAnalyser:
    """`graph.intent_graph` -- the causal graph of the model's design intent.

    Every feature op becomes a node; a feature that consumes a sketch depends on
    it. The causal order is the graph's own topological answer -- not the op order.
    """

    def applies(state: ModelState) -> bool:
        return bool(_op_dicts(state))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.graph.intent_graph import (
            IntentGraph, IntentNode, IntentRelation, RelationKind,
        )

        graph = IntentGraph()
        sketch_nodes: Dict[str, str] = {}
        n_sketch = 0
        for i, op in enumerate(_op_dicts(state)):
            tag = str(op.get("op", ""))
            nid = f"op{i}"
            graph.add_node(IntentNode(id=nid, intent=tag, feature_id=nid,
                                      attributes={"index": i}))
            if tag == "new_sketch":
                n_sketch += 1
                sketch_nodes[f"sk{n_sketch}"] = nid
                continue
            parent = sketch_nodes.get(str(op.get("sketch", "")))
            if parent:
                # CAUSAL: the sketch must exist before the feature that consumes it.
                graph.add_relation(IntentRelation(source=parent, target=nid,
                                                  kind=RelationKind.CAUSAL,
                                                  label=tag))
        payload = graph.to_dict()
        payload["causal_order"] = list(graph.causal_order())
        return payload

    return FunctionAnalyser("graph.intent_graph", "graph",
                            "harnesscad.eval.quality.graph.intent_graph", applies, run)


def _a_graph_abstraction() -> FunctionAnalyser:
    """`graph.abstraction` -- does this op stream collapse into a higher-level primitive?

    Proposal only. ``accept_abstraction`` needs a compiler + an equivalence judge
    the analysis surface does not own (that is a *gate*, and it belongs to the
    verifier fleet), so this reports the proposal and never claims it was accepted.
    """

    def applies(state: ModelState) -> bool:
        return bool(_op_dicts(state))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.graph.abstraction import propose_abstraction

        proposal = propose_abstraction(_op_dicts(state))
        return {"proposal": proposal, "accepted": None,
                "note": "proposal only: acceptance needs a compiler + equivalence judge"}

    return FunctionAnalyser("graph.abstraction", "graph",
                            "harnesscad.eval.quality.graph.abstraction", applies, run)


# --- assembly ---------------------------------------------------------------

def _a_assembly_interactions() -> FunctionAnalyser:
    """`assembly.interactions` -- allowed contact vs clearance violation vs collision.

    Derives the pairwise face distances from the placed part AABBs (negative =
    interpenetration). Needs >= 2 placed parts.
    """

    def applies(state: ModelState) -> bool:
        return len(state.part_bboxes()) >= 2

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.assembly.interactions import classify_interactions

        boxes = state.part_bboxes()
        interactions = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (ida, a), (idb, b) = boxes[i], boxes[j]
                gaps = [max(a[k] - b[k + 3], b[k] - a[k + 3]) for k in range(3)]
                interactions.append({"faces": (ida, idb), "distance": max(gaps)})
        q = state.query("assembly")
        classified = classify_interactions(
            interactions,
            allowed_contacts=tuple(tuple(p) for p in (q.get("allowed_contacts") or ())),
            minimum_clearance=float(q.get("minimum_clearance", 0.0) or 0.0))
        counts: Dict[str, int] = {}
        for item in classified:
            counts[item["classification"]] = counts.get(item["classification"], 0) + 1
        return {"n_pairs": len(classified), "counts": dict(sorted(counts.items())),
                "interactions": [{"faces": list(i["faces"]), "distance": i["distance"],
                                  "classification": i["classification"]}
                                 for i in classified]}

    return FunctionAnalyser("assembly.interactions", "assembly",
                            "harnesscad.eval.quality.assembly.interactions", applies, run)


# --- edit -------------------------------------------------------------------

def _a_edit_revision_delta() -> FunctionAnalyser:
    """`edit.revision_delta` -- what changed (mass / cost / BOM) between two revisions.

    Needs both revisions: ``extras={"revision": {"before": ..., "after": ...}}``
    (backends, PartEstimates, BOMs or plain metric maps -- the module accepts all
    four). One model state cannot be a delta, so this SKIPS without them.
    """

    def applies(state: ModelState) -> bool:
        q = state.query("revision")
        return "before" in q and "after" in q

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.edit.revision_delta import compare_revisions

        q = state.query("revision")
        rep = compare_revisions(q["before"], q["after"],
                                material=str(q.get("material", "aluminium")))
        return rep.to_dict()

    return FunctionAnalyser("edit.revision_delta", "edit",
                            "harnesscad.eval.quality.edit.revision_delta", applies, run)


# --- sketch -----------------------------------------------------------------

def _a_sketch_serialization() -> FunctionAnalyser:
    """`sketch.serialization` -- the canonical serialisation of the sketch + redundancy check."""

    def applies(state: ModelState) -> bool:
        return bool(state.query("sketch_geometry").get("primitives"))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.sketch.serialization import serialize_sketch

        q = state.query("sketch_geometry")
        payload = serialize_sketch(q["primitives"], tuple(q.get("constraints") or ()))
        return payload if isinstance(payload, dict) else {"serialized": payload}

    return FunctionAnalyser("sketch.serialization", "sketch",
                            "harnesscad.eval.quality.sketch.serialization", applies, run)


# --- reward -----------------------------------------------------------------

def _a_reward_composite() -> FunctionAnalyser:
    """`reward.composite_reward` -- RIVAL reward: gated weighted sum of named components.

    Gates on ``code_valid`` / ``execution_valid``: if a gate fails the total is
    zeroed rather than partially credited. A different function from
    ``reward.execution`` on different inputs -- the two are never averaged.
    """

    def applies(state: ModelState) -> bool:
        return bool(state.query("reward").get("components"))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.reward.composite_reward import CompositeReward

        q = state.query("reward")
        engine = CompositeReward(weights=dict(q.get("weights") or {}) or None)
        res = engine.compute({k: float(v) for k, v in q["components"].items()})
        return {"total": res.total, "gated_out": res.gated_out,
                "contributions": dict(sorted(res.contributions.items()))}

    return FunctionAnalyser("reward.composite", "reward",
                            "harnesscad.eval.quality.reward.composite_reward", applies, run)


def _a_reward_execution() -> FunctionAnalyser:
    """`reward.execution_reward` -- RIVAL reward: executability + Chamfer + format.

    Scores the generated TEXT against a target point cloud. Rival of
    ``reward.composite``: same question ("how good is this candidate?"), different
    protocol. Selected by name; never blended.
    """

    def applies(state: ModelState) -> bool:
        q = state.query("reward")
        return bool(q.get("text")) and q.get("cd") is not None

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.reward.execution_reward import format_reward, geometric_reward

        q = state.query("reward")
        cd = float(q["cd"])
        geometric = geometric_reward(cd)
        fmt = format_reward(str(q["text"]))
        weight_g = float(q.get("geometry_weight", 1.0))
        weight_f = float(q.get("format_weight", 1.0))
        return {"total": weight_g * geometric + weight_f * fmt,
                "geometric": geometric, "format": fmt, "cd": cd,
                "executable": bool(q.get("executable", True))}

    return FunctionAnalyser("reward.execution", "reward",
                            "harnesscad.eval.quality.reward.execution_reward", applies, run)


def _a_reward_pareto() -> FunctionAnalyser:
    """`reward.pareto` -- the non-dominated design front over declared objectives.

    Needs candidate designs (``extras={"candidates": {"items": [...],
    "objectives": [{"name","goal"}]}}``): a single model has no front.
    """

    def applies(state: ModelState) -> bool:
        q = state.query("candidates")
        return bool(q.get("items")) and bool(q.get("objectives"))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.reward.pareto import Objective, pareto_front, pareto_rank

        q = state.query("candidates")
        items = list(q["items"])
        objectives = [Objective(name=str(o["name"]), goal=str(o.get("goal", "min")),
                                key=(lambda item, n=str(o["name"]): float(item[n])))
                      for o in q["objectives"]]
        front = pareto_front(items, objectives)
        ranks = pareto_rank(items, objectives)
        return {"n_items": len(items), "n_front": len(front),
                "front": front, "n_ranks": len(ranks)}

    return FunctionAnalyser("reward.pareto", "reward",
                            "harnesscad.eval.quality.reward.pareto", applies, run)


# --- report -----------------------------------------------------------------

def _a_report_parameter_exposure() -> FunctionAnalyser:
    """`report.parameter_exposure` -- every numeric knob in the op stream, labelled."""

    def applies(state: ModelState) -> bool:
        return bool(_op_dicts(state))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.report.parameter_exposure import expose_parameters

        q = state.query("parameters")
        payload = expose_parameters(_op_dicts(state),
                                    ranges=dict(q.get("ranges") or {}),
                                    labels=dict(q.get("labels") or {}))
        return {"schema": payload["schema"], "n_fields": len(payload["fields"]),
                "fields": [dict(f) for f in payload["fields"]]}

    return FunctionAnalyser("report.parameter_exposure", "report",
                            "harnesscad.eval.quality.report.parameter_exposure",
                            applies, run)


def _a_report_traceability() -> FunctionAnalyser:
    """`report.traceability` -- requirement -> op -> element matrix (and the orphans).

    The brief is formalised into a RequirementSet offline (``domain.spec.formalize``
    heuristic, no LLM) and joined to the op history. Needs the brief:
    ``extras={"brief": {"text": "..."}}``.
    """

    def applies(state: ModelState) -> bool:
        return bool(state.query("brief").get("text"))

    def run(state: ModelState) -> dict:
        from harnesscad.domain.spec.formalize import formalize
        from harnesscad.eval.quality.report.traceability import build_traceability

        reqset = formalize(str(state.query("brief")["text"]))
        matrix = build_traceability(reqset, state.opdag, state.backend)
        payload = matrix.to_dict()
        payload["n_rows"] = len(matrix.rows)
        payload["n_satisfied"] = sum(1 for r in matrix.rows if r.satisfied)
        return payload

    return FunctionAnalyser("report.traceability", "report",
                            "harnesscad.eval.quality.report.traceability", applies, run)


def _a_report_suggest_cots() -> FunctionAnalyser:
    """`report.suggest_cots` -- which features should be a bought standard part."""

    def applies(state: ModelState) -> bool:
        return bool(_op_dicts(state))

    def run(state: ModelState) -> dict:
        from harnesscad.eval.quality.report.suggest_cots import suggest_cots

        suggestions = suggest_cots(state.backend)
        return {"n_suggestions": len(suggestions),
                "suggestions": [s.to_dict() for s in suggestions]}

    return FunctionAnalyser("report.suggest_cots", "report",
                            "harnesscad.eval.quality.report.suggest_cots", applies, run)


_ADAPTERS: Tuple[Callable[[], FunctionAnalyser], ...] = (
    _a_sequence_complexity,
    _a_sequence_quantization_risk,
    _a_sequence_macro_collapse,
    _a_sequence_primitive_pooling,
    _a_sequence_code_normalize,
    _a_sequence_confidence,
    _a_geometry_canonical_pose,
    _a_geometry_invariance,
    _a_geometry_mesh_stability,
    _a_geometry_usability_standard,
    lambda: _anomaly_analyser("geometry.anomaly_zscore", "zscore"),
    lambda: _anomaly_analyser("geometry.anomaly_iqr", "iqr"),
    lambda: _anomaly_analyser("geometry.anomaly_isolation", "isolation"),
    _a_physics_mass_properties,
    _a_physics_beam_screening,
    _a_graph_intent_graph,
    _a_graph_abstraction,
    _a_assembly_interactions,
    _a_edit_revision_delta,
    _a_sketch_serialization,
    _a_reward_composite,
    _a_reward_execution,
    _a_reward_pareto,
    _a_report_parameter_exposure,
    _a_report_traceability,
    _a_report_suggest_cots,
)

_FLEET: Optional[Tuple[FunctionAnalyser, ...]] = None


def _summaries() -> Dict[str, str]:
    return {e.dotted: e.summary for e in capability_registry.index()}


def _build_fleet() -> Tuple[FunctionAnalyser, ...]:
    summaries = _summaries()
    fleet: List[FunctionAnalyser] = []
    for build in _ADAPTERS:
        try:
            a = build()
        except Exception:  # noqa: BLE001 - a broken adapter must not kill the fleet
            continue
        fleet.append(FunctionAnalyser(
            name=a.name, kind=a.kind, dotted=a.dotted,
            _applies=a._applies, _run=a._run,  # noqa: SLF001 - re-stamped with its summary
            summary=summaries.get(a.dotted, "")))
    fleet.sort(key=lambda a: (_KIND_ORDER.get(a.kind, len(KINDS)), a.name))
    return tuple(fleet)


def analysers(kind: Optional[str] = None) -> Tuple[FunctionAnalyser, ...]:
    """The whole fleet, deterministically ordered by (kind, name)."""
    global _FLEET
    if _FLEET is None:
        _FLEET = _build_fleet()
    if kind is None:
        return _FLEET
    return tuple(a for a in _FLEET if a.kind == kind)


def analyser(name: str) -> FunctionAnalyser:
    for a in analysers():
        if a.name == name:
            return a
    raise KeyError(f"no such analyser: {name!r} "
                   f"(known: {', '.join(a.name for a in analysers())})")


def kinds() -> Tuple[str, ...]:
    return KINDS


def adapted_modules() -> Tuple[str, ...]:
    return tuple(sorted({a.dotted for a in analysers()}))


def unadapted() -> Tuple[str, ...]:
    """Quality modules in the index that no analyser binds yet (discovery, not silence)."""
    bound = set(adapted_modules())
    return tuple(sorted(e.dotted for e in capability_registry.find(package=QUALITY_PACKAGE)
                        if e.dotted not in bound and e.name != "registry"))


# ---------------------------------------------------------------------------
# Rival families
# ---------------------------------------------------------------------------

#: Analysers in one family answer the SAME question under DIFFERENT protocols.
#: Their numbers are not comparable: expose both, average neither.
RIVAL_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("anomaly_score", (
        "geometry.anomaly_zscore",     # gaussian tail, z > 3.5
        "geometry.anomaly_iqr",        # robust quartile fence, k = 3.0
        "geometry.anomaly_isolation",  # isolation-forest path length
    )),
    ("candidate_reward", (
        "reward.composite",   # gated weighted sum of named components
        "reward.execution",   # executability + Chamfer + format on the generated text
    )),
)


def rivals() -> Dict[str, Tuple[str, ...]]:
    return {name: members for name, members in RIVAL_FAMILIES}


class RivalBlendError(ValueError):
    """Someone tried to pool two rival analysers into one number."""


def _rival_of(name: str) -> Optional[str]:
    for family, members in RIVAL_FAMILIES:
        if name in members:
            return family
    return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def analyse(state: ModelState,
            kinds_: Optional[Sequence[str]] = None,
            only: Optional[Sequence[str]] = None,
            skip: Sequence[str] = (),
            fleet: Optional[Sequence[Analyser]] = None) -> List[AnalysisResult]:
    """Run the fleet over one model state. NEVER raises.

    An analyser that does not apply is ``skipped`` (its input is absent -- it is
    not guessed). An analyser that raises is an ``error`` entry and the run
    continues. Results come back in fleet order, so the output is deterministic.
    """
    wanted = set(kinds_) if kinds_ else None
    onlyset = set(only) if only else None
    skipset = set(skip)
    selected = list(fleet) if fleet is not None else list(analysers())

    results: List[AnalysisResult] = []
    for a in selected:
        name = getattr(a, "name", type(a).__name__)
        kind = getattr(a, "kind", "")
        dotted = getattr(a, "dotted", "")
        if wanted is not None and kind not in wanted:
            continue
        if onlyset is not None and name not in onlyset:
            continue
        if name in skipset:
            continue
        base = dict(name=name, kind=kind, dotted=dotted)
        try:
            if not a.applies_to(state):
                results.append(AnalysisResult(
                    status="skipped", error="state does not carry this analyser's input",
                    **base))
                continue
            value = a.analyse(state)
        except Exception as exc:  # noqa: BLE001 - THE point of this dispatcher
            results.append(AnalysisResult(
                status="error", error=f"{type(exc).__name__}: {exc}", **base))
            continue
        results.append(AnalysisResult(status="ok", value=value, **base))
    return results


#: Named scalar scores an analyser exposes, and how to read them out of its value.
#: Rival analysers land under DISTINCT keys (never a shared "reward" key), so the
#: verbalised summary can never silently average two protocols.
_SCORE_KEYS: Tuple[Tuple[str, str], ...] = (
    ("sequence.complexity_taxonomy", "level"),
    ("reward.composite", "total"),
    ("reward.execution", "total"),
    ("geometry.anomaly_zscore", "score"),
    ("geometry.anomaly_iqr", "score"),
    ("geometry.anomaly_isolation", "score"),
    ("physics.mass_properties", "total_mass"),
)


def _scores(results: Sequence[AnalysisResult]) -> Dict[str, float]:
    """Scalar scores keyed by ANALYSER name -- never pooled across a rival family."""
    out: Dict[str, float] = {}
    seen_family: Dict[str, str] = {}
    for r in results:
        if r.status != "ok":
            continue
        for name, key in _SCORE_KEYS:
            if r.name != name or key not in r.value:
                continue
            val = r.value[key]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                continue
            family = _rival_of(r.name)
            if family:
                seen_family.setdefault(family, r.name)
            out[f"{r.name}.{key}"] = float(val)
    return out


def report(state: ModelState, **kwargs) -> QualityReport:
    """The quality report for one model state: analyses + scores + prose + claim audit.

    The prose is produced by ``report.score_verbalization`` (band templates, one
    line per NAMED score -- rivals stay separate lines, never one blended number),
    and then audited by ``report.claim_audit`` for over-claiming absolutes. The
    harness audits its own report.
    """
    results = analyse(state, **kwargs)
    scores = _scores(results)
    prose = ""
    claims: Dict[str, Any] = {}
    try:
        from harnesscad.eval.quality.report.score_verbalization import verbalise_scores

        prose = verbalise_scores({k: v for k, v in scores.items() if 0.0 <= v <= 1.0}) \
            if any(0.0 <= v <= 1.0 for v in scores.values()) else ""
    except Exception as exc:  # noqa: BLE001
        prose = ""
        claims["verbalisation_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from harnesscad.eval.quality.report.claim_audit import audit_text

        audited = audit_text(prose or "(no verbalisable scores)", is_markdown=False)
        claims["findings"] = [
            {"rule": f.rule, "severity": f.severity, "line": f.line, "message": f.message}
            for f in audited.findings
        ]
        claims["lines_scanned"] = audited.lines_scanned
    except Exception as exc:  # noqa: BLE001
        claims["audit_error"] = f"{type(exc).__name__}: {exc}"
    return QualityReport(results=results, scores=scores, prose=prose, claims=claims)


# ---------------------------------------------------------------------------
# CLI (wired into core.cli as `harnesscad report`)
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ops", default=None,
                        help="path to a JSON array of ops (default: the built-in demo)")
    parser.add_argument("--backend", default="stub", choices=["stub", "cadquery", "frep"])
    parser.add_argument("--brief", default=None,
                        help="the design brief, so the traceability matrix can be built")
    parser.add_argument("--extras", default=None,
                        help="path to a JSON object of extra state projections "
                             "(anomaly reference corpus, reward components, mesh, ...)")
    parser.add_argument("--kind", default=None, choices=list(KINDS),
                        help="restrict the report to one analyser kind")
    parser.add_argument("--only", default=None,
                        help="comma-separated analyser names to run")
    parser.add_argument("--list", action="store_true", help="list every analyser")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival analyser families (never averaged)")
    parser.add_argument("--unadapted", action="store_true",
                        help="list quality modules with no analyser yet")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")


def run(args: argparse.Namespace) -> int:
    if getattr(args, "list", False):
        for a in analysers(kind=getattr(args, "kind", None)):
            print(f"{a.name:<32} {a.kind:<10} {a.dotted}")
            if a.summary:
                print(f"    {a.summary}")
        print(f"-- {len(analysers())} analysers / {len(adapted_modules())} modules "
              f"/ {len(unadapted())} quality modules unadapted")
        return 0

    if getattr(args, "rivals", False):
        for family, members in RIVAL_FAMILIES:
            print(f"{family}: (exposed by name, never averaged)")
            for n in members:
                print(f"    - {n:<32} {analyser(n).dotted}")
        return 0

    if getattr(args, "unadapted", False):
        for dotted in unadapted():
            print(dotted)
        print(f"-- {len(unadapted())} quality modules without an analyser")
        return 0

    from harnesscad.core.cli import DEMO_OPS
    from harnesscad.io.surfaces.server import CISPServer

    if getattr(args, "ops", None):
        try:
            with open(args.ops, "r", encoding="utf-8") as fh:
                ops = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: could not load ops from {args.ops!r}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(ops, list):
            print(f"error: {args.ops!r} must contain a JSON array of ops", file=sys.stderr)
            return 2
    else:
        ops = [dict(op) for op in DEMO_OPS]

    extras: Dict[str, dict] = {}
    if getattr(args, "extras", None):
        try:
            with open(args.extras, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: could not load extras from {args.extras!r}: {exc}",
                  file=sys.stderr)
            return 2
        if not isinstance(loaded, dict):
            print(f"error: {args.extras!r} must contain a JSON object", file=sys.stderr)
            return 2
        extras.update({k: v for k, v in loaded.items() if isinstance(v, dict)})
    if getattr(args, "brief", None):
        extras["brief"] = {"text": args.brief}

    server = CISPServer(backend=args.backend)
    result = server.applyOps(ops)
    if not result["ok"]:
        print("error: the op stream does not build; nothing to analyse", file=sys.stderr)
        for d in result.get("diagnostics") or []:
            print(f"  [{d['severity']}] {d['code']}: {d['message']}", file=sys.stderr)
        return 1

    session = server.session
    state = model_state(session.backend, session.opdag, extras)
    only = [n.strip() for n in args.only.split(",")] if getattr(args, "only", None) else None
    rep = report(state, kinds_=[args.kind] if getattr(args, "kind", None) else None,
                 only=only)

    if getattr(args, "json", False):
        print(rep.to_json())
        return 0 if not rep.errors() else 1

    print(f"digest:   {result['digest']}")
    print(f"analysed: {len(rep.ok())} ok / {len(rep.skipped())} skipped / "
          f"{len(rep.errors())} error")
    print()
    for r in rep.ok():
        print(f"[{r.kind}] {r.name}  ({r.dotted})")
        for key in sorted(r.value):
            val = r.value[key]
            rendered = json.dumps(val, sort_keys=True, default=str)
            if len(rendered) > 100:
                rendered = rendered[:97] + "..."
            print(f"    {key:<22} {rendered}")
    if rep.scores:
        print()
        print("scores (per analyser -- rivals are listed apart, never averaged):")
        for key in sorted(rep.scores):
            print(f"  {key:<40} {rep.scores[key]!r}")
    if rep.prose:
        print()
        print("summary:")
        print(rep.prose)
    findings = (rep.claims or {}).get("findings") or []
    if findings:
        print()
        print("claim audit of this report's own prose:")
        for f in findings:
            print(f"  [{f['severity']}] {f['rule']}: {f['message']}")
    if rep.skipped():
        print()
        print("skipped (input absent -- never fabricated):")
        for r in rep.skipped():
            print(f"  {r.name:<32} {r.error}")
    if rep.errors():
        print()
        print("errors:")
        for r in rep.errors():
            print(f"  {r.name:<32} {r.error}")
    return 0 if not rep.errors() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad report",
        description="quality analysis surface: analyse a model, do not gate it")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
