"""Data-engine pipeline: the runnable seam for ``data/dataengine`` + ``data/datagen``.

The repo carries ~120 data modules (schemas, curation, annotation, augmentation,
reward, preference, self-training, trace, edits, audit). Every one of them is
correct and tested in isolation, and *nothing called any of them*: there was no
seam at all for the data layer -- no place where samples flow

    generate / ingest  ->  annotate  ->  curate  ->  augment  ->  emit

This module is that seam. It follows the three dispatchers already in the tree
(:mod:`harnesscad.eval.verifiers.registry`, :mod:`harnesscad.eval.bench.registry`,
:mod:`harnesscad.io.formats.registry`):

*   **Discovery, not a hardcoded inventory.** The stage catalogue is checked
    against the static AST index (:mod:`harnesscad.registry`, packages
    ``dataengine`` / ``datagen``); :func:`unadapted` reports every data module no
    stage binds yet, so an orphan stays visible instead of silently vanishing.
*   **Adapters only.** Each stage adapts a module's *real* public API. No data
    module is modified, and no stage invents a number a module did not compute.
*   **Composable, named stages.** A pipeline is an ordered tuple of stage names.
    A stage that raises is captured as an ``error`` :class:`StageResult` and the
    run carries on with the records it already had -- a broken stage can never
    take the pipeline down.
*   **Rivals stay selectable, never blended.** Three de-duplication strategies
    (scale-invariant, exact-token, outcome-signature), two complexity filters and
    two reward functions answer the SAME question in mutually incompatible ways.
    :data:`RIVAL_FAMILIES` is enforced when a pipeline is built, so a preset that
    would run two rivals into one dataset cannot even be constructed.
*   **Deterministic.** Every random draw is seeded from :class:`Context.seed`;
    splits/dedup/subsets sort before they choose; nothing reads the wall clock.
    Same records + same seed -> byte-identical :meth:`Dataset.to_json`.

The flywheel (the thing these modules were built for) is wired too:
:func:`records_from_session` folds the trace events a :class:`HarnessSession`
already emits into the canonical :class:`Trajectory` record, and the
``ingest.session_trace`` stage turns a finished session into training records --
STaR / DPO / GRPO rows come out of ``emit.trace_export``.

Typical use::

    from harnesscad.data import pipeline

    ctx = pipeline.Context(seed=7)
    dataset = pipeline.run_preset("text2cad", records, ctx)
    dataset.to_json()          # canonical, sorted, deterministic

Stdlib-only, absolute imports, no wall clock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "KINDS",
    "Context",
    "Stage",
    "StageResult",
    "Dataset",
    "Pipeline",
    "RIVAL_FAMILIES",
    "RivalBlendError",
    "stages",
    "stage",
    "kinds",
    "presets",
    "preset",
    "rivals",
    "unadapted",
    "run_stage",
    "run_pipeline",
    "run_preset",
    "records_from_session",
    "add_arguments",
    "run",
    "main",
]

#: The packages this seam adapts (as the static index names them).
DATA_PACKAGES: Tuple[str, ...] = ("dataengine", "datagen")

#: Stage families, in pipeline order.
KINDS: Tuple[str, ...] = (
    "generate",    # manufacture / ingest raw samples
    "annotate",    # attach prompts, code, metadata, labels
    "curate",      # filter, dedup, tier, split, subset
    "augment",     # variant expansion
    "reward",      # scalar reward per record
    "preference",  # pairwise / binary preference records
    "selftrain",   # pseudo-label + confidence selection
    "audit",       # corpus-level diagnostics (never mutates records)
    "emit",        # materialise the dataset
)
_KIND_ORDER = {k: i for i, k in enumerate(KINDS)}


# ---------------------------------------------------------------------------
# Records, context, results
# ---------------------------------------------------------------------------

#: A *record* is a plain JSON-able dict. The keys a stage needs are declared in
#: ``Stage.requires``; a record that lacks them is passed through untouched (the
#: stage reports how many it skipped -- it never fabricates the missing field).
Record = Dict[str, Any]


@dataclass
class Context:
    """Everything a stage may read besides the records themselves.

    ``seed`` drives every random draw in the run (no stage may create its own
    unseeded RNG). ``options`` carries per-stage settings; ``artifacts`` collects
    the corpus-level outputs (audits, export rows, capture records) that are not
    records.
    """

    seed: int = 0
    target_size: int = 0
    splits: Tuple[Tuple[str, float], ...] = (("train", 0.8), ("val", 0.1), ("test", 0.1))
    options: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def rng(self, salt: str) -> random.Random:
        """A stage-local RNG derived from (seed, stage name). Stable across runs."""
        h = hashlib.sha256(f"{self.seed}|{salt}".encode()).hexdigest()[:16]
        return random.Random(int(h, 16))

    def option(self, stage_name: str, key: str, default: Any = None) -> Any:
        opts = self.options.get(stage_name)
        if isinstance(opts, dict) and key in opts:
            return opts[key]
        return default


@dataclass(frozen=True)
class StageResult:
    """What one stage did. ``status`` is ok | error."""

    name: str
    kind: str
    dotted: str
    status: str
    n_in: int
    n_out: int
    note: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "dotted": self.dotted,
            "status": self.status, "n_in": self.n_in, "n_out": self.n_out,
            "note": self.note, "error": self.error,
        }


# A stage function takes (records, ctx) and returns (records, note). It may write
# corpus-level output into ctx.artifacts. It must not mutate the input list.
StageFn = Callable[[List[Record], Context], Tuple[List[Record], str]]


@dataclass(frozen=True)
class Stage:
    """One adapted data module behind a uniform ``(records, ctx) -> records``."""

    name: str
    kind: str
    dotted: str
    fn: StageFn
    summary: str = ""
    requires: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "dotted": self.dotted,
            "summary": self.summary, "requires": list(self.requires),
        }


@dataclass
class Dataset:
    """The emitted dataset: records + artifacts + the per-stage ledger."""

    pipeline: str
    seed: int
    records: List[Record] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    results: List[StageResult] = field(default_factory=list)

    def errors(self) -> List[StageResult]:
        return [r for r in self.results if r.status == "error"]

    def splits(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self.records:
            counts[str(rec.get("split", ""))] = counts.get(str(rec.get("split", "")), 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline,
            "seed": self.seed,
            "n_records": len(self.records),
            "splits": self.splits(),
            "records": self.records,
            "artifacts": self.artifacts,
            "stages": [r.to_dict() for r in self.results],
        }

    def to_json(self) -> str:
        """Canonical serialisation: sorted keys, fixed separators, no clock."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, default=_jsonable)


def _jsonable(obj: Any) -> Any:
    """Last-resort encoder: dataclass-ish and tuple-ish values become JSON."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in sorted(vars(obj).items())}
    return str(obj)


# ---------------------------------------------------------------------------
# Small shared helpers (thin; they derive nothing a module should derive)
# ---------------------------------------------------------------------------

def _rid(rec: Record, position: int) -> str:
    return str(rec.get("id") or f"rec-{position}")


def _ops(rec: Record) -> List[dict]:
    ops = rec.get("ops")
    return [o for o in ops if isinstance(o, dict)] if isinstance(ops, list) else []


def _commands(rec: Record) -> Tuple[str, ...]:
    return tuple(str(o.get("op", "")) for o in _ops(rec))


def _numeric_features(rec: Record) -> Tuple[float, ...]:
    """The record's numeric op parameters, in op order -- the dedup feature vector.

    Deliberately mechanical: it is the op stream's own numbers, not a learned
    embedding. ``curate.dedup_scale`` normalises it; nothing else interprets it.
    """
    feats = rec.get("features")
    if isinstance(feats, (list, tuple)) and feats:
        return tuple(float(v) for v in feats)
    out: List[float] = []
    for op in _ops(rec):
        for key in sorted(op):
            val = op[key]
            if key != "op" and isinstance(val, (int, float)) and not isinstance(val, bool):
                out.append(float(val))
    return tuple(out)


def _token_sequence(rec: Record) -> Tuple[Tuple[Any, ...], ...]:
    """The record's exact op tokens: (op name, *sorted (key, value) pairs).

    This is the token identity ``curate.dedup_tokens`` compares. Two plates that
    differ only in width are DIFFERENT tokens here -- and identical designs under
    ``curate.dedup_scale``. That disagreement is the rivalry, not a bug.
    """
    return tuple(
        (str(op.get("op", "")),) + tuple((k, op[k]) for k in sorted(op) if k != "op")
        for op in _ops(rec)
    )


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=_jsonable).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Stage adapters.
#
# Every adapter imports its data module INSIDE the closure, so a module with an
# optional dependency degrades to a caught `error` StageResult rather than
# breaking the import of this seam.
# ---------------------------------------------------------------------------

# --- generate / ingest ------------------------------------------------------

def _stage_generate_solver_loop(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`datagen.pipeline` -- the solver-in-the-loop generator (verified ground truth).

    Runs the parametric generators through a real HarnessSession and keeps only
    the candidates that BUILD. Option ``n`` (default 0) -- 0 means "the caller
    supplied the samples; generate nothing".
    """
    from harnesscad.data.datagen.pipeline import generate_dataset_report
    from harnesscad.io.backends.stub import StubBackend

    n = int(ctx.option("generate.solver_loop", "n", 0) or 0)
    if n <= 0:
        return list(records), "no generation requested (n=0)"
    report = generate_dataset_report(n, ctx.seed, StubBackend)
    out = list(records)
    for i, sample in enumerate(report.samples):
        out.append({
            "id": f"gen-{i}-{sample.digest[:8]}",
            "prompt": sample.brief,
            "generator": sample.generator,
            "params": dict(sample.params),
            "ops": [dict(op) for op in sample.ops],
            "digest": sample.digest,
            "summary": dict(sample.summary),
            "ok": True,
            "source": "datagen.pipeline",
        })
    ctx.artifacts["generation_yield"] = {
        "total": report.total, "kept": report.kept, "yield_rate": report.yield_rate,
    }
    return out, f"generated {report.kept}/{report.total} verified samples"


def _stage_ingest_session_trace(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.trace.trajectory` -- fold a HarnessSession's trace into records.

    THE FLYWHEEL. Option ``sessions`` is a list of
    ``{"prompt": str, "events": [tracer events], "plan": Any, "rationales": {i: str}}``
    -- exactly what ``core.trace.InMemoryTracer``/``JsonlTracer`` already write. Each
    session becomes one training record carrying the canonical Trajectory
    (steps = (S_t, A_t, R_t, S_t+1)), its dense rewards and its sub-goals.
    """
    from harnesscad.data.dataengine.trace.intent import attach_intent
    from harnesscad.data.dataengine.trace.trajectory import from_events

    sessions = ctx.option("ingest.session_trace", "sessions", []) or []
    out = list(records)
    trajectories = []
    for i, sess in enumerate(sessions):
        events = list(sess.get("events") or [])
        traj = from_events(events, prompt=sess.get("prompt"), plan=sess.get("plan"),
                           metadata=dict(sess.get("metadata") or {}))
        rationales = sess.get("rationales") or {}
        for step in traj.steps:
            rationale = rationales.get(step.index) or rationales.get(str(step.index))
            if rationale:
                attach_intent(step, str(rationale), op=str(step.action.tool_call.get("op", "")),
                              index=step.index)
        trajectories.append(traj)
        applied = [s.action.tool_call for s in traj.steps if s.outcome == "applied"]
        out.append({
            "id": str(sess.get("id") or f"session-{i}"),
            "prompt": traj.prompt or "",
            "ops": [dict(op) for op in applied],
            "trajectory": traj.to_dict(),
            "reward": {"trajectory_total": traj.total_reward()},
            "ok": bool(traj.success),
            "n_steps": len(traj.steps),
            "n_corrections": int(traj.corrections()),
            "source": "harness.session",
        })
    ctx.artifacts.setdefault("_trajectories", []).extend(trajectories)
    return out, f"ingested {len(trajectories)} session trace(s)"


def _stage_ingest_session_capture(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.trace.session_capture` -- consent-gated capture of the op decisions.

    Records with a trajectory are replayed into a ModelingSessionCapture. Consent
    is REQUIRED and never assumed: option ``consent`` must be true, otherwise the
    stage records nothing (the module itself refuses to export without it).
    Timestamps are the step indices -- no wall clock.
    """
    from harnesscad.data.dataengine.trace.session_capture import Consent, ModelingSessionCapture

    granted = bool(ctx.option("ingest.session_capture", "consent", False))
    if not granted:
        return list(records), "no training consent given; nothing captured"
    captured = 0
    out: List[Record] = []
    for position, rec in enumerate(records):
        traj = rec.get("trajectory")
        if not isinstance(traj, dict) or not traj.get("steps"):
            out.append(rec)
            continue
        capture = ModelingSessionCapture(
            session_id=_rid(rec, position),
            consent=Consent(granted=True, scope="training"),
            provenance={"source": str(rec.get("source", ""))},
        )
        for step in traj["steps"]:
            ts = float(step.get("index", 0))
            decision = capture.propose(ts, [step["action"]["tool_call"]])
            capture.decide(decision.proposal_id, ts + 0.5,
                           accepted=step.get("outcome") == "applied",
                           reason=str(step.get("outcome", "")))
        new = dict(rec)
        new["capture"] = capture.export()
        out.append(new)
        captured += 1
    return out, f"captured {captured} consented session(s)"


# --- annotate ---------------------------------------------------------------

def _named_sketches(ops: Sequence[dict]) -> List[dict]:
    """CISP names sketches implicitly (sk1, sk2, ...); the emitter reads ``name``.

    A pure renaming pass so ``emit_cadquery`` can resolve the references CISP left
    implicit. No numbers change; nothing else is rewritten.
    """
    out: List[dict] = []
    n = 0
    for op in ops:
        op = dict(op)
        if op.get("op") == "new_sketch" and "name" not in op:
            n += 1
            op["name"] = f"sk{n}"
        out.append(op)
    return out


def _stage_annotate_code(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`datagen.cadquery_codegen` -- emit the CadQuery script for the op stream.

    The emitter covers a restricted op vocabulary (sketch / rect / circle / extrude
    / fillet / chamfer). An op stream it cannot express raises, and that record
    simply gets NO code -- the stage never emits a script that does not correspond
    to the op stream.
    """
    from harnesscad.data.datagen.cadquery_codegen import emit_cadquery

    out: List[Record] = []
    done = 0
    unsupported = 0
    for rec in records:
        ops = _ops(rec)
        if not ops:
            out.append(rec)
            continue
        try:
            code = emit_cadquery(_named_sketches(ops))
        except ValueError:
            unsupported += 1
            out.append(rec)
            continue
        new = dict(rec)
        new["code"] = code
        out.append(new)
        done += 1
    return out, (f"emitted code for {done}/{len(records)} records "
                 f"({unsupported} outside the emitter's op vocabulary)")


def _stage_annotate_code_complexity(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.curation.code_complexity` -- token/statement/call/depth bins."""
    from harnesscad.data.dataengine.curation.code_complexity import analyze_code

    out: List[Record] = []
    done = 0
    for rec in records:
        code = rec.get("code")
        if not isinstance(code, str) or not code.strip():
            out.append(rec)
            continue
        c = analyze_code(code)
        new = dict(rec)
        new["code_complexity"] = {
            "tokens": c.tokens, "statements": c.statements, "calls": c.calls,
            "max_depth": c.max_depth, "bin": c.bin,
        }
        out.append(new)
        done += 1
    return out, f"scored {done} code bodies"


def _stage_annotate_instruction_slots(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`datagen.instruction_taxonomy` -- seeded (category, style, length) slots + coverage."""
    from harnesscad.data.datagen.instruction_taxonomy import (
        InstructionSample, seeded_slots, slot_coverage,
    )

    if not records:
        return [], "no records"
    slots = seeded_slots(len(records), ctx.seed)
    out: List[Record] = []
    samples = []
    for i, rec in enumerate(records):
        slot = slots[i % len(slots)]
        new = dict(rec)
        new["instruction_slot"] = {
            "category": slot.category, "style": slot.style,
            "length_bucket": slot.length_bucket,
        }
        out.append(new)
        samples.append(InstructionSample(text=str(rec.get("prompt") or ""), slot=slot))
    ctx.artifacts["instruction_slot_coverage"] = slot_coverage(samples)
    return out, f"assigned {len(slots)} seeded slots"


def _stage_annotate_minimal_metadata(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.annotation.minimal_metadata` -- Text2CAD minimal-metadata reduction.

    Only records that carry a raw ``metadata`` entity map are touched: the module
    strips random keys / redundant entity types. A record without raw metadata is
    passed through -- there is nothing honest to minimise.
    """
    from harnesscad.data.dataengine.annotation.minimal_metadata import generate_minimal_metadata

    out: List[Record] = []
    done = 0
    for rec in records:
        raw = rec.get("metadata")
        if not isinstance(raw, dict) or not raw:
            out.append(rec)
            continue
        meta = generate_minimal_metadata(raw)
        new = dict(rec)
        new["minimal_metadata"] = meta.as_dict()
        out.append(new)
        done += 1
    return out, f"minimised metadata for {done} records"


def _stage_annotate_features(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """The op stream's own numeric parameters, as the feature vector the curators read.

    Not a paper module -- a two-line projection, kept here so ``curate.dedup_scale``
    and ``curate.subset`` have the ``features`` key they document as their input.
    """
    out = []
    for rec in records:
        new = dict(rec)
        new["features"] = list(_numeric_features(rec))
        new["commands"] = list(_commands(rec))
        out.append(new)
    return out, f"projected features for {len(out)} records"


# --- curate -----------------------------------------------------------------

def _stage_curate_dedup_scale(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.curation.scale_dedup` -- RIVAL dedup: scale-INVARIANT signatures.

    Two designs that differ only in dimension collapse to one. Rival of
    ``curate.dedup_tokens`` (exact op-token identity) and ``curate.dedup_outcome``
    (identical benchmark outcomes): they keep different records by design.
    """
    from harnesscad.data.dataengine.curation.scale_dedup import dedup_by_scale

    if not records:
        return [], "no records"
    res = dedup_by_scale(records, int(ctx.option("curate.dedup_scale", "precision", 3)))
    ctx.artifacts["dedup_scale"] = {
        "n_in": res["n_in"], "n_out": res["n_out"],
        "reduction_ratio": res["reduction_ratio"],
    }
    return list(res["kept"]), f"kept {res['n_out']}/{res['n_in']} (scale-invariant)"


def _stage_curate_dedup_tokens(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.curation.sketch_filters` -- RIVAL dedup: EXACT op-token identity.

    Vitruvion's exact-token dedup: only byte-identical command sequences collapse.
    A scale variant survives here and does not survive ``curate.dedup_scale``.
    """
    from harnesscad.data.dataengine.curation.sketch_filters import unique_indices

    if not records:
        return [], "no records"
    seqs = [tuple(_token_sequence(rec)) for rec in records]
    keep = unique_indices(seqs)
    kept = [records[i] for i in keep]
    ctx.artifacts["dedup_tokens"] = {"n_in": len(records), "n_out": len(kept)}
    return kept, f"kept {len(kept)}/{len(records)} (exact token identity)"


def _stage_curate_dedup_outcome(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.curation.outcome_dedup` -- RIVAL dedup: identical per-option outcomes.

    Collapses records whose ``outcomes`` vector is identical. Records without an
    ``outcomes`` key are kept untouched (the signature is undefined for them).
    """
    from harnesscad.data.dataengine.curation.outcome_dedup import dedup_report, deduplicate

    scored = [(i, r) for i, r in enumerate(records) if isinstance(r.get("outcomes"), list)]
    if not scored:
        return list(records), "no record carries an 'outcomes' vector; nothing deduped"
    instances = [tuple(r["outcomes"]) for _, r in scored]
    keep_idx = deduplicate(instances, return_indices=True)
    keep = {scored[i][0] for i in keep_idx}
    kept = [r for i, r in enumerate(records)
            if i in keep or not isinstance(r.get("outcomes"), list)]
    rep = dedup_report(instances)
    ctx.artifacts["dedup_outcome"] = {
        "n_before": rep.n_before, "n_after": rep.n_after,
        "n_duplicate_groups": rep.n_duplicate_groups,
    }
    return kept, f"kept {len(kept)}/{len(records)} (outcome signature)"


def _stage_curate_filter_knit(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.curation.complexity_limits` -- RIVAL filter: KnitCAD B-rep limits.

    Rejects on face/contact/edge counts and multi-solid bodies. Needs a measured
    B-rep summary (``faces``/``contacts``/``max_edges_per_face``/``solids``); a
    record without one is kept (this filter has nothing to say about it).
    """
    from harnesscad.data.dataengine.curation.complexity_limits import (
        KnitLimits, filter_record, rejection_distribution,
    )

    limits = KnitLimits()
    keys = ("faces", "contacts", "max_edges_per_face", "solids")
    judged = [r for r in records if all(k in r for k in keys)]
    if not judged:
        return list(records), "no record carries a B-rep summary; nothing filtered"
    kept = [r for r in records
            if not all(k in r for k in keys) or not filter_record(r, limits)]
    ctx.artifacts["knit_rejections"] = rejection_distribution(judged, limits)
    return kept, f"kept {len(kept)}/{len(records)} (KnitCAD B-rep limits)"


def _stage_curate_filter_code_tokens(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.curation.code_complexity` -- RIVAL filter: code-token budget.

    Routes records whose script exceeds the token budget through the declared
    overflow policy (reject / chunk / long-context-route). A different question
    from the KnitCAD B-rep limits, on a different quantity -- never blend them.
    """
    from harnesscad.data.dataengine.curation.code_complexity import analyze_code, overflow_route

    maximum = int(ctx.option("curate.filter_code_tokens", "max_tokens", 3000))
    policy = str(ctx.option("curate.filter_code_tokens", "policy", "reject"))
    kept: List[Record] = []
    routed: Dict[str, int] = {}
    for rec in records:
        code = rec.get("code")
        if not isinstance(code, str) or not code.strip():
            kept.append(rec)
            continue
        tokens = analyze_code(code).tokens
        route = overflow_route(tokens, maximum, policy)
        routed[route] = routed.get(route, 0) + 1
        if route == "accept":
            kept.append(rec)
        elif route != "reject":
            new = dict(rec)
            new["overflow_route"] = route
            kept.append(new)
    ctx.artifacts["code_token_routes"] = dict(sorted(routed.items()))
    return kept, f"kept {len(kept)}/{len(records)} (code-token budget {maximum}, {policy})"


def _stage_curate_tiers(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.curation.complexity_tiers` -- DST complexity stratification."""
    from harnesscad.data.dataengine.curation.complexity_tiers import (
        ComplexitySample, partition, tier_counts,
    )

    if not records:
        return [], "no records"
    samples = [
        ComplexitySample(
            sample_id=_rid(rec, i),
            nl_length=float(len(str(rec.get("prompt") or "").split())),
            geom=float(len(_numeric_features(rec))),
            ops=float(len(_ops(rec))),
        )
        for i, rec in enumerate(records)
    ]
    scored = partition(samples)
    by_id = {s.sample_id: s for s in scored}
    out = []
    for i, rec in enumerate(records):
        s = by_id.get(_rid(rec, i))
        new = dict(rec)
        if s is not None:
            new["tier"] = s.tier.value if hasattr(s.tier, "value") else str(s.tier)
            new["complexity"] = s.complexity
        out.append(new)
    ctx.artifacts["complexity_tiers"] = {
        (t.value if hasattr(t, "value") else str(t)): n
        for t, n in sorted(tier_counts(scored).items(), key=lambda kv: str(kv[0]))
    }
    return out, f"tiered {len(scored)} records"


def _stage_curate_split(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """Stratified, seeded train/val/test split (deterministic; no wall clock).

    Records are grouped by ``tier`` (or a single stratum when untiered), sorted by
    id, shuffled with a seed derived from ``Context.seed``, then cut by the
    configured ratios. The leakage audit that follows
    (``audit.leakage``) is what proves this split honest.
    """
    if not records:
        return [], "no records"
    ratios = list(ctx.splits)
    strata: Dict[str, List[Tuple[str, int]]] = {}
    for i, rec in enumerate(records):
        strata.setdefault(str(rec.get("tier", "")), []).append((_rid(rec, i), i))
    assignment: Dict[int, str] = {}
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        rng = ctx.rng(f"curate.split|{stratum}")
        rng.shuffle(members)
        n = len(members)
        cut = 0
        for j, (name, frac) in enumerate(ratios):
            take = n - cut if j == len(ratios) - 1 else int(round(n * frac))
            for _, idx in members[cut:cut + take]:
                assignment[idx] = name
            cut += take
    out = []
    for i, rec in enumerate(records):
        new = dict(rec)
        new["split"] = assignment.get(i, ratios[0][0])
        out.append(new)
    return out, f"split {len(records)} records over {len(strata)} strata"


def _stage_curate_subset(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.curation.scale_dedup` -- farthest-point down-selection to a budget.

    Only runs when ``Context.target_size`` is set; otherwise the full corpus stays.
    """
    from harnesscad.data.dataengine.curation.scale_dedup import select_training_subset

    if ctx.target_size <= 0 or len(records) <= ctx.target_size:
        return list(records), "no target size (or corpus already within it)"
    kept = select_training_subset(records, ctx.target_size, ctx.seed)
    return list(kept), f"down-selected {len(kept)}/{len(records)} (farthest-point)"


# --- augment ----------------------------------------------------------------

def _stage_augment_parametric(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`datagen.parametric_augment` -- mirror / transpose / perturb the op stream.

    Structure-preserving variants (same ops, same ids, only the numbers move), so
    every variant is still a buildable op stream on the same backend.
    """
    from harnesscad.data.datagen.parametric_augment import augment
    from harnesscad.data.datagen.pipeline import Sample

    out = list(records)
    made = 0
    for i, rec in enumerate(records):
        ops = _ops(rec)
        if not ops:
            continue
        sample = Sample(
            brief=str(rec.get("prompt") or ""),
            generator=str(rec.get("generator") or "unknown"),
            params=dict(rec.get("params") or {}),
            ops=[dict(op) for op in ops],
            digest=str(rec.get("digest") or ""),
            summary=dict(rec.get("summary") or {}),
        )
        for v, variant in enumerate(augment(sample, ctx.seed + i)):
            new = dict(rec)
            new["id"] = f"{_rid(rec, i)}-aug{v}"
            new["ops"] = [dict(op) for op in variant.ops]
            new["params"] = dict(variant.params)
            new["features"] = list(_numeric_features({"ops": variant.ops}))
            new["augmented_from"] = _rid(rec, i)
            new.pop("digest", None)
            out.append(new)
            made += 1
    return out, f"added {made} parametric variants"


def _stage_augment_geometric_points(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.augment.geometric_augment` -- caption-invariant point augmentation.

    Only for records carrying a 2D ``points`` list (GeoCAD stage-1 shape); a
    record without one is left alone.
    """
    from harnesscad.data.dataengine.augment.geometric_augment import augment_batch, policy_for_branch

    count = int(ctx.option("augment.geometric_points", "count", 1))
    branch = str(ctx.option("augment.geometric_points", "branch", "shape"))
    out = list(records)
    made = 0
    for i, rec in enumerate(records):
        pts = rec.get("points")
        if not isinstance(pts, list) or not pts:
            continue
        rng = ctx.rng(f"augment.geometric_points|{_rid(rec, i)}")
        verts = [(float(p[0]), float(p[1])) for p in pts]
        for v, batch in enumerate(augment_batch(verts, rng, count, policy_for_branch(branch))):
            new = dict(rec)
            new["id"] = f"{_rid(rec, i)}-geo{v}"
            new["points"] = [list(p) for p in batch]
            new["augmented_from"] = _rid(rec, i)
            out.append(new)
            made += 1
    return out, f"added {made} geometric point variants"


def _stage_augment_primitive_noise(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`datagen.primitive_noise` -- Vitruvion truncated-normal sketch noise.

    Needs quantised sketch ``entities``; records without them pass through.
    """
    from harnesscad.data.datagen.primitive_noise import noisify_sketch

    out = list(records)
    made = 0
    for i, rec in enumerate(records):
        entities = rec.get("entities")
        if not isinstance(entities, list) or not entities:
            continue
        new = dict(rec)
        new["id"] = f"{_rid(rec, i)}-noise"
        new["entities"] = noisify_sketch(entities, seed=ctx.seed + i)
        new["augmented_from"] = _rid(rec, i)
        out.append(new)
        made += 1
    return out, f"added {made} noised sketches"


# --- reward -----------------------------------------------------------------

def _stage_reward_executability(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.reward.executability_reward` -- RIVAL reward: exec + IoU + judged eval.

    ``R = r_exec + lambda_g * r_geom(IoU) + lambda_e * r_eval(score)``. A different
    function of a different set of signals from ``reward.geometry_semantics`` --
    the two are never averaged; a preset selects one.
    """
    from harnesscad.data.dataengine.reward.executability_reward import (
        r_eval, r_geom, total_reward,
    )

    out = []
    scored = 0
    for rec in records:
        new = dict(rec)
        executes = bool(rec.get("ok", False))
        inter = float(rec.get("intersection", 0.0) or 0.0)
        union = float(rec.get("union", 0.0) or 0.0)
        geom = r_geom(inter, union) if union > 0 else 0.0
        evaluation = r_eval(float(rec.get("judge_score", 0.0) or 0.0),
                            tuple(rec.get("failures") or ()))
        value = total_reward(executes, geom, evaluation)
        rewards = dict(new.get("reward") or {})
        rewards["executability"] = value
        new["reward"] = rewards
        out.append(new)
        scored += 1
    return out, f"scored {scored} records (executability reward)"


def _stage_reward_geometry_semantics(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.reward.geometry_semantics_reward` -- RIVAL reward: unified geo/semantic.

    ``R = l1 * format(text) + l2 * phi(similarity) * IoU`` -- gated on a semantic
    similarity the executability reward never looks at. Needs ``iou`` + ``similarity``;
    a record without them keeps whatever reward it already had.
    """
    from harnesscad.data.dataengine.reward.geometry_semantics_reward import unified_reward

    out = []
    scored = 0
    for rec in records:
        new = dict(rec)
        if "iou" in rec and "similarity" in rec:
            value = unified_reward(float(rec["iou"]), float(rec["similarity"]),
                                   str(rec.get("code") or ""))
            rewards = dict(new.get("reward") or {})
            rewards["geometry_semantics"] = value
            new["reward"] = rewards
            scored += 1
        out.append(new)
    return out, f"scored {scored}/{len(records)} records (geometry-semantics reward)"


# --- preference / selftrain -------------------------------------------------

def _stage_preference_dpo(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.preference.dpo_pairs` -- DPO rows from same-prompt candidates.

    Candidates are grouped by prompt; within a group every (chosen, rejected) pair
    is emitted from the SELECTED reward (option ``reward``, default the first
    reward key on the record). Ties yield no pair. Rows land in
    ``artifacts["dpo"]`` -- they are training rows, not corpus records.
    """
    from harnesscad.data.dataengine.preference.dpo_pairs import all_preference_pairs, to_dpo_records

    key = ctx.option("preference.dpo", "reward", None)
    groups: Dict[str, List[dict]] = {}
    for i, rec in enumerate(records):
        rewards = rec.get("reward") or {}
        if not isinstance(rewards, dict) or not rewards:
            continue
        rkey = key or sorted(rewards)[0]
        if rkey not in rewards:
            continue
        groups.setdefault(str(rec.get("prompt") or ""), []).append({
            "id": _rid(rec, i),
            "code": str(rec.get("code") or ""),
            "reward": float(rewards[rkey]),
        })
    rows: List[dict] = []
    for prompt in sorted(groups):
        cands = groups[prompt]
        if len(cands) < 2:
            continue
        rows.extend(to_dpo_records(all_preference_pairs(cands), prompt=prompt))
    ctx.artifacts["dpo"] = rows
    return list(records), f"built {len(rows)} DPO rows from {len(groups)} prompt groups"


def _stage_selftrain_confidence(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.selftrain.confidence_score` -- rank pseudo-labels by confidence.

    Needs a ``chamfer`` distance per record (the self-training signal); records
    without one are passed through unranked.
    """
    from harnesscad.data.dataengine.selftrain.confidence_score import confidence_score

    out = []
    scored = 0
    for rec in records:
        new = dict(rec)
        if "chamfer" in rec:
            new["confidence"] = confidence_score(
                float(rec["chamfer"]), bool(rec.get("ok", False)),
                length=len(str(rec.get("code") or "")))
            scored += 1
        out.append(new)
    return out, f"scored {scored}/{len(records)} pseudo-labels"


# --- audit ------------------------------------------------------------------

def _stage_audit_bias(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.audit.bias` -- provenance coverage / imbalance (never mutates)."""
    from harnesscad.data.dataengine.audit.bias import audit_bias

    report = audit_bias(records)
    ctx.artifacts["bias"] = report.to_dict()
    return list(records), f"{len(report.warnings)} coverage warning(s)"


def _stage_audit_command_balance(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.audit.command_balance` -- op-distribution imbalance + rare coverage."""
    from harnesscad.data.dataengine.audit.command_balance import command_balance

    seqs = [list(_commands(rec)) for rec in records if _ops(rec)]
    if not seqs:
        ctx.artifacts["command_balance"] = {}
        return list(records), "no op streams to balance"
    bal = command_balance(seqs)
    ctx.artifacts["command_balance"] = {
        "counts": dict(sorted(bal.counts.items())),
        "frequencies": dict(sorted(bal.frequencies.items())),
        "rare_commands": list(bal.rare_commands),
        "rare_coverage": bal.rare_coverage,
    }
    return list(records), f"{len(bal.rare_commands)} rare command(s)"


def _stage_audit_leakage(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.schemas.script_record` -- split-leakage audit over prompt/script digests.

    Only records that carry BOTH a split and a script are audited: leakage is
    undefined without them. A leak means one prompt/template/artifact digest
    appears in two splits -- the failure mode that invalidates every number a
    benchmark then reports.
    """
    from harnesscad.data.dataengine.schemas.script_record import CFSCRecord, audit_leakage

    rows = []
    for i, rec in enumerate(records):
        code = rec.get("code")
        if not rec.get("split") or not isinstance(code, str) or not code.strip():
            continue
        rows.append(CFSCRecord(
            id=_rid(rec, i),
            prompt=str(rec.get("prompt") or ""),
            script=code,
            artifact_digest=str(rec.get("digest") or _digest(rec.get("ops"))),
            family=str(rec.get("generator") or ""),
            template_digest=_digest(list(_commands(rec))),
            parameters=tuple(sorted((k, v) for k, v in (rec.get("params") or {}).items())),
            dimension=str(rec.get("dimension") or "3d"),
            # The schema admits exactly {"none", "drafting"}; a record that does not
            # declare one carries no drafting annotations, so "none" is the truth.
            annotation_mode=(str(rec.get("annotation_mode"))
                             if rec.get("annotation_mode") in ("none", "drafting")
                             else "none"),
            legal=True,
            built=bool(rec.get("ok", False)),
            roundtrip=bool(rec.get("roundtrip", False)),
            split=str(rec["split"]),
        ))
    if not rows:
        ctx.artifacts["leakage"] = {"audited": 0, "leaks": []}
        return list(records), "no split+script records to audit"
    leaks = audit_leakage(rows)
    ctx.artifacts["leakage"] = {
        "audited": len(rows),
        "leaks": [list(leak) for leak in leaks],
    }
    return list(records), f"{len(leaks)} leak(s) over {len(rows)} records"


# --- emit -------------------------------------------------------------------

def _trajectory_from_dict(payload: dict):
    """Rehydrate a :class:`Trajectory` from the dict a record carries.

    The record must stay JSON-able, so ``ingest.session_trace`` stores the
    trajectory as ``Trajectory.to_dict()``; the exporters need the object back.
    This is the exact inverse -- no field is invented.
    """
    from harnesscad.data.dataengine.trace.trajectory import Action, Step, SubGoal, Trajectory

    steps = []
    for sd in payload.get("steps") or []:
        action = sd.get("action") or {}
        steps.append(Step(
            index=int(sd.get("index", 0)),
            state_before=dict(sd.get("state_before") or {}),
            action=Action(reasoning=str(action.get("reasoning", "")),
                          tool_call=dict(action.get("tool_call") or {})),
            reward=float(sd.get("reward", 0.0)),
            state_after=dict(sd.get("state_after") or {}),
            outcome=str(sd.get("outcome", "")),
            run_id=sd.get("run_id"),
            sub_goal=sd.get("sub_goal"),
            diagnostics=list(sd.get("diagnostics") or []),
        ))
    labels = [SubGoal(index=int(g.get("index", 0)), label=str(g.get("label", "")),
                      reached=bool(g.get("reached", False)),
                      reward=float(g.get("reward", 0.0)), run_id=g.get("run_id"))
              for g in payload.get("sub_goal_labels") or []]
    return Trajectory(steps=steps, final_reward=float(payload.get("final_reward", 0.0)),
                      sub_goal_labels=labels, prompt=payload.get("prompt"),
                      plan=payload.get("plan"), metadata=dict(payload.get("metadata") or {}),
                      run_ids=list(payload.get("run_ids") or []))

def _stage_emit_trace_export(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.trace.export` -- STaR / DPO / GRPO rows + the flywheel metrics.

    These are three FORMATS of the same trajectories (not rivals): STaR keeps only
    verified successes, DPO pairs a success against a failure, GRPO keeps the
    group-relative advantage. All three are emitted; the trainer picks.
    """
    from harnesscad.data.dataengine.trace.export import (
        flywheel_metrics, to_dpo, to_grpo, to_star,
    )

    trajs = list(ctx.artifacts.get("_trajectories") or [])
    if not trajs:
        # The records may already carry their trajectories (records_from_session
        # ingests before the pipeline runs) -- rehydrate those.
        trajs = [_trajectory_from_dict(rec["trajectory"]) for rec in records
                 if isinstance(rec.get("trajectory"), dict)]
    if not trajs:
        return list(records), "no trajectories ingested; nothing to export"
    ctx.artifacts["training_rows"] = {
        "star": to_star(trajs),
        "dpo": to_dpo(trajs),
        "grpo": to_grpo(trajs),
    }
    ctx.artifacts["flywheel"] = flywheel_metrics(trajs)
    return list(records), (f"exported {len(ctx.artifacts['training_rows']['star'])} STaR / "
                           f"{len(ctx.artifacts['training_rows']['dpo'])} DPO / "
                           f"{len(ctx.artifacts['training_rows']['grpo'])} GRPO rows")


def _stage_emit_jsonl(records: List[Record], ctx: Context) -> Tuple[List[Record], str]:
    """`dataengine.trace.export` -- write the records as JSON Lines (option ``path``)."""
    from harnesscad.data.dataengine.trace.export import write_jsonl

    path = ctx.option("emit.jsonl", "path", None)
    if not path:
        return list(records), "no path given; dataset kept in memory"
    n = write_jsonl(str(path), [json.loads(json.dumps(r, sort_keys=True, default=_jsonable))
                                for r in records])
    return list(records), f"wrote {n} records to {path}"


# ---------------------------------------------------------------------------
# The stage catalogue
# ---------------------------------------------------------------------------

_STAGE_DEFS: Tuple[Tuple[str, str, str, StageFn, Tuple[str, ...]], ...] = (
    # name, kind, dotted module it adapts, fn, required record keys
    ("generate.solver_loop", "generate", "harnesscad.data.datagen.pipeline",
     _stage_generate_solver_loop, ()),
    ("ingest.session_trace", "generate", "harnesscad.data.dataengine.trace.trajectory",
     _stage_ingest_session_trace, ()),
    ("ingest.session_capture", "generate", "harnesscad.data.dataengine.trace.session_capture",
     _stage_ingest_session_capture, ("trajectory",)),

    ("annotate.features", "annotate", "harnesscad.data.pipeline",
     _stage_annotate_features, ("ops",)),
    ("annotate.code", "annotate", "harnesscad.data.datagen.cadquery_codegen",
     _stage_annotate_code, ("ops",)),
    ("annotate.code_complexity", "annotate",
     "harnesscad.data.dataengine.curation.code_complexity",
     _stage_annotate_code_complexity, ("code",)),
    ("annotate.instruction_slots", "annotate",
     "harnesscad.data.datagen.instruction_taxonomy",
     _stage_annotate_instruction_slots, ()),
    ("annotate.minimal_metadata", "annotate",
     "harnesscad.data.dataengine.annotation.minimal_metadata",
     _stage_annotate_minimal_metadata, ("metadata",)),

    ("curate.dedup_scale", "curate", "harnesscad.data.dataengine.curation.scale_dedup",
     _stage_curate_dedup_scale, ("features",)),
    ("curate.dedup_tokens", "curate", "harnesscad.data.dataengine.curation.sketch_filters",
     _stage_curate_dedup_tokens, ("ops",)),
    ("curate.dedup_outcome", "curate", "harnesscad.data.dataengine.curation.outcome_dedup",
     _stage_curate_dedup_outcome, ("outcomes",)),
    ("curate.filter_knit", "curate", "harnesscad.data.dataengine.curation.complexity_limits",
     _stage_curate_filter_knit, ("faces",)),
    ("curate.filter_code_tokens", "curate",
     "harnesscad.data.dataengine.curation.code_complexity",
     _stage_curate_filter_code_tokens, ("code",)),
    ("curate.tiers", "curate", "harnesscad.data.dataengine.curation.complexity_tiers",
     _stage_curate_tiers, ()),
    ("curate.split", "curate", "harnesscad.data.pipeline", _stage_curate_split, ()),
    ("curate.subset", "curate", "harnesscad.data.dataengine.curation.scale_dedup",
     _stage_curate_subset, ("features",)),

    ("augment.parametric", "augment", "harnesscad.data.datagen.parametric_augment",
     _stage_augment_parametric, ("ops",)),
    ("augment.geometric_points", "augment",
     "harnesscad.data.dataengine.augment.geometric_augment",
     _stage_augment_geometric_points, ("points",)),
    ("augment.primitive_noise", "augment", "harnesscad.data.datagen.primitive_noise",
     _stage_augment_primitive_noise, ("entities",)),

    ("reward.executability", "reward",
     "harnesscad.data.dataengine.reward.executability_reward",
     _stage_reward_executability, ()),
    ("reward.geometry_semantics", "reward",
     "harnesscad.data.dataengine.reward.geometry_semantics_reward",
     _stage_reward_geometry_semantics, ("iou",)),

    ("preference.dpo", "preference", "harnesscad.data.dataengine.preference.dpo_pairs",
     _stage_preference_dpo, ("reward",)),
    ("selftrain.confidence", "selftrain",
     "harnesscad.data.dataengine.selftrain.confidence_score",
     _stage_selftrain_confidence, ("chamfer",)),

    ("audit.bias", "audit", "harnesscad.data.dataengine.audit.bias",
     _stage_audit_bias, ()),
    ("audit.command_balance", "audit", "harnesscad.data.dataengine.audit.command_balance",
     _stage_audit_command_balance, ("ops",)),
    ("audit.leakage", "audit", "harnesscad.data.dataengine.schemas.script_record",
     _stage_audit_leakage, ("split",)),

    ("emit.trace_export", "emit", "harnesscad.data.dataengine.trace.export",
     _stage_emit_trace_export, ()),
    ("emit.jsonl", "emit", "harnesscad.data.dataengine.trace.export",
     _stage_emit_jsonl, ()),
)

_STAGES: Optional[Dict[str, Stage]] = None


def _summaries() -> Dict[str, str]:
    """dotted -> docstring summary, straight out of the static AST index."""
    return {e.dotted: e.summary for e in capability_registry.index()}


def _build_stages() -> Dict[str, Stage]:
    summaries = _summaries()
    out: Dict[str, Stage] = {}
    for name, kind, dotted, fn, requires in _STAGE_DEFS:
        out[name] = Stage(name=name, kind=kind, dotted=dotted, fn=fn,
                          summary=summaries.get(dotted, ""), requires=requires)
    return out


def _stage_map() -> Dict[str, Stage]:
    global _STAGES
    if _STAGES is None:
        _STAGES = _build_stages()
    return _STAGES


def stages(kind: Optional[str] = None) -> Tuple[Stage, ...]:
    """Every adapted stage, deterministically ordered by (kind, name)."""
    items = list(_stage_map().values())
    if kind is not None:
        items = [s for s in items if s.kind == kind]
    items.sort(key=lambda s: (_KIND_ORDER.get(s.kind, len(KINDS)), s.name))
    return tuple(items)


def stage(name: str) -> Stage:
    try:
        return _stage_map()[name]
    except KeyError:
        raise KeyError(f"no such stage: {name!r} "
                       f"(known: {', '.join(s.name for s in stages())})") from None


def kinds() -> Tuple[str, ...]:
    return KINDS


def adapted_modules() -> Tuple[str, ...]:
    """The data modules this seam actually calls (deduplicated, sorted)."""
    return tuple(sorted({s.dotted for s in stages()
                         if s.dotted != "harnesscad.data.pipeline"}))


def unadapted() -> Tuple[str, ...]:
    """Data modules in the static index that no stage binds yet.

    Discovery, not silence: an orphan the seam does not honestly call stays
    listed here rather than being papered over with a fake call site.
    """
    bound = set(adapted_modules())
    out = []
    for pkg in DATA_PACKAGES:
        for entry in capability_registry.find(package=pkg):
            if entry.dotted not in bound:
                out.append(entry.dotted)
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
# Rival families -- the reason presets exist.
# ---------------------------------------------------------------------------

#: Stages inside one family answer the SAME question under DIFFERENT, mutually
#: incompatible definitions. Running two of them into one dataset would silently
#: blend two protocols. A pipeline may select at most one member of each family.
RIVAL_FAMILIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("dedup", (
        "curate.dedup_scale",     # scale-invariant signature (a scale variant IS a dup)
        "curate.dedup_tokens",    # exact op-token identity   (a scale variant is NOT a dup)
        "curate.dedup_outcome",   # identical benchmark outcome vectors
    )),
    ("complexity_filter", (
        "curate.filter_knit",         # KnitCAD B-rep limits (faces/contacts/solids)
        "curate.filter_code_tokens",  # code-token budget + overflow routing
    )),
    ("reward", (
        "reward.executability",       # exec + IoU + judged eval
        "reward.geometry_semantics",  # format + phi(similarity) * IoU
    )),
    ("augment", (
        "augment.parametric",         # op-stream mirror/transpose/perturb
        "augment.primitive_noise",    # truncated-normal sketch-primitive noise
    )),
)


def rivals() -> Dict[str, Tuple[str, ...]]:
    """family -> the mutually-exclusive stages in it (never run two into one dataset)."""
    return {name: members for name, members in RIVAL_FAMILIES}


class RivalBlendError(ValueError):
    """A pipeline tried to select two rival stages -- their outputs are not comparable."""


def _rival_conflicts(names: Sequence[str]) -> List[Tuple[str, Tuple[str, ...]]]:
    chosen = set(names)
    conflicts = []
    for family, members in RIVAL_FAMILIES:
        hit = tuple(sorted(chosen.intersection(members)))
        if len(hit) > 1:
            conflicts.append((family, hit))
    return conflicts


# ---------------------------------------------------------------------------
# Pipelines + presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pipeline:
    """A named, rival-free, ordered selection of stages."""

    name: str
    description: str
    stage_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        for n in self.stage_names:
            stage(n)  # raises KeyError on an unknown stage
        conflicts = _rival_conflicts(self.stage_names)
        if conflicts:
            raise RivalBlendError(
                f"pipeline {self.name!r} selects rival stages that must never be blended: "
                + "; ".join(f"{fam}: {', '.join(hit)}" for fam, hit in conflicts))

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "stages": list(self.stage_names)}


_PRESET_DEFS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("text2cad",
     "Text-to-CAD SFT corpus: emit code, bin it, EXACT-token dedup (a scale variant "
     "is a distinct design here), code-token budget, complexity tiers, stratified "
     "split, leakage audit.",
     ("annotate.features", "annotate.code", "annotate.code_complexity",
      "annotate.instruction_slots", "curate.dedup_tokens", "curate.filter_code_tokens",
      "curate.tiers", "curate.split", "audit.bias", "audit.command_balance",
      "audit.leakage", "emit.jsonl")),

    ("lowdata",
     "Low-data protocol: SCALE-INVARIANT dedup (a scale variant is a duplicate here -- "
     "the deliberate opposite of text2cad) plus farthest-point down-selection to "
     "Context.target_size.",
     ("annotate.features", "annotate.code", "curate.dedup_scale", "curate.subset",
      "curate.tiers", "curate.split", "audit.command_balance", "emit.jsonl")),

    ("preference",
     "Preference corpus: score candidates with the EXECUTABILITY reward and pair them "
     "into DPO rows. Never run the geometry-semantics reward into the same dataset.",
     ("annotate.features", "annotate.code", "reward.executability", "preference.dpo",
      "curate.tiers", "curate.split", "emit.jsonl")),

    ("preference_semantic",
     "The rival preference corpus: the same DPO construction over the "
     "GEOMETRY-SEMANTICS reward (format + phi(similarity) * IoU).",
     ("annotate.features", "annotate.code", "reward.geometry_semantics",
      "preference.dpo", "curate.tiers", "curate.split", "emit.jsonl")),

    ("flywheel",
     "The harness's own flywheel: fold finished HarnessSession traces into "
     "trajectories, capture the consented op decisions, and emit STaR / DPO / GRPO "
     "training rows.",
     ("ingest.session_trace", "ingest.session_capture", "annotate.features",
      "annotate.code", "curate.tiers", "curate.split", "audit.command_balance",
      "emit.trace_export", "emit.jsonl")),

    ("bootstrap",
     "Cold start: generate verified parts through the solver-in-the-loop generator, "
     "augment them parametrically, then curate and split.",
     ("generate.solver_loop", "annotate.features", "annotate.code",
      "augment.parametric", "curate.dedup_tokens", "curate.tiers", "curate.split",
      "audit.bias", "audit.command_balance", "audit.leakage", "emit.jsonl")),
)

_PRESETS: Optional[Dict[str, Pipeline]] = None


def _preset_map() -> Dict[str, Pipeline]:
    global _PRESETS
    if _PRESETS is None:
        _PRESETS = {name: Pipeline(name, desc, tuple(names))
                    for name, desc, names in _PRESET_DEFS}
    return _PRESETS


def presets() -> Tuple[str, ...]:
    return tuple(sorted(_preset_map()))


def preset(name: str) -> Pipeline:
    try:
        return _preset_map()[name]
    except KeyError:
        raise KeyError(f"no such pipeline: {name!r} "
                       f"(known: {', '.join(presets())})") from None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_stage(s: Stage, records: List[Record], ctx: Context) -> Tuple[List[Record], StageResult]:
    """Run one stage. NEVER raises: a stage that blows up leaves the records alone."""
    n_in = len(records)
    try:
        out, note = s.fn(list(records), ctx)
    except Exception as exc:  # noqa: BLE001 - THE point of this dispatcher
        return list(records), StageResult(
            name=s.name, kind=s.kind, dotted=s.dotted, status="error",
            n_in=n_in, n_out=n_in, error=f"{type(exc).__name__}: {exc}")
    out = [r for r in out if isinstance(r, dict)]
    return out, StageResult(name=s.name, kind=s.kind, dotted=s.dotted, status="ok",
                            n_in=n_in, n_out=len(out), note=note)


def run_pipeline(stage_names: Sequence[str], records: Sequence[Record],
                 ctx: Optional[Context] = None, name: str = "custom") -> Dataset:
    """Run an explicit stage list. Rival-free by check, deterministic by construction."""
    conflicts = _rival_conflicts(list(stage_names))
    if conflicts:
        raise RivalBlendError(
            f"pipeline {name!r} would blend rival stages: "
            + "; ".join(f"{fam}: {', '.join(hit)}" for fam, hit in conflicts))
    ctx = ctx if ctx is not None else Context()
    current = [dict(r) for r in records]
    results: List[StageResult] = []
    for n in stage_names:
        current, result = run_stage(stage(n), current, ctx)
        results.append(result)
    artifacts = {k: v for k, v in sorted(ctx.artifacts.items()) if not k.startswith("_")}
    return Dataset(pipeline=name, seed=ctx.seed, records=current,
                   artifacts=artifacts, results=results)


def run_preset(name: str, records: Sequence[Record],
               ctx: Optional[Context] = None) -> Dataset:
    """Run a named preset pipeline over ``records``."""
    p = preset(name)
    return run_pipeline(p.stage_names, records, ctx, name=p.name)


# ---------------------------------------------------------------------------
# The flywheel entry point: a HarnessSession run -> training records
# ---------------------------------------------------------------------------

def records_from_session(events: Iterable[dict], prompt: str = "", plan: Any = None,
                         session_id: str = "session-0",
                         metadata: Optional[Dict[str, Any]] = None) -> List[Record]:
    """Turn the trace events a HarnessSession emitted into pipeline records.

    ``events`` is exactly what ``core.trace.InMemoryTracer.events`` (or a JSONL
    trace file) contains. The result is a one-record list ready to be fed to
    :func:`run_preset` with the ``flywheel`` preset::

        tracer = InMemoryTracer()
        session = HarnessSession(StubBackend(), tracer=tracer)
        session.apply_ops(ops)
        recs = records_from_session(tracer.events, prompt="a 20x10 plate")
        ds = run_preset("flywheel", recs, Context(seed=1))
        ds.artifacts["training_rows"]["star"]     # SFT rows from the verified run
    """
    ctx = Context(options={"ingest.session_trace": {"sessions": [{
        "id": session_id, "prompt": prompt, "plan": plan,
        "events": list(events), "metadata": dict(metadata or {}),
    }]}})
    out, _ = _stage_ingest_session_trace([], ctx)
    return out


# ---------------------------------------------------------------------------
# CLI (wired into core.cli as `harnesscad dataset`)
# ---------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every discovered stage")
    parser.add_argument("--presets", action="store_true",
                        help="list the named pipelines and the stages each selects")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival stage families that must never be blended")
    parser.add_argument("--unadapted", action="store_true",
                        help="list data modules with no stage yet")
    parser.add_argument("--kind", default=None, choices=list(KINDS),
                        help="filter --list by stage kind")
    parser.add_argument("--pipeline", default=None, help="run this named pipeline")
    parser.add_argument("--input", default=None,
                        help="path to a JSON array of records for --pipeline")
    parser.add_argument("--out", default=None, help="write the emitted records as JSONL")
    parser.add_argument("--seed", type=int, default=0, help="the run seed (default 0)")
    parser.add_argument("--target-size", type=int, default=0, dest="target_size",
                        help="down-selection budget for curate.subset (0 = keep all)")
    parser.add_argument("--generate", type=int, default=0,
                        help="generate N verified samples first (bootstrap pipeline)")
    parser.add_argument("--json", action="store_true", help="print the dataset as JSON")


def _load_records(path: str) -> List[Record]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = payload.get("records", [])
    if not isinstance(payload, list):
        raise ValueError("input must be a JSON array of records, or an object "
                         "with a 'records' array")
    return [r for r in payload if isinstance(r, dict)]


def run(args: argparse.Namespace) -> int:
    if getattr(args, "presets", False):
        for name in presets():
            p = preset(name)
            print(p.name)
            print(f"    {p.description}")
            for n in p.stage_names:
                print(f"    - {n:<28} {stage(n).dotted}")
        print(f"-- {len(presets())} pipelines")
        return 0

    if getattr(args, "rivals", False):
        for family, members in RIVAL_FAMILIES:
            print(f"{family}: (never run two of these into one dataset)")
            for n in members:
                print(f"    - {n:<28} {stage(n).dotted}")
        return 0

    if getattr(args, "unadapted", False):
        for dotted in unadapted():
            print(dotted)
        print(f"-- {len(unadapted())} data modules without a stage")
        return 0

    if getattr(args, "list", False) or not getattr(args, "pipeline", None):
        selected = stages(kind=getattr(args, "kind", None))
        for s in selected:
            need = f" [needs {'+'.join(s.requires)}]" if s.requires else ""
            print(f"{s.name:<30} {s.kind:<10}{need}")
            print(f"    {s.dotted}")
            if s.summary:
                print(f"    {s.summary}")
        print(f"-- {len(selected)} stages / {len(stages())} discovered / "
              f"{len(unadapted())} data modules unadapted")
        return 0

    records: List[Record] = []
    if getattr(args, "input", None):
        try:
            records = _load_records(args.input)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: could not load records from {args.input!r}: {exc}",
                  file=sys.stderr)
            return 2

    ctx = Context(seed=int(args.seed), target_size=int(args.target_size))
    if getattr(args, "generate", 0):
        ctx.options["generate.solver_loop"] = {"n": int(args.generate)}
    if getattr(args, "out", None):
        ctx.options["emit.jsonl"] = {"path": args.out}

    try:
        dataset = run_preset(args.pipeline, records, ctx)
    except (KeyError, RivalBlendError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if getattr(args, "json", False):
        print(dataset.to_json())
        return 0 if not dataset.errors() else 1

    print(f"pipeline: {dataset.pipeline}")
    print(f"seed:     {dataset.seed}")
    print(f"records:  {len(dataset.records)}")
    print(f"splits:   {json.dumps(dataset.splits(), sort_keys=True)}")
    print("stages:")
    for r in dataset.results:
        mark = "ok " if r.status == "ok" else "ERR"
        detail = r.note if r.status == "ok" else r.error
        print(f"  [{mark}] {r.name:<28} {r.n_in:>4} -> {r.n_out:<4} {detail}")
    if dataset.artifacts:
        print("artifacts:", ", ".join(sorted(dataset.artifacts)))
    return 0 if not dataset.errors() else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad dataset",
        description="data-engine pipeline: generate -> annotate -> curate -> augment -> emit")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
