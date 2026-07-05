"""Observability layer — trace / log / metrics triad over the harness spine.

This is HARNESS_BLUEPRINT.md sec.15 made concrete. It sits ON TOP of ``trace.py``
(the ``Tracer`` protocol and its ``{ts, run_id, kind, data}`` event shape) and the
events the loop (``loop.py``) emits at every decision point of the
applyOps -> regen -> verify -> checkpoint cycle:

    run_start     data: {op_count}
    op_applied    data: {op, index, digest}
    verify_result data: {ok, diagnostics}
    rejected      data: {op, reason, diagnostics}   reason in {backend-rejected, verify-failed}
    checkpoint    data: {label, index}
    run_end       data: {ok, applied, digest}

Four capabilities, matching the blueprint's triad + tooling:

  * ``Span`` / ``SpanCollector`` — spans per LLM call / tool op / state transition
    carrying tokens / cost / latency. Latency comes from an *injected clock*; the
    default is a monotonic integer counter (``trace.monotonic_counter``), so there
    is NO wall-clock dependency and tests are fully deterministic.
  * ``Metrics`` — computes the blueprint's targets from a trajectory (a list of
    trace events): task-success-rate, tool-call accuracy, recovery rate,
    escalation rate, mean trajectory efficiency — each with a confidence interval
    (Wilson for proportions, normal approximation for the efficiency mean).
  * ``FailureTaxonomy`` — classifies a failed run into
    {regen, reasoning, hallucination, loop, context-overflow, refusal}, each mapped
    to a remediation string.
  * ``replay`` / ``Replayer`` / ``load_jsonl`` — reconstructs the op sequence and
    per-op outcomes from a JsonlTracer event stream (CAD failures are semantic,
    not syntactic, so you must replay), yielding a readable trajectory report.

``report(events)`` ties spans + metrics + taxonomy into one summary dict.

Design constraints (mirroring trace.py): absolute imports, stdlib only, no
wall-clock in any default path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from trace import EVENT_KINDS, monotonic_counter

# The blueprint's sec.15 numeric targets. Kept as data so tooling can assert
# against them without re-hard-coding literals.
TARGETS: Dict[str, float] = {
    "task_success_rate": 0.85,
    "tool_call_accuracy": 0.90,
    "recovery_rate": 0.60,
    "escalation_rate": 0.15,  # this one is an UPPER bound (lower is better)
}

# Metrics where a HIGHER value is better vs. where a LOWER value is better.
_LOWER_IS_BETTER = frozenset({"escalation_rate"})

# The failure taxonomy categories from sec.15, each with its remediation.
REMEDIATION: Dict[str, str] = {
    "regen": (
        "Kernel regeneration/boolean failed: retry with adjusted parameters "
        "(never the same invalid op), fall back to a simpler modeling strategy, "
        "and roll back the offending feature via the event log."
    ),
    "reasoning": (
        "A spec/geometry predicate was unmet (wrong plan, not a bad reference): "
        "invoke the critic to reflect and re-plan the feature order / constraints "
        "before re-emitting."
    ),
    "hallucination": (
        "An op referenced a non-existent entity/edge/sketch: re-ground the op "
        "against the current feature-tree summary and re-emit with valid handles."
    ),
    "loop": (
        "Repeated identical ops with no state change: break the loop, vary "
        "parameters, raise reasoning effort, and escalate if no progress."
    ),
    "context-overflow": (
        "Token budget exceeded: prune/summarize the op history and re-stage only "
        "the active spec + constraints at the head/tail of the context."
    ),
    "refusal": (
        "The model refused or emitted an empty op: clarify the brief, relax the "
        "guardrail if it is a false positive, or escalate to a human."
    ),
}

CATEGORIES: Tuple[str, ...] = (
    "regen",
    "reasoning",
    "hallucination",
    "loop",
    "context-overflow",
    "refusal",
)


# =====================================================================
# Spans — trace/metrics triad (sec.15: "spans per LLM call, tool op,
# state transition, with tokens/cost/latency").
# =====================================================================

SPAN_KINDS: Tuple[str, ...] = ("llm", "tool", "state")


@dataclass
class Span:
    """One timed unit of work: an LLM call, a tool/kernel op, or a state
    transition. Latency is ``end - start`` in whatever unit the injected clock
    ticks (integer ticks under the default monotonic clock).
    """

    name: str
    kind: str  # one of SPAN_KINDS
    run_id: Optional[str] = None
    tokens: int = 0
    cost_usd: float = 0.0
    start: Optional[int] = None
    end: Optional[int] = None
    attributes: Dict = field(default_factory=dict)

    @property
    def latency(self) -> int:
        if self.start is None or self.end is None:
            return 0
        return self.end - self.start

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "run_id": self.run_id,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "start": self.start,
            "end": self.end,
            "latency": self.latency,
            "attributes": dict(self.attributes),
        }


class _OpenSpan:
    """Context manager returned by ``SpanCollector.span``: stamp start on enter,
    end on exit, then append to the collector. Mutate ``.tokens`` / ``.cost_usd``
    on the yielded span inside the block.
    """

    def __init__(self, collector: "SpanCollector", span: Span) -> None:
        self._collector = collector
        self.span = span

    def __enter__(self) -> Span:
        self.span.start = self._collector._clock()
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.span.end = self._collector._clock()
        self._collector.spans.append(self.span)
        return False  # never swallow exceptions


class SpanCollector:
    """Collects ``Span`` objects. Latency is derived from an injected clock so
    behaviour is deterministic and wall-clock-free by default.
    """

    def __init__(self, clock: Optional[Callable[[], int]] = None) -> None:
        self._clock = clock if clock is not None else monotonic_counter()
        self.spans: List[Span] = []

    def span(self, name: str, kind: str, run_id: Optional[str] = None,
             tokens: int = 0, cost_usd: float = 0.0, **attributes) -> _OpenSpan:
        """Open a span as a context manager: ``with sc.span(...) as s: s.tokens = ...``."""
        if kind not in SPAN_KINDS:
            raise ValueError(f"unknown span kind {kind!r}; expected {SPAN_KINDS}")
        return _OpenSpan(self, Span(name=name, kind=kind, run_id=run_id,
                                    tokens=tokens, cost_usd=cost_usd,
                                    attributes=dict(attributes)))

    def record(self, name: str, kind: str, latency: int, tokens: int = 0,
               cost_usd: float = 0.0, run_id: Optional[str] = None,
               **attributes) -> Span:
        """Record an already-completed span with a known latency (no clock use)."""
        if kind not in SPAN_KINDS:
            raise ValueError(f"unknown span kind {kind!r}; expected {SPAN_KINDS}")
        start = self._clock()
        sp = Span(name=name, kind=kind, run_id=run_id, tokens=tokens,
                  cost_usd=cost_usd, start=start, end=start + latency,
                  attributes=dict(attributes))
        self.spans.append(sp)
        return sp

    # --- aggregation -----------------------------------------------------
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.spans)

    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def total_latency(self) -> int:
        return sum(s.latency for s in self.spans)

    def by_kind(self) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for kind in SPAN_KINDS:
            group = [s for s in self.spans if s.kind == kind]
            if not group:
                continue
            out[kind] = {
                "count": len(group),
                "tokens": sum(s.tokens for s in group),
                "cost_usd": sum(s.cost_usd for s in group),
                "latency": sum(s.latency for s in group),
            }
        return out

    def aggregate(self) -> Dict:
        return {
            "count": len(self.spans),
            "tokens": self.total_tokens(),
            "cost_usd": self.total_cost(),
            "latency": self.total_latency(),
            "by_kind": self.by_kind(),
        }

    # --- construction from an event stream -------------------------------
    @classmethod
    def from_events(cls, events: Iterable[dict],
                    clock: Optional[Callable[[], int]] = None) -> "SpanCollector":
        """Synthesize spans from a recorded trace stream.

        Each op attempt (``op_applied`` / ``rejected``) becomes a ``tool`` span
        and each run (``run_start`` .. ``run_end``) a ``state`` span. Latency is
        taken from event ``ts`` deltas when present, else from a ``latency_ms``
        hook in ``data``, else 0. Tokens/cost come from the same ``data`` hooks.
        """
        collector = cls(clock=clock)
        events = list(events)
        run_start_ts: Dict[str, Optional[int]] = {}
        for ev in events:
            kind = ev.get("kind")
            data = ev.get("data") or {}
            run_id = ev.get("run_id")
            ts = ev.get("ts")
            if kind == "run_start":
                run_start_ts[run_id] = ts
            elif kind in ("op_applied", "rejected"):
                collector.spans.append(Span(
                    name=kind,
                    kind="tool",
                    run_id=run_id,
                    tokens=int(data.get("tokens", 0) or 0),
                    cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
                    start=ts,
                    end=(ts + int(data.get("latency_ms", 0) or 0))
                    if ts is not None else None,
                    attributes={"index": data.get("index"),
                                "reason": data.get("reason")},
                ))
            elif kind == "run_end":
                start = run_start_ts.get(run_id)
                collector.spans.append(Span(
                    name="run",
                    kind="state",
                    run_id=run_id,
                    start=start,
                    end=ts if ts is not None else start,
                    attributes={"ok": data.get("ok"),
                                "applied": data.get("applied")},
                ))
        return collector


# =====================================================================
# Confidence-interval helpers (stdlib math only).
# =====================================================================

def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n. Robust at the
    extremes (0%, 100%) and for small n, unlike the normal approximation.
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def normal_interval(values: List[float], z: float = 1.96) -> Tuple[float, float]:
    """Normal-approximation CI for the mean of ``values`` (sample-sd / sqrt n)."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return (mean - z * se, mean + z * se)


@dataclass
class Proportion:
    """A k-of-n proportion with a Wilson confidence interval."""

    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else 0.0

    def ci(self, z: float = 1.96) -> Tuple[float, float]:
        return wilson_interval(self.k, self.n, z)

    def to_dict(self, z: float = 1.96) -> dict:
        lo, hi = self.ci(z)
        return {"value": self.value, "k": self.k, "n": self.n, "ci": [lo, hi]}


@dataclass
class Mean:
    """A sample mean with a normal-approximation confidence interval."""

    values: List[float]

    @property
    def value(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    def ci(self, z: float = 1.96) -> Tuple[float, float]:
        return normal_interval(self.values, z)

    def to_dict(self, z: float = 1.96) -> dict:
        lo, hi = self.ci(z)
        return {"value": self.value, "n": len(self.values), "ci": [lo, hi]}


# =====================================================================
# Trajectory grouping — split a flat event stream into per-run trajectories,
# preserving order. A "trajectory" is one applyOps run's event slice.
# =====================================================================

@dataclass
class RunTrajectory:
    run_id: str
    events: List[dict]

    def of_kind(self, kind: str) -> List[dict]:
        return [e for e in self.events if e.get("kind") == kind]

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {k: 0 for k in EVENT_KINDS}
        for e in self.events:
            k = e.get("kind")
            if k in c:
                c[k] += 1
        return c

    @property
    def run_end(self) -> Optional[dict]:
        ends = self.of_kind("run_end")
        return ends[-1] if ends else None

    @property
    def ok(self) -> bool:
        end = self.run_end
        return bool(end and end["data"].get("ok"))

    @property
    def failed(self) -> bool:
        end = self.run_end
        # A run is a failure if it ended not-ok. A run with no run_end is
        # treated as incomplete (not counted as a clean success).
        return end is not None and not end["data"].get("ok")

    def backend_rejected(self) -> List[dict]:
        return [e for e in self.of_kind("rejected")
                if (e["data"].get("reason") == "backend-rejected")]

    def verify_rejected(self) -> List[dict]:
        return [e for e in self.of_kind("rejected")
                if (e["data"].get("reason") == "verify-failed")]

    def all_diagnostics(self) -> List[dict]:
        diags: List[dict] = []
        for e in self.events:
            for d in (e["data"].get("diagnostics") or []):
                diags.append(d)
        return diags


def group_runs(events: Iterable[dict]) -> List[RunTrajectory]:
    """Split a flat event stream into ordered per-run trajectories.

    Order of first appearance of each ``run_id`` is preserved, which matters for
    recovery (a failed run recovering in a *later* run).
    """
    order: List[str] = []
    buckets: Dict[str, List[dict]] = {}
    for ev in events:
        rid = ev.get("run_id")
        if rid not in buckets:
            buckets[rid] = []
            order.append(rid)
        buckets[rid].append(ev)
    return [RunTrajectory(rid, buckets[rid]) for rid in order]


# =====================================================================
# Metrics — the blueprint's sec.15 targets from a trajectory.
# =====================================================================

class Metrics:
    """Compute sec.15 targets from a list of trace events (a trajectory or a
    concatenation of several runs).

    Definitions (event-derived, deterministic):
      * task_success_rate  = ok runs / total runs.
      * tool_call_accuracy = (applied - verify_failures) / (applied + backend_rejects).
        Every op emission either applies or is backend-rejected; a verify failure
        is an applied op later rolled back, so it is a failed tool call.
      * recovery_rate      = failed runs that are followed by a later ok run
        / failed runs. (The loop stops a batch at first failure, so recovery is a
        cross-run, retry-level notion.)
      * escalation_rate    = escalated runs / total runs, where a run escalates if
        its ``run_end`` data is explicitly ``escalated`` OR it failed with no later
        recovery (a terminal, unrecovered failure = an escalation).
      * trajectory_efficiency (eta = L*/L_agent, sec.16) averaged over runs. If
        ``optimal_lengths`` maps run_id -> L*, that is used; otherwise a proxy of
        applied / attempts (fraction of emitted ops that stuck) is used.
    """

    def __init__(self, events: Iterable[dict],
                 optimal_lengths: Optional[Dict[str, int]] = None) -> None:
        self.runs = group_runs(events)
        self._optimal = optimal_lengths or {}
        self._compute()

    def _compute(self) -> None:
        runs = self.runs
        total = len(runs)

        # --- task success rate -----------------------------------------
        ok_runs = sum(1 for r in runs if r.ok)
        self.task_success_rate = Proportion(ok_runs, total)

        # --- tool-call accuracy (aggregated over all runs) -------------
        applied = sum(len(r.of_kind("op_applied")) for r in runs)
        backend_rej = sum(len(r.backend_rejected()) for r in runs)
        verify_rej = sum(len(r.verify_rejected()) for r in runs)
        attempts = applied + backend_rej
        successful = applied - verify_rej
        self.tool_call_accuracy = Proportion(successful, attempts)

        # --- recovery + escalation (cross-run) -------------------------
        failed_idx = [i for i, r in enumerate(runs) if r.failed]
        recovered = 0
        escalated = 0
        for i in failed_idx:
            later_ok = any(runs[j].ok for j in range(i + 1, total))
            explicit = bool(runs[i].run_end
                            and runs[i].run_end["data"].get("escalated"))
            if later_ok:
                recovered += 1
            if explicit or not later_ok:
                escalated += 1
        # Also count any run explicitly marked escalated even if it "succeeded".
        for i, r in enumerate(runs):
            if not r.failed and r.run_end and r.run_end["data"].get("escalated"):
                escalated += 1
        self.recovery_rate = Proportion(recovered, len(failed_idx))
        self.escalation_rate = Proportion(escalated, total)

        # --- trajectory efficiency -------------------------------------
        etas: List[float] = []
        for r in runs:
            l_applied = len(r.of_kind("op_applied"))
            l_agent = l_applied + len(r.backend_rejected()) + len(r.verify_rejected())
            if l_agent <= 0:
                continue
            opt = self._optimal.get(r.run_id)
            if opt is not None:
                etas.append(min(1.0, opt / l_agent))
            else:
                etas.append(l_applied / l_agent)
        self.trajectory_efficiency = Mean(etas)

    # --- reporting -------------------------------------------------------
    def summary(self) -> Dict[str, dict]:
        return {
            "task_success_rate": self.task_success_rate.to_dict(),
            "tool_call_accuracy": self.tool_call_accuracy.to_dict(),
            "recovery_rate": self.recovery_rate.to_dict(),
            "escalation_rate": self.escalation_rate.to_dict(),
            "trajectory_efficiency": self.trajectory_efficiency.to_dict(),
        }

    def meets_targets(self) -> Dict[str, bool]:
        vals = {
            "task_success_rate": self.task_success_rate.value,
            "tool_call_accuracy": self.tool_call_accuracy.value,
            "recovery_rate": self.recovery_rate.value,
            "escalation_rate": self.escalation_rate.value,
        }
        out: Dict[str, bool] = {}
        for name, target in TARGETS.items():
            if name in _LOWER_IS_BETTER:
                out[name] = vals[name] <= target
            else:
                out[name] = vals[name] >= target
        return out


# =====================================================================
# Failure taxonomy — classify a failed run into one of six categories.
# =====================================================================

@dataclass
class Classification:
    category: str
    remediation: str
    evidence: str
    run_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "remediation": self.remediation,
            "evidence": self.evidence,
            "run_id": self.run_id,
        }


class FailureTaxonomy:
    """Classify a failed run into {regen, reasoning, hallucination, loop,
    context-overflow, refusal} from its trace events (+ diagnostics), and map it
    to a remediation. Rule order encodes precedence: an explicit signal (refusal,
    context-overflow) wins over inferred ones (loop, then reference vs. plan).
    """

    # Substrings in diagnostic codes / reasons that point at each category.
    _REGEN_CODES = ("regen", "boolean", "non-manifold", "nonmanifold",
                    "self-intersect", "watertight", "rebuild", "empty-body")
    _HALLUCINATION_CODES = ("unknown-ref", "bad-reference", "missing",
                            "no-such", "undefined", "unresolved", "not-found",
                            "unknown-entity")
    _REASONING_CODES = ("constraint", "dof", "under-constrained",
                        "over-constrained", "predicate", "tolerance",
                        "dimension", "spec", "contract")
    _OVERFLOW_CODES = ("context-overflow", "context_overflow", "token-budget",
                       "token_budget", "overflow", "max-context")
    _REFUSAL_CODES = ("refusal", "refused", "empty-op", "declined", "abstain")

    @classmethod
    def _codes_and_reasons(cls, run: RunTrajectory) -> List[str]:
        blob: List[str] = []
        for e in run.events:
            data = e.get("data") or {}
            reason = data.get("reason")
            if reason:
                blob.append(str(reason).lower())
            if data.get("escalated"):
                blob.append("escalated")
            for d in (data.get("diagnostics") or []):
                blob.append(str(d.get("code", "")).lower())
                blob.append(str(d.get("message", "")).lower())
        return blob

    @classmethod
    def _op_signature(cls, op) -> str:
        try:
            return json.dumps(op, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return repr(op)

    @classmethod
    def _detect_loop(cls, run: RunTrajectory) -> Optional[str]:
        # A loop = the same op emitted repeatedly, or applied ops that never move
        # the state digest. Either is the sec.10 "never re-emit the same op" smell.
        # Count distinct emission ATTEMPTS only. A verify-failed rejection
        # re-carries the op that was just applied (it is the rollback of that
        # same op, not a fresh emission), so excluding it avoids a false loop.
        sigs: Dict[str, int] = {}
        for e in run.events:
            kind = e.get("kind")
            data = e.get("data") or {}
            if kind == "op_applied" or (
                    kind == "rejected" and data.get("reason") == "backend-rejected"):
                op = data.get("op")
                if op is None:
                    continue
                s = cls._op_signature(op)
                sigs[s] = sigs.get(s, 0) + 1
        repeated = max(sigs.values()) if sigs else 0
        if repeated >= 2:
            return f"same op emitted {repeated} times"
        digests = [(e.get("data") or {}).get("digest")
                   for e in run.of_kind("op_applied")]
        digests = [d for d in digests if d is not None]
        if len(digests) >= 2 and len(set(digests)) == 1:
            return "state digest never changed across applied ops"
        return None

    @classmethod
    def _match(cls, blob: List[str], needles: Tuple[str, ...]) -> Optional[str]:
        for token in blob:
            for needle in needles:
                if needle in token:
                    return needle
        return None

    @classmethod
    def classify(cls, run: RunTrajectory) -> Classification:
        rid = run.run_id
        blob = cls._codes_and_reasons(run)

        def mk(cat: str, evidence: str) -> Classification:
            return Classification(cat, REMEDIATION[cat], evidence, rid)

        # 1. Refusal — explicit, highest precedence.
        hit = cls._match(blob, cls._REFUSAL_CODES)
        if hit:
            return mk("refusal", f"refusal signal: {hit!r}")

        # 2. Context overflow — explicit budget signal.
        hit = cls._match(blob, cls._OVERFLOW_CODES)
        if hit:
            return mk("context-overflow", f"overflow signal: {hit!r}")

        # 3. Loop — structural (repetition / digest stagnation).
        loop_ev = cls._detect_loop(run)
        if loop_ev:
            return mk("loop", loop_ev)

        # 4. Hallucination — a backend-rejected op with a bad-reference code.
        hit = cls._match(blob, cls._HALLUCINATION_CODES)
        if hit:
            return mk("hallucination", f"bad-reference signal: {hit!r}")

        # 5. Regen — kernel rebuild / boolean / manifold failure.
        hit = cls._match(blob, cls._REGEN_CODES)
        if hit:
            return mk("regen", f"regen/kernel signal: {hit!r}")

        # 6. Reasoning — a spec/constraint predicate went unmet (wrong plan).
        hit = cls._match(blob, cls._REASONING_CODES)
        if hit:
            return mk("reasoning", f"spec/constraint signal: {hit!r}")

        # Fallback: an op was backend-rejected -> treat as hallucination
        # (bad handle), else a verify failure with no clear code -> reasoning.
        if run.backend_rejected():
            return mk("hallucination", "backend rejected an op (unclassified reason)")
        return mk("reasoning", "verify failed with no specific diagnostic code")

    @classmethod
    def classify_events(cls, events: Iterable[dict]) -> List[Classification]:
        """Classify every failed run in a flat event stream."""
        return [cls.classify(r) for r in group_runs(events) if r.failed]


# =====================================================================
# Replay — reconstruct the op sequence and per-op outcomes.
# =====================================================================

@dataclass
class OpOutcome:
    index: int
    op: dict
    outcome: str  # applied | rejected-backend | rolled-back
    digest: Optional[str] = None
    verify_ok: Optional[bool] = None
    checkpointed: bool = False
    diagnostics: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "op": self.op,
            "outcome": self.outcome,
            "digest": self.digest,
            "verify_ok": self.verify_ok,
            "checkpointed": self.checkpointed,
            "diagnostics": self.diagnostics,
        }


@dataclass
class RunReplay:
    run_id: str
    op_count: Optional[int]
    ops: List[OpOutcome]
    ok: Optional[bool]
    applied: Optional[int]
    digest: Optional[str]

    def op_order(self) -> List[dict]:
        """The ops in the order they were attempted."""
        return [o.op for o in self.ops]

    def applied_ops(self) -> List[dict]:
        return [o.op for o in self.ops if o.outcome == "applied"]

    def render(self) -> str:
        status = "ok" if self.ok else "FAILED"
        lines = [f"run {self.run_id}  [{status}]  "
                 f"applied={self.applied} digest={self.digest}"]
        for o in self.ops:
            tag = {
                "applied": "  + ",
                "rejected-backend": "  x ",
                "rolled-back": "  ~ ",
            }.get(o.outcome, "  ? ")
            op_name = o.op.get("op") or o.op.get("kind") or o.op.get("type") or "op"
            line = f"{tag}[{o.index}] {op_name} -> {o.outcome}"
            if o.diagnostics:
                codes = ",".join(d.get("code", "?") for d in o.diagnostics)
                line += f"  ({codes})"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "op_count": self.op_count,
            "ok": self.ok,
            "applied": self.applied,
            "digest": self.digest,
            "ops": [o.to_dict() for o in self.ops],
        }


class Replayer:
    """Reconstruct per-run op sequences and outcomes from a trace stream.

    CAD failures are semantic (a valid-looking op that produces wrong/invalid
    geometry), so a syntactic diff is not enough — you must walk the trajectory
    to see which ops applied, which were blocked, and which were rolled back.
    """

    def replay(self, events: Iterable[dict]) -> List[RunReplay]:
        out: List[RunReplay] = []
        for run in group_runs(events):
            out.append(self._replay_run(run))
        return out

    def _replay_run(self, run: RunTrajectory) -> RunReplay:
        ops: List[OpOutcome] = []
        op_count: Optional[int] = None
        ok: Optional[bool] = None
        applied: Optional[int] = None
        digest: Optional[str] = None

        for ev in run.events:
            kind = ev.get("kind")
            data = ev.get("data") or {}
            if kind == "run_start":
                op_count = data.get("op_count")
            elif kind == "op_applied":
                ops.append(OpOutcome(
                    index=data.get("index", len(ops)),
                    op=data.get("op", {}),
                    outcome="applied",
                    digest=data.get("digest"),
                ))
            elif kind == "verify_result":
                if ops:
                    ops[-1].verify_ok = data.get("ok")
                    if data.get("diagnostics"):
                        ops[-1].diagnostics += list(data["diagnostics"])
            elif kind == "rejected":
                reason = data.get("reason")
                if reason == "verify-failed" and ops:
                    # The last applied op was rolled back.
                    ops[-1].outcome = "rolled-back"
                    if data.get("diagnostics"):
                        ops[-1].diagnostics += list(data["diagnostics"])
                else:
                    # backend-rejected: this op never applied.
                    ops.append(OpOutcome(
                        index=len(ops),
                        op=data.get("op", {}),
                        outcome="rejected-backend",
                        diagnostics=list(data.get("diagnostics") or []),
                    ))
            elif kind == "checkpoint":
                if ops:
                    ops[-1].checkpointed = True
            elif kind == "run_end":
                ok = data.get("ok")
                applied = data.get("applied")
                digest = data.get("digest")

        return RunReplay(run.run_id, op_count, ops, ok, applied, digest)

    def render(self, events: Iterable[dict]) -> str:
        return "\n\n".join(r.render() for r in self.replay(events))


def replay(events: Iterable[dict]) -> List[RunReplay]:
    """Convenience: reconstruct all runs' op sequences/outcomes from a stream."""
    return Replayer().replay(list(events))


def render_trajectory(events: Iterable[dict]) -> str:
    """A readable trajectory report for a whole event stream."""
    return Replayer().render(list(events))


def load_jsonl(path: str, encoding: str = "utf-8") -> List[dict]:
    """Read a ``JsonlTracer`` output file back into a list of event dicts.

    Blank lines are skipped; each non-blank line is one ``{ts, run_id, kind,
    data}`` record.
    """
    events: List[dict] = []
    with open(path, "r", encoding=encoding) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


# =====================================================================
# report — tie spans + metrics + taxonomy into one observability summary.
# =====================================================================

def report(events: Iterable[dict],
           optimal_lengths: Optional[Dict[str, int]] = None) -> dict:
    """One observability summary for a trajectory: run counts, the sec.15 metric
    triad (with CIs and target checks), synthesized span aggregates, per-failure
    taxonomy classifications, and a compact replay of each run.
    """
    events = list(events)
    metrics = Metrics(events, optimal_lengths=optimal_lengths)
    spans = SpanCollector.from_events(events)
    runs = group_runs(events)
    replays = replay(events)

    return {
        "runs": {
            "total": len(runs),
            "ok": sum(1 for r in runs if r.ok),
            "failed": sum(1 for r in runs if r.failed),
        },
        "metrics": metrics.summary(),
        "targets": TARGETS,
        "targets_met": metrics.meets_targets(),
        "spans": spans.aggregate(),
        "failures": [c.to_dict() for c in FailureTaxonomy.classify_events(events)],
        "replay": [r.to_dict() for r in replays],
    }
