"""One (brief, model) run of the harness's block-and-correct loop.

The loop is the harness. It is NOT reimplemented here:

*   the model is asked for ops by :class:`harnesscad.agents.agent.planner.Planner`
    (same system prompt, same op vocabulary, same JSON contract);
*   the ops are applied by :class:`harnesscad.io.surfaces.server.CISPServer` at
    ``verify_level="full"``, i.e. the transactional session plus the whole
    discovered verifier fleet;
*   the TYPED DIAGNOSTICS that come back on failure are handed straight back to
    ``Planner.plan(..., diagnostics=...)``, which is the harness's designed
    correction channel (system prompt rule 7: "fix exactly what they report and
    re-emit the full corrected op sequence").

The only thing this module adds is the outer retry bookkeeping: a FRESH session
per attempt (rule 7 says the model re-emits the *full* sequence, so replaying it
onto a session that already holds the previous attempt's geometry would double
the part), a cap on attempts, and a record of exactly what happened -- which is
what the scoreboard is made of.

Nothing in a `RunRecord` is edited by hand anywhere in this package. If a model
cannot produce a part, the record says so and the part is absent from the
showcase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.agents.agent.planner import Planner
from harnesscad.eval.showcase.briefs import Brief, grade_geometry
from harnesscad.io.surfaces.server import CISPServer

__all__ = ["Attempt", "RunRecord", "run_brief", "apply_ops", "blocking_diagnostics",
           "MAX_ATTEMPTS", "PREFLIGHT_PREFIX"]

MAX_ATTEMPTS = 3
BACKEND = "frep"
VERIFY_LEVEL = "full"

#: The KERNEL PREFLIGHT gate.
#:
#: `HarnessSession` runs the fleet advisorily: at verify_level="full" a
#: `preflight-RADIUS_TOO_LARGE` (a fillet the kernel cannot cut) comes back as a
#: WARNING and the batch still reports ok, because the fleet is designed not to
#: change the transactional semantics of the core verifiers. For a showcase that
#: SHIPS the geometry, that is too lenient: a part whose fillet the kernel
#: preflight says is impossible is not a part.
#:
#: So the showcase PROMOTES kernel-preflight findings to blocking, and feeds them
#: back to the model as the typed diagnostics they already are. This is the only
#: policy this package adds, it is applied identically to every model, and it
#: never edits a model's ops -- it only refuses to accept a bad one and asks
#: again.
PREFLIGHT_PREFIX = "preflight-"


@dataclass
class Attempt:
    """What the model produced on one turn, and what the harness said about it."""

    index: int
    ops: List[dict] = field(default_factory=list)
    #: The model's output could not be parsed into ops at all.
    parse_error: Optional[str] = None
    #: An exception from the provider (timeout, connection refused, OOM).
    call_error: Optional[str] = None
    #: The session accepted the batch (core verifiers passed).
    ok: bool = False
    #: The session accepted it AND the kernel preflight did not veto it. This is
    #: what the loop treats as success.
    accepted: bool = False
    applied: int = 0
    digest: str = ""
    diagnostics: List[dict] = field(default_factory=list)
    #: The subset of `diagnostics` that was handed back to the model.
    blocking: List[dict] = field(default_factory=list)
    rejected: Optional[dict] = None
    seconds: float = 0.0

    @property
    def error_codes(self) -> List[str]:
        """The codes the model was actually asked to fix."""
        return [d["code"] for d in self.blocking]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "ops": self.ops,
            "op_count": len(self.ops),
            "parse_error": self.parse_error,
            "call_error": self.call_error,
            "ok": self.ok,
            "accepted": self.accepted,
            "applied": self.applied,
            "digest": self.digest,
            "diagnostics": self.diagnostics,
            "blocking": self.blocking,
            "error_codes": self.error_codes,
            "rejected": self.rejected,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class RunRecord:
    """The full, auditable story of one (brief, model) pair."""

    brief_id: str
    model: str
    seed: int
    attempts: List[Attempt] = field(default_factory=list)
    solved: bool = False
    ops: List[dict] = field(default_factory=list)
    digest: str = ""
    volume_mm3: Optional[float] = None
    bbox: Optional[List[float]] = None
    grade: Dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    render: Optional[dict] = None
    #: Always False in this package. Present so the claim is machine-checkable:
    #: a hand-edited op stream is DISQUALIFIED and must be reported as such.
    hand_fixed: bool = False

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def diagnostics_seen(self) -> List[str]:
        """Every distinct ERROR code the harness threw at the model, in order."""
        seen: List[str] = []
        for a in self.attempts:
            for code in a.error_codes:
                if code not in seen:
                    seen.append(code)
            if a.parse_error and "plan-parse-error" not in seen:
                seen.append("plan-parse-error")
        return seen

    def to_dict(self) -> dict:
        return {
            "brief_id": self.brief_id,
            "model": self.model,
            "seed": self.seed,
            "solved": self.solved,
            "attempt_count": self.attempt_count,
            "diagnostics_seen": self.diagnostics_seen,
            "failure_reason": self.failure_reason,
            "digest": self.digest,
            "volume_mm3": self.volume_mm3,
            "bbox": self.bbox,
            "grade": self.grade,
            "ops": self.ops,
            "render": self.render,
            "hand_fixed": self.hand_fixed,
            "attempts": [a.to_dict() for a in self.attempts],
            "seconds": round(sum(a.seconds for a in self.attempts), 2),
        }


def apply_ops(ops: Sequence[dict], backend: str = BACKEND,
              verify_level: str = VERIFY_LEVEL):
    """Apply an op stream to a fresh session. Returns (server, result)."""
    server = CISPServer(backend=backend, verify_level=verify_level)
    result = server.applyOps([dict(o) for o in ops])
    return server, result


def blocking_diagnostics(result: Dict[str, Any]) -> List[dict]:
    """The diagnostics the model is asked to fix -- and only those.

    Two cases, and the session's own semantics separate them for us:

    *   the batch was REJECTED (``ok`` false): the session bailed out at the
        offending op and never reached the advisory fleet, so every diagnostic
        it returned is a core/backend one that genuinely blocked the build
        (`bad-ref`, `over-constrained`, ...). All of them go back.
    *   the batch was ACCEPTED: the fleet then ran, advisorily. Most of what it
        says is not actionable with the op set (`missing-metadata`: the op
        vocabulary has no way to name a part) and re-prompting a model with it
        would just push it to hallucinate ops. Only the KERNEL PREFLIGHT findings
        -- an op the geometry kernel could not actually execute -- are promoted
        to blocking and sent back.

    So the model is never asked to fix something the op set cannot express, and
    never allowed to ship geometry the kernel says is impossible.
    """
    diags = list(result.get("diagnostics") or [])
    if not result.get("ok"):
        return [d for d in diags if d.get("severity") == "error"]
    return [d for d in diags
            if str(d.get("code", "")).startswith(PREFLIGHT_PREFIX)
            and d.get("severity") in ("error", "warning")]


def _measure(server: CISPServer) -> Dict[str, Any]:
    try:
        m = server.session.backend.query("measure")
    except Exception:  # noqa: BLE001 - a query must never sink a run
        return {}
    return m or {}


def _failure_reason(attempts: List[Attempt]) -> str:
    """Why the model never got there -- the diagnostic it could not fix."""
    if not attempts:
        return "no attempts were made"
    last = attempts[-1]
    if last.call_error:
        return f"provider error: {last.call_error}"
    if last.parse_error:
        return f"never produced parseable ops: {last.parse_error}"
    errs = last.blocking
    if errs:
        d = errs[0]
        where = f" @{d['where']}" if d.get("where") else ""
        return (f"could not fix [{d['code']}] after {len(attempts)} attempts: "
                f"{d['message']}{where}")
    return "no verified solid and no error diagnostic (empty model)"


def run_brief(brief: Brief, llm: Any, model: str = "", seed: int = 0,
              max_attempts: int = MAX_ATTEMPTS, backend: str = BACKEND,
              verify_level: str = VERIFY_LEVEL) -> RunRecord:
    """Drive brief -> ops -> verify -> (diagnostics -> ops)* for one model.

    `llm` is any `agents.llm.base.LLM`; the tests inject a scripted one, the
    sweep injects a seeded ollama client. No network access happens here that
    the `llm` does not perform.
    """
    import time

    planner = Planner(llm)
    record = RunRecord(brief_id=brief.id, model=model, seed=seed)
    diagnostics: Optional[List[dict]] = None

    for i in range(1, max_attempts + 1):
        attempt = Attempt(index=i)
        started = time.monotonic()
        try:
            # state_summary is deliberately omitted: each attempt rebuilds from
            # an empty session, so the model's own re-emitted stream IS the state.
            parsed = planner.plan_parsed(brief.text, diagnostics=diagnostics)
        except Exception as exc:  # noqa: BLE001 - provider failures are results too
            attempt.call_error = f"{type(exc).__name__}: {exc}"
            attempt.seconds = time.monotonic() - started
            record.attempts.append(attempt)
            break
        attempt.seconds = time.monotonic() - started

        if not parsed.ok:
            attempt.parse_error = parsed.error
            record.attempts.append(attempt)
            # The parse error IS a typed diagnostic as far as the planner is
            # concerned -- the same channel the verifiers use.
            diagnostics = [{
                "severity": "error",
                "code": "plan-parse-error",
                "message": parsed.error or "no valid ops",
            }]
            continue

        ops = [op.to_dict() for op in parsed.ops]
        attempt.ops = ops
        server, result = apply_ops(ops, backend=backend, verify_level=verify_level)
        attempt.ok = bool(result["ok"])
        attempt.applied = int(result["applied"])
        attempt.digest = result["digest"]
        attempt.diagnostics = list(result.get("diagnostics") or [])
        attempt.blocking = blocking_diagnostics(result)
        attempt.rejected = result.get("rejected")
        measured = _measure(server) if attempt.ok else {}
        # An op stream that "applies" but leaves no solid (e.g. sketches only) is
        # not a part. The core SolidPresenceCheck only fires once a feature has
        # run, so guard it here too.
        if attempt.ok and not measured.get("volume"):
            attempt.blocking = attempt.blocking or [{
                "severity": "error",
                "code": "no-solid",
                "message": ("the op stream applied cleanly but produced no solid "
                            "with volume; run a feature (extrude/revolve) on a "
                            "sketch profile."),
            }]
        attempt.accepted = attempt.ok and not attempt.blocking
        record.attempts.append(attempt)

        if attempt.accepted:
            record.solved = True
            record.ops = ops
            record.digest = attempt.digest
            record.volume_mm3 = measured.get("volume")
            bbox = measured.get("bbox")
            record.bbox = list(bbox) if bbox else None
            record.grade = grade_geometry(
                brief, record.volume_mm3, [o.get("op", "") for o in ops])
            return record

        diagnostics = attempt.blocking or [{
            "severity": "error",
            "code": "no-solid",
            "message": "the op stream produced no verified solid",
        }]

    record.failure_reason = _failure_reason(record.attempts)
    return record
