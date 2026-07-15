"""Observability / trace layer — structured events off the harness spine.

The loop (loop.py) emits typed events at every decision point of the
applyOps -> regen -> verify -> checkpoint cycle. A ``Tracer`` receives them.
This is deliberately kernel- and LLM-agnostic: events carry an opaque ``data``
dict, so the LLM layer can later fold token/cost/latency telemetry into the same
stream without touching the loop.

Design constraints (sandbox-friendly, deterministic):
  - No wall-clock at import time and none required at runtime. ``ts`` is
    OPTIONAL and ``None`` by default; a tracer may be handed an injectable
    ``clock`` (defaulting to a monotonic integer counter) when a relative
    ordering timestamp is wanted without depending on ``datetime.now``.
  - The default tracer is a no-op, so tracing is zero-cost and behaviour is
    identical to an untraced loop.

Event kinds (the loop emits exactly these):
  run_start     — a batch begins        data: {run_id, op_count}
  op_applied    — an op accepted        data: {op, digest, index}
  verify_result — verifier ran          data: {ok, diagnostics}
  rejected      — an op was blocked      data: {op, reason, diagnostics}
  checkpoint    — history checkpointed  data: {label, index}
  step_reward   — PER-OP credit          data: {index, op, reward, reason}
  run_end       — batch finished        data: {ok, applied, digest, step_rewards,
                                               mean_step_reward}

PER-STEP CREDIT ASSIGNMENT (``step_reward``)
--------------------------------------------
The loop used to be graded on its final solid alone. A six-op plan that produced
a wrong part got one scalar and attributed nothing: ops 1-4 may have been
perfect, and nothing knew. `agents/agent/tool_reward.py` implemented the
per-step (process) reward — ``R = alpha*R_ORM + beta*mean(R_step) +
gamma*R_format`` — and its only importers were a dispatch table and its own test.

The loop now emits a ``step_reward`` event for every op it decides about: 1.0 for
an op that applied AND verified, 0.0 for the op that broke the trajectory. Ops
after the break are never reached and are never punished — that is the book's
trajectory slicing (negative reward only on the first divergent op), and it is
the instrument that shows a loop being poisoned at op 3 instead of at brief 12.

Per-event token/cost/latency hooks: any tracer accepts these inside ``data``
(e.g. ``{"tokens": ..., "cost_usd": ..., "latency_ms": ...}``) as placeholders
for the LLM layer to populate later; the trace layer just records them verbatim.
"""

from __future__ import annotations

import itertools
import json
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

# The canonical event kinds the loop emits. Exposed so downstream tooling can
# validate/route without hard-coding string literals.
EVENT_KINDS = (
    "run_start",
    "op_applied",
    "verify_result",
    "rejected",
    "checkpoint",
    "step_reward",
    "run_end",
)


def monotonic_counter() -> Callable[[], int]:
    """A deterministic, wall-clock-free clock: 0, 1, 2, ... on each call."""
    counter = itertools.count()
    return lambda: next(counter)


@runtime_checkable
class Tracer(Protocol):
    """Sink for harness trace events.

    Implementations MUST be side-effect-only from the loop's perspective: an
    ``event`` call never raises into or alters the loop's control flow.
    """

    def event(self, kind: str, run_id: str, data: dict) -> None: ...


class NullTracer:
    """No-op tracer — the default. Zero cost, no state, no output."""

    def event(self, kind: str, run_id: str, data: dict) -> None:  # noqa: D401
        return None


class InMemoryTracer:
    """Records every event into ``self.events`` (a list of dicts) — for tests.

    Each recorded event is ``{"ts", "run_id", "kind", "data"}`` with ``ts`` set
    from the injected clock (a monotonic counter by default).
    """

    def __init__(self, clock: Optional[Callable[[], Optional[int]]] = None) -> None:
        self._clock = clock if clock is not None else monotonic_counter()
        self.events: List[Dict] = []

    def event(self, kind: str, run_id: str, data: dict) -> None:
        self.events.append({
            "ts": self._clock(),
            "run_id": run_id,
            "kind": kind,
            "data": data,
        })

    # --- convenience for tests ------------------------------------------
    def kinds(self) -> List[str]:
        return [e["kind"] for e in self.events]

    def of_kind(self, kind: str) -> List[Dict]:
        return [e for e in self.events if e["kind"] == kind]


class JsonlTracer:
    """Appends one JSON object per event to ``path`` (JSON Lines).

    Each line is ``{"ts", "run_id", "kind", "data"}``. ``ts`` is ``None`` by
    default (no wall-clock dependency); pass a ``clock`` to stamp a relative
    ordering value instead. The file is opened in append mode per event so the
    stream survives crashes and interleaves with nothing held open.
    """

    def __init__(self, path: str,
                 clock: Optional[Callable[[], Optional[int]]] = None,
                 encoding: str = "utf-8") -> None:
        self.path = path
        # Default clock yields None: ts is optional and off unless requested.
        self._clock = clock if clock is not None else (lambda: None)
        self._encoding = encoding

    def event(self, kind: str, run_id: str, data: dict) -> None:
        record = {
            "ts": self._clock(),
            "run_id": run_id,
            "kind": kind,
            "data": data,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with open(self.path, "a", encoding=self._encoding) as fh:
            fh.write(line + "\n")
