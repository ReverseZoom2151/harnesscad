"""AsyncOverseer — the async event-stream monitor with authority to halt.

HARNESS_BLUEPRINT sec.12 asks for an "async overseer (watches the event stream for
loops/stagnation, authority to halt)". No real threads are required: we model it as
an event-*consuming* monitor. It plugs into a running session two ways —

  1. as a :class:`trace.Tracer` (``overseer.event(kind, run_id, data)``): drop it in
     via ``HarnessSession(backend, tracer=overseer)`` and it watches live; or
  2. by replaying a recorded stream: ``overseer.observe(event)`` per event.

Either path funnels into ``observe(event) -> Optional[Halt]``. It reuses the existing
:class:`loopdetect.LoopDetector` (fed the ops carried in the events) for oscillation
detection, and tracks digest progress for stagnation. Fully deterministic — no wall
clock, no threads — so a replayed stream always yields the same halt decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from cisp.ops import parse_op
from loopdetect import LoopDetector


@dataclass
class Halt:
    """The HALT signal the overseer emits when it decides to stop a run."""

    kind: str          # "loop" | "stagnation"
    reason: str
    run_id: Optional[str] = None
    at_event: int = -1  # index of the event that triggered the halt

    def to_dict(self) -> dict:
        return {"kind": self.kind, "reason": self.reason,
                "run_id": self.run_id, "at_event": self.at_event}


class AsyncOverseer:
    """Watches the trace event stream and halts on loops or stagnation.

    * **loop** — the same op is *rejected* ``loop_threshold`` times within a sliding
      window (the agent is stuck retrying an op the harness keeps blocking). Detected
      with a reused :class:`loopdetect.LoopDetector`.
    * **stagnation** — ``stagnation_rounds`` consecutive completed runs (``run_end``)
      leave the model digest unchanged: the agent is spending rounds without making
      any progress.

    ``observe(event)`` returns a :class:`Halt` the moment either fires (latched: once
    halted it keeps returning the same Halt), else ``None``. The overseer also
    satisfies the :class:`trace.Tracer` protocol so it can be handed straight to a
    :class:`loop.HarnessSession`.
    """

    def __init__(self, loop_window: int = 6, loop_threshold: int = 3,
                 stagnation_rounds: int = 3) -> None:
        if stagnation_rounds < 1:
            raise ValueError("stagnation_rounds must be >= 1")
        self._loop = LoopDetector(window=loop_window, threshold=loop_threshold)
        self.stagnation_rounds = stagnation_rounds
        self._last_digest: Optional[str] = None
        self._same_digest_runs = 0
        self._event_index = -1
        self.halt: Optional[Halt] = None
        self.halts: List[Halt] = []

    @property
    def halted(self) -> bool:
        return self.halt is not None

    # --- Tracer protocol (live plug-in via HarnessSession(tracer=...)) --------
    def event(self, kind: str, run_id: str, data: dict) -> None:
        """Tracer sink: fold the loop's live event into the monitor. Side-effect
        only (records any Halt into ``self.halts``); never raises into the loop."""
        self.observe({"kind": kind, "run_id": run_id, "data": data or {}})

    # --- the monitor ---------------------------------------------------------
    def observe(self, event: dict) -> Optional[Halt]:
        """Consume one event; return a Halt if the overseer decides to stop."""
        self._event_index += 1
        if self.halt is not None:
            return self.halt  # latched — already halted

        kind = event.get("kind")
        data = event.get("data") or {}
        run_id = event.get("run_id")

        if kind == "op_applied":
            # Real forward progress: a new op landed. Reset the loop window and
            # let the digest bookkeeping (below, on run_end) see fresh state.
            self._loop.reset()

        elif kind == "rejected":
            op_dict = data.get("op")
            op = _maybe_op(op_dict)
            if op is not None and self._loop.observe(op):
                return self._raise(Halt(
                    "loop",
                    f"op repeatedly rejected without change: {op_dict}",
                    run_id, self._event_index))

        elif kind == "run_end":
            digest = data.get("digest")
            if digest is not None and digest == self._last_digest:
                self._same_digest_runs += 1
                if self._same_digest_runs >= self.stagnation_rounds:
                    return self._raise(Halt(
                        "stagnation",
                        f"{self._same_digest_runs} consecutive runs left the model "
                        f"digest unchanged ({digest[:12]}...)",
                        run_id, self._event_index))
            else:
                self._same_digest_runs = 0
                self._last_digest = digest

        return None

    def observe_stream(self, events) -> Optional[Halt]:
        """Convenience: feed an iterable of events, stopping at the first Halt."""
        for ev in events:
            h = self.observe(ev)
            if h is not None:
                return h
        return None

    # --- internals -----------------------------------------------------------
    def _raise(self, halt: Halt) -> Halt:
        self.halt = halt
        self.halts.append(halt)
        return halt


def _maybe_op(op_dict):
    """Reconstruct an Op from its dict form for the LoopDetector; return None if
    the event carries no usable op (so a malformed event degrades, not crashes)."""
    if not isinstance(op_dict, dict):
        return None
    try:
        return parse_op(op_dict)
    except Exception:  # noqa: BLE001 - a bad/partial op dict must not halt the monitor
        return None
