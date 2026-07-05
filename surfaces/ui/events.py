"""Typed SSE event protocol — the UI wire contract (HARNESS_BLUEPRINT.md sec.14).

The harness spine (loop.py) emits *internal* trace events (trace.py:
run_start/op_applied/verify_result/rejected/checkpoint/run_end). This module is
the *outward-facing* layer: the eight typed events a UI consumes over Server-
Sent Events. It is deliberately framework-free — no web server, no async — so it
can feed any transport (an ``http.server`` handler, a websocket bridge, a test
harness). A caller adapts trace events into these UIEvents; this module only
owns the wire shape.

Wire format (the SSE spec: one ``event:`` field, one ``data:`` field, blank
line terminator)::

    event: <type>
    data: <json>
    <blank line>

``data`` is always a single-line JSON object so a frame never spans an
ambiguous number of ``data:`` lines. ``to_sse()`` serialises; ``parse_sse``
is the exact inverse; ``EventStream`` and ``parse_stream`` handle sequences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Iterator, List


class EventType(str, Enum):
    """The eight typed UI events. Values are the SSE ``event:`` field text.

    Subclassing ``str`` makes ``EventType.STATUS == "status"`` and lets the
    value flow straight into JSON without an explicit ``.value`` everywhere.
    """

    STATUS = "status"                        # human-readable progress line
    THINKING = "thinking"                    # reasoning trace (may be hidden)
    TOKEN = "token"                          # incremental assistant token
    TOOL_CALL = "tool_call"                  # an op/tool is about to run
    TOOL_RESULT = "tool_result"              # its result (geometry diff, etc.)
    APPROVAL_REQUIRED = "approval_required"  # Tier-3 gate: human must approve
    ACTION_REJECTED = "action_rejected"      # gate/guardrail blocked an action
    DONE = "done"                            # terminal: the run finished

    def __str__(self) -> str:  # keep f-strings clean: f"{EventType.DONE}" -> "done"
        return self.value


# Tuple of the canonical wire strings, for validation/routing without literals.
EVENT_TYPES = tuple(t.value for t in EventType)


@dataclass
class UIEvent:
    """A single typed UI event: a type + an opaque JSON-serialisable payload.

    The payload is intentionally an open ``dict`` (mirroring trace.py's opaque
    ``data``) so the token/cost/latency and geometry-diff fields the blueprint
    calls for can ride along without schema churn. Per-type factory classmethods
    give ergonomic, self-documenting construction while keeping one dataclass so
    ``to_sse``/``parse_sse`` round-trip uniformly.
    """

    type: EventType
    data: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept a raw wire string ("status") as well as an EventType.
        if not isinstance(self.type, EventType):
            self.type = EventType(self.type)

    # --- wire format ------------------------------------------------------
    def to_sse(self) -> str:
        """Serialise to one SSE frame: ``event: <type>\\ndata: <json>\\n\\n``."""
        payload = json.dumps(self.data, sort_keys=True, separators=(",", ":"))
        return f"event: {self.type.value}\ndata: {payload}\n\n"

    def to_dict(self) -> dict:
        return {"type": self.type.value, "data": self.data}

    # --- typed constructors ----------------------------------------------
    @classmethod
    def status(cls, message: str, **extra) -> "UIEvent":
        return cls(EventType.STATUS, {"message": message, **extra})

    @classmethod
    def thinking(cls, text: str, **extra) -> "UIEvent":
        return cls(EventType.THINKING, {"text": text, **extra})

    @classmethod
    def token(cls, text: str, **extra) -> "UIEvent":
        return cls(EventType.TOKEN, {"text": text, **extra})

    @classmethod
    def tool_call(cls, name: str, args: Dict, call_id: str = "", **extra) -> "UIEvent":
        return cls(EventType.TOOL_CALL,
                   {"name": name, "args": args, "call_id": call_id, **extra})

    @classmethod
    def tool_result(cls, name: str, result: Dict, call_id: str = "",
                    ok: bool = True, **extra) -> "UIEvent":
        return cls(EventType.TOOL_RESULT,
                   {"name": name, "result": result, "call_id": call_id,
                    "ok": ok, **extra})

    @classmethod
    def approval_required(cls, name: str, risk: str, preview: Dict,
                          call_id: str = "", batch: List[Dict] = None,
                          **extra) -> "UIEvent":
        data = {"name": name, "risk": risk, "preview": preview,
                "call_id": call_id, **extra}
        if batch is not None:
            data["batch"] = batch
        return cls(EventType.APPROVAL_REQUIRED, data)

    @classmethod
    def action_rejected(cls, name: str, reason: str, diagnostics: List = None,
                        **extra) -> "UIEvent":
        return cls(EventType.ACTION_REJECTED,
                   {"name": name, "reason": reason,
                    "diagnostics": diagnostics or [], **extra})

    @classmethod
    def done(cls, ok: bool = True, **extra) -> "UIEvent":
        return cls(EventType.DONE, {"ok": ok, **extra})


# --- parsing ---------------------------------------------------------------
def parse_sse(frame: str) -> UIEvent:
    """Parse one SSE frame back into a ``UIEvent`` (inverse of ``to_sse``).

    Tolerant of trailing blank lines and of ``data:`` with or without the
    single leading space the SSE spec permits. Multiple ``data:`` lines are
    joined with newlines (SSE semantics) before JSON decoding.
    """
    event_field = None
    data_lines: List[str] = []
    for raw in frame.splitlines():
        if not raw:
            continue
        if raw.startswith("event:"):
            event_field = raw[len("event:"):].strip()
        elif raw.startswith("data:"):
            chunk = raw[len("data:"):]
            if chunk.startswith(" "):
                chunk = chunk[1:]
            data_lines.append(chunk)
    if event_field is None:
        raise ValueError("SSE frame missing 'event:' field")
    data_text = "\n".join(data_lines).strip()
    data = json.loads(data_text) if data_text else {}
    return UIEvent(EventType(event_field), data)


def parse_stream(text: str) -> List[UIEvent]:
    """Parse a stream of ``\\n\\n``-separated SSE frames into UIEvents."""
    events: List[UIEvent] = []
    for block in text.split("\n\n"):
        if block.strip():
            events.append(parse_sse(block))
    return events


# --- streaming helper ------------------------------------------------------
class EventStream:
    """Turns a sequence of ``UIEvent``s into a stream of SSE wire strings.

    Framework-free: it is just an iterable of strings. A transport (HTTP SSE
    handler, websocket, file) writes each yielded chunk. ``ensure_done`` (on by
    default) appends a terminal ``done`` event if the source did not end with
    one, so every stream a UI sees is well-formed and closes cleanly.
    """

    def __init__(self, events: Iterable[UIEvent] = (), ensure_done: bool = True) -> None:
        self._events: List[UIEvent] = list(events)
        self._ensure_done = ensure_done

    def add(self, event: UIEvent) -> "UIEvent":
        self._events.append(event)
        return event

    def extend(self, events: Iterable[UIEvent]) -> None:
        self._events.extend(events)

    @property
    def events(self) -> List[UIEvent]:
        return list(self._events)

    def __iter__(self) -> Iterator[str]:
        saw_done = False
        for ev in self._events:
            saw_done = ev.type is EventType.DONE
            yield ev.to_sse()
        if self._ensure_done and not saw_done:
            yield UIEvent.done().to_sse()

    def to_sse(self) -> str:
        """The whole stream as one concatenated SSE string."""
        return "".join(self)
