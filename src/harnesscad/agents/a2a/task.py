"""The A2A task lifecycle — async tasks with a guarded state machine.

Per HARNESS_BLUEPRINT.md sec.12, long geometry/meshing/FEA solves run as
*async tasks* whose progress is streamed as SSE-style status updates (or pushed
via webhooks). This module models that lifecycle in-process so remote compute is
a drop-in later:

  - ``TaskState`` — submitted -> working -> input_required -> completed / failed
    / canceled, with a legal-transition guard that *raises* on illegal moves.
  - ``Task``      — a single unit of work grouped by ``contextId`` (related tasks
    share one context), carrying an artifacts list and a status history. It emits
    SSE-style events to subscriber callbacks on every status/artifact change
    (modelled as a plain callback list — no real network).
  - ``TaskStore`` — an in-memory, pluggable registry keyed by taskId, with a
    contextId secondary index for grouping.

Timestamps follow trace.py's sandbox-friendly convention: an injectable
``clock`` (a deterministic monotonic integer counter by default) rather than a
wall-clock dependency.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from harnesscad.agents.a2a.messages import A2AMessage, Part


def monotonic_counter() -> Callable[[], int]:
    """A deterministic, wall-clock-free clock: 0, 1, 2, ... on each call.

    Mirrors trace.monotonic_counter so task timestamps order consistently with
    the trace event stream without depending on ``datetime.now``.
    """
    counter = itertools.count()
    return lambda: next(counter)


class TaskState(str, Enum):
    """The lifecycle states of an A2A task.

    Subclasses ``str`` so ``state.value`` round-trips cleanly through JSON and so
    a state compares equal to its wire string.
    """

    SUBMITTED = "submitted"
    WORKING = "working"
    # NOTE: spec wire value is hyphenated "input-required" (was "input_required",
    # a real wire bug that broke interop with conformant A2A peers).
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"


# Terminal states have no outgoing transitions. REJECTED is terminal (the agent
# declined the task outright).
TERMINAL_STATES = frozenset({
    TaskState.COMPLETED,
    TaskState.FAILED,
    TaskState.CANCELED,
    TaskState.REJECTED,
})

# The legal transition graph. ``None`` is the pre-submission origin so ``submit()``
# is itself guard-driven (nothing -> submitted). Every other edge is an explicit,
# validated move; anything not listed raises IllegalTransition.
LEGAL_TRANSITIONS: Dict[Optional[TaskState], frozenset] = {
    None: frozenset({TaskState.SUBMITTED}),
    TaskState.SUBMITTED: frozenset({
        TaskState.WORKING,
        TaskState.CANCELED,
        TaskState.FAILED,
        TaskState.REJECTED,
    }),
    TaskState.WORKING: frozenset({
        TaskState.INPUT_REQUIRED,
        TaskState.AUTH_REQUIRED,
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.CANCELED,
    }),
    TaskState.INPUT_REQUIRED: frozenset({TaskState.WORKING, TaskState.CANCELED, TaskState.FAILED}),
    TaskState.AUTH_REQUIRED: frozenset({TaskState.WORKING, TaskState.CANCELED, TaskState.FAILED}),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELED: frozenset(),
    TaskState.REJECTED: frozenset(),
    TaskState.UNKNOWN: frozenset(),
}


class IllegalTransition(Exception):
    """Raised when a task is driven along an edge not in LEGAL_TRANSITIONS."""

    def __init__(self, frm: Optional[TaskState], to: TaskState) -> None:
        self.frm = frm
        self.to = to
        frm_name = frm.value if frm is not None else "<unsubmitted>"
        super().__init__(f"illegal task transition: {frm_name} -> {to.value}")


@dataclass
class TaskStatus:
    """One entry in a task's status history.

    ``message`` optionally carries the A2AMessage that accompanied the change
    (e.g. the prompt for INPUT_REQUIRED or the error detail for FAILED). ``ts``
    is stamped from the task's injected clock.
    """

    state: TaskState
    message: Optional[A2AMessage] = None
    ts: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "message": self.message.to_dict() if self.message is not None else None,
            "ts": self.ts,
        }


# SSE-style event kinds a task emits to subscribers. Aligns with the typed SSE
# protocol in the blueprint (sec.14) and trace.py's event-dict shape.
EVENT_STATUS_UPDATE = "status_update"
EVENT_ARTIFACT_UPDATE = "artifact_update"

# Wire-level (spec) event kinds are hyphenated: "status-update"/"artifact-update"
# (TaskStatusUpdateEvent / TaskArtifactUpdateEvent). We keep the internal
# underscore names above for back-compat with existing subscribers and map to the
# spec strings only at the network seam via ``to_wire_event``.
WIRE_EVENT_KINDS = {
    EVENT_STATUS_UPDATE: "status-update",
    EVENT_ARTIFACT_UPDATE: "artifact-update",
}


def to_wire_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Map an internal event dict to its spec wire shape (hyphenated ``kind``)."""
    wire = dict(event)
    wire["kind"] = WIRE_EVENT_KINDS.get(event.get("kind"), event.get("kind"))
    return wire

# A subscriber is any callable that receives one event dict. This is the
# in-process stand-in for an SSE stream / webhook POST — no real network.
Subscriber = Callable[[Dict[str, Any]], None]


class Task:
    """A single async unit of work with a guarded lifecycle.

    Construct a task (state is ``None`` = not yet submitted), then drive it with
    the ``submit``/``start``/``require_input``/``complete``/``fail``/``cancel``
    helpers. Each helper routes through the transition guard and, on success,
    appends to ``history`` and emits an SSE-style ``status_update`` event to all
    subscribers. ``add_artifact`` appends to ``artifacts`` and emits an
    ``artifact_update`` event.
    """

    def __init__(
        self,
        taskId: Optional[str] = None,
        contextId: Optional[str] = None,
        clock: Optional[Callable[[], Optional[int]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.taskId: str = taskId if taskId is not None else uuid.uuid4().hex
        self.contextId: str = contextId if contextId is not None else uuid.uuid4().hex
        self._clock = clock if clock is not None else monotonic_counter()
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.state: Optional[TaskState] = None
        self.history: List[TaskStatus] = []
        self.artifacts: List[Part] = []
        self._subscribers: List[Subscriber] = []

    # --- subscription (SSE / webhook stand-in) --------------------------
    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """Register a callback for lifecycle events. Returns an unsubscribe fn."""
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:  # pragma: no cover - already removed
                pass

        return _unsubscribe

    def _emit(self, event: Dict[str, Any]) -> None:
        # Side-effect-only from the task's perspective (cf. trace.Tracer): a
        # subscriber must not alter control flow. We deliberately do not guard
        # against subscriber exceptions here so test/wiring bugs surface loudly.
        for cb in list(self._subscribers):
            cb(event)

    # --- the guard ------------------------------------------------------
    def _transition(self, to: TaskState, message: Optional[A2AMessage] = None) -> "Task":
        allowed = LEGAL_TRANSITIONS.get(self.state, frozenset())
        if to not in allowed:
            raise IllegalTransition(self.state, to)
        self.state = to
        entry = TaskStatus(state=to, message=message, ts=self._clock())
        self.history.append(entry)
        self._emit({
            "kind": EVENT_STATUS_UPDATE,
            "taskId": self.taskId,
            "contextId": self.contextId,
            "ts": entry.ts,
            "data": {
                "state": to.value,
                "message": message.to_dict() if message is not None else None,
                "final": to in TERMINAL_STATES,
            },
        })
        return self

    # --- lifecycle helpers ----------------------------------------------
    def submit(self, message: Optional[A2AMessage] = None) -> "Task":
        """Move an unsubmitted task to SUBMITTED (the lifecycle entry point)."""
        return self._transition(TaskState.SUBMITTED, message)

    def start(self, message: Optional[A2AMessage] = None) -> "Task":
        """Begin work: SUBMITTED/INPUT_REQUIRED -> WORKING."""
        return self._transition(TaskState.WORKING, message)

    def require_input(self, message: Optional[A2AMessage] = None) -> "Task":
        """Pause for the caller: WORKING -> INPUT_REQUIRED."""
        return self._transition(TaskState.INPUT_REQUIRED, message)

    def complete(self, message: Optional[A2AMessage] = None) -> "Task":
        """Finish successfully: WORKING -> COMPLETED (terminal)."""
        return self._transition(TaskState.COMPLETED, message)

    def fail(self, message: Optional[A2AMessage] = None) -> "Task":
        """Finish with error: -> FAILED (terminal)."""
        return self._transition(TaskState.FAILED, message)

    def cancel(self, message: Optional[A2AMessage] = None) -> "Task":
        """Abort: -> CANCELED (terminal)."""
        return self._transition(TaskState.CANCELED, message)

    def reject(self, message: Optional[A2AMessage] = None) -> "Task":
        """Decline the task outright: SUBMITTED -> REJECTED (terminal)."""
        return self._transition(TaskState.REJECTED, message)

    def require_auth(self, message: Optional[A2AMessage] = None) -> "Task":
        """Pause for authentication: WORKING -> AUTH_REQUIRED."""
        return self._transition(TaskState.AUTH_REQUIRED, message)

    # --- artifacts ------------------------------------------------------
    def add_artifact(self, artifact: Part) -> "Task":
        """Attach a produced artefact and emit an artifact_update event."""
        self.artifacts.append(artifact)
        self._emit({
            "kind": EVENT_ARTIFACT_UPDATE,
            "taskId": self.taskId,
            "contextId": self.contextId,
            "ts": self._clock(),
            "data": {"artifact": artifact.to_dict()},
        })
        return self

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> Dict[str, Any]:
        """Internal serialisation (unchanged): flat ``taskId``/``state``/``history``.

        Kept intact for existing in-process callers/tests. For the on-the-wire
        A2A ``Task`` object shape use ``to_a2a`` instead.
        """
        return {
            "taskId": self.taskId,
            "contextId": self.contextId,
            "state": self.state.value if self.state is not None else None,
            "history": [h.to_dict() for h in self.history],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": dict(self.metadata),
        }

    def to_a2a(self) -> Dict[str, Any]:
        """Emit the spec A2A ``Task`` object shape.

        Differs from ``to_dict`` per a2a-protocol.org: top-level ``id`` (not
        ``taskId``); a nested ``status:{state, message?, timestamp?}`` holding the
        CURRENT status; ``history`` as the Message[] conversation (the messages
        attached to status changes, in order); ``artifacts`` as Artifact[]; and a
        ``kind:"task"`` discriminator.
        """
        current = self.history[-1] if self.history else None
        status: Dict[str, Any] = {
            "state": self.state.value if self.state is not None else TaskState.UNKNOWN.value,
        }
        if current is not None:
            if current.message is not None:
                status["message"] = current.message.to_dict()
            if current.ts is not None:
                status["timestamp"] = current.ts
        return {
            "id": self.taskId,
            "contextId": self.contextId,
            "status": status,
            "history": [h.message.to_dict() for h in self.history if h.message is not None],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metadata": dict(self.metadata),
            "kind": "task",
        }


class TaskStore:
    """In-memory, pluggable registry of tasks, keyed by taskId.

    A secondary index maps ``contextId`` -> ordered taskIds so related tasks
    (the blueprint's ``contextId`` grouping) are retrievable together. A real
    backing store (Redis, a DB) can implement the same handful of methods.
    """

    def __init__(self, clock: Optional[Callable[[], Optional[int]]] = None) -> None:
        self._clock = clock
        self._tasks: Dict[str, Task] = {}
        self._by_context: Dict[str, List[str]] = {}

    def create(
        self,
        contextId: Optional[str] = None,
        taskId: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Mint a new task (sharing the store's clock) and register it."""
        task = Task(taskId=taskId, contextId=contextId, clock=self._clock, metadata=metadata)
        self.put(task)
        return task

    def put(self, task: Task) -> Task:
        """Register (or replace) a task and update the context index."""
        self._tasks[task.taskId] = task
        ids = self._by_context.setdefault(task.contextId, [])
        if task.taskId not in ids:
            ids.append(task.taskId)
        return task

    def get(self, taskId: str) -> Optional[Task]:
        return self._tasks.get(taskId)

    def by_context(self, contextId: str) -> List[Task]:
        """All tasks sharing ``contextId``, in insertion order."""
        return [self._tasks[i] for i in self._by_context.get(contextId, []) if i in self._tasks]

    def contexts(self) -> List[str]:
        return list(self._by_context.keys())

    def all(self) -> List[Task]:
        return list(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, taskId: object) -> bool:
        return taskId in self._tasks
