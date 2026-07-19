"""process_state — agent state, tools, and long-running processes as a state model.

This module models an agent not as a straight-line function call but as a state
machine that can dispatch tools, some of
which are LONG-RUNNING: a process that outlives the turn that started it (an export,
a mesh boolean, a solver run) and whose completion the agent must come back and
OBSERVE later. Three abstractions carry that:

* the agent's own state (idle -> thinking -> acting -> waiting -> done/failed),
* a tool (a named capability; a long-running tool starts a process instead of
  returning a value inline),
* a long-running process (started, then polled to a terminal status).

The value of stating this as an EXPLICIT, deterministic model — rather than letting
it live implicitly inside a loop — is that every transition is legal-or-refused.
An agent cannot "observe" a process it never started, cannot go from ``done`` back
to ``acting``, and cannot report success while a process it launched is still
running. Those are exactly the bugs an ad-hoc loop makes; here they raise.

Why HarnessCAD wants it
-----------------------
A CAD CUA's most expensive actions are long-running: exporting a solid, recomputing
a document, running a boolean. Driving them through :mod:`harnesscad.io.cua.uia`
means dispatch -> WAIT -> verify the outcome, which is precisely a long-running
process with a polled terminal status. This model is the turn-spanning bookkeeping
that sits above the single-action verification uia already does: it says which
process the agent is waiting on and refuses to declare victory early.

Pure stdlib, deterministic, import-safe. No threads, no clock, no OS — a process is
advanced by explicitly OBSERVING a status, so a test drives the machine exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class AgentState(str, Enum):
    """The agent's lifecycle states."""

    IDLE = "idle"          # nothing in flight
    THINKING = "thinking"  # deciding the next tool
    ACTING = "acting"      # a tool is executing inline
    WAITING = "waiting"    # blocked on one or more long-running processes
    DONE = "done"          # terminal success
    FAILED = "failed"      # terminal failure


class ProcessStatus(str, Enum):
    """A long-running process's status."""

    PENDING = "pending"      # created, not started
    RUNNING = "running"      # started, not yet resolved
    SUCCEEDED = "succeeded"  # terminal, with a result
    FAILED = "failed"        # terminal, with an error
    CANCELLED = "cancelled"  # terminal, cancelled by the agent


_TERMINAL_PROCESS = {ProcessStatus.SUCCEEDED, ProcessStatus.FAILED,
                     ProcessStatus.CANCELLED}

#: Legal process transitions. Anything not listed is refused.
_PROCESS_TRANSITIONS: Dict[ProcessStatus, Tuple[ProcessStatus, ...]] = {
    ProcessStatus.PENDING: (ProcessStatus.RUNNING, ProcessStatus.CANCELLED),
    ProcessStatus.RUNNING: (ProcessStatus.SUCCEEDED, ProcessStatus.FAILED,
                            ProcessStatus.CANCELLED),
    ProcessStatus.SUCCEEDED: (),
    ProcessStatus.FAILED: (),
    ProcessStatus.CANCELLED: (),
}

#: Legal agent transitions.
_AGENT_TRANSITIONS: Dict[AgentState, Tuple[AgentState, ...]] = {
    AgentState.IDLE: (AgentState.THINKING, AgentState.DONE, AgentState.FAILED),
    AgentState.THINKING: (AgentState.ACTING, AgentState.WAITING, AgentState.DONE,
                          AgentState.FAILED),
    AgentState.ACTING: (AgentState.THINKING, AgentState.WAITING, AgentState.DONE,
                        AgentState.FAILED),
    AgentState.WAITING: (AgentState.THINKING, AgentState.WAITING, AgentState.DONE,
                         AgentState.FAILED),
    AgentState.DONE: (),
    AgentState.FAILED: (),
}


class StateError(RuntimeError):
    """An illegal transition, or an operation invalid for the current state."""


@dataclass
class LongRunningProcess:
    """A process that spans turns: started, then observed to a terminal status.

    Deterministic by construction — it never consults a clock or a thread. The
    agent advances it by calling :meth:`observe` with the status it read from the
    world (an export finished, a recompute failed). Every status change is checked
    against :data:`_PROCESS_TRANSITIONS`, and the transition history is retained so
    a trace can show exactly how the process resolved.
    """

    id: str
    name: str
    status: ProcessStatus = ProcessStatus.PENDING
    result: Any = None
    error: str = ""
    history: List[ProcessStatus] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(self.status)

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_PROCESS

    @property
    def succeeded(self) -> bool:
        return self.status is ProcessStatus.SUCCEEDED

    def _to(self, status: ProcessStatus) -> None:
        if status not in _PROCESS_TRANSITIONS[self.status]:
            raise StateError("illegal process transition %s -> %s for %r"
                             % (self.status.value, status.value, self.name))
        self.status = status
        self.history.append(status)

    def start(self) -> "LongRunningProcess":
        self._to(ProcessStatus.RUNNING)
        return self

    def observe(self, status: ProcessStatus, *, result: Any = None,
                error: str = "") -> "LongRunningProcess":
        """Advance the process to an observed ``status``. Refuses illegal moves.

        This is the ONLY way a process resolves — there is no implicit completion,
        so a test (and a real agent) resolves a process exactly when it has read the
        evidence that it resolved. Observing a terminal process, or an illegal
        move, raises rather than silently overwriting a settled result.
        """
        self._to(status)
        if status is ProcessStatus.SUCCEEDED:
            self.result = result
        elif status is ProcessStatus.FAILED:
            self.error = error or "process failed"
        return self

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "status": self.status.value,
                "result": self.result, "error": self.error,
                "history": [s.value for s in self.history]}


@dataclass(frozen=True)
class Tool:
    """A named capability. ``long_running`` tools START a process instead of
    returning inline; the ``handler`` (injected) does the real work in production
    and is a fake in a test. The model owns only the SHAPE — whether a tool blocks."""

    name: str
    long_running: bool = False
    handler: Optional[Callable[..., Any]] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "long_running": self.long_running}


class AgentRuntime:
    """The deterministic agent state machine with a tool registry and a process set.

    Enforces the two things an ad-hoc loop gets wrong:

    1. Every agent-state change is validated against :data:`_AGENT_TRANSITIONS`.
    2. The agent cannot be marked ``DONE`` while any process it started is still
       running — :meth:`finish` refuses, which is the turn-spanning analogue of
       uia's "an unverified action is not an action".

    Dispatching an inline tool runs it and returns to THINKING; dispatching a
    long-running tool registers a :class:`LongRunningProcess` and moves to WAITING.
    """

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self.state: AgentState = AgentState.IDLE
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.processes: Dict[str, LongRunningProcess] = {}
        self.history: List[AgentState] = [self.state]
        self._proc_seq = 0

    # -- agent transitions -------------------------------------------------
    def _to(self, state: AgentState) -> None:
        if state not in _AGENT_TRANSITIONS[self.state]:
            raise StateError("illegal agent transition %s -> %s"
                             % (self.state.value, state.value))
        self.state = state
        self.history.append(state)

    def think(self) -> "AgentRuntime":
        self._to(AgentState.THINKING)
        return self

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    # -- process bookkeeping ----------------------------------------------
    @property
    def pending_processes(self) -> List[LongRunningProcess]:
        """Processes that are started but not yet terminal — what WAITING waits on."""
        return [p for p in self.processes.values()
                if p.status is ProcessStatus.RUNNING]

    def _new_process_id(self, name: str) -> str:
        self._proc_seq += 1
        return "%s#%d" % (name, self._proc_seq)

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, tool_name: str, **kwargs: Any) -> Any:
        """Run a tool. An inline tool returns its value and the agent goes back to
        THINKING; a long-running tool returns a :class:`LongRunningProcess` in the
        RUNNING state and the agent goes to WAITING.

        Dispatch is only legal from THINKING (or WAITING, to launch another process
        while already waiting) — a settled agent cannot act.
        """
        if self.state not in (AgentState.THINKING, AgentState.WAITING):
            raise StateError("cannot dispatch a tool from state %s" % self.state.value)
        tool = self.tools.get(tool_name)
        if tool is None:
            raise StateError("no such tool %r" % tool_name)
        if tool.long_running:
            proc = LongRunningProcess(id=self._new_process_id(tool.name),
                                      name=tool.name)
            proc.start()
            self.processes[proc.id] = proc
            if self.state is not AgentState.WAITING:
                self._to(AgentState.WAITING)
            return proc
        # inline tool
        if self.state is not AgentState.ACTING:
            self._to(AgentState.ACTING)
        result = tool.handler(**kwargs) if tool.handler is not None else None
        self._to(AgentState.THINKING)
        return result

    def observe_process(self, process_id: str, status: ProcessStatus, *,
                        result: Any = None, error: str = "") -> LongRunningProcess:
        """Resolve (or advance) a registered process, then reconcile agent state.

        When the LAST running process resolves, the agent leaves WAITING for
        THINKING (there is a decision to make about the result); while any remain
        running it stays WAITING. Observing an unknown process id raises — the
        agent cannot observe what it never launched.
        """
        proc = self.processes.get(process_id)
        if proc is None:
            raise StateError("no registered process %r" % process_id)
        proc.observe(status, result=result, error=error)
        if self.state is AgentState.WAITING and not self.pending_processes:
            self._to(AgentState.THINKING)
        return proc

    # -- terminal ----------------------------------------------------------
    def finish(self) -> "AgentRuntime":
        """Mark the agent DONE — but REFUSE while any process is still running.

        Declaring success with a launched process still in flight is the exact
        turn-spanning bug this model exists to prevent; a caller must observe every
        process to a terminal status first.
        """
        if self.pending_processes:
            raise StateError("cannot finish: %d process(es) still running (%s)"
                             % (len(self.pending_processes),
                                ", ".join(p.name for p in self.pending_processes)))
        self._to(AgentState.DONE)
        return self

    def fail(self, reason: str = "") -> "AgentRuntime":
        """Mark the agent FAILED. Legal from any non-terminal state."""
        self._to(AgentState.FAILED)
        return self

    def to_dict(self) -> dict:
        return {"state": self.state.value,
                "tools": [t.to_dict() for t in self.tools.values()],
                "processes": [p.to_dict() for p in self.processes.values()],
                "history": [s.value for s in self.history]}
