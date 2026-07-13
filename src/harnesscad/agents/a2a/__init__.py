"""A2A (agent-to-agent) internal message vocabulary and task lifecycle.

Per docs/blueprint.md sec.2 and sec.12, agents in HarnessCAD speak the A2A
message format as their internal wire format — even in-process — so a remote
transport (HTTP + SSE streaming / webhooks) is a drop-in later. MCP is used for
each agent's *tools*; A2A is used *between* agents (trust-boundary separation).

Two layers:
  - ``a2a.messages`` — the value objects: ``AgentCard``, ``A2AMessage``, ``Part``.
  - ``a2a.task``     — the async lifecycle: ``Task``/``TaskState``/``TaskStore``
    with a guarded state machine and SSE-style subscriber callbacks.
"""

from harnesscad.agents.a2a.messages import (
    AgentCard,
    A2AMessage,
    Part,
    PART_TEXT,
    PART_DATA,
    PART_ARTIFACT,
    PART_KINDS,
    ROLE_USER,
    ROLE_AGENT,
    user_message,
    agent_message,
)
from harnesscad.agents.a2a.task import (
    Task,
    TaskState,
    TaskStatus,
    TaskStore,
    IllegalTransition,
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    EVENT_STATUS_UPDATE,
    EVENT_ARTIFACT_UPDATE,
    monotonic_counter,
)

__all__ = [
    # messages
    "AgentCard",
    "A2AMessage",
    "Part",
    "PART_TEXT",
    "PART_DATA",
    "PART_ARTIFACT",
    "PART_KINDS",
    "ROLE_USER",
    "ROLE_AGENT",
    "user_message",
    "agent_message",
    # task
    "Task",
    "TaskState",
    "TaskStatus",
    "TaskStore",
    "IllegalTransition",
    "LEGAL_TRANSITIONS",
    "TERMINAL_STATES",
    "EVENT_STATUS_UPDATE",
    "EVENT_ARTIFACT_UPDATE",
    "monotonic_counter",
]
