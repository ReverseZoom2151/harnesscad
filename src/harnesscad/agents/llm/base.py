"""The swappable provider seam.

This module defines our *own* thin vocabulary for talking to a chat LLM — a
`Message`, a `ToolSpec`, a `CompletionResult`, and the `LLM` Protocol. Nothing
here imports a vendor SDK: concrete backends (litellm, a test mock, a future
direct client) implement `LLM` and translate to/from their provider's shapes.
Keeping this layer vendor-neutral is the whole point — swap the backend, keep
the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, runtime_checkable


# --- messages --------------------------------------------------------------
@dataclass
class Message:
    """One chat turn. `role` is 'system' | 'user' | 'assistant' | 'tool'.

    `tool_call_id`/`name` are only used for tool-result messages; most callers
    just set `role` and `content`.
    """

    role: str
    content: str = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


# --- tools -----------------------------------------------------------------
@dataclass
class ToolSpec:
    """A callable the model may invoke, described by a JSON-Schema parameter set.

    This is the provider-neutral form; backends translate it into whatever the
    vendor expects (e.g. litellm/OpenAI's ``{"type": "function", ...}``).
    """

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


@dataclass
class ToolCall:
    """A tool invocation the model asked for. `arguments` is the raw JSON string
    the model produced (parsed lazily by callers, e.g. llm.structured)."""

    name: str
    arguments: str
    id: Optional[str] = None


@dataclass
class CompletionResult:
    """What every `LLM.complete` returns, regardless of provider.

    `text` is the assistant's free-text content (may be empty when the model
    answered purely with tool calls). `tool_calls` carries any structured calls.
    `raw` is the untouched provider response for escape hatches/debugging.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# --- the seam --------------------------------------------------------------
@runtime_checkable
class LLM(Protocol):
    """The one interface the rest of the harness depends on.

    Any object with `complete` and `stream` is an LLM — the planner never knows
    (or cares) which provider is behind it.
    """

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **opts: Any,
    ) -> CompletionResult:
        """Run one non-streaming completion and return a `CompletionResult`."""
        ...

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **opts: Any,
    ) -> Iterator[str]:
        """Yield text chunks as they arrive."""
        ...


def messages_to_dicts(messages: Iterable[Message]) -> List[Dict[str, Any]]:
    """Normalise a list of Message (or already-dict) items to provider dicts."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        out.append(m.to_dict() if isinstance(m, Message) else dict(m))
    return out
