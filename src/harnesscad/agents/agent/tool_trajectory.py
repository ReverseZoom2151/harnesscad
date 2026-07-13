"""TOOLCAD tool-using trajectory — the ReAct think/tool_call/tool_response trace.

TOOLCAD structures a CAD-CoT rollout as a sequence of ReAct steps, each wrapping
the agent's reasoning and tool integration in special tokens (Sec. 3.3, App.
A.3):

    <think> ... </think>  ->  <tool_call> ... </tool_call>  ->  <tool_response>

A full trajectory is the ordered sequence of such steps building geometric state
via the CAD engine, terminated by an ``<answer>COMPLETED</answer>`` (App. A.1).
This module provides a deterministic, stdlib-only representation of that trace:

  * ``TrajectoryStep`` — one (think, tool_call, tool_response) triple with the
    engine's success/fail label.
  * ``ToolTrajectory`` — the ordered rollout, tracking the running geometric
    object list (Sec. 3.3) as tools succeed, and whether it is completed.
  * ``parse_react_trajectory`` — parse the paper's tagged textual format
    (``<think>``/``<tool_call>``/``<tool_response>``/``<answer>``) into a
    structured trajectory, using a real JSON tool-call body
    (``{"name": ..., "arguments": ...}``) exactly as the prompt template
    prescribes.
  * ``check_format_order`` — verify the strict per-step tag order the paper's
    format reward enforces (think -> tool_call -> tool_response).

Execution against a :class:`agent.toolcad_tool_schema.ToolExecutionState`
produces the tool_response labels, so a trajectory can be *rolled out*
deterministically from a list of tool calls.

Stdlib only, deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from harnesscad.agents.agent.tool_schema import (
    InterfaceResult,
    ToolCall,
    ToolExecutionState,
    ToolLibrary,
)


@dataclass(frozen=True)
class TrajectoryStep:
    """One ReAct step: reasoning, a typed tool call, and the engine response."""

    think: str
    call: ToolCall
    result: InterfaceResult

    @property
    def label(self) -> str:
        return self.result.label

    @property
    def succeeded(self) -> bool:
        return self.result.success


@dataclass
class ToolTrajectory:
    """An ordered tool-using rollout with a running geometric-object list."""

    steps: List[TrajectoryStep] = field(default_factory=list)
    completed: bool = False

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def tool_calls(self) -> Tuple[ToolCall, ...]:
        return tuple(s.call for s in self.steps)

    @property
    def num_success(self) -> int:
        return sum(1 for s in self.steps if s.succeeded)

    @property
    def num_fail(self) -> int:
        return sum(1 for s in self.steps if not s.succeeded)

    def add(self, step: TrajectoryStep) -> None:
        self.steps.append(step)


def rollout(
    calls: Sequence[Tuple[str, ToolCall]],
    library: ToolLibrary,
    *,
    completed: bool = True,
) -> ToolTrajectory:
    """Deterministically roll out ``(think, ToolCall)`` pairs against the engine.

    Each call is executed against a fresh :class:`ToolExecutionState`; the
    engine's success/fail InterfaceResult becomes the step's tool_response.
    """
    state = ToolExecutionState(library)
    traj = ToolTrajectory(completed=completed)
    for think, call in calls:
        result = state.execute(call)
        traj.add(TrajectoryStep(think=think, call=call, result=result))
    return traj


# --- Textual (tagged) parsing ----------------------------------------------

_THINK = re.compile(r"<think>(.*?)</think>", re.S)
_TOOLCALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_TOOLRESP = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
_ANSWER = re.compile(r"<answer>\s*COMPLETED\s*</answer>", re.S | re.I)

# Ordered tag scan for format checking.
_TAG = re.compile(r"<(/?)(think|tool_call|tool_response)>")


def check_format_order(text: str) -> bool:
    """Return True iff every step follows think -> tool_call -> tool_response.

    Mirrors the paper's format-reward structural check (App. B.3): all tags
    present, correctly opened/closed, and in the prescribed order per step.
    """
    expected = [
        ("", "think"), ("/", "think"),
        ("", "tool_call"), ("/", "tool_call"),
        ("", "tool_response"), ("/", "tool_response"),
    ]
    tags = _TAG.findall(text)
    if not tags or len(tags) % len(expected) != 0:
        return False
    for i, tag in enumerate(tags):
        if tag != expected[i % len(expected)]:
            return False
    return True


def _parse_tool_call_body(body: str) -> ToolCall:
    data = json.loads(body.strip())
    if not isinstance(data, dict) or "name" not in data:
        raise ValueError("tool_call body must be a JSON object with a 'name'")
    args = data.get("arguments", {})
    if not isinstance(args, dict):
        raise ValueError("tool_call 'arguments' must be a JSON object")
    return ToolCall(name=str(data["name"]), arguments=dict(args))


def parse_react_trajectory(text: str) -> ToolTrajectory:
    """Parse a tagged ReAct transcript into a structured trajectory.

    The tool_response label is read from the response text: a body containing
    the word 'fail' (case-insensitive) marks the step failed, else success —
    matching the paper's success/fail structured messages.
    """
    thinks = _THINK.findall(text)
    calls = _TOOLCALL.findall(text)
    resps = _TOOLRESP.findall(text)
    if not (len(thinks) == len(calls) == len(resps)):
        raise ValueError(
            "mismatched think/tool_call/tool_response counts: "
            f"{len(thinks)}/{len(calls)}/{len(resps)}"
        )
    traj = ToolTrajectory(completed=bool(_ANSWER.search(text)))
    for think, call_body, resp in zip(thinks, calls, resps):
        tool_call = _parse_tool_call_body(call_body)
        success = "fail" not in resp.lower()
        result = InterfaceResult(success, resp.strip())
        traj.add(TrajectoryStep(think.strip(), tool_call, result))
    return traj


def render_step(step: TrajectoryStep) -> str:
    """Render a step back to the paper's tagged textual format."""
    body = json.dumps(
        {"name": step.call.name, "arguments": dict(step.call.arguments)},
        sort_keys=True,
    )
    return (
        f"<think>{step.think}</think>\n"
        f"<tool_call>{body}</tool_call>\n"
        f"<tool_response>{step.result.description}</tool_response>"
    )
