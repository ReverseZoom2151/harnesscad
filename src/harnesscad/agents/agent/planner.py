"""`Planner` — natural-language brief -> validated CISP ops.

The planner is the NL->ops step: it assembles the message stack (system prompt +
the brief + a snapshot of current model state + any diagnostics from a failed
prior attempt), calls the `LLM`, and funnels the response through
`llm.structured` so it only ever hands back validated `cisp.ops.Op` objects. It
is deliberately provider-blind — it takes any `LLM`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from harnesscad.core.cisp.ops import Op
from harnesscad.agents.agent.system_prompt import SYSTEM_PROMPT
from harnesscad.agents.llm.base import LLM, Message, ToolSpec, system, user
from harnesscad.agents.llm.structured import ParsedOps, validate_ops


# A tool the model may call instead of replying in free text. Either path works;
# llm.structured pulls ops from tool-call args OR the text body.
EMIT_OPS_TOOL = ToolSpec(
    name="emit_ops",
    description="Emit the CISP op sequence that builds the requested design.",
    parameters={
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Ordered list of CISP op objects, each with an 'op' tag.",
            }
        },
        "required": ["ops"],
    },
)


class PlanError(RuntimeError):
    """Raised when the model's response could not be parsed into valid ops."""

    def __init__(self, message: str, raw: Any = None) -> None:
        super().__init__(message)
        self.raw = raw


class Planner:
    def __init__(self, llm: LLM, use_tool: bool = False) -> None:
        self.llm = llm
        self.use_tool = use_tool

    # --- message assembly ------------------------------------------------
    def build_messages(
        self,
        brief: str,
        state_summary: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[List[Any]] = None,
    ) -> List[Message]:
        msgs: List[Message] = [system(SYSTEM_PROMPT)]
        parts = [f"DESIGN BRIEF:\n{brief}"]
        if state_summary:
            parts.append(
                "CURRENT MODEL STATE:\n" + json.dumps(state_summary, sort_keys=True, indent=2)
            )
        if diagnostics:
            parts.append(
                "PRIOR ATTEMPT FAILED — fix these diagnostics and re-emit the "
                "full corrected op sequence:\n" + _format_diagnostics(diagnostics)
            )
        msgs.append(user("\n\n".join(parts)))
        return msgs

    # --- planning --------------------------------------------------------
    def plan(
        self,
        brief: str,
        state_summary: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[List[Any]] = None,
    ) -> List[Op]:
        """Return validated CISP ops for `brief`. Raises PlanError on bad output."""
        parsed = self.plan_parsed(brief, state_summary, diagnostics)
        if not parsed.ok:
            raise PlanError(parsed.error or "planner produced no valid ops")
        return parsed.ops

    def plan_parsed(
        self,
        brief: str,
        state_summary: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[List[Any]] = None,
    ) -> ParsedOps:
        """Like `plan`, but returns a ParsedOps (never raises on parse failure)."""
        messages = self.build_messages(brief, state_summary, diagnostics)
        tools = [EMIT_OPS_TOOL] if self.use_tool else None
        result = self.llm.complete(messages, tools=tools)
        return validate_ops(result)


def _format_diagnostics(diagnostics: List[Any]) -> str:
    lines: List[str] = []
    for d in diagnostics:
        if hasattr(d, "to_dict"):
            d = d.to_dict()
        if isinstance(d, dict):
            sev = d.get("severity", "error")
            code = d.get("code", "")
            msg = d.get("message", "")
            where = d.get("where")
            loc = f" @ {where}" if where else ""
            lines.append(f"- [{sev}] {code}: {msg}{loc}")
        else:
            lines.append(f"- {d}")
    return "\n".join(lines)
