"""`Planner` — natural-language brief -> validated CISP ops.

The planner is the NL->ops step: it assembles the message stack (system prompt +
the brief + a snapshot of current model state + any diagnostics from a failed
prior attempt), calls the `LLM`, and funnels the response through
`llm.structured` so it only ever hands back validated `cisp.ops.Op` objects. It
is deliberately provider-blind — it takes any `LLM`.

THE FEEDBACK GATE
-----------------
This is the one place in the product where a diagnostic becomes an INSTRUCTION
TO A MODEL, so this is where the soundness policy is enforced. Only PROVEN and
MEASURED diagnostics (`verifiers.soundness`) are written into the retry prompt.
HEURISTIC ones are still produced, still returned in the `ApplyOpsResult`, still
logged and still shown to humans — they are simply not spoken to the model.

The reason is measured, not aesthetic. `assets/pressure/report.md`: the typed
loop lost to blind resampling by 8.3 points and lost hardest on the strongest
model, because every one of its net losses was a REGRESSION — a correct part
that the model broke *because a wrong diagnostic told it to*. A typed diagnostic
is a lever, and a lever amplifies whichever way it is pushed. The value of a
diagnostic is bounded above by its truth, and the tighter a model's
instruction-following, the tighter that bound binds. A wrong instruction is
worse than no instruction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from harnesscad.core.cisp.ops import Op
from harnesscad.agents.agent.system_prompt import SYSTEM_PROMPT
from harnesscad.agents.llm.base import LLM, Message, ToolSpec, system, user
from harnesscad.agents.llm.structured import ParsedOps, validate_ops
from harnesscad.eval.verifiers.soundness import MODEL_FACING_TIERS, model_facing


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
    """NL brief -> ops, with the soundness gate on the feedback channel.

    ``feedback_tiers`` is the policy: the soundness tiers whose diagnostics are
    allowed into the model's retry prompt. It defaults to PROVEN + MEASURED.
    Pass ``feedback_tiers=soundness.TIERS`` to restore the pre-tiering behaviour
    (everything is fed back) — that is the configuration the pressure experiment
    measured at -8.3 points, and it exists so the comparison can be re-run, not
    because anyone should ship it.
    """

    def __init__(self, llm: LLM, use_tool: bool = False,
                 feedback_tiers: Iterable[str] = MODEL_FACING_TIERS) -> None:
        self.llm = llm
        self.use_tool = use_tool
        self.feedback_tiers = tuple(feedback_tiers)

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
        # THE GATE. Heuristic findings are dropped here and nowhere else: they
        # remain in the caller's ApplyOpsResult for the log and the human.
        trusted = model_facing(diagnostics or [], self.feedback_tiers)
        if trusted:
            parts.append(
                "PRIOR ATTEMPT FAILED — these are OBSERVATIONS about what was "
                "built, with the evidence for each. Reason from them, re-emit "
                "the full corrected op sequence, and change only what the "
                "evidence requires:\n" + _format_diagnostics(trusted)
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
