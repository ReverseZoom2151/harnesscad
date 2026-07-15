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
from harnesscad.agents.agent.feedback import (
    MODEL_FACING_TIERS,
    PRIOR_ATTEMPT_HEADER,
    gate,
    render,
)
from harnesscad.agents.agent.system_prompt import SYSTEM_PROMPT, build_system_prompt
from harnesscad.agents.llm.base import LLM, Message, ToolSpec, system, user
from harnesscad.agents.llm.structured import ParsedOps, validate_ops
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.io.surfaces.mcp.tools import ToolCatalog


# --- THE TOOL SURFACE -------------------------------------------------------
# The model gets the SAME tool catalogue we export to external MCP clients
# (`surfaces.mcp.tools.ToolCatalog`): five-component descriptions (what /
# when-to-use / when-NOT-to-use / side-effects / output), typed params with
# enums, and annotations -- all generated from the op registry, so they cannot
# drift from the code.
#
# It used to get one tool called `emit_ops` described in a single line, while
# the good descriptions were shipped to strangers. The book is emphatic that a
# tool description IS the interface (selection accuracy moves 10-20% on the
# description alone), and we were withholding ours from ourselves.
#
# The op stream stays a SINGLE transactional tool call rather than one tool call
# per op. That is not a compromise, it is the CISP contract: ops are ordered and
# order-dependent, the session applies them as one block-and-correct batch, and
# the digest/replay/trace machinery is defined over the batch. What changes is
# that the batch's item schema and its documentation are now the catalogue's,
# not a bare `{"type": "object"}`.
_CATALOG = ToolCatalog()


def _op_schemas() -> List[dict]:
    """One JSON schema per CISP op, with the op tag pinned as a const."""
    schemas: List[dict] = []
    for tool in sorted(_CATALOG.op_tools(), key=lambda t: t.name):
        schema = tool.input_schema()
        props = dict(schema.get("properties") or {})
        props["op"] = {"type": "string", "const": tool.name,
                       "description": tool.description.what}
        required = ["op"] + list(schema.get("required") or [])
        schemas.append({
            "type": "object",
            "title": tool.name,
            "description": tool.description.text(),
            "properties": props,
            "required": required,
        })
    return schemas


def _emit_ops_description() -> str:
    """The five-component description of the batch tool itself."""
    return (
        "Emit the ordered CISP op sequence that builds the requested design.\n"
        "When to use: every turn. This is the only way to change the model; the "
        "whole sequence is applied as one transaction.\n"
        "When NOT to use: do not emit a partial sequence to 'continue' a prior "
        "one -- always re-emit the FULL corrected sequence from the current "
        "state, because rejected ops were rolled back and never touched the "
        "model.\n"
        "Side effects: mutates the model. Each op is verified as it is applied; "
        "any op that fails verification is rolled back and reported back to you.\n"
        "Output: an applyOps result -- ok, the number of ops applied, the model "
        "digest, and any diagnostics.\n"
        "Each element of 'ops' is one op object; the per-op schemas below carry "
        "what each op does, when to use it, when NOT to use it, and its side "
        "effects."
    )


def build_emit_ops_tool(catalog: Optional[ToolCatalog] = None) -> ToolSpec:
    """The batch tool, documented from the MCP catalogue."""
    global _CATALOG
    if catalog is not None:
        _CATALOG = catalog
    return ToolSpec(
        name="emit_ops",
        description=_emit_ops_description(),
        parameters={
            "type": "object",
            "properties": {
                "ops": {
                    "type": "array",
                    "description": (
                        "Ordered list of CISP op objects, each with an 'op' tag. "
                        "Applied as one transaction, in order."),
                    "items": {"anyOf": _op_schemas()},
                }
            },
            "required": ["ops"],
        },
    )


#: A tool the model may call instead of replying in free text. Either path works;
#: llm.structured pulls ops from tool-call args OR the text body.
EMIT_OPS_TOOL = build_emit_ops_tool()


class PlanError(RuntimeError):
    """Raised when the model's response could not be parsed into valid ops."""

    def __init__(self, message: str, raw: Any = None) -> None:
        super().__init__(message)
        self.raw = raw


def prioritize_diagnostics(items: List[Any], top_k: int = 0) -> List[Any]:
    """Rank model-facing diagnostics ERROR -> WARNING -> INFO, then take the top-k.

    Delegates the ordering to `agents/agents/roles.py:prioritize`, which existed,
    was tested, and was never called from anywhere: the loop handed the model
    every diagnostic it had, in whatever order the fleet happened to run. A model
    facing many demands without a ranking fixes the last thing it read.

    Ordering is STABLE inside a severity band, so pipeline order is preserved and
    the prompt is a pure function of the diagnostics. The items handed back are
    the ORIGINAL objects (Diagnostic or dict) -- nothing is rewritten, only
    reordered and capped -- so the renderer and the soundness codes are untouched.
    """
    if not items:
        return []
    # Lazy: roles.py imports Planner (the Designer wraps one), so a module-level
    # import here would be circular.
    from harnesscad.agents.agents.roles import Finding, prioritize

    findings: List[Finding] = []
    for i, item in enumerate(items):
        d = item.to_dict() if hasattr(item, "to_dict") else item
        sev = d.get("severity") if isinstance(d, dict) else None
        if isinstance(sev, str):
            try:
                sev = Severity(sev)
            except ValueError:
                sev = Severity.INFO
        elif not isinstance(sev, Severity):
            sev = Severity.INFO
        findings.append(Finding(
            severity=sev,
            code=str((d.get("code") if isinstance(d, dict) else "") or ""),
            message=str(i),           # carries the index; never rendered
            source="fleet",
        ))
    order = [int(f.message) for f in prioritize(findings)]
    ranked = [items[i] for i in order]
    return ranked[:top_k] if top_k and top_k > 0 else ranked


class Planner:
    """NL brief -> ops, with the soundness gate on the feedback channel.

    ``feedback_tiers`` is the policy: the soundness tiers whose diagnostics are
    allowed into the model's retry prompt. It defaults to PROVEN + MEASURED.
    Pass ``feedback_tiers=soundness.TIERS`` to restore the pre-tiering behaviour
    (everything is fed back) — that is the configuration the pressure experiment
    measured at -8.3 points, and it exists so the comparison can be re-run, not
    because anyone should ship it.
    """

    def __init__(self, llm: LLM, use_tool: bool = True,
                 feedback_tiers: Iterable[str] = MODEL_FACING_TIERS,
                 memory: Any = None,
                 exemplars: int = 3,
                 max_diagnostics: int = 5) -> None:
        self.llm = llm
        self.use_tool = use_tool
        self.feedback_tiers = tuple(feedback_tiers)
        # FEW-SHOT. The system prompt was pure zero-shot for a strict-format
        # structured-output task while `agents/rag/exemplar_select.py` (greedy
        # submodular DST retrieval) sat orphaned for want of an exemplar bank.
        # `exemplars=k` pins the k VERIFIED worked examples that best tile THIS
        # brief (agents/context/cisp_exemplars.py, every stream executed against
        # the kernel in its test). `exemplars=0` restores the zero-shot prompt
        # byte for byte -- that is the OFF arm, and it must exist.
        #
        # This is orthogonal to `memory`: memory recalls what THIS deployment has
        # built and passed the gate; the bank is the floor that exists on run one,
        # when memory is empty, which is when the weak models fail at FORMAT.
        self.exemplars = int(exemplars)
        # PRIORITISATION. When nine diagnostics came back, all nine went to the
        # model, unranked, and a model fixes whichever it read last.
        # `agents/agents/roles.py:prioritize` existed and was never called.
        # 0 means "no cap"; the default hands back the top 5, worst first.
        self.max_diagnostics = int(max_diagnostics)
        # RETRIEVAL IS PART OF PROMPT COMPOSITION, NOT A BOLT-ON. The planner is
        # the one place a prompt is built, so it is the place recall happens.
        # ``memory`` is a HarnessMemory (anything with ``.recall(brief)`` whose
        # result has ``.prompt_block()``). ``None`` reproduces the pre-memory
        # prompt BYTE-FOR-BYTE — that is the OFF arm of the A/B, and it must.
        self.memory = memory
        self.last_recalled: Any = None

    # --- message assembly ------------------------------------------------
    def build_messages(
        self,
        brief: str,
        state_summary: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[List[Any]] = None,
    ) -> List[Message]:
        msgs: List[Message] = [system(
            build_system_prompt(brief if self.exemplars else None,
                                k=self.exemplars))]
        parts: List[str] = []

        # Memory at the HEAD of the user turn, ahead of the brief: recalled,
        # oracle-verified exemplars are context to read the brief *with*, not an
        # afterthought appended to it (blueprint sec.7 — learned context first).
        # Distinct from `self.exemplars`: that is the STATIC bank (the floor on
        # run one, when memory is empty); this is what THIS deployment actually
        # built and the measured gate actually passed.
        recalled = self._recall(brief)
        if recalled is not None:
            block = recalled.prompt_block()
            if block:
                parts.append(
                    "MEMORY — retrieved from past runs that PASSED the measured "
                    "output gate:\n" + block)

        parts.append(f"DESIGN BRIEF:\n{brief}")
        if state_summary:
            parts.append(
                "CURRENT MODEL STATE:\n" + json.dumps(state_summary, sort_keys=True, indent=2)
            )
        # THE GATE (agent.feedback.gate). The harness already gated before it
        # handed these back; gating again is a no-op, and it keeps the policy
        # true for any caller that drives the Planner directly.
        trusted = gate(diagnostics or [], self.feedback_tiers)
        # THE RANKING. `roles.prioritize` orders ERROR -> WARNING -> INFO, stably
        # (so pipeline order survives inside a severity band), and only the top-k
        # are spoken. An unranked list of nine is not feedback, it is noise.
        trusted = prioritize_diagnostics(trusted, self.max_diagnostics)
        if trusted:
            parts.append(PRIOR_ATTEMPT_HEADER + "\n" + render(trusted))
        msgs.append(user("\n\n".join(parts)))
        return msgs

    def _recall(self, brief: str) -> Any:
        """Retrieve from memory, if a memory is attached. Never fatal.

        A memory that raises must not take the run down with it: retrieval is an
        enhancement, and an enhancement that can fail the build is a regression.
        """
        if self.memory is None:
            return None
        try:
            self.last_recalled = self.memory.recall(brief)
        except Exception:  # noqa: BLE001 - a broken memory degrades to no memory
            self.last_recalled = None
        return self.last_recalled

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


#: The formatter now lives in `agent.feedback` -- ONE renderer for every loop and
#: surface. Kept as an alias so existing callers do not break.
_format_diagnostics = render
