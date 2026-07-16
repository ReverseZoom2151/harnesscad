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
from harnesscad.io.surfaces.mcp.tools import ToolCatalog, ToolDescription


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


# --- AGENTIC RETRIEVAL (blueprint sec.16.7) --------------------------------
# Retrieval used to be a fixed pre-step: the harness decided, once, to paste a
# memory block ahead of the brief. That is not agentic -- the MODEL never gets to
# say "I need the ISO torque table before I can size this bolt". This tool puts
# that decision in the model's hands: it is offered ONLY when a doc/skill corpus
# (a retriever) is attached, carries the same 5-component description contract as
# every other tool, and its result is injected back into context so the model can
# plan WITH what it pulled. No corpus attached -> the tool is never offered and
# the loop is byte-for-byte the pre-retrieval one.
_RETRIEVE_DESCRIPTION = ToolDescription(
    what="Search the attached document / skill corpus (standards, API docs, "
         "worked examples) and read back the most relevant passages BEFORE you "
         "commit to ops.",
    when="When the brief turns on a fact you should ground rather than guess -- a "
         "standard torque/tolerance, an API signature, a dimension table, a "
         "material spec -- or when unsure which feature/op a term maps to. "
         "Retrieve, read the passages, THEN emit_ops.",
    when_not="Do not retrieve when the brief is fully specified and needs no "
             "external fact (a bare 'extrude a 20x10x5 plate'); do not retrieve "
             "to change the model (only emit_ops mutates); do not re-run the same "
             "query hoping for different passages -- refine the query terms "
             "instead.",
    side_effects="Read-only: retrieval never mutates the model. It appends the "
                 "retrieved passages to your context for this turn only.",
    output="A ranked list of passages, each with its source, heading breadcrumb "
           "and text; use them to inform the op sequence you emit next.")


def _retrieve_tool_description() -> str:
    """The five-component description of the retrieve tool, rendered to text."""
    return _RETRIEVE_DESCRIPTION.text()


def build_retrieve_tool() -> ToolSpec:
    """The corpus-retrieval tool the model may call to ground itself mid-plan."""
    return ToolSpec(
        name="retrieve",
        description=_retrieve_tool_description(),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": ("What to search the corpus for -- use the "
                                    "engineer's terms (standard/part numbers, "
                                    "symbol names, dimensions) for best recall."),
                },
                "k": {
                    "type": "integer",
                    "description": "How many passages to read back (default 5).",
                    "default": 5,
                },
                "source": {
                    "type": "string",
                    "description": ("Optional: restrict to sources whose name "
                                    "contains this substring (e.g. a filename)."),
                },
                "heading": {
                    "type": "string",
                    "description": ("Optional: restrict to passages under a "
                                    "heading breadcrumb containing this text."),
                },
            },
            "required": ["query"],
        },
    )


#: Offered to the model ONLY when a retriever is attached to the Planner.
RETRIEVE_TOOL = build_retrieve_tool()


class PlanError(RuntimeError):
    """Raised when the model's response could not be parsed into valid ops."""

    def __init__(self, message: str, raw: Any = None) -> None:
        super().__init__(message)
        self.raw = raw


def _diag_code(d: Any) -> str:
    """The code of a diagnostic, whether it is a dict or a Diagnostic object.

    Both forms reach the planner (the harness feeds back dicts; a caller driving
    it directly may pass Diagnostics), so the reader tolerates either and
    returns "" for anything unrecognisable.
    """
    if isinstance(d, dict):
        return str(d.get("code") or "")
    return str(getattr(d, "code", "") or "")


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
                 max_diagnostics: int = 5,
                 retriever: Any = None,
                 max_retrieval_rounds: int = 3,
                 quality_references: bool = False,
                 max_iterations: int = 5) -> None:
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
        # AGENTIC RETRIEVAL. ``retriever`` is anything with
        # ``retrieve(query, k, source=, heading=) -> [hit, ...]`` (a
        # ``rag.HybridRetriever``). When attached, the model is offered the
        # ``retrieve`` tool and may DECIDE to pull grounding mid-plan; the results
        # are injected into context and the model is re-prompted. ``None`` (the
        # default) offers no such tool -- the loop is the pre-retrieval one,
        # unchanged. ``max_retrieval_rounds`` bounds how many times the model may
        # retrieve before it must emit ops, so a model cannot loop on search.
        self.retriever = retriever
        self.max_retrieval_rounds = int(max_retrieval_rounds)
        self.last_retrievals: List[Dict[str, Any]] = []
        # STATE-CONDITIONED REFERENCES (agents/agent/quality_references.py).
        # The loop's repair advice used to depend only on the LAST diagnostics;
        # the injector conditions on the SHAPE of the trajectory instead (first
        # iteration, the same error twice, the same error three times, running
        # out of iterations). The planner is the only place a prompt is built,
        # so it is where the snippets are injected -- and it already sees every
        # iteration's diagnostics, so it can keep the error history itself
        # without the harness threading anything through.
        #
        # OFF by default: it appends to the prompt, and an unchanged prompt is
        # the contract every existing caller has. `quality_references=True`
        # opts in.
        self.quality_references = bool(quality_references)
        self.max_iterations = int(max_iterations)
        #: Advisory-reference bookkeeping: how many prompts we have built (the
        #: iteration index) and the ordered error-code history driving the
        #: stuck-loop snippets. Only touched when quality_references is on.
        self._qr_iteration = 0
        self._qr_error_history: List[str] = []
        #: The snippets injected into the LAST prompt (introspection/tests).
        self.last_references: List[str] = []

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
        # State-conditioned references LAST: they are advice about how to act on
        # everything above (the brief, the state, the diagnostics), so they read
        # as the closing instruction rather than preamble. No-op unless opted in.
        parts.extend(self._references(diagnostics))
        msgs.append(user("\n\n".join(parts)))
        return msgs

    def _references(self, diagnostics: Optional[List[Any]]) -> List[str]:
        """State-conditioned reference snippets for THIS iteration.

        Advances the planner's own iteration/error-history bookkeeping and asks
        `quality_references.references_for_state` what to inject. Returns []
        when `quality_references` is off -- the default -- so the assembled
        prompt is byte-for-byte the pre-references one.

        Never fatal: injected context is an enhancement, and a crash here must
        not cost the caller their plan.
        """
        self.last_references = []
        if not self.quality_references:
            return []
        try:
            from harnesscad.agents.agent.quality_references import (
                QUALITY_FIX_MAP, AgentLoopState, references_for_state)

            self._qr_iteration += 1
            codes = [_diag_code(d) for d in (diagnostics or [])]
            codes = [c for c in codes if c]
            self._qr_error_history.extend(codes)
            # Only codes the fix map actually knows earn a targeted repair line;
            # an unknown code would otherwise spend prompt budget on a generic
            # "inspect the reported issue" that says nothing the diagnostics
            # above have not already said. Unknown codes still count toward the
            # error HISTORY, which is what detects a stuck loop.
            failed = [c for c in codes if c in QUALITY_FIX_MAP]
            state = AgentLoopState(
                iteration=self._qr_iteration,
                max_iterations=self.max_iterations,
                failed_quality_codes=failed,
                error_history=tuple(self._qr_error_history),
            )
            self.last_references = references_for_state(state)
        except Exception:  # noqa: BLE001 - advisory context, never a gate
            self.last_references = []
        return self.last_references

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
        tools = self._tools_for_call()
        result = self.llm.complete(messages, tools=tools)
        if self.retriever is not None:
            result = self._resolve_retrievals(messages, tools, result)
        return validate_ops(result)

    # --- agentic retrieval -----------------------------------------------
    def _tools_for_call(self) -> Optional[List[ToolSpec]]:
        """The tool surface for this call: emit_ops always, retrieve if gated on."""
        if not self.use_tool:
            return None
        tools = [EMIT_OPS_TOOL]
        if self.retriever is not None:
            tools.append(RETRIEVE_TOOL)
        return tools

    def _resolve_retrievals(self, messages: List[Message],
                            tools: Optional[List[ToolSpec]],
                            result: Any) -> Any:
        """Service any ``retrieve`` tool calls, re-prompting with the results.

        Loops while the model asks to retrieve (bounded by
        ``max_retrieval_rounds``): each round runs the retriever, injects the
        passages into context, and re-completes. As soon as the model stops
        retrieving -- it emitted ops or plain text -- the result is handed back to
        ``validate_ops`` unchanged. Retrieval is never fatal: a broken retriever
        degrades to a note in context, exactly like memory recall.
        """
        self.last_retrievals = []
        rounds = 0
        while getattr(result, "has_tool_calls", False):
            retrieve_calls = [tc for tc in result.tool_calls
                              if tc.name == "retrieve"]
            if not retrieve_calls:
                return result  # emit_ops (or other) -> let validate_ops handle it
            if rounds >= self.max_retrieval_rounds:
                return result  # budget spent; force the model to commit to ops
            messages = list(messages)
            if result.text:
                messages.append(Message("assistant", result.text))
            for tc in retrieve_calls:
                messages.append(user(self._run_retrieve(tc.arguments)))
            rounds += 1
            result = self.llm.complete(messages, tools=tools)
        return result

    def _run_retrieve(self, arguments: Any) -> str:
        """Execute one retrieve tool call, rendering its passages as a context block."""
        query, k, source, heading = _parse_retrieve_args(arguments)
        if not query:
            return "RETRIEVED CONTEXT: (no query supplied; nothing retrieved)"
        try:
            hits = self.retriever.retrieve(query, k, source=source, heading=heading)
        except TypeError:
            # A retriever without the optional filter kwargs still works.
            try:
                hits = self.retriever.retrieve(query, k)
            except Exception:  # noqa: BLE001 - retrieval must never fail the run
                hits = []
        except Exception:  # noqa: BLE001 - retrieval must never fail the run
            hits = []
        self.last_retrievals.append({"query": query, "k": k, "n": len(hits)})
        return _render_retrieval_block(query, hits)


def _parse_retrieve_args(arguments: Any):
    """Coerce raw retrieve tool-call arguments (JSON string or dict) into fields.

    Returns ``(query, k, source, heading)`` with defensive defaults; a malformed
    arguments blob yields an empty query rather than raising.
    """
    data: Dict[str, Any] = {}
    if isinstance(arguments, str):
        try:
            data = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            data = {}
    elif isinstance(arguments, dict):
        data = arguments
    query = str(data.get("query") or "").strip()
    try:
        k = int(data.get("k") or 5)
    except (TypeError, ValueError):
        k = 5
    if k <= 0:
        k = 5
    source = data.get("source") or None
    heading = data.get("heading") or None
    return query, k, source, heading


def _render_retrieval_block(query: str, hits: Iterable[Any]) -> str:
    """Format retrieved hits into the context block the model reads next turn."""
    lines = [f"RETRIEVED CONTEXT for query {query!r} "
             "(read-only grounding; use it, then emit_ops):"]
    hits = list(hits)
    if not hits:
        lines.append("  (no matching passages in the corpus)")
        return "\n".join(lines)
    for i, h in enumerate(hits, 1):
        source = getattr(h, "source", "") or ""
        crumb = " > ".join(getattr(h, "heading_path", []) or [])
        text = getattr(h, "text", None)
        if text is None:
            text = str(h)
        header = f"[{i}] {source}"
        if crumb:
            header += f" -- {crumb}"
        lines.append(header)
        lines.append(str(text).strip())
    return "\n".join(lines)


#: The formatter now lives in `agent.feedback` -- ONE renderer for every loop and
#: surface. Kept as an alias so existing callers do not break.
_format_diagnostics = render
