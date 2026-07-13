"""ContextManager — owns the token window (HARNESS_BLUEPRINT.md sec.3 + sec.7).

The window is a hard budget, not a suggestion:

    C >= S(system) + M(memory/RAG) + T(tools) + H(history) + R(reserved)

Everything here is stdlib-only. Token counting is *pluggable*: the `TokenCounter`
protocol is a single `count(text) -> int` method. The default `HeuristicCounter`
is deliberately better than the naive 4-char rule (CAD emits code/JSON, which the
4-char rule mis-estimates by 20-40%): it counts word runs, digit runs, and
individual punctuation marks as separate tokens, which tracks real BPE token
counts far more closely on structured text. To get exact counts, drop in a
tokenizer-backed counter, e.g.::

    import tiktoken
    class TiktokenCounter:
        def __init__(self, model="gpt-4o"):
            self._enc = tiktoken.encoding_for_model(model)
        def count(self, text: str) -> int:
            return len(self._enc.encode(text))

    cm = ContextManager(budget=128_000, counter=TiktokenCounter())

Nothing else in this module changes — the counter is the only swap point.

Two guarantees the manager enforces:
  1. PRE-FLIGHT on every call: `preflight()` reports/raises BEFORE the LLM sees
     an over-budget request, so the provider never silently truncates the middle.
  2. PINNED HEAD + PREFIX CACHE: `assemble()` keeps the system prompt and the
     first user message pinned, keeps a stable system(+tool) prefix leading every
     request for prefix-cache friendliness, and evicts from the MIDDLE (trailing
     tool-result messages first), preserving the most recent turns at the tail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Protocol, Sequence, Union, runtime_checkable

from harnesscad.agents.llm.base import Message, ToolSpec


# --- token counting --------------------------------------------------------
@runtime_checkable
class TokenCounter(Protocol):
    """The single seam for token counting. Implement `count` and you're done."""

    def count(self, text: str) -> int:
        ...


# One regex, three alternatives: a word/identifier run (letters + underscore),
# a digit run, or any single non-space punctuation character. This mirrors how
# BPE tokenizers segment structured text far more closely than "len//4".
_TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+|[^\sA-Za-z0-9_]")


class HeuristicCounter:
    """Stdlib default counter. Better than the 4-char rule for code/JSON/CAD.

    Counts three token classes: word/identifier runs, digit runs, and each
    individual punctuation mark. On structured payloads (JSON op arrays, kernel
    logs) this tracks real tokenizer counts much more closely than `len//4`,
    because punctuation-dense text (`{"op":"extrude","distance":5}`) is *under*-
    counted by the 4-char rule. It is still an estimate — swap in an exact
    tokenizer (see module docstring) when precision matters.
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(_TOKEN_RE.findall(text))


# --- budget report ---------------------------------------------------------
@dataclass
class BudgetReport:
    """The result of a budget computation: the `C >= S+M+T+H+R` breakdown.

    `ok` is True iff `total <= budget`. `overflow` is how many tokens over budget
    the request is (0 when within budget). This is what pre-flight surfaces and
    what `assemble` returns alongside the kept messages.
    """

    budget: int
    system: int = 0
    memory: int = 0
    tools: int = 0
    history: int = 0
    reserved: int = 0

    @property
    def total(self) -> int:
        return self.system + self.memory + self.tools + self.history + self.reserved

    @property
    def ok(self) -> bool:
        return self.total <= self.budget

    @property
    def overflow(self) -> int:
        return max(0, self.total - self.budget)

    @property
    def remaining(self) -> int:
        return self.budget - self.total

    def to_dict(self) -> dict:
        return {
            "budget": self.budget,
            "system": self.system,
            "memory": self.memory,
            "tools": self.tools,
            "history": self.history,
            "reserved": self.reserved,
            "total": self.total,
            "ok": self.ok,
            "overflow": self.overflow,
            "remaining": self.remaining,
        }

    def __str__(self) -> str:
        return (
            f"budget C={self.budget}: S={self.system} M={self.memory} "
            f"T={self.tools} H={self.history} R={self.reserved} "
            f"-> total={self.total} ({'OK' if self.ok else f'OVER by {self.overflow}'})"
        )


class ContextOverflowError(RuntimeError):
    """Raised by `preflight(strict=True)` when a request exceeds the budget.

    Carries the `BudgetReport` so the caller can see the exact S/M/T/H/R
    breakdown and decide how to shed load (summarize history, drop docs, raise
    the reserved floor) rather than let the provider truncate silently.
    """

    def __init__(self, report: BudgetReport) -> None:
        self.report = report
        super().__init__(
            f"context over budget by {report.overflow} tokens ({report})"
        )


@dataclass
class AssembledContext:
    """What `assemble` returns: the kept messages + the budget report + notes.

    `messages` is the final, ordered window to send. `report` is the post-eviction
    budget breakdown. `evicted` counts how many history messages were dropped.
    `stable_prefix_tokens` is the size of the cache-friendly leading prefix
    (system prompt + tool definitions) that should be identical across turns.
    """

    messages: List[Message]
    report: BudgetReport
    evicted: int = 0
    stable_prefix_tokens: int = 0


# --- the manager -----------------------------------------------------------
_ToolLike = Union[ToolSpec, dict, str]
_OpsSource = Any  # OpDAG | Iterable[Op] — kept loose to avoid a hard import.


class ContextManager:
    """Owns the window: budget math, pre-flight, pinned+prefix-cached assembly.

    `budget` is C (total tokens the window admits). `reserved` is the default R
    (space held back for the model's completion) applied when a call doesn't
    override it. `counter` is any `TokenCounter`; the default heuristic counter
    is stdlib-only.
    """

    def __init__(
        self,
        budget: int,
        counter: Optional[TokenCounter] = None,
        reserved: int = 0,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget (C) must be positive")
        self.budget = budget
        self.counter = counter if counter is not None else HeuristicCounter()
        self.default_reserved = reserved

    # --- counting helpers -------------------------------------------------
    def count_text(self, text: str) -> int:
        return self.counter.count(text)

    def count_message(self, m: Message) -> int:
        """Tokens for one message: content + a small envelope for role/name/id.

        The envelope approximates the per-message formatting overhead real chat
        APIs add (role markers, delimiters). It is deliberately a fixed small
        constant rather than provider-specific magic.
        """
        n = self.counter.count(m.content or "")
        if m.name:
            n += self.counter.count(m.name)
        # role token + structural delimiters, per message.
        return n + _MESSAGE_ENVELOPE

    def count_messages(self, messages: Iterable[Message]) -> int:
        return sum(self.count_message(m) for m in messages)

    def count_tool(self, tool: _ToolLike) -> int:
        """Tokens a single tool definition contributes to the window."""
        if isinstance(tool, str):
            return self.counter.count(tool)
        if isinstance(tool, ToolSpec):
            text = tool.name + "\n" + tool.description + "\n" + json.dumps(
                tool.parameters or {}, sort_keys=True, separators=(",", ":")
            )
            return self.counter.count(text) + _TOOL_ENVELOPE
        # a raw dict (already-serialised tool schema)
        return self.counter.count(
            json.dumps(tool, sort_keys=True, separators=(",", ":"))
        ) + _TOOL_ENVELOPE

    def count_tools(self, tools: Optional[Sequence[_ToolLike]]) -> int:
        if not tools:
            return 0
        return sum(self.count_tool(t) for t in tools)

    # --- pre-flight -------------------------------------------------------
    def preflight(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[_ToolLike]] = None,
        reserved: Optional[int] = None,
        strict: bool = True,
    ) -> BudgetReport:
        """Guard a request BEFORE the call. Returns a `BudgetReport`; raises
        `ContextOverflowError` when over budget and `strict` (the default).

        This is the silent-truncation guard from the blueprint: run it on *every*
        call. The system message (first, if `role == 'system'`) is attributed to
        S; the rest of the messages to H; tools to T; `reserved` to R. Memory is
        counted under H here unless the caller assembles it separately.
        """
        r = reserved if reserved is not None else self.default_reserved
        msgs = list(messages)
        system_tokens = 0
        history_msgs = msgs
        if msgs and msgs[0].role == "system":
            system_tokens = self.count_message(msgs[0])
            history_msgs = msgs[1:]
        report = BudgetReport(
            budget=self.budget,
            system=system_tokens,
            tools=self.count_tools(tools),
            history=self.count_messages(history_msgs),
            reserved=r,
        )
        if strict and not report.ok:
            raise ContextOverflowError(report)
        return report

    # --- assembly (pin head, prefix-cache, evict middle) ------------------
    def assemble(
        self,
        system: Union[str, Message],
        first_user: Union[str, Message],
        history: Optional[Sequence[Message]] = None,
        memory: Optional[Union[str, Message, Sequence[Message]]] = None,
        tools: Optional[Sequence[_ToolLike]] = None,
        reserved: Optional[int] = None,
    ) -> AssembledContext:
        """Build a within-budget window that PINS the head and evicts the middle.

        Layout of the returned messages (prefix-cache friendly):

            [system]            <- pinned, part of the stable prefix (with tools)
            [memory...]         <- optional RAG/memory block (M)
            [first_user]        <- pinned (the spec/brief — never lost-in-middle)
            [ ...kept history ] <- most recent turns; middle evicted as needed

        Pinned = system + first_user (never evicted). Tools are counted in T but
        travel out-of-band (the LLM takes them as a separate argument), yet they
        form part of the *conceptual* stable prefix with the system prompt, so
        `stable_prefix_tokens` reports system+tools.

        Eviction, when H won't fit, removes from the MIDDLE: trailing tool-result
        messages first (they're the bulkiest and most stale — mesh/log dumps),
        then other middle turns, oldest-first, always preserving the most recent
        message at the tail. If even the pinned head + memory + tools + reserved
        exceed C, the returned report has `ok == False` (caller should shed load).
        """
        r = reserved if reserved is not None else self.default_reserved
        sys_msg = _as_message(system, "system")
        user_msg = _as_message(first_user, "user")
        mem_msgs = _as_messages(memory, "system")
        hist = list(history or [])

        S = self.count_message(sys_msg)
        M = self.count_messages(mem_msgs)
        T = self.count_tools(tools)
        pinned_user = self.count_message(user_msg)

        # Budget left for the evictable history after fixed costs.
        fixed = S + M + T + r + pinned_user
        avail_for_history = self.budget - fixed

        kept, evicted = _evict_middle(
            hist, avail_for_history, self.count_message
        )

        H = self.count_messages(kept) + pinned_user
        report = BudgetReport(
            budget=self.budget, system=S, memory=M, tools=T,
            history=H, reserved=r,
        )
        messages = [sys_msg, *mem_msgs, user_msg, *kept]
        return AssembledContext(
            messages=messages,
            report=report,
            evicted=evicted,
            stable_prefix_tokens=S + T,
        )

    # --- feature-tree summary (sec.7: NOT a full B-rep dump) --------------
    def feature_tree_summary(self, opdag_or_ops: _OpsSource) -> str:
        """Render a compact, deterministic feature-tree text summary from ops.

        Takes an `OpDAG` (anything with an `.ops()` method) or a plain iterable
        of ops. Reconstructs the deterministic ids the backend assigns (sketches
        `sk1..`, entities `e1..`, features `f1..`) and prints a nested outline —
        sketches with their primitives/constraints, then features. This is the
        feature-tree-as-truth view (intent), never a B-rep face/edge dump.
        """
        return feature_tree_summary(opdag_or_ops)


# Per-message / per-tool structural overhead (role markers, delimiters). Small
# fixed constants, not provider-specific magic — swap the counter for exactness.
_MESSAGE_ENVELOPE = 4
_TOOL_ENVELOPE = 6


def _as_message(x: Union[str, Message], role: str) -> Message:
    if isinstance(x, Message):
        return x
    return Message(role, x)


def _as_messages(
    x: Optional[Union[str, Message, Sequence[Message]]], role: str
) -> List[Message]:
    if x is None:
        return []
    if isinstance(x, Message):
        return [x]
    if isinstance(x, str):
        return [Message(role, x)] if x else []
    return list(x)


def _evict_middle(history, avail, count_fn):
    """Keep as much of `history` as fits in `avail`, evicting from the middle.

    Priority: (1) trailing tool-result messages first, oldest-first; (2) then
    other messages, oldest-first; always preserving the single most-recent
    message (the tail) until nothing else can be dropped. Deterministic.

    Returns (kept_messages, evicted_count).
    """
    kept = list(history)
    if not kept:
        return kept, 0
    total = sum(count_fn(m) for m in kept)
    evicted = 0
    if avail < 0:
        avail = 0

    def pick_index():
        # (1) earliest tool message that is not the last element.
        for i, m in enumerate(kept):
            if i == len(kept) - 1:
                break
            if m.role == "tool":
                return i
        # (2) earliest non-last message.
        if len(kept) > 1:
            return 0
        # (3) forced: the last remaining message.
        return 0

    while total > avail and kept:
        idx = pick_index()
        total -= count_fn(kept[idx])
        del kept[idx]
        evicted += 1
    return kept, evicted


# --- module-level feature-tree summary (importable directly) ---------------
_SKETCH_OPS = {"new_sketch"}
_ENTITY_OPS = {"add_point", "add_line", "add_circle", "add_rectangle"}
_FEATURE_OPS = {"extrude", "fillet", "boolean"}


def _op_to_dict(op) -> dict:
    if hasattr(op, "to_dict"):
        return op.to_dict()
    return dict(op)


def _entity_desc(d: dict, eid: str) -> str:
    tag = d["op"]
    if tag == "add_point":
        return f"    + point ({d.get('x', 0)},{d.get('y', 0)}) [{eid}]"
    if tag == "add_line":
        return (
            f"    + line ({d.get('x1', 0)},{d.get('y1', 0)})->"
            f"({d.get('x2', 0)},{d.get('y2', 0)}) [{eid}]"
        )
    if tag == "add_circle":
        return f"    + circle r={d.get('r', 0)} @({d.get('cx', 0)},{d.get('cy', 0)}) [{eid}]"
    if tag == "add_rectangle":
        return (
            f"    + rectangle {d.get('w', 0)}x{d.get('h', 0)} "
            f"@({d.get('x', 0)},{d.get('y', 0)}) [{eid}]"
        )
    return f"    + {tag} [{eid}]"


def _feature_desc(d: dict, fid: str) -> str:
    tag = d["op"]
    if tag == "extrude":
        return f"  {fid}: extrude {d.get('sketch', '?')} distance={d.get('distance', 0)}"
    if tag == "fillet":
        edges = d.get("edges", []) or []
        return f"  {fid}: fillet radius={d.get('radius', 0)} edges={list(edges)}"
    if tag == "boolean":
        return (
            f"  {fid}: boolean {d.get('kind', '?')} "
            f"target={d.get('target', '?')} tool={d.get('tool', '?')}"
        )
    return f"  {fid}: {tag}"


def feature_tree_summary(opdag_or_ops: _OpsSource) -> str:
    """Compact, deterministic feature-tree summary (module-level entry point).

    See `ContextManager.feature_tree_summary`. Importable directly for callers
    that don't hold a manager instance.
    """
    if hasattr(opdag_or_ops, "ops"):
        ops = list(opdag_or_ops.ops())
    else:
        ops = list(opdag_or_ops)

    if not ops:
        return "Feature tree (empty)"

    dicts = [_op_to_dict(op) for op in ops]
    lines: List[str] = [f"Feature tree ({len(ops)} ops):"]

    sk_n = 0
    e_n = 0
    f_n = 0
    cur_sketch: Optional[str] = None

    for d in dicts:
        tag = d["op"]
        if tag in _SKETCH_OPS:
            sk_n += 1
            cur_sketch = f"sk{sk_n}"
            lines.append(f"  {cur_sketch}: sketch on {d.get('plane', 'XY')}")
        elif tag in _ENTITY_OPS:
            e_n += 1
            lines.append(_entity_desc(d, f"e{e_n}"))
        elif tag == "constrain":
            val = d.get("value")
            val_s = "" if val is None else f"={val}"
            b = d.get("b")
            refs = d.get("a", "")
            if b:
                refs = f"{refs},{b}"
            lines.append(f"    ~ constrain {d.get('kind', '?')}{val_s} ({refs})")
        elif tag in _FEATURE_OPS:
            f_n += 1
            lines.append(_feature_desc(d, f"f{f_n}"))
        else:
            lines.append(f"  ? {tag}")

    lines.append(
        f"  [totals: {sk_n} sketch(es), {e_n} entit(ies), {f_n} feature(s)]"
    )
    return "\n".join(lines)
