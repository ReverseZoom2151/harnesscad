"""Agent lifecycle hook bus: block / modify / observe with error isolation.

The hook system exposes five lifecycle events (``pre_tool_use``, ``post_tool_use``,
``user_prompt_submit``, ``post_response``, ``file_attach``); handlers fire in
a deterministic order; a handler returning ``{"block": True, ...}`` stops the
chain immediately (a veto); a handler returning ``{"modify": <text>}`` has its
replacement threaded into the next handler's context; a raising handler is
isolated (logged, skipped) so one broken hook cannot take down the loop; and
hooks can be disabled by name without unregistering them.

The harness's agent loop needs a safe interception point instead of hard-wired
policy gates at every call site. This module provides a registration-based bus;
it intentionally does not scan and execute arbitrary hook files from a data
directory, because that would create an unverified-code channel.
Handlers are plain callables registered by the composing code; discovery
metadata is kept so the wiring stays inspectable.

Determinism: handlers fire in registration order; ``fire`` returns both the
merged result and a per-handler trace so a blocked call can name its vetoer.

Stdlib only, absolute imports. ``--selfcheck`` covers ordering, block
short-circuit, modify threading, error isolation, and the disable list.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "VALID_EVENTS",
    "HookInfo",
    "HookOutcome",
    "FireReport",
    "HookBus",
    "main",
]

#: The lifecycle events freecad-ai defines, unchanged.
VALID_EVENTS: Tuple[str, ...] = (
    "pre_tool_use",
    "post_tool_use",
    "user_prompt_submit",
    "post_response",
    "file_attach",
)

Handler = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


@dataclass(frozen=True)
class HookInfo:
    """Discovery metadata for one registered hook."""
    name: str
    events: Tuple[str, ...]
    builtin: bool = False
    provenance: str = ""


@dataclass(frozen=True)
class HookOutcome:
    """What one handler did during a fire."""
    name: str
    status: str          # "ok" | "blocked" | "error" | "disabled"
    result: Optional[Dict[str, Any]] = None
    error: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "result": self.result, "error": self.error}


@dataclass
class FireReport:
    """The full result of firing one event."""
    event: str
    merged: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    blocked_by: str = ""
    trace: List[HookOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"event": self.event, "merged": dict(self.merged),
                "blocked": self.blocked, "blocked_by": self.blocked_by,
                "trace": [t.to_dict() for t in self.trace]}


class HookBus:
    """A deterministic, error-isolated event hook bus.

    Hook dispatch semantics:

    * handlers for an event run in registration order;
    * before each handler, if an earlier handler produced ``modify`` and the
      context carries ``text``, the context ``text`` is replaced by the
      modified value (so rewrites chain);
    * a handler returning a dict with truthy ``block`` stops the chain and
      the report says who vetoed;
    * a raising handler is recorded as an error and skipped;
    * non-dict / falsy returns are observations (merged with nothing);
    * disabled hooks are skipped but appear in the trace as ``disabled``.
    """

    def __init__(self, events: Sequence[str] = VALID_EVENTS) -> None:
        self._events: Tuple[str, ...] = tuple(events)
        self._handlers: Dict[str, List[Tuple[str, Handler]]] = {}
        self._info: List[HookInfo] = []
        self._disabled: set = set()

    # -- registration -----------------------------------------------------
    def register(self, name: str, event: str, handler: Handler,
                 builtin: bool = False, provenance: str = "") -> HookInfo:
        if event not in self._events:
            raise ValueError(f"unknown event '{event}' (valid: {self._events})")
        if not name:
            raise ValueError("a hook needs a name")
        self._handlers.setdefault(event, []).append((name, handler))
        existing = next((i for i in self._info if i.name == name), None)
        if existing is None:
            info = HookInfo(name=name, events=(event,), builtin=builtin,
                            provenance=provenance)
            self._info.append(info)
        else:
            info = HookInfo(name=name, events=existing.events + (event,),
                            builtin=existing.builtin, provenance=existing.provenance)
            self._info[self._info.index(existing)] = info
        return info

    def disable(self, name: str) -> None:
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        self._disabled.discard(name)

    @property
    def discovered_hooks(self) -> List[HookInfo]:
        return list(self._info)

    def is_disabled(self, name: str) -> bool:
        return name in self._disabled

    # -- firing -----------------------------------------------------------
    def fire(self, event: str, context: Dict[str, Any]) -> FireReport:
        if event not in self._events:
            raise ValueError(f"unknown event '{event}' (valid: {self._events})")
        report = FireReport(event=event)
        handlers = self._handlers.get(event, [])
        ctx = dict(context)  # never mutate the caller's dict
        for name, handler in handlers:
            if name in self._disabled:
                report.trace.append(HookOutcome(name=name, status="disabled"))
                continue
            if "modify" in report.merged and "text" in ctx:
                ctx["text"] = report.merged["modify"]
            try:
                result = handler(ctx)
            except Exception as exc:
                report.trace.append(HookOutcome(
                    name=name, status="error",
                    error=f"{type(exc).__name__}: {exc}"))
                continue
            if result and isinstance(result, dict):
                if result.get("block"):
                    report.merged.update(result)
                    report.blocked = True
                    report.blocked_by = name
                    report.trace.append(HookOutcome(
                        name=name, status="blocked", result=dict(result)))
                    return report
                report.merged.update(result)
                report.trace.append(HookOutcome(
                    name=name, status="ok", result=dict(result)))
            else:
                report.trace.append(HookOutcome(name=name, status="ok"))
        return report


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    bus = HookBus()

    calls: List[str] = []

    def uppercase(ctx: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("uppercase")
        return {"modify": str(ctx.get("text", "")).upper()}

    def suffixer(ctx: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("suffixer")
        return {"modify": str(ctx.get("text", "")) + "!"}

    def observer(_ctx: Dict[str, Any]) -> None:
        calls.append("observer")
        return None

    def broken(_ctx: Dict[str, Any]) -> Dict[str, Any]:
        calls.append("broken")
        raise RuntimeError("boom")

    def veto_shell(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        calls.append("veto")
        if ctx.get("tool") == "shell":
            return {"block": True, "reason": "shell is not allowed"}
        return None

    bus.register("uppercase", "user_prompt_submit", uppercase)
    bus.register("broken", "user_prompt_submit", broken)
    bus.register("suffixer", "user_prompt_submit", suffixer)
    bus.register("observer", "user_prompt_submit", observer)
    bus.register("veto", "pre_tool_use", veto_shell, builtin=True)

    # Modify threading + error isolation + ordering.
    report = bus.fire("user_prompt_submit", {"text": "hello"})
    check(report.merged.get("modify") == "HELLO!", "modify chained through")
    check(not report.blocked, "no block")
    check(calls == ["uppercase", "broken", "suffixer", "observer"],
          "registration order preserved")
    statuses = [t.status for t in report.trace]
    check(statuses == ["ok", "error", "ok", "ok"], "error isolated, chain continues")

    # Block short-circuit.
    calls.clear()
    blocked = bus.fire("pre_tool_use", {"tool": "shell"})
    check(blocked.blocked and blocked.blocked_by == "veto", "veto names itself")
    check(blocked.merged.get("reason") == "shell is not allowed", "veto reason")
    allowed = bus.fire("pre_tool_use", {"tool": "measure"})
    check(not allowed.blocked, "non-matching tool passes")

    # Disable list.
    bus.disable("suffixer")
    report2 = bus.fire("user_prompt_submit", {"text": "hi"})
    check(report2.merged.get("modify") == "HI", "disabled hook skipped")
    check(any(t.status == "disabled" and t.name == "suffixer"
              for t in report2.trace), "disabled hook traced")
    bus.enable("suffixer")
    check(bus.fire("user_prompt_submit", {"text": "hi"}).merged["modify"] == "HI!",
          "re-enable works")

    # Caller's context is never mutated.
    ctx = {"text": "orig"}
    bus.fire("user_prompt_submit", ctx)
    check(ctx == {"text": "orig"}, "caller context untouched")

    # Discovery + validation.
    names = [i.name for i in bus.discovered_hooks]
    check(names == ["uppercase", "broken", "suffixer", "observer", "veto"],
          "discovery order stable")
    try:
        bus.register("x", "no_such_event", observer)
        check(False, "unknown event must be rejected")
    except ValueError:
        pass
    try:
        bus.fire("no_such_event", {})
        check(False, "firing unknown event must be rejected")
    except ValueError:
        pass

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("hook_bus selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Agent lifecycle hook bus (freecad-ai)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
