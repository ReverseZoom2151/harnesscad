"""THE feedback channel: one gate, one formatter, for every loop and surface.

A diagnostic becomes an INSTRUCTION the moment it is written into a model's
retry prompt. That is the only channel where being wrong destroys work, so it
gets exactly one implementation and every caller uses it.

Before this module there were two formatters and the gate lived inside one
planner class. `agent.planner.Planner` gated; `eval.pressure.prompts.format_typed`
did not, and it was the formatter that produced the published -8.3
(`assets/pressure/report.md`). A gate that is a property of one class is not a
policy -- any planner that is not that class silently bypasses it, which is
exactly what the A2A surface's `_PlatePlanner` did.

So the gate moved OUT of the planner and UP to the harness boundary
(`core.harness.AgentHarness`), which is the one place every write path passes
through. `gate` and `render` below are what it calls, and what `Planner` calls,
and what any future arm of the pressure experiment must call. Applying the gate
twice is a no-op: it is a filter, and it is idempotent.

Deterministic, stdlib-only.
"""

from __future__ import annotations

from typing import Any, Iterable, List

from harnesscad.eval.verifiers.soundness import (
    MODEL_FACING_TIERS,
    human_facing,
    model_facing,
    tier_of,
)

__all__ = [
    "MODEL_FACING_TIERS",
    "gate",
    "withheld",
    "render",
    "format_diagnostics",
    "tier_of",
]


def gate(diagnostics: Iterable[Any],
         tiers: Iterable[str] = MODEL_FACING_TIERS) -> List[Any]:
    """THE gate: the diagnostics allowed to instruct a model.

    Only PROVEN and MEASURED tiers pass by default. HEURISTIC findings are still
    produced, still returned in the `ApplyOpsResult`, still traced and still
    shown to humans -- they are simply not spoken to the model.

    Idempotent, so it is safe for the harness to gate and the planner to gate
    again: a filter applied twice filters the same set.
    """
    return model_facing(diagnostics or [], tiers)


def withheld(diagnostics: Iterable[Any],
             tiers: Iterable[str] = MODEL_FACING_TIERS) -> List[Any]:
    """The complement of :func:`gate` -- what was kept FROM the model.

    This is precisely the set a human should be shown: findings that might be
    right and might be wrong, and which we declined to turn into orders.
    """
    return human_facing(diagnostics or [], tiers)


def render(diagnostics: Iterable[Any]) -> str:
    """The ONE model-facing rendering of a diagnostic list.

    Each line carries the tier, so the model can see how much weight a finding
    is entitled to rather than receiving every line as equally authoritative.
    """
    lines: List[str] = []
    for d in diagnostics or []:
        tier = tier_of(d)
        if hasattr(d, "to_dict"):
            d = d.to_dict()
        if isinstance(d, dict):
            sev = d.get("severity", "error")
            code = d.get("code", "")
            msg = d.get("message", "")
            where = d.get("where")
            loc = f" @ {where}" if where else ""
            lines.append(f"- [{sev}/{tier}] {code}: {msg}{loc}")
        else:
            lines.append(f"- [{tier}] {d}")
    return "\n".join(lines)


#: The header the model sees above a gated diagnostic list. Observation-led, not
#: imperative: `soundness.observe` exists because "Fix exactly these problems"
#: is an order, and an order is executed even when it is wrong.
PRIOR_ATTEMPT_HEADER = (
    "PRIOR ATTEMPT FAILED -- these are OBSERVATIONS about what was built, with "
    "the evidence for each. Reason from them, re-emit the full corrected op "
    "sequence, and change only what the evidence requires:"
)


def format_diagnostics(diagnostics: Iterable[Any]) -> str:
    """Backwards-compatible alias for :func:`render`."""
    return render(diagnostics)
