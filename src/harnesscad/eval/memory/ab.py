"""The memory A/B: same briefs, same model, same seed, memory ON vs OFF.

WHY THIS EXISTS
---------------
Agent-S shipped an experience-augmented memory as its headline ICLR contribution
and then removed it in the version that set OSWorld SOTA -- "simpler, better, and
faster". Retrieved experience was NET-NEGATIVE against a strong base model,
because their store had no way to know whether a remembered trajectory had ever
worked, so it filled with plausible garbage and poisoned the loop.

We already shipped one mechanism (typed diagnostics) that we ASSUMED helped and
that measurably hurt by 8.3 points. The point of this repository is not to do
that again. So memory does not get to be assumed either. It gets measured.

THE DESIGN
----------
Two arms over the pressure corpus (``eval/pressure/briefs.py`` -- 28 briefs, each
carrying hidden geometric ground truth that NEITHER arm ever sees):

  ``off``  the harness prompt exactly as it was: system prompt + brief + state +
           gated diagnostics. Byte-identical to the pre-memory planner.
  ``on``   the same, plus a MEMORY block at the head of the user turn, holding
           the top-k ORACLE-VERIFIED past solutions, verified skills, lessons,
           and known verifier false positives, retrieved by the brief.

Everything else is pinned: the same model, the same seed, temperature 0, the same
brief ORDER (memory is order-dependent by nature, so the order is fixed and
reported), the same attempt budget, the same parser, the same grader.

THE WRITE GATE
--------------
An op stream enters memory only if ``io/gate.py`` MEASURED the part it builds --
closed, manifold, non-degenerate, honouring its own declared intent. The model's
claim that it succeeded buys nothing. ``metrics.grade`` runs that gate on every
attempt already, so the gate verdict here is the same instrument the grader uses,
read at the write boundary.

The grader's ``solved`` (the hidden ground truth) is NEVER shown to memory. If it
were, memory would be a channel for the answer key and the experiment would be
worthless.

Deterministic and cached: rerunning costs nothing and reproduces byte-identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from harnesscad.agents.agent.planner import Planner
from harnesscad.agents.memory.harness_memory import HarnessMemory, OracleVerdict
from harnesscad.eval.pressure.briefs import Brief, briefs_for
from harnesscad.eval.pressure.cache import CompletionCache
from harnesscad.eval.pressure.metrics import grade
from harnesscad.eval.pressure.model import (
    CachedClient,
    Client,
    OllamaClient,
    extract_ops,
    ops_to_dicts,
)

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_SEED = 20260714

ARM_OFF = "off"
ARM_ON = "on"
ARMS = (ARM_OFF, ARM_ON)


class _NoLLM:
    """The Planner's ``llm`` is only touched by ``plan_parsed``; this rig drives
    the client itself (the same client the pressure experiment used, so the
    parser and the sampling are identical to the published baseline) and calls
    only ``build_messages``. Passing a real LLM here would be a second, silently
    different code path."""

    def complete(self, messages, tools=None):  # pragma: no cover - never called
        raise AssertionError("the A/B drives the client directly")


@dataclass
class BriefRun:
    """One brief in one arm."""

    brief_id: str
    arm: str
    solved: bool = False
    solved_shape: bool = False
    built: bool = False
    gate_ok: bool = False
    attempts: int = 0
    recalled_episodes: int = 0
    recalled_false_positives: int = 0
    memory_admitted: int = 0
    memory_refused: int = 0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ArmResult:
    model: str
    arm: str
    runs: List[BriefRun] = field(default_factory=list)
    memory_stats: Dict[str, int] = field(default_factory=dict)
    false_positive_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def solved(self) -> int:
        return sum(1 for r in self.runs if r.solved)

    @property
    def solved_shape(self) -> int:
        return sum(1 for r in self.runs if r.solved_shape)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def rate(self) -> float:
        return self.solved / self.n if self.n else 0.0

    @property
    def mean_attempts(self) -> float:
        return (sum(r.attempts for r in self.runs) / self.n) if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model, "arm": self.arm,
            "n": self.n, "solved": self.solved, "rate": self.rate,
            "solved_shape": self.solved_shape,
            "mean_attempts": self.mean_attempts,
            "memory_stats": self.memory_stats,
            "false_positive_counts": self.false_positive_counts,
            "runs": [r.to_dict() for r in self.runs],
        }


@dataclass
class ABReport:
    seed: int
    max_attempts: int
    brief_order: List[str]
    arms: List[ArmResult] = field(default_factory=list)

    def by_model(self) -> Dict[str, Dict[str, ArmResult]]:
        out: Dict[str, Dict[str, ArmResult]] = {}
        for a in self.arms:
            out.setdefault(a.model, {})[a.arm] = a
        return out

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "max_attempts": self.max_attempts,
            "brief_order": self.brief_order,
            "arms": [a.to_dict() for a in self.arms],
        }


# --------------------------------------------------------------------------- #
# one brief, one arm
# --------------------------------------------------------------------------- #
def run_brief(
    client: Client,
    brief: Brief,
    planner: Planner,
    memory: Optional[HarnessMemory],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> BriefRun:
    """Plan -> apply -> grade -> (oracle-gated) remember, up to ``max_attempts``.

    The repair channel is the fleet's MODEL-FACING diagnostics only (the
    soundness gate the planner already enforces). The grader's ground truth never
    enters the prompt, and never enters memory.
    """
    run = BriefRun(brief_id=brief.id, arm=(ARM_ON if memory else ARM_OFF))
    diagnostics: Optional[List[dict]] = None
    last = None

    for attempt in range(max_attempts):
        messages = planner.build_messages(brief.text, None, diagnostics)
        if attempt == 0:
            rec = planner.last_recalled
            if rec is not None:
                run.recalled_episodes = len(rec.episodes)
                run.recalled_false_positives = len(rec.false_positives)

        raw = client.complete([m.to_dict() for m in messages], attempt)
        ops = ops_to_dicts(extract_ops(raw))
        g = grade(brief, ops)
        run.attempts = attempt + 1
        last = (ops, g)

        if memory is not None:
            # THE WRITE GATE. `g.gate_ok` is io/gate.py's measurement of the part
            # this op stream actually builds. `g.solved` (the answer key) is NOT
            # passed and must never be.
            verdict = OracleVerdict(
                ok=bool(g.gate_ok),
                failures=tuple(str(f.get("code", "failure"))
                               for f in g.gate_failures),
                source="gate",
            )
            w = memory.commit(brief.text, ops, verdict,
                              fleet_diagnostics=g.fleet_actionable,
                              summary=brief.id)
            if w["admitted"]:
                run.memory_admitted += 1
            else:
                run.memory_refused += 1

        if g.apply_ok and not g.fleet_model_facing:
            break
        diagnostics = list(g.fleet_model_facing)

    if last is not None:
        _, g = last
        run.solved = bool(g.solved)
        run.solved_shape = bool(g.solved_shape)
        run.built = bool(g.built)
        run.gate_ok = bool(g.gate_ok)
        run.reasons = list(g.reasons)
    return run


# --------------------------------------------------------------------------- #
# one arm
# --------------------------------------------------------------------------- #
def run_arm(
    client: Client,
    briefs: Sequence[Brief],
    memory_on: bool,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    memory_factory: Optional[Callable[[], HarnessMemory]] = None,
) -> ArmResult:
    """Run every brief IN ORDER in one arm.

    Order matters and is therefore pinned: in the ON arm memory accumulates
    ACROSS briefs, which is the whole hypothesis (brief 27 gets to stand on the
    verified solutions of briefs 1..26). In the OFF arm nothing accumulates and
    the order is irrelevant, so the same order costs nothing.
    """
    memory = (memory_factory or HarnessMemory)() if memory_on else None
    planner = Planner(_NoLLM(), use_tool=False, memory=memory)
    arm = ArmResult(model=getattr(client, "name", "client"),
                    arm=ARM_ON if memory_on else ARM_OFF)
    for brief in briefs:
        arm.runs.append(run_brief(client, brief, planner, memory, max_attempts))
    if memory is not None:
        arm.memory_stats = dict(memory.stats)
        arm.false_positive_counts = memory.false_positive_counts()
    return arm


def run(
    models: Sequence[str],
    selector: str = "all",
    seed: int = DEFAULT_SEED,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    cache_dir: str = ".memory_ab_cache",
    client_factory: Optional[Callable[[str], Client]] = None,
) -> ABReport:
    """The full grid: every model x {memory off, memory on}."""
    briefs = briefs_for(selector)
    cache = CompletionCache(cache_dir)
    report = ABReport(seed=seed, max_attempts=max_attempts,
                      brief_order=[b.id for b in briefs])
    for name in models:
        base = (client_factory(name) if client_factory
                else OllamaClient(name, seed=seed, temperature=0.0))
        for memory_on in (False, True):
            client = CachedClient(base, cache, seed=seed, temperature=0.0)
            report.arms.append(
                run_arm(client, briefs, memory_on, max_attempts=max_attempts))
    return report


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def format_text(report: ABReport) -> str:
    """The result, stated plainly -- INCLUDING when memory loses."""
    lines: List[str] = []
    lines.append("MEMORY A/B - same briefs, same model, same seed, memory ON vs OFF")
    lines.append(f"seed={report.seed}  max_attempts={report.max_attempts}  "
                 f"briefs={len(report.brief_order)}")
    lines.append("")
    lines.append(f"{'model':<24} {'arm':<4} {'solved':>7} {'rate':>7} "
                 f"{'shape':>6} {'atts':>5} {'recalled':>9} {'admitted':>9}")
    lines.append("-" * 80)
    deltas: List[float] = []
    for model, arms in report.by_model().items():
        for name in ARMS:
            a = arms.get(name)
            if a is None:
                continue
            recalled = sum(r.recalled_episodes for r in a.runs)
            admitted = sum(r.memory_admitted for r in a.runs)
            lines.append(
                f"{model:<24} {name:<4} {a.solved:>3}/{a.n:<3} "
                f"{100 * a.rate:>6.1f}% {a.solved_shape:>6} "
                f"{a.mean_attempts:>5.2f} {recalled:>9} {admitted:>9}")
        if ARM_OFF in arms and ARM_ON in arms:
            d = 100 * (arms[ARM_ON].rate - arms[ARM_OFF].rate)
            deltas.append(d)
            lines.append(f"{'':<24} {'d':<4} {'':>7} {d:>+6.1f}pp")
        lines.append("")

    if deltas:
        mean = sum(deltas) / len(deltas)
        lines.append(f"MEAN DELTA (on - off): {mean:+.1f} pp across "
                     f"{len(deltas)} model(s)")
        if mean > 0:
            lines.append("Memory helped. The burden of proof is discharged FOR "
                         "this corpus, this model set and this retrieval.")
        elif mean < 0:
            lines.append("MEMORY HURT. This replicates Agent-S's result in a new "
                         "domain: retrieved experience was net-negative even with "
                         "an oracle on every write. That is a real finding and it "
                         "is reported as one. Do not ship memory on.")
        else:
            lines.append("Memory did nothing measurable. A mechanism that costs "
                         "tokens and buys nothing is a mechanism to delete.")

    fps: Dict[str, int] = {}
    for a in report.arms:
        for code, n in a.false_positive_counts.items():
            fps[code] = fps.get(code, 0) + n
    lines.append("")
    lines.append("VERIFIER FALSE POSITIVES observed (fleet said broken, the "
                 "measured gate said the part was fine):")
    if fps:
        for code, n in sorted(fps.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"  {code:<40} {n:>4}")
        lines.append("  -> this is the signal eval/selftest/fleet_audit.py needs, "
                     "and the harness had no way to see it before.")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="memory ON vs OFF A/B")
    ap.add_argument("--model", action="append",
                    help="ollama model; repeat for several")
    ap.add_argument("--briefs", default="all")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    ap.add_argument("--cache-dir", default=".memory_ab_cache")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args(list(argv) if argv is not None else None)

    models = args.model or ["qwen2.5-coder:3b"]
    report = run(models, selector=args.briefs, seed=args.seed,
                 max_attempts=args.max_attempts, cache_dir=args.cache_dir)
    print(format_text(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
