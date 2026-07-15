"""run — the campaign. Put each model in the GUI, drive the briefs, score.

This is the runnable that answers the question the coordinator asked: what is the
SOLVE RATE, with its denominator, per model, and what is the per-tier action
breakdown? It launches a REAL FreeCAD GUI (killed on teardown), drives each brief
through the accessibility tree and dialogs, exports through the harness channel,
and grades the built part against the scripted backend.

Safety is not optional here because this drives a real application on a real
machine: every brief gets a FRESH environment (fresh app, document, body) via a
scratch directory the harness owns; no user file is ever opened; FreeCAD is killed
on teardown even if a brief blows up. The guardrail deny-list means the agent
never saves.

Run it::

    python -m harnesscad.agents.cua.run --live --briefs 4 --models 2

Without ``--live`` it prints the plan (models, briefs) and exits, so it is safe to
invoke anywhere. With ``--live`` it needs FreeCAD + uiautomation + Ollama.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from harnesscad.agents.cua.briefs import BRIEFS, Brief, buildable
from harnesscad.agents.cua.loop import CuaSolve, solve
from harnesscad.agents.cua.models import discover_models, make_llm, select_models


@dataclass
class ModelScore:
    model: str
    solved: int = 0
    attempted: int = 0
    solves: List[CuaSolve] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.solved / self.attempted if self.attempted else 0.0

    def to_dict(self) -> dict:
        # Aggregate the per-tier tally across every brief this model attempted.
        tiers = {"script": 0, "semantic_gui": 0, "viewport_pick": 0, "refused": 0}
        for s in self.solves:
            for k, v in s.tier_counts.to_dict().items():
                tiers[k] += v
        return {
            "model": self.model,
            "solved": self.solved, "attempted": self.attempted,
            "solve_rate": "%d/%d" % (self.solved, self.attempted),
            "rate": round(self.rate, 3),
            "tier_totals": tiers,
            "briefs": [s.to_dict() for s in self.solves],
        }


def run_one(env_factory: Any, llm: Any, brief: Brief, *,
            max_iterations: int = 3) -> CuaSolve:
    """Drive one brief in a FRESH environment. Always tears the GUI down."""
    env = env_factory()
    try:
        env.reset()
        return solve(env, llm, brief, max_iterations=max_iterations)
    finally:
        try:
            env.close()
        finally:
            env.scratch.cleanup()


def run_campaign(models: Optional[List[str]] = None,
                 briefs: Optional[List[Brief]] = None,
                 *, max_iterations: int = 3, base: str = "http://localhost:11434",
                 env_factory: Any = None, report_path: Optional[str] = None,
                 log: Any = None) -> Dict[str, Any]:
    """Every (model, brief): plan, drive, measure, score. Returns the scorecard.

    ``env_factory`` builds a fresh Environment (default: a real
    ``FreeCADGuiEnvironment``). Injectable so a test can drive a fake. ``log`` is
    an optional ``print``-like sink for progress.
    """
    if env_factory is None:
        from harnesscad.io.cua.environment_freecad import FreeCADGuiEnvironment
        env_factory = FreeCADGuiEnvironment
    briefs = briefs if briefs is not None else buildable()
    models = models if models is not None else select_models(base)
    say = log or (lambda *a: None)

    scores: List[ModelScore] = []
    t0 = time.time()
    for model in models:
        say("=== model %s ===" % model)
        llm = make_llm(model, base=base)
        score = ModelScore(model=model)
        for brief in briefs:
            say("  brief %s ..." % brief.id)
            ts = time.time()
            s = run_one(env_factory, llm, brief, max_iterations=max_iterations)
            score.attempted += 1
            if s.solved:
                score.solved += 1
            score.solves.append(s)
            g = s.grade
            say("    -> solved=%s (%.0fs) %s"
                % (s.solved, time.time() - ts,
                   ("" if g is None else "diff.max=%s" % (
                       g.diff.max_delta if g.diff else "n/a"))))
        scores.append(score)

    card = {
        "elapsed_s": round(time.time() - t0, 1),
        "denominator_per_model": len(briefs),
        "briefs": [b.id for b in briefs],
        "models": [s.to_dict() for s in scores],
        "installed_models": [m.__dict__ for m in discover_models(base)],
    }
    if report_path:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(card, fh, indent=2, default=str)
    return card


def format_scorecard(card: Dict[str, Any]) -> str:
    """A terse, human-readable scorecard: solve rate + tier breakdown per model."""
    lines: List[str] = []
    n = card["denominator_per_model"]
    lines.append("CUA campaign: %d briefs/model, %.0fs total"
                 % (n, card["elapsed_s"]))
    lines.append("briefs: " + ", ".join(card["briefs"]))
    for m in card["models"]:
        t = m["tier_totals"]
        lines.append("")
        lines.append("%s  SOLVE RATE %s (%.0f%%)"
                     % (m["model"], m["solve_rate"], 100 * m["rate"]))
        lines.append("  tiers: script=%d semantic_gui=%d viewport_pick=%d refused=%d"
                     % (t["script"], t["semantic_gui"], t["viewport_pick"],
                        t["refused"]))
        for b in m["briefs"]:
            g = b.get("grade") or {}
            diff = (g.get("diff") or {}).get("max_delta", "n/a")
            lines.append("    %-22s solved=%-5s diff.max=%s  %s"
                         % (b["brief_id"], b["solved"], diff,
                            "" if b["solved"] else g.get("reason", b.get("error", ""))[:80]))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Put a model in the GUI and score it.")
    ap.add_argument("--live", action="store_true",
                    help="actually launch FreeCAD and drive it (needs the GUI)")
    ap.add_argument("--models", type=int, default=2, help="how many models")
    ap.add_argument("--briefs", type=int, default=0,
                    help="cap the brief count (0 = all buildable)")
    ap.add_argument("--iters", type=int, default=3, help="max repair iterations")
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--report", default=None, help="write the JSON scorecard here")
    args = ap.parse_args(argv)

    chosen = select_models(args.base, limit=args.models)
    all_briefs = buildable()
    briefs = all_briefs[:args.briefs] if args.briefs else all_briefs

    if not args.live:
        print("DRY RUN (pass --live to drive the GUI).")
        print("installed models:", [m.name for m in discover_models(args.base)])
        print("would run models:", chosen)
        print("would run briefs:", [b.id for b in briefs])
        return 0

    card = run_campaign(models=chosen, briefs=briefs,
                        max_iterations=args.iters, base=args.base,
                        report_path=args.report, log=lambda *a: print(*a, flush=True))
    print()
    print(format_scorecard(card))
    return 0


if __name__ == "__main__":
    sys.exit(main())
