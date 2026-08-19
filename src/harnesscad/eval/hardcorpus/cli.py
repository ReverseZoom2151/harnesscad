"""`harnesscad hardcorpus` -- run a real model against the held-out hard corpus.

    harnesscad hardcorpus --model ornith:9b --limit 2 --out smoke.json
    harnesscad hardcorpus --model qwen3.6:27b --model ornith:9b --out results.json
    harnesscad hardcorpus --reference --limit 4         (no model; the ceiling)
    harnesscad hardcorpus --report results.json [--json] [--full-board]

Every run writes a ``HeldOutReport`` per model and prints the leaderboard row
that :mod:`harnesscad.eval.leaderboard.hardcorpus_board` builds from it -- BOTH
columns (what the field's grader would have believed, what the oracle measured)
and the gap between them.

A ``--limit`` run is a SMOKE TEST, not a result: its ``n`` is the number of
briefs actually scored and every row prints it, so a two-brief rate cannot be
mistaken for the corpus's.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.eval.hardcorpus import runner as runner_mod
from harnesscad.eval.hardcorpus.solver import DEFAULT_MAX_ATTEMPTS

__all__ = ["add_arguments", "run", "render_rows"]


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME",
        help="ollama model to score; repeat to score several (default: %s)"
             % ", ".join(runner_mod.DEFAULT_MODELS))
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="score only the first N held-out briefs (0 = all). A smoke-test "
             "knob: a limited run is reported with its own smaller n and is "
             "cached as a different cell from the full sweep.")
    parser.add_argument("--out", default=runner_mod.DEFAULT_OUT,
                        metavar="RESULTS.JSON",
                        help="write/append the submissions JSON here (resumable)")
    parser.add_argument("--cache", default=runner_mod.DEFAULT_CACHE, metavar="DIR",
                        help="directory for the model-output cache (resume is "
                             "free and byte-identical)")
    parser.add_argument("--seed", type=int, default=runner_mod.DEFAULT_SEED,
                        help="model seed (recorded in the results file)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, dest="max_attempts",
                        default=DEFAULT_MAX_ATTEMPTS,
                        help="model calls per brief; attempts after the first "
                             "exist only to repair an UNPARSEABLE response")
    parser.add_argument("--max-tokens", type=int, dest="max_tokens",
                        default=runner_mod.DEFAULT_MAX_TOKENS,
                        help="completion budget per call (default: %d). A "
                             "reasoning model spends this thinking before it "
                             "reaches the array, and a truncated array is an "
                             "INVALID attempt. NOTE: the completion cache is "
                             "keyed by model+seed+temperature+attempt+messages "
                             "and NOT by this, so point --cache somewhere fresh "
                             "when you change it."
                             % runner_mod.DEFAULT_MAX_TOKENS)
    parser.add_argument("--no-resume", action="store_true", dest="no_resume",
                        help="ignore any existing cells in --out and re-run them")
    parser.add_argument("--reference", action="store_true",
                        help="also score the corpus's own reference solutions -- "
                             "the ceiling and the self-test; needs no model")
    parser.add_argument("--no-models", action="store_true", dest="no_models",
                        help="run no model at all (use with --reference)")
    parser.add_argument("--report", default=None, metavar="RESULTS.JSON",
                        help="print the board for an existing results file and exit")
    parser.add_argument("--full-board", action="store_true", dest="full_board",
                        help="print the whole leaderboard, including the "
                             "near-miss and contract-residual proofs (needs the "
                             "exact kernel and is slow)")
    parser.add_argument("--json", action="store_true",
                        help="emit the submissions/board as JSON")
    return parser


def render_rows(standings: Sequence[Any]) -> str:
    """The compact board: both columns, the gap, and what was unmeasurable."""
    from harnesscad.eval.leaderboard import hardcorpus_board as board

    lines: List[str] = []
    lines.append("HARD CORPUS (held out) -- the field's grader vs the measured oracle")
    lines.append("=" * 78)
    lines.append("%-4s %-18s %5s %7s %7s %7s %6s %6s"
                 % ("#", "submission", "n", "weak", "oracle", "gap", "fooled",
                    "built"))
    lines.append("-" * 78)
    ranked = board.ranking(list(standings))
    if not ranked:
        lines.append("     (no submission)")
    for i, s in enumerate(ranked, start=1):
        lines.append("%-4d %-18s %5d %7.3f %7.3f %7.3f %6d %6d"
                     % (i, s.name[:18], s.n, s.weak_rate, s.oracle_rate,
                        s.gap, s.field_fooled, s.built))
    lines.append("-" * 78)
    lines.append("'weak' is what a Text2CAD-Bench-style board would print alone;")
    lines.append("'oracle' is what measurement says; 'fooled' is the count the")
    lines.append("field PASSED and the oracle FAILED -- parts it would overrate.")
    return "\n".join(lines)


def _render_invalid(payload: Dict[str, Any]) -> str:
    """What never became geometry at all. Unmeasurable output is a finding."""
    lines: List[str] = []
    lines.append("")
    lines.append("UNMEASURABLE OUTPUT -- briefs on which no parseable op stream arrived")
    lines.append("-" * 78)
    lines.append("%-20s %6s %8s %8s %8s" % ("submission", "n", "invalid",
                                            "errored", "calls"))
    for row in payload.get("submissions", []):
        stats = ((row.get("solver") or {}).get("stats") or {})
        lines.append("%-20s %6d %8d %8d %8d"
                     % (str(row.get("name", ""))[:20], int(row.get("n", 0) or 0),
                        int(stats.get("invalid", 0) or 0),
                        int(stats.get("errored", 0) or 0),
                        int(stats.get("model_calls", 0) or 0)))
    lines.append("-" * 78)
    lines.append("An 'invalid' brief is counted AGAINST the model in both columns:")
    lines.append("an empty op stream builds nothing, so neither grader passes it.")
    lines.append("'errored' means the model could not be reached -- an incomplete")
    lines.append("run, not a bad model.")
    return "\n".join(lines)


def _print(payload: Dict[str, Any], as_json: bool, full_board: bool) -> int:
    from harnesscad.eval.leaderboard import hardcorpus_board as board

    standings = runner_mod.standings(payload)
    if as_json:
        out = {"meta": payload.get("meta", {}),
               "ranking": [s.to_dict() for s in board.ranking(standings)],
               "solvers": [{"name": r.get("name"),
                            "stats": (r.get("solver") or {}).get("stats", {})}
                           for r in payload.get("submissions", [])]}
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
        return 0
    if full_board:
        print(board.Board(standings=list(standings)).render())
    else:
        print(render_rows(standings))
    print(_render_invalid(payload))
    return 0


def run(args: argparse.Namespace) -> int:
    as_json = bool(getattr(args, "json", False))
    full_board = bool(getattr(args, "full_board", False))

    if getattr(args, "report", None):
        payload = runner_mod.load_results(args.report)
        if not payload or not payload.get("submissions"):
            print("error: no submissions in %r" % args.report, file=sys.stderr)
            return 2
        return _print(payload, as_json, full_board)

    reference = bool(getattr(args, "reference", False))
    if getattr(args, "no_models", False):
        models: List[str] = []
        if not reference:
            print("error: --no-models with no --reference would run nothing",
                  file=sys.stderr)
            return 2
    else:
        models = list(args.model or runner_mod.DEFAULT_MODELS)

    limit: Optional[int] = int(getattr(args, "limit", 0) or 0) or None
    payload = runner_mod.run(
        models=models,
        out=args.out,
        limit=limit,
        seed=args.seed,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
        max_tokens=args.max_tokens,
        cache_dir=args.cache,
        resume=not args.no_resume,
        reference=reference,
    )
    print()
    rc = _print(payload, as_json, full_board)
    if not as_json:
        print()
        print("wrote: %s" % args.out)
        if limit:
            print("NOTE: --limit %d -- a smoke test, not a corpus result "
                  "(%d briefs available)."
                  % (limit, int(payload.get("meta", {}).get("n_available", 0) or 0)))
    return rc
