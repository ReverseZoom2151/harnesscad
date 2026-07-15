"""`harnesscad pressure` — run the A/B, or print the report from a results file.

    harnesscad pressure --loop all --briefs all --out results.json
        (defaults to the four frontier tags: qwen3.6:27b, qwen3.6:35b,
         ornith:9b, ornith:35b -- pass --model to override)
    harnesscad pressure --report results.json
    harnesscad pressure --list-briefs
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from harnesscad.eval.pressure import report as report_mod
from harnesscad.eval.pressure import runner as runner_mod
from harnesscad.eval.pressure.briefs import BRIEFS, CATEGORIES, briefs_for
from harnesscad.eval.pressure.loops import (
    ALL_LOOPS, BLIND, HARNESS, LOOPS, ORACLE_BON, SELF_CONSISTENCY,
)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model", action="append", default=None, metavar="NAME",
        help="ollama model to test; repeat to test several (default: the four "
             "frontier tags -- %s)" % ", ".join(runner_mod.DEFAULT_MODELS))
    parser.add_argument(
        "--loop", action="append", default=None,
        choices=list(ALL_LOOPS) + ["both", "all"],
        help="'blind' = raw kernel errors, 'harness' = SOUND typed diagnostics, "
             "'oracle_bon' = Best-of-N selected by the differential oracle + the "
             "output gate, 'self_consistency' = majority vote over the SAME N "
             "draws (free), 'both' = blind+harness (the v1 A/B), 'all' = every "
             "arm. Repeatable. Default: both")
    parser.add_argument(
        "--briefs", default="all",
        help="'all', 'traps', 'notraps', a category (%s), or a CSV of brief ids"
             % "/".join(CATEGORIES))
    parser.add_argument("--seed", type=int, default=runner_mod.DEFAULT_SEED,
                        help="model seed (recorded in the results file)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, dest="max_attempts",
                        default=4, help="attempt budget per brief per arm")
    parser.add_argument("--out", default="pressure_results.json",
                        help="write/append the results JSON here (resumable)")
    parser.add_argument("--cache", default=runner_mod.DEFAULT_CACHE,
                        help="directory for the model-output cache")
    parser.add_argument("--no-resume", action="store_true", dest="no_resume",
                        help="ignore any existing cells in --out and re-run them")
    parser.add_argument("--report", default=None, metavar="RESULTS.JSON",
                        help="print the report for an existing results file and exit")
    parser.add_argument("--list-briefs", action="store_true", dest="list_briefs",
                        help="print the brief corpus and exit")
    parser.add_argument("--json", action="store_true",
                        help="with --report, dump the aggregate as JSON")
    return parser


def _cmd_list_briefs() -> int:
    print(f"{len(BRIEFS)} briefs")
    print("-" * 92)
    print(f"{'id':<24} {'category':<12} {'diff':>4} {'trap':>5}  brief")
    print("-" * 92)
    for b in BRIEFS:
        text = b.text if len(b.text) <= 44 else b.text[:41] + "..."
        print(f"{b.id:<24} {b.category:<12} {b.difficulty:>4} "
              f"{'Y' if b.trap else '':>5}  {text}")
    print("-" * 92)
    return 0


def _cmd_report(path: str, as_json: bool) -> int:
    payload = runner_mod.load_results(path)
    if not payload or not payload.get("results"):
        print(f"error: no results in {path!r}", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(
            {"meta": payload.get("meta", {}),
             "cells": {f"{k[0]}|{k[1]}": v for k, v in
                       report_mod.aggregate(payload["results"])["cells"].items()}},
            sort_keys=True, indent=2, default=str))
        return 0
    print(report_mod.render_v2(payload))
    return 0


def run(args: argparse.Namespace) -> int:
    if getattr(args, "list_briefs", False):
        return _cmd_list_briefs()
    if getattr(args, "report", None):
        return _cmd_report(args.report, bool(getattr(args, "json", False)))

    models: List[str] = args.model or list(runner_mod.DEFAULT_MODELS)
    loops: List[str] = []
    for sel in (args.loop or ["both"]):
        for l in (LOOPS if sel == "both" else
                  ALL_LOOPS if sel == "all" else (sel,)):
            if l not in loops:
                loops.append(l)
    try:
        briefs = briefs_for(args.briefs)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not briefs:
        print(f"error: --briefs {args.briefs!r} selected nothing", file=sys.stderr)
        return 2

    payload = runner_mod.run(
        models=models,
        briefs=briefs,
        loops=loops,
        seed=args.seed,
        temperature=args.temperature,
        max_attempts=args.max_attempts,
        out=args.out,
        cache_dir=args.cache,
        resume=not args.no_resume,
    )
    print()
    print(report_mod.render_v2(payload))
    print()
    print(f"wrote: {args.out}")
    return 0
