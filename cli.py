"""harnesscad CLI — drive a HarnessSession from the command line.

    python cli.py apply <ops.json> [--backend stub|cadquery]
        Load a JSON array of ops, run them through a session, and pretty-print
        the ApplyOpsResult (ok, applied, digest, diagnostics).

    python cli.py demo [--backend stub|cadquery]
        Run a built-in sample (constrained plate -> extrude) and print the
        result plus a model summary.

Exit code is non-zero when the resulting model is not ok, so the CLI composes
in scripts / CI (`cli.py apply plan.json && next-step`).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from server import CISPServer


# The deterministic sample: sk1 = first sketch, e1 = first entity (rectangle).
DEMO_OPS: List[dict] = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 20.0},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]


def _print_result(result: dict) -> None:
    print(f"ok:       {result['ok']}")
    print(f"applied:  {result['applied']}")
    print(f"digest:   {result['digest']}")
    diags = result.get("diagnostics") or []
    if diags:
        print("diagnostics:")
        for d in diags:
            where = f" @{d['where']}" if d.get("where") else ""
            print(f"  [{d['severity']}] {d['code']}: {d['message']}{where}")
    else:
        print("diagnostics: (none)")
    if result.get("rejected"):
        print(f"rejected:  {json.dumps(result['rejected'])}")


def _run_ops(ops: List[dict], backend: str) -> dict:
    server = CISPServer(backend=backend)
    return server.applyOps(ops)


def cmd_apply(args: argparse.Namespace) -> int:
    try:
        with open(args.ops, "r", encoding="utf-8") as fh:
            ops = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not load ops from {args.ops!r}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(ops, list):
        print(f"error: {args.ops!r} must contain a JSON array of ops", file=sys.stderr)
        return 2
    result = _run_ops(ops, args.backend)
    _print_result(result)
    return 0 if result["ok"] else 1


def cmd_demo(args: argparse.Namespace) -> int:
    server = CISPServer(backend=args.backend)
    result = server.applyOps([dict(op) for op in DEMO_OPS])
    _print_result(result)
    print("summary:", json.dumps(server.query("summary")["result"], sort_keys=True))
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="harnesscad CISP CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_apply = sub.add_parser("apply", help="run a JSON array of ops")
    p_apply.add_argument("ops", help="path to a JSON array of ops")
    p_apply.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
    p_apply.set_defaults(func=cmd_apply)

    p_demo = sub.add_parser("demo", help="run the built-in constrained-plate sample")
    p_demo.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
