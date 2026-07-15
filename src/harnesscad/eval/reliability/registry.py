"""The RELIABILITY surface -- detect a bad build, then act on it.

``eval/reliability`` carried a whole repair-loop subsystem: a B-rep healer and a
diagnostic-to-repair advisor, the GenCAD self-repair convergence loop and its
feasibility metrics, a retrieval fallback tier, two error normalisers, the
corrected-CoT grammar gate, and an MCTS search over op sequences. None of it was
reachable behind one surface. This module is the dispatcher: the route the
harness takes AFTER verification rejects a build, on the way back to a valid one.

    advise(diagnostics, opdag)      -> concrete repair suggestions per diagnostic
    heal(backend)                   -> OCCT ShapeFix/ShapeUpgrade over a B-rep
    converge(commands, ...)         -> the detect -> repair -> re-check loop
    feasibility(flags)              -> feasibility rate F = V / (V + I)
    benchmark(before, after)        -> baseline-vs-repaired repair-success report
    fallback(brief, ...)            -> the nearest known-good precedent (last tier)
    normalize_code_error(exc, ...)  -> a stable CodeError category
    normalize_compiler(error, ...)  -> a stable compiler diagnostic code
    grammar_gate(trajectory, ...)   -> the sGC/rGC verify of a corrected CoT
    search(planner, factory, brief) -> MCTS over op sequences (AlphaCAD/LATS)

Adapters only: the reliability modules are never modified, and every one is
imported LAZILY inside its route (some pull OCCT, the memory store, or the
verifier fleet, and must not be dragged in just to enumerate the surface).
Deterministic; nothing here shells out or touches the network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "ReliabilityError",
    "advise",
    "heal",
    "converge",
    "feasibility",
    "benchmark",
    "fallback",
    "normalize_code_error",
    "normalize_compiler",
    "grammar_gate",
    "search",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_REL = "harnesscad.eval.reliability."


class ReliabilityError(ValueError):
    """Base class for every reliability-surface failure."""


# --------------------------------------------------------------------------- #
# Diagnostic-to-repair advisor + geometric heal (brep_repair)
# --------------------------------------------------------------------------- #
def advise(diagnostics, opdag=None) -> List[dict]:
    """Map verifier diagnostics to concrete, agent-facing repair suggestions.

    Each suggestion names candidate CISP ops/edits to try next and its rung on
    the ErrorRecovery ladder. Deterministic and order-preserving; ``opdag`` is
    optional (used to name the exact op to drop/edit when present).
    """
    from harnesscad.eval.reliability.brep_repair import RepairAdvisor

    return [s.to_dict() for s in RepairAdvisor().suggest(diagnostics, opdag)]


def heal(backend) -> dict:
    """Heal a backend's current B-rep with OCCT ShapeFix/ShapeUpgrade.

    Returns the :class:`brep_repair.RepairResult` as a dict. Degrades to a clean
    no-op (with a note) when OCCT is unavailable or there is nothing to fix.
    """
    from harnesscad.eval.reliability.brep_repair import repair_solid

    return repair_solid(backend).to_dict()


# --------------------------------------------------------------------------- #
# The repair-loop driver (repair_loop)
# --------------------------------------------------------------------------- #
def converge(commands: Sequence[Any], max_iterations: int = 8, checker=None) -> dict:
    """Iterate detect -> repair -> re-check on a command sequence until feasible.

    Returns the :class:`repair_loop.LoopResult` as a dict. ``checker`` defaults
    to the structural taxonomy check; pass an OCCT-backed predicate for the real
    kernel.
    """
    from harnesscad.eval.reliability.repair_loop import repair_until_feasible

    return repair_until_feasible(commands, max_iterations=int(max_iterations),
                                 checker=checker).to_dict()


# --------------------------------------------------------------------------- #
# Feasibility + repair-success metrics (repair_metrics)
# --------------------------------------------------------------------------- #
def feasibility(flags: Sequence[Any]) -> dict:
    """Feasibility rate ``F = V / (V + I)`` and the valid/invalid tally."""
    from harnesscad.eval.reliability.repair_metrics import feasibility_report

    return feasibility_report(flags).to_dict()


def benchmark(before: Sequence[Any], after: Sequence[Any]) -> dict:
    """Paired baseline-vs-repaired feasibility benchmark (fixed, regressions, rates)."""
    from harnesscad.eval.reliability.repair_metrics import benchmark_repair

    return benchmark_repair(before, after).to_dict()


# --------------------------------------------------------------------------- #
# The retrieval fallback tier (fallback)
# --------------------------------------------------------------------------- #
def fallback(brief_or_features, reason: str = "", *, catalog=None,
             retriever=None, memory=None, min_confidence: float = 0.0) -> dict:
    """The last, always-succeeds recovery tier: the nearest known-good precedent.

    Any of ``catalog`` / ``retriever`` / ``memory`` may be None; with none
    configured the generic buildable prismatic block is returned at confidence 0.
    """
    from harnesscad.eval.reliability.fallback import RetrievalFallback

    tier = RetrievalFallback(catalog=catalog, retriever=retriever, memory=memory,
                             min_confidence=float(min_confidence))
    return tier.fallback(brief_or_features, reason=reason).to_dict()


# --------------------------------------------------------------------------- #
# Error normalisers (code_error, compiler_diagnostics)
# --------------------------------------------------------------------------- #
def normalize_code_error(exc: BaseException, operation: str = "",
                         signature: Optional[str] = None) -> dict:
    """Normalise a Python exception into a stable :class:`code_error.CodeError`."""
    from harnesscad.eval.reliability.code_error import normalize

    err = normalize(exc, operation=operation, signature=signature)
    return {"category": err.category, "operation": err.operation,
            "parameter": err.parameter, "expected": err.expected,
            "hint": err.hint}


def normalize_compiler(error: Any, provider: str = "") -> dict:
    """Normalise a raw CAD-compiler error string into a stable diagnostic code."""
    from harnesscad.eval.reliability.compiler_diagnostics import normalize_compiler_error

    diag = normalize_compiler_error(error, provider=provider)
    return {"code": diag.code, "raw": diag.raw, "provider": diag.provider}


# --------------------------------------------------------------------------- #
# Corrected-CoT grammar gate (cot_grammar_gate)
# --------------------------------------------------------------------------- #
def grammar_gate(trajectory: str, allowed: Sequence[str],
                 variant: str = "sGC") -> dict:
    """Grammar-constraint verify one corrected reasoning trajectory against P."""
    from harnesscad.eval.reliability.cot_grammar_gate import gc_check

    return gc_check(trajectory, allowed, variant=variant).to_dict()


# --------------------------------------------------------------------------- #
# MCTS search over op sequences (strategies.mcts)
# --------------------------------------------------------------------------- #
def search(planner, session_factory, brief: str, **kwargs):
    """Monte-Carlo Tree Search over CISP op sequences (AlphaCAD / LATS tier).

    Returns the :class:`strategies.mcts.MctsResult` (not serialised: it carries
    live ``Op`` objects and the search tree the caller inspects).
    """
    from harnesscad.eval.reliability.strategies.mcts import mcts_search

    return mcts_search(planner, session_factory, brief, **kwargs)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index():
    return {e.dotted: e
            for e in capability_registry.find(package="reliability")}


def _available(dotted: str) -> bool:
    return dotted in _index()


# (group, route, module, doc) -- the dispatch table for the repair subsystem.
_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("repair", "advise", _REL + "brep_repair",
     "diagnostic codes -> concrete candidate CISP ops/edits + recovery rung"),
    ("repair", "heal", _REL + "brep_repair",
     "OCCT ShapeFix/ShapeUpgrade heal of a backend's B-rep (cadquery-guarded)"),
    ("repair", "converge", _REL + "repair_loop",
     "the GenCAD detect -> repair -> re-check convergence loop"),
    ("repair", "converge", _REL + "sequence_repair",
     "the idempotent structural repair one loop iteration applies"),
    ("repair", "converge", _REL + "infeasibility_taxonomy",
     "the structural infeasibility diagnosis + default feasibility check"),
    ("metric", "feasibility", _REL + "repair_metrics",
     "feasibility rate F = V/(V+I) and the baseline-vs-repaired benchmark"),
    ("fallback", "fallback", _REL + "fallback",
     "the retrieval fallback tier: the nearest known-good precedent"),
    ("error", "normalize_code_error", _REL + "code_error",
     "normalise a Python exception into a stable CodeError category"),
    ("error", "normalize_compiler", _REL + "compiler_diagnostics",
     "normalise a raw CAD-compiler error into a stable diagnostic code"),
    ("cot", "grammar_gate", _REL + "cot_grammar_gate",
     "the sGC/rGC grammar-constraint verify of a corrected reasoning trajectory"),
    ("search", "search", _REL + "strategies.mcts",
     "MCTS over CISP op sequences (AlphaCAD/LATS) -- the verifiable-reward tier"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, "no route yet") for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every reliability route")
    parser.add_argument("--feasibility", default=None, metavar="FLAGS",
                        help="comma-separated 0/1 feasibility flags -> the rate")
    parser.add_argument("--compiler-error", default=None, metavar="TEXT",
                        help="normalise a raw compiler error string to a code")
    parser.add_argument("--grammar-gate", default=None, metavar="TRAJ",
                        help="a corrected CoT (text or @file) to grammar-verify")
    parser.add_argument("--allowed", default="", metavar="A,B,C",
                        help="the allowed filename set P for --grammar-gate")
    parser.add_argument("--variant", default="sGC", choices=("sGC", "rGC"),
                        help="grammar-gate variant (strict sGC or relaxed rGC)")
    parser.add_argument("--unadapted", action="store_true",
                        help="list reliability modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _read(text: str) -> str:
    if text.startswith("@"):
        with open(text[1:], "r", encoding="utf-8") as fh:
            return fh.read()
    return text


def _flags(text: str) -> List[bool]:
    return [tok.strip() not in ("", "0", "false", "False", "no")
            for tok in text.split(",") if tok.strip() != ""]


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "feasibility", None) is not None:
        report = feasibility(_flags(args.feasibility))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if getattr(args, "compiler_error", None):
        report = normalize_compiler(args.compiler_error)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if getattr(args, "grammar_gate", None):
        allowed = [a for a in args.allowed.split(",") if a.strip()]
        report = grammar_gate(_read(args.grammar_gate), allowed,
                              variant=getattr(args, "variant", "sGC"))
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("accepted") else 1

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-9s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad reliability",
        description="reliability surface: repair advisor, B-rep heal, repair "
                    "loop, feasibility metrics, fallback, error normalisers, "
                    "CoT grammar gate, MCTS search")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
