"""The JUDGE surface -- score a generated part with checkable properties, not a VLM.

``eval/judge`` carried the deterministic CAD graders mined from cad-judge,
CADGenBench and cadrille: a structural compiler-review, the CAD Score composite,
Betti topology matching, and best-of-N invalidity aggregation. Only the
compiler-review had a consumer (the agent's CRM refine loop); the rest were
correct, tested and unreachable. This module is the dispatcher: one surface over
the whole grading family, replacing the weak VLM judge (r=0.71) with checkable
math.

    review(operations)                    -> the structural compiler-review verdict
    feedback(review)                      -> the re-promptable feedback string
    cad_score(...)                        -> the gated CADGenBench composite
    topology_match(cand, gt)              -> Betti-triple agreement in [0, 1]
    betti(faces, n_voids)                 -> (b0, b1, b2) of a tessellated boundary
    select_best(candidates)               -> best-of-N min-CD / max-IoU per sample
    aggregate(samples)                    -> a run's invalidity ratio + skip curve

Adapters only: the judge modules are never modified and are imported LAZILY
inside their routes. Everything here is stdlib-only, deterministic, and returns
values a reward signal or a gate can trust.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, List, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "JudgeError",
    "review",
    "feedback",
    "cad_score",
    "topology_match",
    "betti",
    "select_best",
    "aggregate",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_JUDGE = "harnesscad.eval.judge."


class JudgeError(ValueError):
    """Base class for every judge-surface failure."""


# --------------------------------------------------------------------------- #
# Compiler-as-a-review (compiler_review)
# --------------------------------------------------------------------------- #
def review(operations: Sequence[dict]) -> dict:
    """Structurally review a sketch-extrude op sequence (format/geometry/extrusion/boolean)."""
    from harnesscad.eval.judge.compiler_review import review_sequence

    return review_sequence(operations).to_dict()


def feedback(review_result: Any) -> str:
    """Render a re-promptable feedback string from a failing review verdict.

    Accepts a :class:`compiler_review.ReviewResult` or its dict form.
    """
    from harnesscad.eval.judge.compiler_review import ReviewResult, feedback_message

    if isinstance(review_result, ReviewResult):
        result = review_result
    else:
        d = dict(review_result)
        result = ReviewResult(ok=bool(d.get("ok")), category=d.get("category"),
                              op_index=d.get("op_index"), reason=d.get("reason", ""))
    return feedback_message(result)


# --------------------------------------------------------------------------- #
# CAD Score composite + topology match (cad_score)
# --------------------------------------------------------------------------- #
def cad_score(*, is_valid: bool, shape: float, interface: float,
              topology: float, editing: bool = False) -> dict:
    """Compose the CADGenBench CAD Score from its axis scores (validity-gated)."""
    from harnesscad.eval.judge.cad_score import cad_score as _cad_score

    b = _cad_score(is_valid=is_valid, shape=shape, interface=interface,
                   topology=topology, editing=editing)
    return {"cad_score": b.cad_score, "is_valid": b.is_valid,
            "weights": dict(b.weights), "components": dict(b.components)}


def topology_match(candidate: Sequence[int], ground_truth: Sequence[int],
                   *, alpha: Optional[float] = None) -> float:
    """Topology-match score from two Betti triples (fuzzy log-ratio product)."""
    from harnesscad.eval.judge.cad_score import BETTI_SHARPNESS, topology_match as _tm

    return _tm(candidate, ground_truth,
               alpha=BETTI_SHARPNESS if alpha is None else float(alpha))


# --------------------------------------------------------------------------- #
# Betti numbers from a mesh (betti)
# --------------------------------------------------------------------------- #
def betti(faces: Sequence[Sequence[int]], *, n_voids: int = 0) -> Tuple[int, int, int]:
    """Full Betti triple ``(b0, b1, b2)`` of the solid bounded by a triangle mesh."""
    from harnesscad.eval.judge.betti import betti_from_mesh

    return betti_from_mesh(faces, n_voids=int(n_voids))


# --------------------------------------------------------------------------- #
# Best-of-N aggregation (best_of_n)
# --------------------------------------------------------------------------- #
def select_best(candidates: Sequence[Any]) -> dict:
    """Best-of-N over one sample's candidates: min-CD, max-IoU, valid count.

    ``candidates`` are :class:`best_of_n.Candidate` objects (or mappings with
    ``valid`` / ``cd`` / ``iou``).
    """
    from harnesscad.eval.judge.best_of_n import Candidate, select_best as _select_best

    cands = [c if isinstance(c, Candidate)
             else Candidate(valid=bool(c.get("valid")), cd=c.get("cd"),
                            iou=c.get("iou"))
             for c in candidates]
    r = _select_best(cands)
    return {"n_candidates": r.n_candidates, "n_valid": r.n_valid,
            "best_cd": r.best_cd, "best_iou": r.best_iou, "any_valid": r.any_valid}


def aggregate(samples: Sequence[Any], *, skip_max: int = 4) -> dict:
    """Aggregate per-sample best-of-N results into a run report (IR, means, skip curve).

    ``samples`` are :class:`best_of_n.SampleResult` objects (e.g. the objects
    :func:`best_of_n.select_best` returns).
    """
    from harnesscad.eval.judge.best_of_n import SampleResult, aggregate_run

    rep = aggregate_run(list(samples), skip_max=int(skip_max))
    return {"n_samples": rep.n_samples, "invalidity_ratio": rep.invalidity_ratio,
            "mean_iou": rep.mean_iou, "mean_cd": rep.mean_cd,
            "median_cd": rep.median_cd,
            "skip_curve": [list(row) for row in rep.skip_curve]}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index():
    return {e.dotted: e
            for e in capability_registry.find(layer="eval", package="judge")}


def _available(dotted: str) -> bool:
    return dotted in _index()


# (group, route, module, doc) -- the dispatch table for the grading family.
_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("compiler", "review", _JUDGE + "compiler_review",
     "structural compiler-review of a sketch-extrude sequence + CRM feedback"),
    ("score", "cad_score", _JUDGE + "cad_score",
     "the gated CADGenBench CAD Score composite + Betti topology match"),
    ("topology", "betti", _JUDGE + "betti",
     "Betti numbers (b0, b1, b2) of a tessellated boundary, kernel-free"),
    ("bestofn", "aggregate", _JUDGE + "best_of_n",
     "best-of-N selection + invalidity ratio + skip-worst curve (cadrille)"),
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
                        help="list every judge route")
    parser.add_argument("--review", default=None, metavar="JSON",
                        help="a sketch-extrude op sequence (JSON or @file) to review")
    parser.add_argument("--betti", default=None, metavar="JSON",
                        help="a triangle-face list (JSON or @file) -> Betti triple")
    parser.add_argument("--n-voids", type=int, default=0,
                        help="void-shell count for --betti (default 0)")
    parser.add_argument("--unadapted", action="store_true",
                        help="list judge modules with no route")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def _load(text: str) -> Any:
    if text.startswith("@"):
        with open(text[1:], "r", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(text)


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "review", None):
        result = review(_load(args.review))
        result["feedback"] = feedback(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("ok") else 1

    if getattr(args, "betti", None):
        faces = _load(args.betti)
        triple = betti(faces, n_voids=getattr(args, "n_voids", 0))
        print(json.dumps({"betti": list(triple)}, indent=2, sort_keys=True))
        return 0

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
        prog="harnesscad judge",
        description="judge surface: compiler-review, CAD Score, Betti topology "
                    "match, best-of-N invalidity aggregation")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
