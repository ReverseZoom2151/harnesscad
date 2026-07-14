"""Do our intrinsic metrics predict whether the part actually SOLVED?

The repository ships ~200 intrinsic benchmark metric modules and had **never**
measured whether any of them correlates with the one outcome it cares about.
`eval/bench/harness/metric_correlation.py` implements exactly that computation
(Pearson r over aligned per-shape series) and its only importer was its own test.

This module points it at the one labelled corpus the harness owns:
`assets/pressure/results.json` -- 208 graded attempts from the controlled A/B,
each carrying the built geometry's measurements AND the grader's `solved` verdict.
Correlating a 0/1 outcome with a continuous metric is the point-biserial
correlation, which IS Pearson r; `metric_correlation.pearson` is therefore the
right function, unchanged.

    python -m harnesscad.eval.bench.harness.pressure_correlation

An intrinsic metric that does not correlate with correctness is decoration. This
prints which ones are which, and it is honest about the caveats: n=208 attempts
over 12 briefs is a small, brief-confounded sample, and a correlation here is
evidence, not proof.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.eval.bench.harness.metric_correlation import pearson

__all__ = ["DEFAULT_RESULTS", "features", "rows", "correlate", "format_text", "main"]

DEFAULT_RESULTS = os.path.join("assets", "pressure", "results.json")

#: Op tags counted as their own metric (op-mix is intrinsic and free).
_OP_TAGS = ("new_sketch", "add_rectangle", "add_circle", "add_line", "constrain",
            "extrude", "revolve", "hole", "fillet", "chamfer", "shell", "boolean",
            "linear_pattern", "circular_pattern", "mirror")


def features(record: Dict[str, Any], cell: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Intrinsic metrics of ONE graded attempt. None when the attempt was not graded.

    Everything here is computable without the brief's ground truth -- that is what
    makes it *intrinsic*, and what makes the question ("does it predict `solved`?")
    non-trivial.
    """
    grade = record.get("grade")
    if not grade:
        return None
    measure = grade.get("measure") or {}
    validity = measure.get("validity") or {}
    bbox = measure.get("bbox") or [0.0, 0.0, 0.0]
    vol = float(measure.get("volume") or 0.0)
    ops = record.get("ops") or []
    diags = grade.get("diagnostics") or []

    bx, by, bz = (float(bbox[0]), float(bbox[1]), float(bbox[2])) if len(bbox) == 3 \
        else (0.0, 0.0, 0.0)
    bvol = bx * by * bz
    extents = sorted([bx, by, bz])

    f: Dict[str, float] = {
        # -- plan shape
        "n_ops": float(len(ops)),
        "raw_chars": float(len(record.get("raw") or "")),
        "parse_ok": 1.0 if record.get("parse_ok") else 0.0,
        "attempt_index": float(record.get("attempt") or 0),
        "applied": float(grade.get("applied") or 0),
        "apply_ok": 1.0 if grade.get("apply_ok") else 0.0,
        "applied_fraction": (float(grade.get("applied") or 0) / len(ops)) if ops else 0.0,
        "built": 1.0 if grade.get("built") else 0.0,
        # -- geometry (intrinsic: no reference needed)
        "volume": vol,
        "bbox_volume": bvol,
        "fill_ratio": (vol / bvol) if bvol > 0 else 0.0,
        "min_extent": extents[0],
        "max_extent": extents[2],
        "aspect_ratio": (extents[2] / extents[0]) if extents[0] > 0 else 0.0,
        # -- kernel validity
        "is_valid": 1.0 if validity.get("is_valid") else 0.0,
        "manifold": 1.0 if validity.get("manifold") else 0.0,
        "watertight": 1.0 if validity.get("watertight") else 0.0,
        "solid_present": 1.0 if validity.get("solid_present") else 0.0,
        "genus": float(validity.get("genus") or 0),
        "euler_characteristic": float(validity.get("euler_characteristic") or 0),
        "validity_issues": float(validity.get("issues") or 0),
        # -- the fleet's own opinion (is the VERIFIER FLEET a predictor of truth?)
        "n_diagnostics": float(len(diags)),
        "n_error_diags": float(sum(1 for d in diags if d.get("severity") == "error")),
        "n_warning_diags": float(sum(1 for d in diags if d.get("severity") == "warning")),
        "n_distinct_codes": float(len({d.get("code") for d in diags})),
        "n_fleet_actionable": float(len(grade.get("fleet_actionable") or [])),
        "fleet_caught": 1.0 if grade.get("fleet_caught") else 0.0,
        # -- cost
        "seconds": float(record.get("seconds") or 0.0),
    }
    tags = [str(o.get("op")) for o in ops if isinstance(o, dict)]
    for tag in _OP_TAGS:
        f["n_op_" + tag] = float(tags.count(tag))
    return f


def rows(results: Dict[str, Any]) -> Tuple[List[Dict[str, float]], List[float]]:
    """(feature dicts, solved labels) over every GRADED attempt in the corpus."""
    xs: List[Dict[str, float]] = []
    ys: List[float] = []
    for cell in results.get("results", []):
        for rec in cell.get("records", []):
            f = features(rec, cell)
            if f is None:
                continue
            xs.append(f)
            ys.append(1.0 if (rec["grade"] or {}).get("solved") else 0.0)
    return xs, ys


def correlate(xs: Sequence[Dict[str, float]],
              ys: Sequence[float]) -> List[Tuple[str, float, float, int]]:
    """(metric, r, |r|, n_nonconstant) per metric, sorted by |r| descending.

    r is Pearson over the aligned series -- the point-biserial correlation with the
    0/1 `solved` label. `metric_correlation.pearson` returns 0.0 for a constant
    series, which is the honest answer: a metric that never varies cannot predict
    anything.
    """
    if not xs:
        return []
    names = sorted(xs[0])
    out: List[Tuple[str, float, float, int]] = []
    for name in names:
        col = [float(x.get(name, 0.0)) for x in xs]
        r = pearson(col, list(ys))
        n_distinct = len(set(col))
        out.append((name, r, abs(r), n_distinct))
    out.sort(key=lambda t: (-t[2], t[0]))
    return out


def _band(ar: float) -> str:
    if ar >= 0.5:
        return "STRONG"
    if ar >= 0.3:
        return "moderate"
    if ar >= 0.1:
        return "weak"
    return "NOISE"


def format_text(scored: Sequence[Tuple[str, float, float, int]], n: int) -> str:
    lines: List[str] = []
    lines.append("INTRINSIC METRICS vs `solved` -- point-biserial r over %d graded "
                 "attempts" % n)
    lines.append("=" * 78)
    lines.append("%-26s %8s %10s  %s" % ("metric", "r", "|r|", "verdict"))
    lines.append("-" * 78)
    for name, r, ar, distinct in scored:
        note = "" if distinct > 1 else "  (constant: never varied)"
        lines.append("%-26s %8.3f %10.3f  %s%s" % (name, r, ar, _band(ar), note))
    lines.append("")
    lines.append("A metric that does not correlate with correctness is decoration.")
    lines.append("Caveat, stated: n=%d attempts over 12 briefs, one seed, T=0. The "
                 "sample is brief-confounded (a metric can track brief difficulty "
                 "rather than plan quality). This is evidence, not proof." % n)
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("results", nargs="?", default=DEFAULT_RESULTS,
                        help="pressure results.json (default: %s)" % DEFAULT_RESULTS)
    parser.add_argument("--json", action="store_true", dest="as_json")


def run(args: argparse.Namespace) -> int:
    path = getattr(args, "results", DEFAULT_RESULTS)
    if not os.path.exists(path):
        print("error: no results at %r" % path, file=sys.stderr)
        return 2
    with open(path, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    xs, ys = rows(results)
    if not xs:
        print("error: no graded attempts in %r" % path, file=sys.stderr)
        return 2
    scored = correlate(xs, ys)
    if getattr(args, "as_json", False):
        print(json.dumps({
            "n": len(xs),
            "solved_rate": sum(ys) / len(ys),
            "metrics": [{"metric": m, "r": r, "abs_r": ar, "distinct": d}
                        for m, r, ar, d in scored],
        }, indent=2, sort_keys=True))
    else:
        print(format_text(scored, len(xs)))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pressure_correlation",
        description="Which intrinsic metrics predict `solved`, and which are noise.")
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
