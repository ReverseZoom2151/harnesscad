"""Sanity check: does the computed complexity agree with the hand labels?

``eval/pressure/briefs.py`` carries a hand-assigned ``difficulty`` column (1 easy
.. 4 trap) written by a human when the corpus was authored. It is the only
independent difficulty signal in the repo, so it is the obvious way to check
whether the structural metric in :mod:`harnesscad.eval.curriculum.complexity`
measures anything real -- and, where it disagrees, WHY.

The headline is a split, and the split is the finding:

  * on the 23 NON-TRAP briefs the agreement is high -- the metric reproduces the
    author's tiering almost exactly;
  * over all 28 it collapses, and it collapses for a reason that no reweighting
    can fix. The five ``trap_*`` briefs are labelled difficulty 4 because their
    stated dimensions are geometrically infeasible, not because their geometry
    is elaborate. ``trap_shell_too_thick`` is a rectangle, an extrude and a
    shell -- structurally the same four ops as ``shell_box_3mm``, which is
    labelled 2. A metric that counts ops CANNOT separate them, and one tuned
    until it did would be fitting the label, not measuring the task.

That is exactly the gap :mod:`harnesscad.eval.curriculum.difficulty` exists to
close: the traps are hard EMPIRICALLY, and the ReCAD max-reward rule finds them
from observed outcomes without any structural signal at all.

Deterministic and offline: it reads the checked-in corpus and computes counts.
Nothing here runs a model or touches the network.

    python -m harnesscad.eval.curriculum.validate
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from harnesscad.eval.curriculum import complexity as _cx

__all__ = ["ABLATIONS", "pressure_briefs", "correlation", "report", "render", "main"]


#: Weight vectors compared in the ablation table. ``full`` is the shipped
#: metric; the others isolate what each term contributes.
ABLATIONS: Dict[str, Dict[str, float]] = {
    "full": dict(_cx.WEIGHTS),
    "op_count_only": {"op_count": 1.0},
    "feature_depth_only": {"feature_depth": 1.0},
    "no_curve_count": {k: v for k, v in _cx.WEIGHTS.items() if k != "curve_count"},
    "no_feature_depth": {k: v for k, v in _cx.WEIGHTS.items() if k != "feature_depth"},
}


def pressure_briefs() -> Sequence[Any]:
    """The 28 checked-in pressure briefs.

    Imported lazily and from the SUBMODULE, so importing this package never
    drags in the pressure loop's model client.
    """
    from harnesscad.eval.pressure.briefs import BRIEFS

    return BRIEFS


def _weighted(features: _cx.ComplexityFeatures, weights: Dict[str, float]) -> float:
    return round(
        sum(w * getattr(features, name) for name, w in sorted(weights.items())), 6
    )


def correlation(computed: Sequence[float], labelled: Sequence[float]) -> Dict[str, Any]:
    """Spearman rho, Kendall tau-b and the discordant-pair count."""
    from harnesscad.eval.bench.judges.judge_human_agreement import (
        kendall_tau_b,
        spearman,
    )

    concordant = discordant = 0
    n = len(computed)
    for i in range(n):
        for j in range(i + 1, n):
            dx = computed[i] - computed[j]
            dy = labelled[i] - labelled[j]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    return {
        "n": n,
        "spearman": round(spearman(list(computed), list(labelled)), 6),
        "kendall_tau_b": round(kendall_tau_b(list(computed), list(labelled)), 6),
        "concordant_pairs": concordant,
        "discordant_pairs": discordant,
    }


def report(briefs: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """The full agreement report over the pressure corpus."""
    briefs = list(pressure_briefs() if briefs is None else briefs)
    feats = [_cx.task_features(b) for b in briefs]
    labels = [float(getattr(b, "difficulty", 0)) for b in briefs]
    traps = [bool(getattr(b, "trap", False)) for b in briefs]

    out: Dict[str, Any] = {
        "n_briefs": len(briefs),
        "n_traps": sum(1 for t in traps if t),
        "ablations": {},
    }
    for name, weights in sorted(ABLATIONS.items()):
        scores = [_weighted(f, weights) for f in feats]
        keep = [i for i in range(len(briefs)) if not traps[i]]
        out["ablations"][name] = {
            "all": correlation(scores, labels),
            "non_trap": correlation(
                [scores[i] for i in keep], [labels[i] for i in keep]
            ),
        }

    rows: List[Dict[str, Any]] = []
    for brief, f in zip(briefs, feats):
        row = f.to_dict()
        row["id"] = _cx.task_id(brief)
        row["labelled_difficulty"] = int(getattr(brief, "difficulty", 0))
        row["trap"] = bool(getattr(brief, "trap", False))
        rows.append(row)
    rows.sort(key=lambda r: (r["score"], r["id"]))
    out["rows"] = rows
    return out


def render(payload: Optional[Dict[str, Any]] = None) -> str:
    """A plain-text table of the report."""
    payload = report() if payload is None else payload
    lines = [
        "curriculum complexity vs the pressure corpus's hand-assigned difficulty",
        "briefs: %d (%d traps)" % (payload["n_briefs"], payload["n_traps"]),
        "",
        "%-20s %18s %18s" % ("weights", "all (rho / tau)", "non-trap (rho / tau)"),
    ]
    for name in sorted(payload["ablations"]):
        entry = payload["ablations"][name]
        lines.append(
            "%-20s %8.4f %8.4f  %8.4f %8.4f"
            % (
                name,
                entry["all"]["spearman"],
                entry["all"]["kendall_tau_b"],
                entry["non_trap"]["spearman"],
                entry["non_trap"]["kendall_tau_b"],
            )
        )
    lines += ["", "%-8s %-24s %5s %6s %s" % ("score", "brief", "label", "trap", "level")]
    from harnesscad.eval.curriculum import ordering as _ord

    by_id = {_cx.task_id(b): b for b in pressure_briefs()}
    for row in payload["rows"]:
        brief = by_id.get(row["id"])
        lines.append(
            "%8.2f %-24s %5d %6s %s"
            % (
                row["score"],
                row["id"],
                row["labelled_difficulty"],
                "yes" if row["trap"] else "no",
                _ord.structural_level(brief) if brief is not None else "?",
            )
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
