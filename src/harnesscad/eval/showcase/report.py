"""The scoreboard: which models can actually build a part from a plain brief.

Two numbers per model, and they are not the same number:

    solved    -- the harness reached a verified solid (no ERROR diagnostics).
    on brief  -- that solid is the part that was asked for (analytic volume
                 match + the required features are genuinely in the op stream).

A model that scores 8 solved / 1 on-brief is building *something* eight times and
the right thing once. Reporting only the first number would be a lie of omission,
so both are always printed, and the gap is the headline.

`unaided` counts the parts a model produced with NO human intervention at all --
which, in this package, is all of them: nothing here hand-edits an op stream. Any
record with `hand_fixed` true is DISQUALIFIED and listed separately.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from harnesscad.eval.showcase.briefs import BRIEFS, brief_by_id
from harnesscad.eval.showcase.models import MODELS

__all__ = ["scoreboard", "write_results", "render_markdown"]


def _pct(n: int, d: int) -> str:
    return "0%" if not d else "%d%%" % round(100.0 * n / d)


def scoreboard(runs: Sequence[dict]) -> Dict[str, Any]:
    """Aggregate raw run records into the model/brief scoreboard."""
    models: List[str] = []
    for r in runs:
        if r["model"] not in models:
            models.append(r["model"])
    briefs = [b.id for b in BRIEFS if any(r["brief_id"] == b.id for r in runs)]

    by_pair: Dict[str, dict] = {f"{r['model']}|{r['brief_id']}": r for r in runs}

    per_model: List[dict] = []
    for m in models:
        rs = [r for r in runs if r["model"] == m]
        solved = [r for r in rs if r["solved"]]
        on_brief = [r for r in solved if (r.get("grade") or {}).get("on_brief")]
        images = [r for r in solved if (r.get("render") or {}).get("ok")]
        disq = [r for r in rs if r.get("hand_fixed")]
        first_try = [r for r in solved if r["attempt_count"] == 1]
        corrected = [r for r in solved if r["attempt_count"] > 1]
        per_model.append({
            "model": m,
            "attempted": len(rs),
            "solved": len(solved),
            "on_brief": len(on_brief),
            "unaided": len(on_brief) - len(disq),
            "disqualified": len(disq),
            "images_valid": len(images),
            "first_attempt": len(first_try),
            "needed_correction": len(corrected),
            "mean_attempts_when_solved": (
                round(sum(r["attempt_count"] for r in solved) / len(solved), 2)
                if solved else None),
            "seconds": round(sum(r.get("seconds") or 0.0 for r in rs), 1),
            "diagnostics_seen": sorted({c for r in rs for c in r["diagnostics_seen"]}),
        })
    per_model.sort(key=lambda d: (-d["on_brief"], -d["solved"], d["model"]))

    per_brief: List[dict] = []
    for b in briefs:
        brief = brief_by_id(b)
        rs = [r for r in runs if r["brief_id"] == b]
        solved = [r for r in rs if r["solved"]]
        on_brief = [r for r in solved if (r.get("grade") or {}).get("on_brief")]
        per_brief.append({
            "brief_id": b,
            "tier": brief.tier,
            "text": brief.text,
            "attempted": len(rs),
            "solved_by": sorted(r["model"] for r in solved),
            "on_brief_by": sorted(r["model"] for r in on_brief),
            "nobody_solved": not solved,
            "nobody_on_brief": not on_brief,
        })
    per_brief.sort(key=lambda d: (d["tier"], d["brief_id"]))

    # Every ERROR diagnostic the fleet threw at a model, and how often.
    diag_counts: Dict[str, int] = {}
    for r in runs:
        for a in r.get("attempts", []):
            for code in a.get("error_codes", []):
                diag_counts[code] = diag_counts.get(code, 0) + 1
            if a.get("parse_error"):
                diag_counts["plan-parse-error"] = diag_counts.get("plan-parse-error", 0) + 1

    totals = {
        "pairs": len(runs),
        "solved": sum(1 for r in runs if r["solved"]),
        "on_brief": sum(1 for r in runs if (r.get("grade") or {}).get("on_brief")),
        "images_valid": sum(1 for r in runs if (r.get("render") or {}).get("ok")),
        "hand_fixed": sum(1 for r in runs if r.get("hand_fixed")),
        "models": len(models),
        "briefs": len(briefs),
    }
    return {
        "totals": totals,
        "per_model": per_model,
        "per_brief": per_brief,
        "diagnostic_counts": dict(sorted(diag_counts.items(),
                                         key=lambda kv: (-kv[1], kv[0]))),
        "matrix": {
            r["model"] + "|" + r["brief_id"]: {
                "solved": r["solved"],
                "on_brief": bool((r.get("grade") or {}).get("on_brief")),
                "attempts": r["attempt_count"],
                "why": r["failure_reason"] or "",
                "image": (r.get("render") or {}).get("path"),
            }
            for r in by_pair.values()
        },
    }


def best_per_brief(runs: Sequence[dict]) -> Dict[str, Optional[dict]]:
    """The best successful run for each brief.

    Ranking (honest, not flattering): on-brief first, then the smallest volume
    error, then the fewest attempts, then the smaller model (a 1.5b that got it
    right beats a 14b that got it right). Ties break on model name for
    determinism.
    """
    order = {m: i for i, m in enumerate(MODELS)}
    best: Dict[str, Optional[dict]] = {}
    for b in BRIEFS:
        cands = [r for r in runs if r["brief_id"] == b.id and r["solved"]]
        if not cands:
            best[b.id] = None
            continue

        def key(r: dict):
            g = r.get("grade") or {}
            err = g.get("volume_rel_error")
            return (
                0 if g.get("on_brief") else 1,
                err if err is not None else 9.0,
                r["attempt_count"],
                order.get(r["model"], 99),
                r["model"],
            )

        best[b.id] = sorted(cands, key=key)[0]
    return best


def write_results(runs: Sequence[dict], out_dir: str) -> Dict[str, Any]:
    """Write results.json (raw + scoreboard) and return the scoreboard."""
    os.makedirs(out_dir, exist_ok=True)
    board = scoreboard(runs)
    payload = {
        "briefs": [b.to_dict() for b in BRIEFS],
        "models": list(MODELS),
        "scoreboard": board,
        "runs": list(runs),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return board


# --- markdown --------------------------------------------------------------
def render_markdown(runs: Sequence[dict], board: Dict[str, Any],
                    drawings: Optional[Sequence[dict]] = None) -> str:
    t = board["totals"]
    briefs = [b["brief_id"] for b in board["per_brief"]]
    lines: List[str] = []
    add = lines.append

    add("# Showcase: plain English in, verified solid out")
    add("")
    add("Every part below was produced by a LOCAL open-weight model driving the "
        "harness's own loop: `agents.agent.planner` writes CISP ops, "
        "`CISPServer(backend='frep', verify_level='full')` applies and verifies "
        "them, and the TYPED DIAGNOSTICS from a failed apply are fed straight "
        "back to the planner for the next attempt (up to 3). Nothing in this "
        "package edits a model's op stream.")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| pairs run (model x brief) | {t['pairs']} |")
    add(f"| verified solid produced | {t['solved']} ({_pct(t['solved'], t['pairs'])}) |")
    add(f"| ...and it was the part that was asked for | {t['on_brief']} "
        f"({_pct(t['on_brief'], t['pairs'])}) |")
    add(f"| renders decoded and validated | {t['images_valid']} |")
    add(f"| hand-fixed (disqualified) | {t['hand_fixed']} |")
    add("")
    add("**solved** means the harness reached a verified solid with no ERROR "
        "diagnostics. **on brief** means that solid's measured volume matches "
        "the analytic volume of the briefed part AND the required features are "
        "actually in the op stream. The gap between the two columns is models "
        "building *a* part instead of *the* part -- a bracket with no holes in "
        "it verifies perfectly.")
    add("")

    add("## Scoreboard (model)")
    add("")
    add("| model | solved | on brief | 1st attempt | needed correction | valid images | mean attempts | diagnostics it saw |")
    add("|---|---|---|---|---|---|---|---|")
    for m in board["per_model"]:
        add("| `%s` | %d/%d | **%d** | %d | %d | %d | %s | %s |" % (
            m["model"], m["solved"], m["attempted"], m["on_brief"],
            m["first_attempt"], m["needed_correction"], m["images_valid"],
            m["mean_attempts_when_solved"] if m["mean_attempts_when_solved"] else "-",
            ", ".join("`%s`" % c for c in m["diagnostics_seen"][:6]) or "-",
        ))
    add("")
    add("`on brief` is the honest column: parts produced UNAIDED that are the "
        "part the brief described.")
    add("")

    add("## Scoreboard (brief)")
    add("")
    add("| brief | tier | solved by | on brief by |")
    add("|---|---|---|---|")
    for b in board["per_brief"]:
        add("| `%s` | %d | %s | %s |" % (
            b["brief_id"], b["tier"],
            ", ".join("`%s`" % m for m in b["solved_by"]) or "**nobody**",
            ", ".join("`%s`" % m for m in b["on_brief_by"]) or "**nobody**",
        ))
    add("")
    unsolved = [b["brief_id"] for b in board["per_brief"] if b["nobody_solved"]]
    unbriefed = [b["brief_id"] for b in board["per_brief"] if b["nobody_on_brief"]]
    add("Briefs no model could build at all: %s" %
        (", ".join("`%s`" % b for b in unsolved) if unsolved else "none"))
    add("")
    add("Briefs no model built CORRECTLY: %s" %
        (", ".join("`%s`" % b for b in unbriefed) if unbriefed else "none"))
    add("")

    add("## The matrix")
    add("")
    add("`+` on brief, `o` verified solid but not the briefed part, `.` failed. "
        "The number is the attempt count.")
    add("")
    header = "| model | " + " | ".join("`%s`" % b for b in briefs) + " |"
    add(header)
    add("|" + "---|" * (len(briefs) + 1))
    for m in board["per_model"]:
        cells = []
        for b in briefs:
            cell = board["matrix"].get(m["model"] + "|" + b)
            if not cell:
                cells.append("-")
            elif cell["on_brief"]:
                cells.append("+%d" % cell["attempts"])
            elif cell["solved"]:
                cells.append("o%d" % cell["attempts"])
            else:
                cells.append(".%d" % cell["attempts"])
        add("| `%s` | %s |" % (m["model"], " | ".join(cells)))
    add("")

    add("## What the harness caught (typed diagnostics fed back to the models)")
    add("")
    add("| diagnostic | times raised |")
    add("|---|---|")
    for code, n in board["diagnostic_counts"].items():
        add("| `%s` | %d |" % (code, n))
    add("")

    add("## Why the failures failed")
    add("")
    add("| model | brief | attempts | the diagnostic it could not fix |")
    add("|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["brief_id"], r["model"])):
        if not r["solved"]:
            add("| `%s` | `%s` | %d | %s |" % (
                r["model"], r["brief_id"], r["attempt_count"],
                (r["failure_reason"] or "").replace("|", "/")[:160]))
    add("")

    add("## Verified solids that are NOT the briefed part")
    add("")
    add("| model | brief | what is wrong |")
    add("|---|---|---|")
    any_off = False
    for r in sorted(runs, key=lambda r: (r["brief_id"], r["model"])):
        g = r.get("grade") or {}
        if r["solved"] and not g.get("on_brief"):
            any_off = True
            add("| `%s` | `%s` | %s |" % (
                r["model"], r["brief_id"],
                "; ".join(g.get("reasons") or ["unknown"]).replace("|", "/")))
    if not any_off:
        add("| - | - | none |")
    add("")

    add("## The images")
    add("")
    add("Every PNG was decoded back off disk with stdlib `zlib` "
        "(`eval/showcase/image.py`) and measured: silhouette fraction, luminance "
        "variance, distinct shades. A render that failed any check was deleted, "
        "not shipped.")
    add("")
    add("| brief | best model | attempts | volume (measured / briefed) | image | silhouette | variance |")
    add("|---|---|---|---|---|---|---|")
    for b in BRIEFS:
        best = best_per_brief(runs).get(b.id)
        if not best:
            add("| `%s` | **no model produced one** | - | - | - | - | - |" % b.id)
            continue
        g = best.get("grade") or {}
        img = best.get("render") or {}
        add("| `%s` | `%s` | %d | %s / %s mm3 | %s | %s | %s |" % (
            b.id, best["model"], best["attempt_count"],
            "%.0f" % (g.get("volume_mm3") or 0.0),
            "%.0f" % (g.get("expected_mm3") or 0.0),
            "`%s`" % img["path"] if img.get("path") else "FAILED VALIDATION",
            img.get("silhouette", "-"), img.get("variance", "-"),
        ))
    add("")
    if drawings:
        add("## Engineering drawings")
        add("")
        add("| brief | model | drawing |")
        add("|---|---|---|")
        for d in drawings:
            add("| `%s` | `%s` | `%s` |" % (d["brief_id"], d["model"], d["path"]))
        add("")

    add("## Reproducing any part")
    add("")
    add("Every record in `results.json` carries the model tag, the seed, the "
        "brief text, the attempt count and the exact op stream that verified. "
        "The models run at temperature 0 with `seed=7`; the frep backend and the "
        "renderer are deterministic.")
    add("")
    add("```")
    add("python -m harnesscad.eval.showcase.cli sweep --model qwen2.5-coder:7b --brief bracket")
    add("```")
    add("")
    add("## Honesty notes")
    add("")
    add("* Nothing in this package hand-edits a model's ops. The `hand_fixed` "
        "flag exists so the claim is machine-checkable; it is false on every "
        "record (`hand_fixed: %d`)." % t["hand_fixed"])
    add("* `on brief` is graded on the frep backend's MESH volume (marching cubes "
        "over an SDF), so it carries a few percent of sampling error; the "
        "per-brief tolerance absorbs that and nothing wider.")
    add("* A part can render beautifully and still be wrong. Any solid whose "
        "volume or features do not match the brief is listed above under "
        "\"Verified solids that are NOT the briefed part\", however good the "
        "picture looks.")
    return "\n".join(lines) + "\n"
