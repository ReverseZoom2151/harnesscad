"""Aggregation and the tables.

Everything here is pure: it takes the list of BriefResult dicts a run produced
and turns it into numbers. No model is called, no geometry is built, so a report
can be regenerated from a results.json forever.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from harnesscad.eval.pressure.loops import BLIND, HARNESS, LOOPS


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _fmt(x: Optional[float], spec: str = "5.2f") -> str:
    return "  -  " if x is None else format(x, spec)


def _pct(n: int, d: int) -> str:
    return "  -  " if not d else f"{100.0 * n / d:5.1f}%"


def aggregate(results: Sequence[dict]) -> Dict[str, Any]:
    """Roll a flat list of BriefResult dicts up into (model, loop) cells."""
    cells: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in results:
        key = (r["model"], r["loop"])
        c = cells.setdefault(key, {
            "model": r["model"], "loop": r["loop"],
            "n": 0, "solved": 0, "attempts": [], "attempts_to_solve": [],
            "invalid_ops": 0, "total_attempts": 0,
            "fleet_caught": 0, "fleet_missed": 0, "seconds": 0.0,
            "n_trap": 0, "solved_trap": 0,
            "n_plain": 0, "solved_plain": 0,
        })
        c["n"] += 1
        c["solved"] += int(r["solved"])
        c["attempts"].append(r["attempts_used"])
        c["total_attempts"] += r["attempts_used"]
        if r["attempts_to_solve"] is not None:
            c["attempts_to_solve"].append(r["attempts_to_solve"])
        c["invalid_ops"] += r["invalid_ops"]
        c["fleet_caught"] += r["fleet_caught"]
        c["fleet_missed"] += r["fleet_missed"]
        c["seconds"] += r["seconds"]
        if r["trap"]:
            c["n_trap"] += 1
            c["solved_trap"] += int(r["solved"])
        else:
            c["n_plain"] += 1
            c["solved_plain"] += int(r["solved"])

    for c in cells.values():
        c["solve_rate"] = c["solved"] / c["n"] if c["n"] else 0.0
        c["mean_attempts"] = _mean(c["attempts"])
        c["mean_attempts_to_solve"] = _mean(c["attempts_to_solve"])
        c["invalid_op_rate"] = (c["invalid_ops"] / c["total_attempts"]
                                if c["total_attempts"] else 0.0)
        c["trap_solve_rate"] = (c["solved_trap"] / c["n_trap"]) if c["n_trap"] else None
        c["plain_solve_rate"] = (c["solved_plain"] / c["n_plain"]) if c["n_plain"] else None
    return {"cells": cells}


def render_table(results: Sequence[dict]) -> str:
    agg = aggregate(results)
    cells = agg["cells"]
    lines: List[str] = []
    lines.append("model x loop")
    lines.append("-" * 108)
    lines.append(
        f"{'model':<20} {'loop':<8} {'n':>3} {'solved':>7} {'solve%':>7} "
        f"{'mean att':>9} {'att2solve':>10} {'invalid%':>9} "
        f"{'catches':>8} {'MISSES':>7} {'wall s':>8}")
    lines.append("-" * 108)
    for key in sorted(cells):
        c = cells[key]
        lines.append(
            f"{c['model']:<20} {c['loop']:<8} {c['n']:>3} {c['solved']:>7} "
            f"{_pct(c['solved'], c['n']):>7} "
            f"{_fmt(c['mean_attempts'], '9.2f')} "
            f"{_fmt(c['mean_attempts_to_solve'], '10.2f')} "
            f"{_pct(c['invalid_ops'], c['total_attempts']):>9} "
            f"{c['fleet_caught']:>8} {c['fleet_missed']:>7} "
            f"{c['seconds']:>8.0f}")
    lines.append("-" * 108)
    lines.append("solve%   = briefs whose FINAL plan matched the brief's geometric ground truth")
    lines.append("att2sol  = mean attempts on the briefs that were solved (lower is better)")
    lines.append("invalid% = share of attempts whose output would not parse into ops")
    lines.append("catches  = attempts where the fleet raised an actionable diagnostic")
    lines.append("MISSES   = attempts where the geometry was WRONG and the fleet said NOTHING")
    lines.append("           (fleet misses are bugs in the verifier fleet, not model failures)")
    return "\n".join(lines)


def render_split(results: Sequence[dict]) -> str:
    """Solve rate split by trap / non-trap. This is where the claim is decided."""
    agg = aggregate(results)
    cells = agg["cells"]
    lines: List[str] = []
    lines.append("solve rate, split by brief kind")
    lines.append("-" * 78)
    lines.append(f"{'model':<20} {'loop':<8} {'plain (n)':>16} {'TRAP (n)':>16} {'overall':>10}")
    lines.append("-" * 78)
    for key in sorted(cells):
        c = cells[key]
        plain = (f"{100.0 * c['plain_solve_rate']:5.1f}% ({c['n_plain']:>2})"
                 if c["plain_solve_rate"] is not None else "-")
        trap = (f"{100.0 * c['trap_solve_rate']:5.1f}% ({c['n_trap']:>2})"
                if c["trap_solve_rate"] is not None else "-")
        lines.append(
            f"{c['model']:<20} {c['loop']:<8} {plain:>16} {trap:>16} "
            f"{100.0 * c['solve_rate']:9.1f}%")
    lines.append("-" * 78)
    return "\n".join(lines)


def headline(results: Sequence[dict]) -> str:
    """The one number the whole exercise exists to produce: harness minus blind."""
    agg = aggregate(results)
    cells = agg["cells"]
    models = sorted({k[0] for k in cells})
    lines: List[str] = []
    lines.append("HEADLINE: does the typed-diagnostic loop beat the blind loop?")
    lines.append("-" * 90)
    lines.append(f"{'model':<20} {'blind solve%':>13} {'harness solve%':>15} "
                 f"{'delta':>9} {'blind att':>10} {'harness att':>12}")
    lines.append("-" * 90)
    tot_b = tot_h = tot_n = 0
    for m in models:
        b = cells.get((m, BLIND))
        h = cells.get((m, HARNESS))
        if not b or not h:
            continue
        db = 100.0 * b["solve_rate"]
        dh = 100.0 * h["solve_rate"]
        lines.append(
            f"{m:<20} {db:12.1f}% {dh:14.1f}% {dh - db:+8.1f}pp "
            f"{_fmt(b['mean_attempts'], '10.2f')} {_fmt(h['mean_attempts'], '12.2f')}")
        tot_b += b["solved"]
        tot_h += h["solved"]
        tot_n += b["n"]
    lines.append("-" * 90)
    if tot_n:
        pb = 100.0 * tot_b / tot_n
        ph = 100.0 * tot_h / tot_n
        lines.append(
            f"{'POOLED':<20} {pb:12.1f}% {ph:14.1f}% {ph - pb:+8.1f}pp "
            f"  ({tot_b}/{tot_n} vs {tot_h}/{tot_n} briefs)")
        verdict = ("TYPED DIAGNOSTICS WIN" if ph > pb else
                   "NO DIFFERENCE" if ph == pb else
                   "TYPED DIAGNOSTICS LOSE")
        lines.append("")
        lines.append(f"VERDICT: {verdict} ({ph - pb:+.1f} percentage points, "
                     f"{tot_h - tot_b:+d} briefs out of {tot_n})")
    return "\n".join(lines)


def per_brief(results: Sequence[dict]) -> str:
    """A per-brief A/B, so a reader can see exactly where the delta came from."""
    by: Dict[Tuple[str, str], Dict[str, dict]] = {}
    for r in results:
        by.setdefault((r["model"], r["brief"]), {})[r["loop"]] = r
    lines = ["per-brief A/B  (. = solved, X = failed)", "-" * 88]
    lines.append(f"{'model':<20} {'brief':<24} {'trap':>5} {'blind':>7} {'harness':>9} {'note':>12}")
    lines.append("-" * 88)
    for key in sorted(by):
        m, bid = key
        cell = by[key]
        b = cell.get(BLIND)
        h = cell.get(HARNESS)
        if not b or not h:
            continue
        bs = "." if b["solved"] else "X"
        hs = "." if h["solved"] else "X"
        note = ""
        if h["solved"] and not b["solved"]:
            note = "HARNESS WIN"
        elif b["solved"] and not h["solved"]:
            note = "HARNESS LOSS"
        lines.append(
            f"{m:<20} {bid:<24} {'Y' if b['trap'] else '':>5} "
            f"{bs:>7} {hs:>9} {note:>12}")
    lines.append("-" * 88)
    return "\n".join(lines)


def fleet_holes(results: Sequence[dict]) -> str:
    """The most valuable output: geometry the fleet passed but the ground truth
    rejected. Each of these is a verifier the harness does not have."""
    seen: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for rec in r["records"]:
            g = rec.get("grade")
            if not g or not g.get("fleet_missed"):
                continue
            for reason in g["reasons"]:
                key = f"{r['brief']}::{reason[:60]}"
                e = seen.setdefault(key, {
                    "brief": r["brief"], "reason": reason, "count": 0,
                    "ops": rec["ops"], "models": set(),
                })
                e["count"] += 1
                e["models"].add(r["model"])
    if not seen:
        return "fleet holes: none found (every wrong solid drew a diagnostic)"
    lines = [f"FLEET HOLES: {len(seen)} distinct wrong-geometry-but-silent-fleet findings",
             "(the fleet built these, called them fine, and they are not what the brief asked for)",
             "-" * 96]
    for key in sorted(seen):
        e = seen[key]
        lines.append(f"  brief {e['brief']}  (x{e['count']})")
        lines.append(f"    ground truth says: {e['reason']}")
        lines.append(f"    fleet said:        nothing actionable")
    lines.append("-" * 96)
    return "\n".join(lines)


def render_all(payload: dict) -> str:
    results = payload["results"]
    meta = payload.get("meta", {})
    out = [
        "=" * 96,
        "HARNESSCAD PRESSURE TEST -- 'typed diagnostics beat blind resampling'",
        "=" * 96,
        f"seed:      {meta.get('seed')}",
        f"backend:   {meta.get('backend')}",
        f"attempts:  {meta.get('max_attempts')}",
        f"temp:      {meta.get('temperature')}",
        f"briefs:    {meta.get('n_briefs')}",
        f"models:    {', '.join(meta.get('models', []))}",
        "",
        headline(results),
        "",
        render_table(results),
        "",
        render_split(results),
        "",
        per_brief(results),
        "",
        fleet_holes(results),
    ]
    return "\n".join(out)
