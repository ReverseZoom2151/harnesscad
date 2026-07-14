"""`python -m harnesscad.eval.showcase.cli` -- run the showcase.

    sweep   every brief x every model, through the block-and-correct loop, with
            a validated render per success. Streams to assets/showcase/runs.jsonl.
    best    for each brief, take the best successful run, render it at 1600x1000
            and (for the top few) emit a multi-view engineering drawing.
    report  turn runs.jsonl into results.json + report.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from harnesscad.eval.showcase import report as report_mod
from harnesscad.eval.showcase.briefs import BRIEFS, brief_by_id
from harnesscad.eval.showcase.image import validate_png
from harnesscad.eval.showcase.loop import MAX_ATTEMPTS, apply_ops
from harnesscad.eval.showcase.models import MODELS, model_slug, resolve_models
from harnesscad.eval.showcase.runner import (
    HERO_HEIGHT, HERO_WIDTH, load_runs, render_record, run_sweep,
)

DEFAULT_ASSETS = os.path.join("assets", "showcase")
#: How many of the best parts get a full multi-view engineering drawing.
DRAWING_COUNT = 4


def _briefs(names: Optional[List[str]]):
    if not names:
        return list(BRIEFS)
    return [brief_by_id(n) for n in names]


def cmd_sweep(args: argparse.Namespace) -> int:
    models = resolve_models(args.model)
    briefs = _briefs(args.brief)
    print(f"sweep: {len(models)} model(s) x {len(briefs)} brief(s), "
          f"max {args.attempts} attempts each")
    runs = run_sweep(models=models, briefs=briefs, assets_dir=args.assets,
                     max_attempts=args.attempts, render=not args.no_render)
    solved = sum(1 for r in runs if r.solved)
    on_brief = sum(1 for r in runs if (r.grade or {}).get("on_brief"))
    print(f"\n{solved}/{len(runs)} verified a solid; {on_brief} were the briefed part")
    return 0


def cmd_best(args: argparse.Namespace) -> int:
    """Hero renders (1600x1000) + multi-view drawings for the best result per brief."""
    runs = load_runs(os.path.join(args.assets, "runs.jsonl"))
    best = report_mod.best_per_brief(runs)
    heroes, drawings = [], []
    ranked = [b for b in BRIEFS if best.get(b.id)]
    # Drawings go to the briefs whose best result is on-brief, hardest first.
    drawable = sorted(
        [b for b in ranked if (best[b.id].get("grade") or {}).get("on_brief")],
        key=lambda b: -b.tier)[:DRAWING_COUNT]

    for brief in ranked:
        rec = best[brief.id]
        slug = model_slug(rec["model"])
        png = os.path.join(args.assets, f"hero-{brief.id}-{slug}.png")
        print(f"hero: {brief.id} <- {rec['model']} ({HERO_WIDTH}x{HERO_HEIGHT})")
        img = render_record(rec["ops"], png, view="hero",
                            width=HERO_WIDTH, height=HERO_HEIGHT, ssaa=args.ssaa)
        if img.get("ok"):
            heroes.append({"brief_id": brief.id, "model": rec["model"], **img})
            print(f"    ok  silhouette={img['silhouette']} variance={img['variance']}")
        else:
            print(f"    DROPPED: {img.get('failures')}")

        if brief in drawable:
            svg = os.path.join(args.assets, f"drawing-{brief.id}-{slug}.svg")
            path = _write_drawing(rec["ops"], svg, brief.text)
            if path:
                drawings.append({"brief_id": brief.id, "model": rec["model"],
                                 "path": path.replace("\\", "/")})
                print(f"    drawing: {path}")
            else:
                print("    drawing: FAILED")

    with open(os.path.join(args.assets, "best.json"), "w", encoding="utf-8") as fh:
        json.dump({"heroes": heroes, "drawings": drawings}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\n{len(heroes)} hero image(s), {len(drawings)} drawing(s)")
    return 0


def _write_drawing(ops, path: str, title: str) -> Optional[str]:
    """A real multi-view engineering drawing (front/top/side, hidden lines, dims)."""
    from harnesscad.io.formats import registry as formats

    server, result = apply_ops(ops)
    if not result["ok"]:
        return None
    try:
        formats.write(server.session, path, views=("front", "top", "side"),
                      title=title[:60])
    except Exception as exc:  # noqa: BLE001
        print(f"    drawing error: {exc}")
        return None
    return path


def cmd_report(args: argparse.Namespace) -> int:
    runs = load_runs(os.path.join(args.assets, "runs.jsonl"))
    board = report_mod.write_results(runs, args.assets)
    drawings = None
    best_path = os.path.join(args.assets, "best.json")
    if os.path.exists(best_path):
        with open(best_path, "r", encoding="utf-8") as fh:
            drawings = json.load(fh).get("drawings")
    md = report_mod.render_markdown(runs, board, drawings=drawings)
    with open(os.path.join(args.assets, "report.md"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(md)
    t = board["totals"]
    print(f"wrote {args.assets}/results.json and {args.assets}/report.md")
    print(f"{t['solved']}/{t['pairs']} solved, {t['on_brief']} on brief, "
          f"{t['images_valid']} valid images")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="harnesscad-showcase", description=__doc__)
    p.add_argument("--assets", default=DEFAULT_ASSETS, help="output directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="every brief x every model through the loop")
    s.add_argument("--model", action="append", choices=list(MODELS),
                   help="restrict to these models (repeatable)")
    s.add_argument("--brief", action="append",
                   choices=[b.id for b in BRIEFS],
                   help="restrict to these briefs (repeatable)")
    s.add_argument("--attempts", type=int, default=MAX_ATTEMPTS)
    s.add_argument("--no-render", action="store_true")
    s.set_defaults(func=cmd_sweep)

    b = sub.add_parser("best", help="hero renders + drawings for the best results")
    b.add_argument("--ssaa", type=int, default=2)
    b.set_defaults(func=cmd_best)

    r = sub.add_parser("report", help="results.json + report.md")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
