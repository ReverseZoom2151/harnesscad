"""The sweep: every brief x every model, through the loop, to a rendered PNG.

For each pair it runs :func:`harnesscad.eval.showcase.loop.run_brief` (the real
block-and-correct loop), and on success renders the verified solid and decodes
the PNG back off disk to prove it is a picture of a part
(:mod:`harnesscad.eval.showcase.image`). Results stream to a JSONL as they land,
so a sweep that is interrupted still yields everything it proved.

Determinism: the models are seeded and run at temperature 0; the renderer is
deterministic; the op streams are hashed by the session (`digest`). Re-running a
pair reproduces its record modulo ollama's own sampling floor.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from harnesscad.eval.showcase.briefs import BRIEFS, Brief, brief_by_id
from harnesscad.eval.showcase.image import validate_png
from harnesscad.eval.showcase.loop import MAX_ATTEMPTS, RunRecord, apply_ops, run_brief
from harnesscad.eval.showcase.models import MODELS, SEED, make_llm, model_slug

__all__ = ["render_record", "run_pair", "run_sweep", "SWEEP_WIDTH", "SWEEP_HEIGHT"]

SWEEP_WIDTH = 800
SWEEP_HEIGHT = 500
HERO_WIDTH = 1600
HERO_HEIGHT = 1000


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def render_record(ops: Sequence[dict], path: str, view: str = "hero",
                  width: int = SWEEP_WIDTH, height: int = SWEEP_HEIGHT,
                  backend: str = "frep", ssaa: int = 1) -> Dict[str, Any]:
    """Rebuild `ops` on a fresh session, render to `path`, then VALIDATE the PNG.

    Returns the image record: path, dims, the decoded statistics and whether the
    image passed. A render that fails validation is reported, not shipped.
    """
    from harnesscad.io import render as render_route

    _ensure_dir(os.path.dirname(path))
    server, result = apply_ops(ops, backend=backend)
    if not result["ok"]:
        return {"path": path, "ok": False,
                "failures": ["ops did not re-verify on replay"]}
    try:
        render_route.render_session(server.session, path, view=view, width=width,
                                    height=height, ssaa=ssaa)
    except Exception as exc:  # noqa: BLE001 - a render failure is a result
        return {"path": path, "ok": False, "failures": [f"{type(exc).__name__}: {exc}"]}
    stats = validate_png(path)
    stats["path"] = path.replace("\\", "/")
    stats["view"] = view
    if not stats.get("ok"):
        # Never ship a broken image: drop it from disk so it cannot be mistaken
        # for a result.
        try:
            os.remove(path)
        except OSError:
            pass
        stats["path"] = None
    return stats


def run_pair(brief: Brief, model: str, assets_dir: str, seed: int = SEED,
             max_attempts: int = MAX_ATTEMPTS, render: bool = True) -> RunRecord:
    """One (brief, model): plan -> correct -> verify -> render -> validate."""
    llm = make_llm(model, seed=seed)
    record = run_brief(brief, llm, model=model, seed=seed, max_attempts=max_attempts)
    if record.solved and render:
        png = os.path.join(assets_dir, f"{brief.id}-{model_slug(model)}.png")
        record.render = render_record(record.ops, png)
    return record


def run_sweep(models: Optional[Sequence[str]] = None,
              briefs: Optional[Sequence[Brief]] = None,
              assets_dir: str = "assets/showcase",
              max_attempts: int = MAX_ATTEMPTS,
              jsonl: Optional[str] = None,
              render: bool = True,
              log=print) -> List[RunRecord]:
    """Run the full matrix. Streams each record to `jsonl` as it completes."""
    models = list(models or MODELS)
    briefs = list(briefs or BRIEFS)
    _ensure_dir(assets_dir)
    jsonl = jsonl or os.path.join(assets_dir, "runs.jsonl")
    _ensure_dir(os.path.dirname(jsonl))

    records: List[RunRecord] = []
    total = len(models) * len(briefs)
    i = 0
    with open(jsonl, "w", encoding="utf-8") as fh:
        for model in models:
            for brief in briefs:
                i += 1
                started = time.monotonic()
                log(f"[{i}/{total}] {model} x {brief.id} ...")
                rec = run_pair(brief, model, assets_dir, max_attempts=max_attempts,
                               render=render)
                records.append(rec)
                fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
                fh.flush()
                took = time.monotonic() - started
                if rec.solved:
                    grade = rec.grade or {}
                    img = (rec.render or {}).get("ok")
                    log("    solved in %d attempt(s), %s, image=%s [%.0fs]"
                        % (rec.attempt_count,
                           "ON BRIEF" if grade.get("on_brief") else
                           "off brief: " + "; ".join(grade.get("reasons") or []),
                           "ok" if img else "FAILED" if render else "skipped",
                           took))
                else:
                    log("    FAILED after %d attempt(s): %s [%.0fs]"
                        % (rec.attempt_count, rec.failure_reason, took))
    return records


def load_runs(jsonl: str) -> List[dict]:
    """Read a runs.jsonl back into plain dicts (for the report/scoreboard)."""
    out: List[dict] = []
    with open(jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
