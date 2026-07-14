"""The orchestrator: (models x briefs x loops), resumable, seeded, cached.

Resumability is per-cell. Every (model, brief, loop) cell that already exists in
the output file is skipped on a re-run, and every model call underneath is
content-addressed in the completion cache, so an interrupted 400-call run picks
up exactly where it stopped and a completed run re-executes for free.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from harnesscad.eval.pressure.briefs import Brief, briefs_for
from harnesscad.eval.pressure.cache import CompletionCache
from harnesscad.eval.pressure.loops import (
    BLIND, DEFAULT_MAX_ATTEMPTS, HARNESS, LOOPS, run_brief,
)
from harnesscad.eval.pressure.model import CachedClient, Client, OllamaClient

DEFAULT_SEED = 20260713
DEFAULT_CACHE = ".pressure_cache"


def _cell_id(model: str, brief: str, loop: str) -> str:
    return f"{model}|{brief}|{loop}"


def load_results(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def save_results(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=1)
    os.replace(tmp, path)


def run(models: Sequence[str],
        briefs: Sequence[Brief],
        loops: Sequence[str] = LOOPS,
        seed: int = DEFAULT_SEED,
        temperature: float = 0.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        out: str = "pressure_results.json",
        cache_dir: str = DEFAULT_CACHE,
        client_factory: Optional[Callable[[str], Client]] = None,
        resume: bool = True,
        log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Run the grid and write `out`. Returns the full payload."""
    say = log if log is not None else (lambda s: print(s, file=sys.stderr, flush=True))

    payload = load_results(out) if resume else {}
    existing: Dict[str, dict] = {
        _cell_id(r["model"], r["brief"], r["loop"]): r
        for r in payload.get("results", [])
    }

    cache = CompletionCache(cache_dir)
    factory = client_factory or (lambda m: OllamaClient(m, seed=seed,
                                                        temperature=temperature))

    results: List[dict] = list(existing.values())
    total = len(models) * len(briefs) * len(loops)
    done = 0
    t0 = time.perf_counter()

    for model in models:
        inner = factory(model)
        client: Client = CachedClient(inner, cache, seed=seed, temperature=temperature)
        for brief in briefs:
            for loop in loops:
                done += 1
                cid = _cell_id(model, brief.id, loop)
                if cid in existing:
                    say(f"[{done}/{total}] {cid} (cached cell, skipped)")
                    continue
                say(f"[{done}/{total}] {cid} ...")
                res = run_brief(client, brief, loop, seed=seed,
                                max_attempts=max_attempts)
                results.append(res.to_dict())
                existing[cid] = res.to_dict()
                say(f"    -> solved={res.solved} attempts={res.attempts_used} "
                    f"invalid={res.invalid_ops} missed={res.fleet_missed} "
                    f"{res.seconds:.1f}s")
                payload = {
                    "meta": _meta(models, briefs, loops, seed, temperature,
                                  max_attempts, cache),
                    "results": results,
                }
                save_results(out, payload)

    payload = {
        "meta": _meta(models, briefs, loops, seed, temperature, max_attempts, cache),
        "results": results,
    }
    payload["meta"]["wall_seconds"] = time.perf_counter() - t0
    save_results(out, payload)
    return payload


def _meta(models, briefs, loops, seed, temperature, max_attempts, cache) -> dict:
    return {
        "seed": seed,
        "temperature": temperature,
        "max_attempts": max_attempts,
        "backend": "frep",
        "models": list(models),
        "loops": list(loops),
        "n_briefs": len(briefs),
        "briefs": [b.id for b in briefs],
        "cache": cache.stats(),
    }
