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
    ALL_LOOPS, BLIND, DEFAULT_MAX_ATTEMPTS, HARNESS, LOOPS, SAMPLING_TEMPERATURE,
    SELECTION_LOOPS, run_brief, run_sampling,
)
from harnesscad.eval.pressure.model import CachedClient, Client, OllamaClient

DEFAULT_SEED = 20260713
DEFAULT_CACHE = ".pressure_cache"

#: THE FRONTIER LINEUP. v2 runs on these four tags and NOTHING ELSE. The old
#: six-model set that a previous agent ran the v2 code against (qwen2.5-coder and
#: friends) has been DELETED from the machine; its partial results in
#: assets/pressure_v2/obsolete_deleted_lineup/ are obsolete and must never be
#: turned into a published number. The run is parameterised by `models` -- pass
#: `--model` to override -- but the default is this and only this.
DEFAULT_MODELS: List[str] = [
    "qwen3.6:27b",
    "qwen3.6:35b",
    "ornith:9b",
    "ornith:35b",
]


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

    iterative = [l for l in loops if l not in SELECTION_LOOPS]
    selection = [l for l in loops if l in SELECTION_LOOPS]

    def flush() -> Dict[str, Any]:
        payload = {
            "meta": _meta(models, briefs, loops, seed, temperature,
                          max_attempts, cache),
            "results": results,
        }
        save_results(out, payload)
        return payload

    for model in models:
        inner = factory(model)
        client: Client = CachedClient(inner, cache, seed=seed, temperature=temperature)
        for brief in briefs:
            for loop in iterative:
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
                say(f"    -> solved={res.solved} shape={res.solved_shape} "
                    f"calls={res.model_calls} invalid={res.invalid_ops} "
                    f"missed={res.fleet_missed} {res.seconds:.1f}s")
                flush()

            # The selection arms SHARE their N draws, so they are run together
            # and cost, between them, exactly what one of them costs.
            if selection:
                done += len(selection)
                missing = [l for l in selection
                           if _cell_id(model, brief.id, l) not in existing]
                if not missing:
                    say(f"[{done}/{total}] {model}|{brief.id}|selection "
                        f"(cached cells, skipped)")
                    continue
                say(f"[{done}/{total}] {model}|{brief.id}|{'+'.join(missing)} "
                    f"(N={max_attempts} draws, T={SAMPLING_TEMPERATURE}) ...")
                arms = run_sampling(client, brief, seed=seed, n=max_attempts,
                                    temperature=SAMPLING_TEMPERATURE)
                for loop in missing:
                    res = arms[loop]
                    results.append(res.to_dict())
                    existing[_cell_id(model, brief.id, loop)] = res.to_dict()
                    say(f"    -> {loop}: solved={res.solved} "
                        f"shape={res.solved_shape} calls={res.model_calls}")
                flush()

    payload = flush()
    payload["meta"]["wall_seconds"] = time.perf_counter() - t0
    save_results(out, payload)
    return payload


def _meta(models, briefs, loops, seed, temperature, max_attempts, cache) -> dict:
    return {
        "version": 2,
        "seed": seed,
        "temperature": temperature,
        "sampling_temperature": SAMPLING_TEMPERATURE,
        "max_attempts": max_attempts,
        "backend": "frep",
        "models": list(models),
        "loops": list(loops),
        "n_briefs": len(briefs),
        "briefs": [b.id for b in briefs],
        "cache": cache.stats(),
    }
