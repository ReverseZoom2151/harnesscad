"""Drive real models across the hard corpus's held-out split, resumably.

One cell is one MODEL. For each model the runner builds a
:class:`~harnesscad.eval.hardcorpus.solver.ModelSolver` over a cached ollama
client and hands it to :func:`harnesscad.eval.hardcorpus.score.score`, which is
the only door onto the held-out briefs. What comes back is a ``HeldOutReport``:
both columns and the gap, and no brief text ever.

RESUME, ON TWO LEVELS
---------------------
Exactly the shape ``eval/pressure/runner.py`` uses, and for the same reason -- a
sweep is hours long and gets interrupted:

* the RESULTS file is keyed by every knob that determines the answer (model,
  limit, seed, temperature, attempt budget, token budget), so a finished model
  is skipped on a re-run, and
* every model call underneath goes through
  :class:`~harnesscad.eval.pressure.model.CachedClient`, so even a model that is
  re-scored (a changed oracle, say) costs no completions.

Changing ``--limit`` or ``--seed`` therefore makes a NEW cell rather than
silently reusing the old one: a 2-brief smoke run can never be mistaken for, or
merged into, the full sweep.

HONEST FAILURE
--------------
The solver never raises, so a model that emits prose forever produces a report
with a real ``n``, zero solves and an ``invalid`` count beside it. The runner
copies the solver's counters into the payload next to the report, because
"the model produced nothing measurable on 14 of 40 briefs" is the finding, and
a bare 0.0 solve rate does not say it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from harnesscad.eval.hardcorpus import score as hc_score
from harnesscad.eval.hardcorpus.solver import DEFAULT_MAX_ATTEMPTS, ModelSolver
from harnesscad.eval.pressure.cache import CompletionCache
from harnesscad.eval.pressure.model import CachedClient, Client, OllamaClient

__all__ = ["DEFAULT_SEED", "DEFAULT_CACHE", "DEFAULT_MODELS", "DEFAULT_OUT",
           "DEFAULT_MAX_TOKENS", "cell_id", "load_results", "save_results",
           "run", "standings", "reference_submission"]

DEFAULT_SEED = 20260819
DEFAULT_CACHE = ".hardcorpus_cache"
DEFAULT_OUT = "hardcorpus_results.json"

#: The completion budget per call. The pressure experiment runs at 1024, and the
#: first smoke run against ``ornith:9b`` showed exactly what that costs HERE: an
#: EMPTY completion on attempt 1 of both briefs, because a reasoning model spends
#: the budget thinking and never reaches the array. This corpus asks for 10-20 op
#: chains where pressure asks for 4, so the budget is raised rather than the
#: finding being blamed on the model.
DEFAULT_MAX_TOKENS = 2048

#: The local lineup, same tags the pressure experiment runs on.
DEFAULT_MODELS: List[str] = ["qwen3.6:27b", "ornith:9b"]


def cell_id(model: str, limit: Optional[int], seed: int, temperature: float,
            max_attempts: int, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Everything that determines a submission. A different knob is a new cell."""
    return "%s|limit=%s|seed=%d|t=%s|attempts=%d|tokens=%d" % (
        model, "all" if not limit else int(limit), int(seed),
        repr(float(temperature)), int(max_attempts), int(max_tokens))


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


def reference_submission(limit: Optional[int] = None) -> Dict[str, Any]:
    """The corpus's own reference solutions, scored as a submission.

    Not a model result -- the CEILING, and the self-test: if this row is not a
    clean sweep the corpus is broken and no model's row beneath it means
    anything. It needs no model and no network.
    """
    report = hc_score.reference_score(limit=limit)
    row = dict(report.to_dict())
    row["name"] = "reference"
    row["solver"] = {"stats": {"model": "reference", "briefs": report.n,
                               "invalid": 0, "errored": 0, "model_calls": 0,
                               "max_attempts": 0}, "records": []}
    return row


def run(models: Sequence[str],
        out: str = DEFAULT_OUT,
        limit: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        temperature: float = 0.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache_dir: str = DEFAULT_CACHE,
        client_factory: Optional[Callable[[str], Client]] = None,
        resume: bool = True,
        reference: bool = False,
        log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Score every model on the held-out split; write ``out``; return the payload."""
    say = log if log is not None else (lambda s: print(s, file=sys.stderr, flush=True))

    payload = load_results(out) if resume else {}
    existing: Dict[str, dict] = {
        r.get("cell", r.get("name", "")): r for r in payload.get("submissions", [])
    }

    cache = CompletionCache(cache_dir)
    factory = client_factory or (lambda m: OllamaClient(m, seed=seed,
                                                        temperature=temperature,
                                                        max_tokens=max_tokens))
    submissions: List[dict] = list(existing.values())
    t0 = time.perf_counter()

    def flush() -> Dict[str, Any]:
        p = {
            "meta": {
                "version": 1,
                "corpus": "hardcorpus/heldout",
                "n_available": hc_score.size(),
                "limit": int(limit) if limit else None,
                "seed": int(seed),
                "temperature": float(temperature),
                "max_attempts": int(max_attempts),
                "max_tokens": int(max_tokens),
                "models": list(models),
                "cache": cache.stats(),
            },
            "submissions": submissions,
        }
        save_results(out, p)
        return p

    if reference:
        cid = cell_id("reference", limit, seed, temperature, max_attempts,
                      max_tokens)
        if cid in existing:
            say("reference (cached cell, skipped)")
        else:
            say("reference (the corpus's own solutions -- the ceiling) ...")
            row = reference_submission(limit)
            row["cell"] = cid
            submissions.append(row)
            existing[cid] = row
            say("    -> oracle=%d/%d weak=%d" % (row["oracle_solved"], row["n"],
                                                 row["weak_passed"]))
            flush()

    total = len(models)
    for i, model in enumerate(models, start=1):
        cid = cell_id(model, limit, seed, temperature, max_attempts, max_tokens)
        if cid in existing:
            say("[%d/%d] %s (cached cell, skipped)" % (i, total, cid))
            continue
        say("[%d/%d] %s ..." % (i, total, cid))
        client: Client = CachedClient(factory(model), cache, seed=seed,
                                      temperature=temperature)
        solver = ModelSolver(client, seed=seed, max_attempts=max_attempts)
        started = time.perf_counter()
        report = hc_score.score(solver, limit=limit)
        row = dict(report.to_dict())
        row["name"] = model
        row["cell"] = cid
        row["solver"] = solver.to_dict()
        row["seconds"] = time.perf_counter() - started
        submissions.append(row)
        existing[cid] = row
        say("    -> oracle=%d/%d weak=%d fooled=%d invalid=%d calls=%d %.1fs"
            % (report.oracle_solved, report.n, report.weak_passed,
               report.field_fooled, solver.invalid, solver.model_calls,
               row["seconds"]))
        flush()

    payload = flush()
    payload["meta"]["wall_seconds"] = time.perf_counter() - t0
    save_results(out, payload)
    return payload


def standings(payload: Dict[str, Any]) -> List[Any]:
    """The payload's submissions as leaderboard rows (both columns and the gap)."""
    from harnesscad.eval.leaderboard import hardcorpus_board as board

    return [board.Standing.from_dict(row, name=row.get("name") or "submission")
            for row in payload.get("submissions", [])]
