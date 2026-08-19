"""An LLM-backed :data:`~harnesscad.eval.hardcorpus.score.Solver`.

``score.score`` wants ONE thing: a callable from a brief's TEXT to an op stream.
This module supplies that callable, backed by a real model, and it does so by
REUSING the machinery ``eval/pressure`` already proved against local ollama
rather than growing a second model layer beside it:

    ``pressure.model.OllamaClient``   the seeded litellm/ollama caller
    ``pressure.model.CachedClient``   the content-addressed disk cache (resume)
    ``pressure.model.ScriptedClient`` the offline stand-in the tests drive
    ``pressure.model.extract_ops``    raw completion text -> validated CISP ops
    ``pressure.prompts.SYSTEM_PROMPT``/``user_prompt``/``format_parse_error``

Nothing about the prompt or the parser is re-specified here; if the pressure
experiment's prompt changes, this changes with it, which is the point -- one op
grammar, one parser, one leniency policy, applied to both benchmarks.

WHAT THIS SOLVER DOES WITH A BAD ANSWER
---------------------------------------
The hard corpus's thesis is that UNMEASURABLE OUTPUT IS A FINDING. So a model
that emits prose, half a JSON array, or nothing at all must be RECORDED, never
crash the run and never be quietly skipped:

* a completion that will not parse is fed back once per remaining attempt with
  :func:`~harnesscad.eval.pressure.prompts.format_parse_error` (the same repair
  turn the pressure loops use), and
* when the budget is exhausted the solver returns an EMPTY op stream and marks
  the brief ``invalid``. An empty stream builds nothing, so the oracle fails it
  and the field's weak metric fails it too -- the brief counts against the model
  in both columns, which is the honest accounting.
* a transport failure (ollama down, timeout) is recorded the same way, as an
  ``error`` attempt, so a half-finished sweep is visibly half-finished instead
  of silently scoring zero.

WHAT IT REFUSES TO REMEMBER
---------------------------
The solver is handed held-out brief TEXT and must not leak it into a results
file. Its per-brief records carry an ORDINAL and a short digest of the text and
never the text itself, nor the ops, nor any reference. (The completion cache is
a different matter: it is keyed by the exact message list and therefore stores
the prompt on disk, exactly as the pressure cache does. That is a local working
file, not a published artifact, and ``--cache`` chooses where it lives.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.pressure.model import Client, extract_ops
from harnesscad.eval.pressure.prompts import (
    SYSTEM_PROMPT, format_parse_error, user_prompt,
)

__all__ = ["Attempt", "SolveRecord", "ModelSolver", "brief_digest",
           "DEFAULT_MAX_ATTEMPTS"]

#: Attempt 1 is the answer; the second attempt exists only to repair a response
#: that could not be parsed at all. Raise it with ``--max-attempts`` to give the
#: model more parse-repair turns; there is no geometry feedback in either case,
#: because the hard corpus scores what a model PRODUCES, not what it can be
#: talked into.
DEFAULT_MAX_ATTEMPTS = 2


def brief_digest(text: str) -> str:
    """A short, stable id for a brief that is not the brief.

    Held-out text may not be written to a results file, but a run still has to
    name its cells. A truncated sha256 of the text does that: stable across
    runs, meaningless to a reader, and impossible to invert into the brief.
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


@dataclass
class Attempt:
    """One model call. Sizes and errors only -- never the completion text."""

    attempt: int
    chars: int = 0
    ops: int = 0
    ok: bool = False
    #: the parse error, or the transport error, or None.
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"attempt": self.attempt, "chars": self.chars, "ops": self.ops,
                "ok": self.ok, "error": self.error}


@dataclass
class SolveRecord:
    """What the solver did on one brief. Counts, not content."""

    ordinal: int
    digest: str
    attempts: List[Attempt] = field(default_factory=list)
    #: True when NO attempt produced parseable ops -- the corpus's "unmeasurable".
    invalid: bool = True
    #: True when the model could not be reached at all on every attempt.
    errored: bool = False
    ops: int = 0

    @property
    def model_calls(self) -> int:
        return len(self.attempts)

    def to_dict(self) -> dict:
        return {"ordinal": self.ordinal, "digest": self.digest,
                "invalid": self.invalid, "errored": self.errored,
                "ops": self.ops, "model_calls": self.model_calls,
                "attempts": [a.to_dict() for a in self.attempts]}


class ModelSolver:
    """A :data:`~harnesscad.eval.hardcorpus.score.Solver` backed by a Client.

    Call it with a brief's text; it returns the op stream the model produced, or
    ``[]`` when the model never produced one. It NEVER raises: an unparseable
    answer and an unreachable model are both findings, and a finding that kills
    the sweep is a finding nobody gets to read.

    ``client`` is anything satisfying :class:`~harnesscad.eval.pressure.model.Client`
    -- an ``OllamaClient`` wrapped in ``CachedClient`` in a real run, a
    ``ScriptedClient`` in the tests.
    """

    def __init__(self, client: Client, seed: int = 0,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> None:
        self.client = client
        self.name = getattr(client, "name", "model")
        self.seed = int(seed)
        self.max_attempts = max(1, int(max_attempts))
        self.records: List[SolveRecord] = []
        self._seen: Dict[str, int] = {}

    # -- aggregate counters the runner reports -------------------------------
    @property
    def invalid(self) -> int:
        """Briefs on which the model never emitted parseable ops."""
        return sum(1 for r in self.records if r.invalid)

    @property
    def errored(self) -> int:
        """Briefs on which every attempt failed to reach the model."""
        return sum(1 for r in self.records if r.errored)

    @property
    def model_calls(self) -> int:
        return sum(r.model_calls for r in self.records)

    def stats(self) -> Dict[str, Any]:
        return {"model": self.name, "briefs": len(self.records),
                "invalid": self.invalid, "errored": self.errored,
                "model_calls": self.model_calls,
                "max_attempts": self.max_attempts}

    def to_dict(self) -> dict:
        return {"stats": self.stats(),
                "records": [r.to_dict() for r in self.records]}

    # -- the Solver protocol -------------------------------------------------
    def __call__(self, text: str) -> Sequence[Op]:
        digest = brief_digest(text)
        ordinal = self._seen.setdefault(digest, len(self._seen))
        record = SolveRecord(ordinal=ordinal, digest=digest)
        self.records.append(record)

        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(text)}]
        errors = 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.client.complete(messages, attempt)
            except Exception as exc:                              # noqa: BLE001
                errors += 1
                record.attempts.append(Attempt(
                    attempt=attempt, error="%s: %s" % (type(exc).__name__, exc)))
                continue
            parsed = extract_ops(raw or "")
            if parsed.ok:
                record.attempts.append(Attempt(
                    attempt=attempt, chars=len(raw or ""),
                    ops=len(parsed.ops), ok=True))
                record.invalid = False
                record.ops = len(parsed.ops)
                return list(parsed.ops)
            record.attempts.append(Attempt(
                attempt=attempt, chars=len(raw or ""), error=parsed.error))
            # The one repair turn the pressure loops also allow: the response
            # was not ops, so say so and ask again. No geometry is revealed --
            # nothing has been built yet.
            messages = messages + [
                {"role": "assistant", "content": raw or ""},
                {"role": "user",
                 "content": format_parse_error(parsed.error or "unparseable",
                                               "hardcorpus")},
            ]
        record.errored = errors == len(record.attempts) and errors > 0
        return []
