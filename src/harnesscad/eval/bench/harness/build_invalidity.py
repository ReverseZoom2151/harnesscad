"""Build-time Invalidity Ratio (IR) and Executability over a generated set.

Sources
-------
* **ReCAD** (arXiv:2512.06328) reports **IR**, the *Invalidity Ratio*: the share
  of generated outputs that do not parse/build into a valid model. Their table
  puts GPT-4o at ``IR 15.14`` and their trained model at ``IR 0.81`` -- i.e. the
  number is quoted on a **0-100 percent** scale, lower is better.
* **CAD-RL** (arXiv:2508.10118) reports the same population quantity with the
  opposite polarity and calls it **Executability** (GPT-4o ``72.72%``, theirs
  ``99.63%``), higher is better.

Because both are counts over one population, ``executability_percent`` and
``ir_percent`` are exact complements here::

    ir_percent + executability_percent == 100.0

so this module derives both from a single pass and never lets them drift.

Exact variant and normalisation
-------------------------------
* **Population**: every generated output of ONE system over ONE task set, one
  output per task (single-sample decoding, no correction loop -- the correction
  loop is the subject of ``bench.harness.correction_budget``, not of this
  module). Nothing is dropped: an output that is empty, truncated, or absent is
  counted as an INVALID member of the population, never excluded from the
  denominator. Excluding non-outputs would silently improve IR.
* **Numerator (IR)**: outputs whose recorded build outcome is *not* success.
* **Denominator**: the population size.
* **Scale**: ``* 100``, i.e. PERCENT, matching both papers' tables. The
  ``*_ratio`` fields carry the identical quantity as a fraction in ``[0, 1]``
  for callers that would rather not multiply by a hundred; they are the same
  measurement, not a second one.

What "builds" means, and what this module refuses to decide
-----------------------------------------------------------
Deciding whether an output parses and builds requires a CAD kernel and code
execution. This module runs neither. It consumes *recorded* build outcomes, one
per generated output::

    {"id": "task-07", "built": False, "error": "SyntaxError"}

``built`` is authoritative when present. When it is absent, an ``error`` key is
accepted as the signal (``None`` / ``""`` means it built). A record carrying
NEITHER key raises :class:`ValueError`: guessing that a silent record is a
success is exactly the fabrication that would make the number meaningless.

Not to be confused with ``bench.sequence.invalidity_ratio``
------------------------------------------------------------
That module is also called "invalidity ratio" and also divides by the number of
sequences, but it measures something else: *structural degeneracy* of a decoded
CAD sequence (zero-length lines, degenerate arcs, zero-depth extrusions), on a
0-1 fraction scale, decidable from the sequence alone with no kernel. This
module measures whether the output BUILDS at all. The two are registered as
rivals (family ``validity_rate``) and must never be pooled: a sequence can be
perfectly buildable and structurally degenerate, or unparseable and therefore
never scored by the structural rule at all.

Stdlib only, deterministic, no execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "BuildReport",
    "is_invalid",
    "invalidity_ratio_percent",
    "executability_percent",
    "failure_census",
    "build_report",
]


def _built(record: Mapping) -> bool:
    """Whether one recorded output built, per the module's stated contract."""
    if "built" in record:
        return bool(record["built"])
    if "error" in record:
        err = record["error"]
        return err is None or (isinstance(err, str) and not err.strip())
    raise ValueError(
        "build outcome record carries neither 'built' nor 'error'; refusing to "
        "assume it built (record keys: %s)" % (sorted(record),))


def is_invalid(record: Mapping) -> bool:
    """True when this generated output did NOT parse/build into a valid model."""
    return not _built(record)


def invalidity_ratio_percent(records: Sequence[Mapping]) -> float:
    """ReCAD IR: ``100 * #(did not build) / #(generated outputs)``.

    Raises on an empty population -- an IR over nothing is not 0.0, it is
    undefined, and returning 0.0 would read as a perfect score.
    """
    if not records:
        raise ValueError("invalidity ratio is undefined over an empty population")
    bad = sum(1 for r in records if is_invalid(r))
    return 100.0 * bad / len(records)


def executability_percent(records: Sequence[Mapping]) -> float:
    """CAD-RL Executability: ``100 - IR`` on the same population."""
    return 100.0 - invalidity_ratio_percent(records)


def failure_census(records: Sequence[Mapping]) -> Dict[str, int]:
    """How many invalid outputs carried each ``error`` label (sorted by key).

    An invalid record with no usable ``error`` string is counted under
    ``"unlabelled"``. Valid records contribute nothing.
    """
    counts: Dict[str, int] = {}
    for r in records:
        if not is_invalid(r):
            continue
        err = r.get("error")
        label = err.strip() if isinstance(err, str) and err.strip() else "unlabelled"
        counts[label] = counts.get(label, 0) + 1
    return {k: counts[k] for k in sorted(counts)}


@dataclass(frozen=True)
class BuildReport:
    """IR and Executability over one generated population, reported together."""

    n_outputs: int
    n_invalid: int
    ir_percent: float
    executability_percent: float
    ir_ratio: float
    executability_ratio: float
    failures: Tuple[Tuple[str, int], ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_outputs": self.n_outputs,
            "n_invalid": self.n_invalid,
            "ir_percent": self.ir_percent,
            "executability_percent": self.executability_percent,
            "ir_ratio": self.ir_ratio,
            "executability_ratio": self.executability_ratio,
            "failures": {k: v for k, v in self.failures},
        }

    def summary(self) -> str:
        return ("IR %.2f%% / Executability %.2f%% over %d outputs "
                "(%d did not build)" % (self.ir_percent,
                                        self.executability_percent,
                                        self.n_outputs, self.n_invalid))


def build_report(records: Sequence[Mapping]) -> BuildReport:
    """Full IR / Executability report over one generated population."""
    rows: List[Mapping] = list(records)
    ir = invalidity_ratio_percent(rows)
    census = failure_census(rows)
    return BuildReport(
        n_outputs=len(rows),
        n_invalid=sum(1 for r in rows if is_invalid(r)),
        ir_percent=ir,
        executability_percent=100.0 - ir,
        ir_ratio=ir / 100.0,
        executability_ratio=1.0 - ir / 100.0,
        failures=tuple(sorted(census.items())),
    )
