"""Eval-scored prompt hill-climbing: keep-best, no ratchet, infra-blind.

The optimizer applies keep/discard logic, a composite score, and version
management through a clean-room, deterministic control policy.

THE CONTROL POLICY
------------------
The source loop is: evaluate a candidate SKILL.md against test cases, score it,
keep or discard against the ALL-TIME best, version it, repeat. Three properties
make it worth porting, and all three are policy -- not model:

  * **All-time best comparison, never last-iteration.** Comparing to the
    previous score lets a sequence of small regressions ratchet the reference
    point downward until the loop has walked away from its own peak. Comparing
    to the all-time best cannot.
  * **Tolerance permits lateral moves but never moves the bar.** A candidate
    within ``tolerance`` of the best is KEPT (it may be the plateau step that
    enables the next improvement) but the best score is **not** updated. That
    is the specific rule that makes lateral moves safe.
  * **Infra-failure exclusion.** A run that failed on network/timeout/provider
    error scored nothing about the *candidate*. The source retries such runs
    with exponential backoff and admits only non-``llm_error`` runs into the
    scored set. Charging infrastructure noise to the candidate would discard
    good candidates for reasons that have nothing to do with them.

MODEL-FREE BY CONSTRUCTION
--------------------------
The source drives this with an LLM proposing each new SKILL.md and a headless
FreeCAD/vision evaluator scoring it. Neither belongs in the harness. Here BOTH
are **injected callables**:

  * ``propose(current, history) -> str`` -- any candidate generator. A model, a
    mutation table, or a scripted list. The loop does not care.
  * ``score(candidate) -> ScoreReport`` -- any evaluator. It returns a score in
    [0, 1], or sets ``infra_error`` to say "this told us nothing about the
    candidate".

So the module contains zero model dependency and the whole policy is
deterministic given deterministic injections -- which is what ``--selfcheck``
exercises. :func:`composite_score` ports the source's weighted-metric roll-up
(re-normalising over the metrics actually present, so a skipped metric
redistributes its weight rather than scoring zero) for callers that want it,
but the loop accepts any scorer.

Stdlib-only, deterministic, absolute imports. ``--selfcheck`` proves keep-best,
the no-ratchet invariant, the tolerance rule, infra exclusion, version history,
and that the result never regresses below the starting candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_WEIGHTS",
    "composite_score",
    "ScoreReport",
    "Version",
    "HillclimbResult",
    "hillclimb",
    "main",
]

#: The source's default metric weights. Only metrics actually reported
#: participate; the rest are re-normalised over what is present.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "completion": 0.30,
    "error_rate": 0.25,
    "correctness": 0.20,
    "efficiency": 0.10,
    "retries": 0.10,
    "visual": 0.05,
}


def composite_score(metrics: Mapping[str, float],
                    weights: Optional[Mapping[str, float]] = None) -> float:
    """Weighted roll-up of per-metric scores, re-normalised over what is present.

    A metric that was not measured (no expected bbox, no reference image) is
    absent from ``metrics`` and its weight is redistributed -- it never scores
    zero, which would punish a candidate for a check the caller declined to
    run. Unknown metric names are ignored. No metrics -> 0.0.
    """
    w = dict(weights or DEFAULT_WEIGHTS)
    present = {k: float(v) for k, v in metrics.items() if k in w}
    total = sum(w[k] for k in present)
    if not present or total <= 0.0:
        return 0.0
    return sum(present[k] * w[k] for k in present) / total


@dataclass(frozen=True)
class ScoreReport:
    """One evaluation of one candidate.

    ``infra_error`` marks a run that failed for reasons unrelated to the
    candidate (provider down, timeout, rate limit). Such a report is EXCLUDED:
    it is not scored, not versioned, and cannot displace the best.
    """

    score: float = 0.0
    infra_error: bool = False
    detail: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Version:
    """One recorded iteration of the climb (the source's ``.optimize/vN``)."""

    iteration: int
    candidate: str
    score: float
    kept: bool
    is_best: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "score": self.score,
            "kept": self.kept,
            "is_best": self.is_best,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HillclimbResult:
    """The outcome: the best candidate ever seen, and how it was reached."""

    best: str
    best_score: float
    iterations: int
    history: Tuple[Version, ...] = ()
    infra_failures: int = 0

    def to_dict(self) -> dict:
        return {
            "best_score": self.best_score,
            "iterations": self.iterations,
            "infra_failures": self.infra_failures,
            "history": [v.to_dict() for v in self.history],
        }


def hillclimb(
    initial: str,
    propose: Callable[[str, Sequence[Version]], str],
    score: Callable[[str], ScoreReport],
    *,
    iterations: int = 10,
    tolerance: float = 0.05,
    max_infra_retries: int = 3,
) -> HillclimbResult:
    """Hill-climb ``initial`` under an injected proposer and scorer.

    The baseline is scored first, so the returned ``best`` is never worse than
    ``initial``: a proposer that only ever makes things worse yields the input
    back. Each iteration then proposes from the CURRENT candidate (the last
    kept one) with the full history available, scores it, and applies the
    source's keep/discard rule against the ALL-TIME best:

      * ``score >= best``            -> keep, and the best moves up;
      * ``best - tolerance <= score`` -> keep as a lateral candidate, but the
        best score does **not** move (no ratchet);
      * otherwise                    -> discard, and the current candidate
        reverts to the all-time best.

    An ``infra_error`` report is retried up to ``max_infra_retries`` times; if
    every attempt fails the iteration is abandoned without scoring, versioning
    or displacing anything, and the climb moves on. Infra noise cannot cost a
    candidate its place.

    Both callables are injected: this function contains no model dependency and
    is fully deterministic given deterministic injections.
    """
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    if tolerance < 0.0:
        raise ValueError("tolerance must be >= 0")

    history: List[Version] = []
    infra_failures = 0

    baseline = _score_excluding_infra(initial, score, max_infra_retries)
    if baseline is None:
        # The baseline could not be measured at all -- return it untouched
        # rather than let infra noise pick a winner.
        return HillclimbResult(initial, 0.0, 0, (), max_infra_retries + 1)

    best = initial
    best_score = baseline.score
    current = initial
    history.append(Version(0, initial, best_score, True, True, "baseline"))

    for i in range(1, iterations + 1):
        candidate = propose(current, tuple(history))
        report = _score_excluding_infra(candidate, score, max_infra_retries)
        if report is None:
            infra_failures += 1
            continue  # told us nothing about the candidate; charge it nothing

        if report.score >= best_score:
            best, best_score, current = candidate, report.score, candidate
            history.append(Version(i, candidate, report.score, True, True,
                                   "improved"))
        elif report.score >= best_score - tolerance:
            # Lateral move: kept as a jumping-off point, but the bar holds.
            current = candidate
            history.append(Version(i, candidate, report.score, True, False,
                                   "lateral-within-tolerance"))
        else:
            current = best  # restore the all-time best
            history.append(Version(i, candidate, report.score, False, False,
                                   "regressed"))

    return HillclimbResult(best, best_score, iterations, tuple(history),
                           infra_failures)


def _score_excluding_infra(candidate: str, score: Callable[[str], ScoreReport],
                           max_retries: int) -> Optional[ScoreReport]:
    """Score with infra-failure retries. ``None`` == never measured."""
    for _attempt in range(max_retries + 1):
        report = score(candidate)
        if not report.infra_error:
            return report
    return None


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Eval-scored prompt hill-climbing with keep-best, the "
                    "no-ratchet tolerance rule and infra-failure exclusion "
                    "(freecad-ai skill-optimizer reimplementation).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove keep-best, no-ratchet, tolerance, infra "
                             "exclusion and the never-regress guarantee.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    def scored(table):
        """A deterministic scorer: candidate name -> score."""
        return lambda c: ScoreReport(score=table[c])

    def sequence(names):
        """A deterministic proposer: emit `names` in order."""
        it = iter(names)
        return lambda _current, _history: next(it)

    # 1. Climbing keeps the best and reports it.
    table = {"v0": 0.5, "v1": 0.7, "v2": 0.9}
    r = hillclimb("v0", sequence(["v1", "v2"]), scored(table), iterations=2)
    assert r.best == "v2" and r.best_score == 0.9, r.to_dict()
    assert [v.reason for v in r.history] == ["baseline", "improved", "improved"]
    print("[selfcheck] climb keeps the improving candidate")

    # 2. Never regress below the baseline, however bad the proposals.
    table = {"v0": 0.8, "bad1": 0.1, "bad2": 0.2}
    r = hillclimb("v0", sequence(["bad1", "bad2"]), scored(table), iterations=2)
    assert r.best == "v0" and r.best_score == 0.8
    assert [v.kept for v in r.history] == [True, False, False]
    print("[selfcheck] a purely-harmful proposer returns the input unchanged")

    # 3. Zero iterations = score the baseline, change nothing.
    r = hillclimb("v0", sequence([]), scored({"v0": 0.42}), iterations=0)
    assert r.best == "v0" and r.best_score == 0.42 and len(r.history) == 1
    print("[selfcheck] zero iterations is a no-op baseline score")

    # 4. Tolerance: a lateral move is KEPT but the bar does NOT move.
    table = {"v0": 0.8, "lat": 0.77, "up": 0.79}
    r = hillclimb("v0", sequence(["lat", "up"]), scored(table), iterations=2,
                  tolerance=0.05)
    assert r.history[1].kept and not r.history[1].is_best
    assert r.history[1].reason == "lateral-within-tolerance"
    assert r.best == "v0" and r.best_score == 0.8  # 0.79 < 0.8: bar held
    print("[selfcheck] lateral move kept as a candidate; best score unmoved")

    # 5. NO RATCHET: a chain of within-tolerance regressions cannot walk the
    #    bar down. Each step is compared to the all-time best, not the last.
    steps = {"v0": 0.90, "a": 0.86, "b": 0.82, "c": 0.78, "d": 0.74}
    r = hillclimb("v0", sequence(["a", "b", "c", "d"]), scored(steps),
                  iterations=4, tolerance=0.05)
    assert r.best == "v0" and r.best_score == 0.90, r.to_dict()
    # 'a' is within tolerance of 0.90 and is kept; b/c/d are NOT within
    # tolerance of the ALL-TIME best (only of their predecessor) -> discarded.
    assert [v.kept for v in r.history] == [True, True, False, False, False]
    print("[selfcheck] no ratchet: 0.90 -> .86 -> .82 -> .78 -> .74 cannot "
          "walk the bar down; best stays 0.90")

    # A last-iteration comparison WOULD have ratcheted -- prove the contrast.
    prev, ratcheted = 0.90, True
    for s in (0.86, 0.82, 0.78, 0.74):
        if s < prev - 0.05:
            ratcheted = False
        prev = s
    assert ratcheted and prev == 0.74  # each step passes vs its predecessor
    print("[selfcheck] ...whereas comparing to the PREVIOUS score would have "
          "accepted every step down to 0.74")

    # 6. Infra-failure exclusion: an infra error is retried, then abandoned --
    #    it never displaces the best and never enters history.
    calls = []

    def flaky(candidate):
        calls.append(candidate)
        if candidate == "infra":
            return ScoreReport(infra_error=True, detail="provider timeout")
        return ScoreReport(score={"v0": 0.5, "good": 0.9}[candidate])

    r = hillclimb("v0", sequence(["infra", "good"]), flaky, iterations=2,
                  max_infra_retries=3)
    assert r.infra_failures == 1
    assert calls.count("infra") == 4          # 1 attempt + 3 retries
    assert r.best == "good" and r.best_score == 0.9
    assert [v.iteration for v in r.history] == [0, 2]  # iteration 1 not recorded
    print("[selfcheck] infra failure retried then abandoned: unscored, "
          "unversioned, best untouched")

    # An infra failure is NOT a zero -- it must not discard a good candidate.
    r = hillclimb("v0", sequence(["infra"]),
                  lambda c: (ScoreReport(infra_error=True) if c == "infra"
                             else ScoreReport(score=0.8)),
                  iterations=1)
    assert r.best == "v0" and r.best_score == 0.8 and len(r.history) == 1
    print("[selfcheck] infra failure is not scored as zero")

    # An unmeasurable baseline yields the input, not a guess.
    r = hillclimb("v0", sequence([]), lambda c: ScoreReport(infra_error=True),
                  iterations=0)
    assert r.best == "v0" and r.best_score == 0.0 and r.history == ()
    print("[selfcheck] unmeasurable baseline returns the input untouched")

    # 7. Version history records every scored iteration.
    table = {"v0": 0.5, "v1": 0.7, "v2": 0.3}
    r = hillclimb("v0", sequence(["v1", "v2"]), scored(table), iterations=2)
    assert len(r.history) == 3
    assert [v.is_best for v in r.history] == [True, True, False]
    assert r.history[-1].reason == "regressed"
    print("[selfcheck] versioned history with per-iteration keep/best flags")

    # 8. The proposer sees the current best-kept candidate and the history.
    seen = []

    def watcher(current, history):
        seen.append((current, len(history)))
        return {"v0": "v1", "v1": "v2"}[current]

    hillclimb("v0", watcher, scored({"v0": 0.1, "v1": 0.2, "v2": 0.3}),
              iterations=2)
    assert seen == [("v0", 1), ("v1", 2)], seen
    print("[selfcheck] proposer receives the current candidate + history")

    # After a regression the proposer is handed the all-time best back.
    seen.clear()

    def watcher2(current, history):
        seen.append(current)
        return {"v0": "bad", "bad": "x"}[current] if current in ("v0", "bad") \
            else "x"

    hillclimb("v0", watcher2, scored({"v0": 0.9, "bad": 0.1, "x": 0.2}),
              iterations=2)
    assert seen == ["v0", "v0"], seen  # reverted, not left on the bad branch
    print("[selfcheck] a discarded candidate reverts to the all-time best")

    # 9. Composite score: weights re-normalise over present metrics.
    assert composite_score({}) == 0.0
    assert abs(composite_score({"completion": 1.0}) - 1.0) < 1e-9
    assert abs(composite_score({"completion": 1.0, "error_rate": 0.0})
               - (0.30 / 0.55)) < 1e-9
    # An absent metric redistributes rather than scoring zero.
    assert composite_score({"completion": 1.0, "error_rate": 1.0}) == 1.0
    assert composite_score({"nonsense": 1.0}) == 0.0  # unknown ignored
    print("[selfcheck] composite score re-normalises over present metrics; "
          "an unmeasured metric is not a zero")

    # 10. Determinism + model-freedom.
    a = hillclimb("v0", sequence(["v1", "v2"]), scored({"v0": 0.5, "v1": 0.7,
                                                        "v2": 0.9}),
                  iterations=2)
    b = hillclimb("v0", sequence(["v1", "v2"]), scored({"v0": 0.5, "v1": 0.7,
                                                        "v2": 0.9}),
                  iterations=2)
    assert a.to_dict() == b.to_dict()
    import sys
    assert not any("openai" in m or "anthropic" in m for m in sys.modules
                   if m.startswith(("openai", "anthropic")))
    print("[selfcheck] deterministic; scorer and proposer injected, no model "
          "dependency")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
