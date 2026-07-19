"""Tri-state "no silent green" scorecard: the aggregator that makes a skipped
check impossible to render as a pass.

The invariant, made concrete: a check that COULD NOT RUN never rolls up as
green. It reports a distinct third state -- ``UNKNOWN`` -- and that state is
CONTAGIOUS to green (any unknown keeps the whole scorecard off green) yet
DISTINCT from fail (unknown is a gap, not a defect). This is the harness's
"silence is not success" rule applied to a SET of checks: the exact bug this
project exists to prevent is a skipped check displayed as success.

The harness already carries the tri-state at the CHECK level -- io/gate.py
leaves an unmeasured field ``None`` and refuses rather than defaulting a pass
(``validation-rules-unevaluated``), eval/verifiers marks results ``skipped``,
and eval/reliability/error_contract.py distinguishes retry from abstain. What
was missing, and what this module supplies, is the general AGGREGATOR that
takes a set of typed {pass, fail, unknown} results and enforces
"any unknown => not green" over the whole set, with every unknown carrying its
reason (never a bare "unknown" -- always "unknown BECAUSE x", mirroring the
HONEST_RESIDUAL discipline).

Composition, not restatement:

  * This scorecard answers ONE question -- *did every check actually run and
    pass, so I may show green?* -- the RUN-STATE axis.
  * ``governance/credibility_tier.py`` answers an ORTHOGONAL question -- *how
    strong is the evidence behind a result that DID run?* -- the
    EVIDENCE-STRENGTH axis, with its own never-upgrade invariant.

The two axes stack: a scorecard can be fully green on run-state (every check
ran and passed) while its merged credibility tier is still weak. A result is
only truly ship-ready when it clears BOTH. To make that composition real (not
just documented), a :class:`CheckResult` may carry an optional
``CredibilityStamp`` and :meth:`Scorecard.credibility` folds them with
``credibility_tier.merge_credibility`` -- inheriting the weakest tier, exactly
as that module's own corollary requires. credibility_tier is imported
read-only; nothing here modifies it.

The scorecard is implemented in pure stdlib with dataclasses and an enum. Its
roll-up is deterministic: FAIL if any check fails, UNKNOWN if any could not
run or there are no checks, otherwise PASS. Every UNKNOWN requires a reason
and composes with the harness's credibility tiers; no kernel or model is
required.
"""

from __future__ import annotations

import argparse
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.governance.credibility_tier import (
    UNVERIFIED,
    CredibilityStamp,
    merge_credibility,
)

__all__ = [
    "TriState",
    "CheckResult",
    "Scorecard",
    "tri_state_rollup",
]


class TriState(enum.Enum):
    """A check's run-state outcome, and the scorecard verdict rolled up from a
    set of them. ``UNKNOWN`` is never silently treated as ``PASS`` -- a check
    that could not run reports it, and it keeps the whole scorecard off green.
    """

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"

    @property
    def is_green(self) -> bool:
        """Green means only one thing: the check ran and passed."""
        return self is TriState.PASS


@dataclass(frozen=True)
class CheckResult:
    """One check's typed result: a name, a tri-state, and a reason.

    The reason-required invariant is the honesty rule made structural: an
    ``UNKNOWN`` (and a ``FAIL``) MUST carry a non-empty reason. A bare
    "unknown" is forbidden -- it must always be "unknown BECAUSE x", so a gap
    can never be displayed without the reason it is a gap. A ``PASS`` may carry
    a detail but need not.

    ``credibility`` optionally attaches the EVIDENCE-STRENGTH stamp from
    ``credibility_tier`` for a check that ran; the run-state tri-state above is
    independent of it (a check can PASS on a weak tier). :meth:`Scorecard.
    credibility` folds the present stamps to the weakest tier.
    """

    name: str
    state: TriState
    reason: str = ""
    credibility: Optional[CredibilityStamp] = None

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("a check result needs a name")
        if not isinstance(self.state, TriState):
            raise TypeError(f"state must be a TriState, got {self.state!r}")
        if self.state is TriState.UNKNOWN and not str(self.reason).strip():
            raise ValueError(
                f"check {self.name!r} is UNKNOWN but carries no reason; an "
                "unknown must always be 'unknown BECAUSE x', never bare")
        if self.state is TriState.FAIL and not str(self.reason).strip():
            raise ValueError(
                f"check {self.name!r} is FAIL but carries no reason; a "
                "failure must say why it failed")

    # -- constructors: one per state, so a caller states intent explicitly -- #
    @classmethod
    def passing(cls, name: str, detail: str = "",
                credibility: Optional[CredibilityStamp] = None) -> "CheckResult":
        """A check that ran and passed."""
        return cls(name=name, state=TriState.PASS, reason=detail,
                   credibility=credibility)

    @classmethod
    def failing(cls, name: str, reason: str,
                credibility: Optional[CredibilityStamp] = None) -> "CheckResult":
        """A check that ran and failed. ``reason`` is required."""
        return cls(name=name, state=TriState.FAIL, reason=reason,
                   credibility=credibility)

    @classmethod
    def unknown(cls, name: str, reason: str) -> "CheckResult":
        """A check that COULD NOT RUN. ``reason`` -- why it could not -- is
        required; there is no credibility stamp because nothing was evidenced."""
        return cls(name=name, state=TriState.UNKNOWN, reason=reason)

    @property
    def passed(self) -> bool:
        """True only when the check actually ran and passed -- never for an
        unknown, which is the whole point."""
        return self.state is TriState.PASS

    @property
    def ran(self) -> bool:
        """Whether the check ran at all (pass or fail, not unknown)."""
        return self.state is not TriState.UNKNOWN

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {
            "name": self.name,
            "state": self.state.value,
            "reason": self.reason,
        }
        if self.credibility is not None:
            out["credibility"] = self.credibility.to_dict()
        return out

    def __str__(self) -> str:
        because = f": {self.reason}" if self.reason else ""
        return f"[{self.state.value.upper()}] {self.name}{because}"


def tri_state_rollup(results: Sequence[CheckResult]) -> TriState:
    """Roll a set of typed results up to ONE verdict, honouring no-silent-green.

    The aggregation rule, in precedence order:

      1. any ``FAIL`` -> ``FAIL`` (a real defect dominates; it is louder than a
         gap);
      2. else any ``UNKNOWN`` -- OR no checks at all -> ``UNKNOWN`` (a gap, or
         a total absence of evidence, is contagious to green: it can never
         render as pass);
      3. only when EVERY check ran and passed -> ``PASS``.

    So ``PASS`` is never returned while any check is unknown, and ``UNKNOWN`` is
    never collapsed into ``FAIL`` -- unknown is contagious-to-green but distinct
    from fail. An empty set is ``UNKNOWN``, never ``PASS``: nothing checked
    proves nothing.
    """
    if any(r.state is TriState.FAIL for r in results):
        return TriState.FAIL
    if not results or any(r.state is TriState.UNKNOWN for r in results):
        return TriState.UNKNOWN
    return TriState.PASS


@dataclass(frozen=True)
class Scorecard:
    """A set of check results with a rolled-up tri-state verdict.

    :attr:`verdict` is green (:attr:`is_green`) ONLY when there is at least one
    check and every one ran and passed. Any unknown keeps it off green while
    staying distinct from a failure; the reasons behind every unknown and
    failure are preserved and surfaced, never dropped.
    """

    results: Tuple[CheckResult, ...] = ()

    @classmethod
    def of(cls, results: Sequence[CheckResult]) -> "Scorecard":
        return cls(results=tuple(results))

    @property
    def verdict(self) -> TriState:
        return tri_state_rollup(self.results)

    @property
    def is_green(self) -> bool:
        """True only when the verdict is PASS -- the single honest 'show green'."""
        return self.verdict is TriState.PASS

    def failures(self) -> Tuple[CheckResult, ...]:
        """The checks that ran and failed -- the blocking defects."""
        return tuple(r for r in self.results if r.state is TriState.FAIL)

    def unknowns(self) -> Tuple[CheckResult, ...]:
        """The checks that could not run -- the gaps, never silently passed."""
        return tuple(r for r in self.results if r.state is TriState.UNKNOWN)

    def unknown_reasons(self) -> Tuple[str, ...]:
        """Every gap as 'name: reason' -- the "unknown BECAUSE x" residual the
        verdict carries so a non-green scorecard always says why it is not green."""
        return tuple(f"{r.name}: {r.reason}" for r in self.unknowns())

    def credibility(self) -> str:
        """The merged EVIDENCE-STRENGTH tier across results that carry a stamp.

        Composes with ``credibility_tier`` on the orthogonal axis: even a fully
        green scorecard (every check ran and passed) is bounded by its weakest
        evidence tier here, so "green" on run-state and "credible" on evidence
        remain two separate clearances. No stamps -> ``UNVERIFIED``.
        """
        stamps = [r.credibility for r in self.results if r.credibility is not None]
        if not stamps:
            return UNVERIFIED
        return merge_credibility(stamps)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "is_green": self.is_green,
            "counts": {
                "pass": sum(1 for r in self.results if r.state is TriState.PASS),
                "fail": len(self.failures()),
                "unknown": len(self.unknowns()),
                "total": len(self.results),
            },
            "credibility": self.credibility(),
            "results": [r.to_dict() for r in self.results],
            "failures": [r.name for r in self.failures()],
            "unknown_reasons": list(self.unknown_reasons()),
        }

    def __str__(self) -> str:
        c = self.to_dict()["counts"]
        return (f"scorecard {self.verdict.value.upper()} "
                f"(pass={c['pass']} fail={c['fail']} unknown={c['unknown']})")


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def _demo_scorecards() -> Dict[str, Scorecard]:
    """The synthetic scorecards the selfcheck asserts against."""
    return {
        "all_pass": Scorecard.of([
            CheckResult.passing("watertight", "closed 2-manifold"),
            CheckResult.passing("safety_factor", "3.1 vs 2.0 required"),
        ]),
        "one_unknown_among_pass": Scorecard.of([
            CheckResult.passing("watertight", "closed 2-manifold"),
            CheckResult.passing("safety_factor", "3.1 vs 2.0 required"),
            CheckResult.unknown(
                "self_intersection",
                "mesh too large to check honestly; not measured"),
        ]),
        "one_fail_among_pass": Scorecard.of([
            CheckResult.passing("watertight", "closed 2-manifold"),
            CheckResult.failing("safety_factor", "1.2 below 2.0 required"),
        ]),
        "fail_and_unknown": Scorecard.of([
            CheckResult.failing("safety_factor", "1.2 below 2.0 required"),
            CheckResult.unknown("fatigue", "S-N curve unavailable for alloy"),
        ]),
        "empty": Scorecard.of([]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad-scorecard",
        description="Tri-state 'no silent green' scorecard aggregator: any "
                    "unknown keeps the set off green, distinct from fail, with "
                    "the reason always carried (anvilate scorecard.py port).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove the no-silent-green invariant on synthetic "
                             "scorecards: one unknown among all-pass is NOT "
                             "green, unknown != fail, a reason is always kept.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    cards = _demo_scorecards()

    # 1. All checks ran and passed -> green.
    green = cards["all_pass"]
    assert green.verdict is TriState.PASS, green.to_dict()
    assert green.is_green
    print(f"[selfcheck] all-pass -> {green}")

    # 2. THE INVARIANT: one unknown among all-pass is NOT green -- and it is
    #    UNKNOWN, not FAIL. Unknown is contagious to green but distinct from
    #    fail.
    mixed = cards["one_unknown_among_pass"]
    assert mixed.verdict is TriState.UNKNOWN, mixed.to_dict()
    assert not mixed.is_green, "an unknown check must keep the scorecard off green"
    assert mixed.verdict is not TriState.FAIL, "unknown must not be treated as fail"
    print(f"[selfcheck] one unknown among all-pass -> {mixed} "
          f"(not green, not fail)")

    # 3. The reason is always carried -- never a bare unknown. A non-green
    #    scorecard says why it is not green.
    reasons = mixed.unknown_reasons()
    assert reasons and all(": " in r and r.split(": ", 1)[1] for r in reasons), reasons
    print(f"[selfcheck] unknown carries its reason -> {reasons[0]!r}")

    # 4. A bare unknown is structurally impossible: the constructor refuses it.
    try:
        CheckResult.unknown("no_reason", "")
    except ValueError:
        print("[selfcheck] a reasonless UNKNOWN is rejected at construction")
    else:  # pragma: no cover - defensive
        raise AssertionError("a reasonless UNKNOWN must be rejected")

    # 5. A real defect dominates: fail beats pass, and fail beats unknown too
    #    (fail is louder than a gap).
    assert cards["one_fail_among_pass"].verdict is TriState.FAIL
    assert cards["fail_and_unknown"].verdict is TriState.FAIL
    print("[selfcheck] fail dominates pass and unknown")

    # 6. Nothing checked proves nothing: an empty scorecard is UNKNOWN, not
    #    green.
    empty = cards["empty"]
    assert empty.verdict is TriState.UNKNOWN and not empty.is_green
    print("[selfcheck] empty scorecard -> UNKNOWN (no checks proves nothing)")

    # 7. Composition with credibility_tier: the two honesty axes are orthogonal
    #    and stack. A fully GREEN scorecard on run-state can still be bounded by
    #    a weak evidence tier -- green != credible.
    from harnesscad.governance.credibility_tier import classify_credibility
    solver = classify_credibility("solver", solver_executed=True)  # top tier
    critique = classify_credibility("critique")                    # weakest tier
    composed = Scorecard.of([
        CheckResult.passing("stress", "within allowable", credibility=solver),
        CheckResult.passing("dfm_clearance", "ok", credibility=critique),
    ])
    assert composed.is_green, "run-state: every check ran and passed"
    assert composed.credibility() == "critique_finding", composed.credibility()
    print(f"[selfcheck] green on run-state yet credibility bounded to weakest "
          f"tier {composed.credibility()!r} (axes stack, not restated)")

    # 8. Determinism: same inputs -> identical serialisation.
    assert mixed.to_dict() == cards["one_unknown_among_pass"].to_dict()
    print("[selfcheck] deterministic scorecard serialisation")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
