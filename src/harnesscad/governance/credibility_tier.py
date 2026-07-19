"""V&V-40 credibility tiering + regression-diff verdicts for result-bearing outputs.

Implements a credibility-tiering scheme plus a ``regression_diff`` verdict
taxonomy for parametric-edit governance.

ASME V&V-40 / NAFEMS principle: *model credibility must be commensurate with
the risk of the decision it informs.* Honesty signals (solver_executed,
is_solver_evidence, uncertainty, production_ready) are consolidated by the
classifier into ONE ordered tier every result can stamp. Two invariants:

  * **never-upgrade**: a tier is never more credible than its evidence. An
    output CLAIMING an executed-solver result whose ``solver_executed`` flag
    is not True is downgraded to ``unverified`` (rank 0) with an explicit
    ``downgrade_reason``; a surrogate marked as solver evidence stays a
    surrogate unless a solver actually ran.
  * **downgrade-on-insufficient-evidence**: unknown or unsupported evidence
    kinds land at ``unverified``; ``production_ready`` is forced False unless
    explicitly certified.

Tiers, low -> high: ``critique_finding`` < ``surrogate_prediction`` <
``proxy_assembly_result`` < ``executed_solver_result``.

The ``regression_diff`` verdict enum classifies what a parametric edit did to
the model's part topology (returned with every ``cad.edit_parameter`` response
as the agent's safety net):

  * ``clean``            -- only the intended part(s) changed;
  * ``collateral_change`` -- parts NOT targeted also moved (shared constant);
  * ``topology_changed`` -- the part set itself changed (part appeared or
    disappeared; unexpected for a pure dimensional edit);
  * ``identical``        -- nothing changed (wrong constant or no-op value).

Pure stdlib, deterministic; no kernel, no model, matching the governance
package's style.
"""

from __future__ import annotations

import argparse
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# credibility tiers
# --------------------------------------------------------------------------- #

#: Ordered low -> high credibility. Rank is index + 1; rank 0 is unverified.
CREDIBILITY_TIERS: Tuple[str, ...] = (
    "critique_finding",
    "surrogate_prediction",
    "proxy_assembly_result",
    "executed_solver_result",
)

UNVERIFIED = "unverified"

_TIER_META: Dict[str, Dict[str, Any]] = {
    "critique_finding": {
        "rank": 1,
        "label": "Critique finding",
        "evidence_basis": ("deterministic geometric / manufacturability "
                           "heuristic; no physics simulated"),
    },
    "surrogate_prediction": {
        "rank": 2,
        "label": "Surrogate prediction",
        "evidence_basis": ("data-driven surrogate estimate with an "
                           "uncertainty band; not solver evidence"),
    },
    "proxy_assembly_result": {
        "rank": 3,
        "label": "Proxy-assembly result",
        "evidence_basis": ("simplified proxy model; contact physics and bolt "
                           "preload not modeled"),
    },
    "executed_solver_result": {
        "rank": 4,
        "label": "Executed-solver result",
        "evidence_basis": "result from an executed solver run",
    },
}

_UNVERIFIED_META: Dict[str, Any] = {
    "rank": 0,
    "label": "Unverified",
    "evidence_basis": "no executed evidence; setup or draft only",
}

#: Producer-supplied evidence_kind aliases -> canonical tier (source table).
_KIND_TO_TIER: Dict[str, str] = {
    "critique": "critique_finding",
    "critique_finding": "critique_finding",
    "design_rule": "critique_finding",
    "geometry": "critique_finding",
    "surrogate": "surrogate_prediction",
    "surrogate_prediction": "surrogate_prediction",
    "proxy_assembly": "proxy_assembly_result",
    "proxy_assembly_result": "proxy_assembly_result",
    "assembly_proxy": "proxy_assembly_result",
    "solver": "executed_solver_result",
    "executed_solver": "executed_solver_result",
    "executed_solver_result": "executed_solver_result",
}


def credibility_rank(tier: str) -> int:
    """Rank of a tier (higher = more credible). Unknown / unverified -> 0."""
    meta = _TIER_META.get(tier)
    return int(meta["rank"]) if meta else 0


@dataclass(frozen=True)
class CredibilityStamp:
    """One self-describing credibility stamp for a result-bearing output."""

    tier: str
    rank: int
    label: str
    evidence_basis: str
    production_ready: bool
    signals: Dict[str, Any] = field(default_factory=dict)
    downgrade_reason: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {
            "tier": self.tier,
            "rank": self.rank,
            "label": self.label,
            "evidence_basis": self.evidence_basis,
            "production_ready": self.production_ready,
            "tier_order": list(CREDIBILITY_TIERS),
            "signals": dict(self.signals),
        }
        if self.downgrade_reason:
            out["downgrade_reason"] = self.downgrade_reason
        if self.notes:
            out["notes"] = self.notes
        return out


def classify_credibility(
    evidence_kind: str,
    solver_executed: Optional[bool] = None,
    is_solver_evidence: Optional[bool] = None,
    uncertainty_std: Optional[float] = None,
    production_ready: Optional[bool] = None,
    notes: Optional[str] = None,
) -> CredibilityStamp:
    """Map an evidence kind + honesty flags to ONE credibility tier.

    ``evidence_kind`` is what the producer BELIEVES the result is; the honesty
    flags downgrade that claim when they contradict it (never-upgrade
    invariant). ``production_ready`` is forced False unless explicitly True:
    the harness never certifies by default.
    """
    kind = (evidence_kind or "").strip().lower()
    base: Optional[str] = _KIND_TO_TIER.get(kind)
    downgrade_reason: Optional[str] = None

    if base == "executed_solver_result" and solver_executed is not True:
        base = None
        downgrade_reason = ("evidence_kind claims a solver result but "
                            "solver_executed is not true")
    elif base == "surrogate_prediction" and is_solver_evidence is True:
        # Contradictory claim: a surrogate marked as solver evidence only
        # earns the solver tier if a solver actually ran.
        base = ("executed_solver_result" if solver_executed is True
                else "surrogate_prediction")

    if base is None:
        meta = _UNVERIFIED_META
        tier = UNVERIFIED
        if downgrade_reason is None and kind:
            downgrade_reason = f"unknown evidence_kind {kind!r}"
    else:
        meta = _TIER_META[base]
        tier = base

    signals = {
        "evidence_kind": kind or None,
        "solver_executed": solver_executed,
        "is_solver_evidence": is_solver_evidence,
        "uncertainty_std": uncertainty_std,
    }
    signals = {k: v for k, v in signals.items() if v is not None}

    return CredibilityStamp(
        tier=tier,
        rank=int(meta["rank"]),
        label=str(meta["label"]),
        evidence_basis=str(meta["evidence_basis"]),
        production_ready=production_ready is True,
        signals=signals,
        downgrade_reason=downgrade_reason,
        notes=notes,
    )


def merge_credibility(stamps: Sequence[CredibilityStamp]) -> str:
    """The tier of a combined claim is the WEAKEST contributing tier.

    Corollary of never-upgrade: aggregating evidence cannot raise credibility
    above its least-credible input.
    """
    if not stamps:
        return UNVERIFIED
    weakest = min(stamps, key=lambda s: s.rank)
    return weakest.tier


# --------------------------------------------------------------------------- #
# regression-diff verdicts
# --------------------------------------------------------------------------- #

class RegressionVerdict(enum.Enum):
    """What a parametric edit did to the model, judged by per-part topology."""

    CLEAN = "clean"
    COLLATERAL_CHANGE = "collateral_change"
    TOPOLOGY_CHANGED = "topology_changed"
    IDENTICAL = "identical"


@dataclass(frozen=True)
class RegressionDiff:
    """A regression-diff result: the verdict plus the parts behind it."""

    verdict: RegressionVerdict
    changed_parts: Tuple[str, ...] = ()
    collateral_parts: Tuple[str, ...] = ()
    appeared_parts: Tuple[str, ...] = ()
    disappeared_parts: Tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "changed_parts": list(self.changed_parts),
            "collateral_parts": list(self.collateral_parts),
            "appeared_parts": list(self.appeared_parts),
            "disappeared_parts": list(self.disappeared_parts),
        }


def judge_regression(
    before_parts: Iterable[str],
    after_parts: Iterable[str],
    changed_parts: Iterable[str],
    targeted_parts: Iterable[str],
    global_edit: bool = False,
) -> RegressionDiff:
    """Classify an edit's before/after per-part comparison into a verdict.

    Inputs are part NAMES: the part sets before and after the edit, the parts
    whose geometry measurably changed, and the parts the edit intended to
    touch. Rules, in source precedence order:

      1. part set changed -> ``topology_changed`` (a part appeared or
         disappeared; unexpected for a pure dimensional edit);
      2. nothing changed -> ``identical`` (wrong constant or no-op value);
      3. non-targeted parts changed -> ``collateral_change``, naming them --
         UNLESS ``global_edit`` is True (edits to a shared global constant
         are MEANT to move many parts, so collateral is not judged);
      4. otherwise -> ``clean``.
    """
    before: FrozenSet[str] = frozenset(before_parts)
    after: FrozenSet[str] = frozenset(after_parts)
    changed: FrozenSet[str] = frozenset(changed_parts)
    targeted: FrozenSet[str] = frozenset(targeted_parts)

    if before != after:
        return RegressionDiff(
            verdict=RegressionVerdict.TOPOLOGY_CHANGED,
            changed_parts=tuple(sorted(changed)),
            appeared_parts=tuple(sorted(after - before)),
            disappeared_parts=tuple(sorted(before - after)),
        )
    if not changed:
        return RegressionDiff(verdict=RegressionVerdict.IDENTICAL)
    collateral = changed - targeted
    if collateral and not global_edit:
        return RegressionDiff(
            verdict=RegressionVerdict.COLLATERAL_CHANGE,
            changed_parts=tuple(sorted(changed)),
            collateral_parts=tuple(sorted(collateral)),
        )
    return RegressionDiff(
        verdict=RegressionVerdict.CLEAN,
        changed_parts=tuple(sorted(changed)),
    )


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="V&V-40 credibility tiering with never-upgrade invariant "
                    "+ regression_diff verdicts.",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="assert tier ordering, downgrade invariants, and "
                             "verdict classification on synthetic inputs.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. Ordering: ranks strictly increase along the tier tuple.
    ranks = [credibility_rank(t) for t in CREDIBILITY_TIERS]
    assert ranks == sorted(ranks) and len(set(ranks)) == len(ranks)
    assert credibility_rank(UNVERIFIED) == 0
    print(f"[selfcheck] tier order {CREDIBILITY_TIERS} ranks {ranks}")

    # 2. Honest solver claim earns the top tier.
    s = classify_credibility("solver", solver_executed=True)
    assert s.tier == "executed_solver_result" and s.rank == 4

    # 3. Never-upgrade: dishonest solver claim falls to unverified.
    s = classify_credibility("solver", solver_executed=False)
    assert s.tier == UNVERIFIED and s.rank == 0 and s.downgrade_reason
    s = classify_credibility("solver")  # flag missing entirely
    assert s.tier == UNVERIFIED and s.downgrade_reason
    print("[selfcheck] never-upgrade: unverified solver claim downgraded")

    # 4. Contradictory surrogate stays a surrogate without a real run.
    s = classify_credibility("surrogate", is_solver_evidence=True)
    assert s.tier == "surrogate_prediction"
    s = classify_credibility("surrogate", is_solver_evidence=True,
                             solver_executed=True)
    assert s.tier == "executed_solver_result"
    print("[selfcheck] surrogate-claiming-solver resolved by solver_executed")

    # 5. Unknown kind -> unverified; production_ready never defaults True.
    s = classify_credibility("vibes")
    assert s.tier == UNVERIFIED and not s.production_ready
    assert not classify_credibility("critique").production_ready
    assert classify_credibility("critique",
                                production_ready=True).production_ready
    print("[selfcheck] downgrade-on-insufficient-evidence + no default "
          "certification")

    # 6. Merge takes the weakest tier.
    stamps = [classify_credibility("solver", solver_executed=True),
              classify_credibility("critique")]
    assert merge_credibility(stamps) == "critique_finding"
    assert merge_credibility([]) == UNVERIFIED
    print("[selfcheck] merged claim inherits weakest tier")

    # 7. Regression verdicts.
    parts = ("base", "boss", "rib")
    d = judge_regression(parts, parts, ("boss",), ("boss",))
    assert d.verdict is RegressionVerdict.CLEAN, d.to_dict()
    d = judge_regression(parts, parts, ("boss", "rib"), ("boss",))
    assert d.verdict is RegressionVerdict.COLLATERAL_CHANGE
    assert d.collateral_parts == ("rib",)
    d = judge_regression(parts, parts, (), ("boss",))
    assert d.verdict is RegressionVerdict.IDENTICAL
    d = judge_regression(parts, ("base", "boss"), ("boss",), ("boss",))
    assert d.verdict is RegressionVerdict.TOPOLOGY_CHANGED
    assert d.disappeared_parts == ("rib",)
    d = judge_regression(parts, parts, ("boss", "rib"), ("boss",),
                         global_edit=True)
    assert d.verdict is RegressionVerdict.CLEAN  # global constants may fan out
    print("[selfcheck] regression_diff verdicts: clean / collateral / "
          "identical / topology_changed (+ global-edit exemption)")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
