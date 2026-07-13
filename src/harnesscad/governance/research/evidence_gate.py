"""Deterministic governance for evidence-backed engineering research.

No network, model, or vendor service is required.  The package records claims
and evidence, ensembles independent reviewer scores, applies explicit stage
gates, and supports checkpoint/rollback of governance state.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    checksum: str
    kind: str = "experiment"
    reproducible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "source": self.source, "checksum": self.checksum,
            "kind": self.kind, "reproducible": self.reproducible,
            "metadata": dict(sorted(self.metadata.items())),
        }


@dataclass(frozen=True)
class Claim:
    id: str
    statement: str
    evidence_ids: tuple[str, ...]
    expected_result: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "statement": self.statement,
            "evidence_ids": list(self.evidence_ids),
            "expected_result": self.expected_result,
        }


@dataclass(frozen=True)
class Review:
    reviewer: str
    scores: Mapping[str, float]
    recommendation: str = "refine"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.recommendation not in {"advance", "refine", "reject"}:
            raise ValueError("recommendation must be advance, refine, or reject")
        if not self.scores:
            raise ValueError("review must contain scores")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.scores.values()):
            raise ValueError("review scores must be in [0, 1]")

    @property
    def mean(self) -> float:
        return sum(float(v) for v in self.scores.values()) / len(self.scores)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateDecision:
    outcome: str
    stage_before: str
    stage_after: str
    ensemble_score: float
    checks: CheckResult
    rationale: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome, "stage_before": self.stage_before,
            "stage_after": self.stage_after, "ensemble_score": self.ensemble_score,
            "checks": {
                "ok": self.checks.ok, "errors": list(self.checks.errors),
                "warnings": list(self.checks.warnings),
            },
            "rationale": list(self.rationale),
        }


class ResearchGovernance:
    def __init__(self, stages: Sequence[str] = ("discovery", "validation", "release")) -> None:
        if not stages or len(set(stages)) != len(stages):
            raise ValueError("stages must be non-empty and unique")
        self.stages = tuple(stages)
        self.stage_index = 0
        self.evidence: dict[str, Evidence] = {}
        self.claims: dict[str, Claim] = {}
        self.decisions: list[GateDecision] = []
        self._checkpoints: dict[str, dict] = {}

    @property
    def stage(self) -> str:
        return self.stages[self.stage_index]

    def add_evidence(self, evidence: Evidence) -> None:
        if not evidence.id or not evidence.source or not evidence.checksum:
            raise ValueError("evidence id, source, and checksum are required")
        if evidence.id in self.evidence and self.evidence[evidence.id] != evidence:
            raise ValueError(f"conflicting evidence id: {evidence.id}")
        self.evidence[evidence.id] = evidence

    def add_claim(self, claim: Claim) -> None:
        if not claim.id or not claim.statement or not claim.evidence_ids:
            raise ValueError("claim id, statement, and evidence links are required")
        if claim.id in self.claims and self.claims[claim.id] != claim:
            raise ValueError(f"conflicting claim id: {claim.id}")
        self.claims[claim.id] = claim

    def check(self) -> CheckResult:
        errors: list[str] = []
        warnings: list[str] = []
        for claim in sorted(self.claims.values(), key=lambda item: item.id):
            missing = sorted(set(claim.evidence_ids) - self.evidence.keys())
            if missing:
                errors.append(f"claim {claim.id} missing evidence: {', '.join(missing)}")
            if len(set(claim.evidence_ids)) != len(claim.evidence_ids):
                warnings.append(f"claim {claim.id} has duplicate evidence links")
        linked = {eid for claim in self.claims.values() for eid in claim.evidence_ids}
        for evidence in sorted(self.evidence.values(), key=lambda item: item.id):
            if evidence.id not in linked:
                warnings.append(f"evidence {evidence.id} is unlinked")
            if not evidence.reproducible:
                errors.append(f"evidence {evidence.id} is not reproducible")
            expected = evidence.metadata.get("expected_result")
            observed = evidence.metadata.get("observed_result")
            if expected is not None and observed is not None and expected != observed:
                errors.append(f"evidence {evidence.id} result is inconsistent")
        if not self.claims:
            errors.append("no claims registered")
        return CheckResult(not errors, tuple(errors), tuple(warnings))

    @staticmethod
    def ensemble(reviews: Sequence[Review]) -> float:
        if not reviews:
            raise ValueError("at least one review is required")
        # Each reviewer gets equal weight regardless of criterion count.
        return round(sum(review.mean for review in reviews) / len(reviews), 6)

    def evaluate_gate(
        self, reviews: Sequence[Review], *, advance_threshold: float = 0.75,
        reject_threshold: float = 0.35,
    ) -> GateDecision:
        checks = self.check()
        score = self.ensemble(reviews)
        recommendations = [review.recommendation for review in reviews]
        reject_votes = recommendations.count("reject")
        reasons = list(checks.errors)
        before = self.stage

        if reject_votes > len(reviews) / 2 or score < reject_threshold:
            outcome = "reject"
            reasons.append("review ensemble rejected the work")
        elif checks.ok and score >= advance_threshold and recommendations.count("advance") >= len(reviews) / 2:
            outcome = "advance"
            reasons.append("evidence checks and review threshold passed")
        else:
            outcome = "refine"
            reasons.append("gate requirements need refinement")

        if outcome == "advance" and self.stage_index < len(self.stages) - 1:
            self.stage_index += 1
        after = self.stage
        decision = GateDecision(outcome, before, after, score, checks, tuple(reasons))
        self.decisions.append(decision)
        return decision

    def checkpoint(self, label: str) -> str:
        if not label:
            raise ValueError("checkpoint label is required")
        state = {
            "stage_index": self.stage_index,
            "evidence": copy.deepcopy(self.evidence),
            "claims": copy.deepcopy(self.claims),
            "decisions": copy.deepcopy(self.decisions),
        }
        self._checkpoints[label] = state
        return self.state_digest()

    def rollback(self, label: str) -> None:
        state = copy.deepcopy(self._checkpoints[label])
        self.stage_index = state["stage_index"]
        self.evidence = state["evidence"]
        self.claims = state["claims"]
        self.decisions = state["decisions"]

    def state_digest(self) -> str:
        payload = {
            "stage": self.stage,
            "evidence": [e.to_dict() for e in sorted(self.evidence.values(), key=lambda x: x.id)],
            "claims": [c.to_dict() for c in sorted(self.claims.values(), key=lambda x: x.id)],
            "decisions": [d.to_dict() for d in self.decisions],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
