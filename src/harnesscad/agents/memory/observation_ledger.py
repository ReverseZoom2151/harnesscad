"""Validation-failure observation ledger: aggregated per-family rule stats.

Source: ``resources/cad_repos/AgentSCAD-main`` (``skills/scad-generation/
learned-observations.jsonl``). AgentSCAD's generation skill carries a small
append-only JSONL ledger of *aggregated* validation outcomes: one record per
(part family, validation rule) with ``failure_count`` / ``total_checks`` /
``failure_rate`` / ``repair_success_rate`` and an integer ``confidence``
that grows with evidence. Before generating a part of a known family, the
highest-confidence failure patterns become advisory warnings ("this family
keeps failing R001 minimum wall thickness"), steering generation away from
repeat mistakes without any training.

This complements the harness's existing memory:
:mod:`harnesscad.agents.memory.error_notebook` stores *individual* corrected
error trajectories for retrieval; nothing aggregates outcome *statistics*
per (family, rule). The ledger is that aggregate: cheap, monotone evidence
counting with deterministic advisory emission.

VERIFICATION-FIRST INVARIANT: an advisory is a *statistic about past
failures*, not a verified fact about the current part. Advisories are
emitted for the planner/report channel, flagged ``unverified``; nothing here
feeds the model channel as trusted construction knowledge, and nothing here
ever marks a plan as passing or failing -- verifiers do that.

Stdlib only, deterministic, absolute imports. ``--selfcheck`` covers merge
arithmetic, JSONL round-trip, and advisory ordering.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Observation",
    "Advisory",
    "ObservationLedger",
    "main",
]

#: Confidence saturates: past this much evidence, more of the same failure
#: does not make the warning stronger (AgentSCAD stores small integers).
MAX_CONFIDENCE = 100


@dataclass
class Observation:
    """Aggregated outcomes for one (family, rule) pair."""
    family: str
    rule_id: str
    rule_name: str = ""
    failure_count: int = 0
    total_checks: int = 0
    repair_success_count: int = 0
    repair_attempt_count: int = 0
    confidence: int = 0
    source: str = "validation_pattern"

    @property
    def failure_rate(self) -> float:
        return self.failure_count / self.total_checks if self.total_checks else 0.0

    @property
    def repair_success_rate(self) -> float:
        if not self.repair_attempt_count:
            return 0.0
        return self.repair_success_count / self.repair_attempt_count

    def key(self) -> Tuple[str, str]:
        return (self.family, self.rule_id)

    def to_dict(self) -> dict:
        return {
            "observation_type": "validation_failure",
            "family": self.family,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "failure_count": self.failure_count,
            "total_checks": self.total_checks,
            "failure_rate": round(self.failure_rate, 6),
            "repair_success_count": self.repair_success_count,
            "repair_attempt_count": self.repair_attempt_count,
            "repair_success_rate": round(self.repair_success_rate, 6),
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Observation":
        return cls(
            family=str(d.get("family", "unknown")),
            rule_id=str(d.get("rule_id", "")),
            rule_name=str(d.get("rule_name", "")),
            failure_count=int(d.get("failure_count", 0)),
            total_checks=int(d.get("total_checks", 0)),
            repair_success_count=int(d.get("repair_success_count", 0)),
            repair_attempt_count=int(d.get("repair_attempt_count", 0)),
            confidence=int(d.get("confidence", 0)),
            source=str(d.get("source", "validation_pattern")),
        )


@dataclass(frozen=True)
class Advisory:
    """One warning derived from the ledger. Always unverified."""
    family: str
    rule_id: str
    rule_name: str
    failure_rate: float
    repair_success_rate: float
    confidence: int
    weight: float
    message: str
    unverified: bool = True

    def to_dict(self) -> dict:
        return {"family": self.family, "rule_id": self.rule_id,
                "rule_name": self.rule_name, "failure_rate": self.failure_rate,
                "repair_success_rate": self.repair_success_rate,
                "confidence": self.confidence, "weight": self.weight,
                "message": self.message, "unverified": self.unverified}


class ObservationLedger:
    """Monotone aggregation of validation outcomes per (family, rule)."""

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], Observation] = {}

    # -- recording ----------------------------------------------------------
    def record_check(self, family: str, rule_id: str, passed: bool,
                     rule_name: str = "") -> Observation:
        """Record one validation-rule outcome for a family."""
        obs = self._entries.get((family, rule_id))
        if obs is None:
            obs = Observation(family=family, rule_id=rule_id, rule_name=rule_name)
            self._entries[obs.key()] = obs
        if rule_name and not obs.rule_name:
            obs.rule_name = rule_name
        obs.total_checks += 1
        if not passed:
            obs.failure_count += 1
            obs.confidence = min(MAX_CONFIDENCE, obs.confidence + 1)
        return obs

    def record_repair(self, family: str, rule_id: str, succeeded: bool) -> Observation:
        """Record one repair attempt against a previously failing rule."""
        obs = self._entries.get((family, rule_id))
        if obs is None:
            obs = Observation(family=family, rule_id=rule_id)
            self._entries[obs.key()] = obs
        obs.repair_attempt_count += 1
        if succeeded:
            obs.repair_success_count += 1
        return obs

    def merge(self, other: "ObservationLedger") -> "ObservationLedger":
        """Fold another ledger in: counts add, confidence saturates."""
        for key, theirs in other._entries.items():
            mine = self._entries.get(key)
            if mine is None:
                self._entries[key] = Observation.from_dict(theirs.to_dict())
                continue
            mine.failure_count += theirs.failure_count
            mine.total_checks += theirs.total_checks
            mine.repair_success_count += theirs.repair_success_count
            mine.repair_attempt_count += theirs.repair_attempt_count
            mine.confidence = min(MAX_CONFIDENCE, mine.confidence + theirs.confidence)
            if theirs.rule_name and not mine.rule_name:
                mine.rule_name = theirs.rule_name
        return self

    # -- queries -------------------------------------------------------------
    @property
    def entries(self) -> List[Observation]:
        return [self._entries[k] for k in sorted(self._entries)]

    def get(self, family: str, rule_id: str) -> Optional[Observation]:
        return self._entries.get((family, rule_id))

    def advisories(self, family: str, min_confidence: int = 2,
                   min_failure_rate: float = 0.25,
                   limit: int = 5) -> List[Advisory]:
        """Warnings for a family, strongest first.

        Weight is ``failure_rate * (confidence / MAX_CONFIDENCE)``: a pattern
        must be both frequent and well-evidenced to rank. Ties break on
        rule id. A low ``repair_success_rate`` is called out in the message
        because it means the failure is also hard to fix after the fact.
        """
        out: List[Advisory] = []
        for obs in self.entries:
            if obs.family != family:
                continue
            if obs.confidence < min_confidence:
                continue
            if obs.failure_rate < min_failure_rate:
                continue
            weight = obs.failure_rate * (obs.confidence / MAX_CONFIDENCE)
            rule = obs.rule_name or obs.rule_id
            message = (f"family '{family}' has failed {rule} in "
                       f"{obs.failure_count}/{obs.total_checks} past checks")
            if obs.repair_attempt_count and obs.repair_success_rate < 0.5:
                message += (f"; repairs succeed only "
                            f"{obs.repair_success_rate:.0%} of the time -- "
                            "get it right at generation time")
            out.append(Advisory(
                family=family, rule_id=obs.rule_id, rule_name=obs.rule_name,
                failure_rate=round(obs.failure_rate, 6),
                repair_success_rate=round(obs.repair_success_rate, 6),
                confidence=obs.confidence, weight=round(weight, 6),
                message=message))
        out.sort(key=lambda a: (-a.weight, a.rule_id))
        return out[:max(0, limit)]

    # -- persistence ----------------------------------------------------------
    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(obs.to_dict(), sort_keys=True)
                         for obs in self.entries)

    @classmethod
    def from_jsonl(cls, text: str) -> "ObservationLedger":
        ledger = cls()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obs = Observation.from_dict(json.loads(line))
            existing = ledger._entries.get(obs.key())
            if existing is None:
                ledger._entries[obs.key()] = obs
            else:
                tmp = cls()
                tmp._entries[obs.key()] = obs
                ledger.merge(tmp)
        return ledger


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    ledger = ObservationLedger()
    # Ten enclosure checks: R001 fails 6 times, C001 once.
    for i in range(10):
        ledger.record_check("electronics_enclosure", "R001",
                            passed=i >= 6, rule_name="Minimum Wall Thickness")
    ledger.record_check("electronics_enclosure", "C001", passed=False,
                        rule_name="OpenSCAD Compile")
    for _ in range(9):
        ledger.record_check("electronics_enclosure", "C001", passed=True)
    # Repairs for R001 mostly fail.
    ledger.record_repair("electronics_enclosure", "R001", succeeded=False)
    ledger.record_repair("electronics_enclosure", "R001", succeeded=False)
    ledger.record_repair("electronics_enclosure", "R001", succeeded=True)

    r001 = ledger.get("electronics_enclosure", "R001")
    check(r001 is not None and abs(r001.failure_rate - 0.6) < 1e-12,
          "failure rate arithmetic")
    check(r001 is not None and abs(r001.repair_success_rate - 1 / 3) < 1e-12,
          "repair success rate arithmetic")
    check(r001 is not None and r001.confidence == 6, "confidence counts failures")

    advisories = ledger.advisories("electronics_enclosure")
    check(len(advisories) == 1 and advisories[0].rule_id == "R001",
          "low-rate C001 filtered; R001 advised")
    check("get it right at generation time" in advisories[0].message,
          "hard-to-repair pattern called out")
    check(advisories[0].unverified, "advisories are unverified")
    check(ledger.advisories("spur_gear") == [], "unknown family: no advisories")

    # JSONL round trip preserves the aggregate.
    text = ledger.to_jsonl()
    reloaded = ObservationLedger.from_jsonl(text)
    check(reloaded.to_jsonl() == text, "round-trip stable")

    # Merge: counts add, confidence saturates.
    other = ObservationLedger()
    for _ in range(3):
        other.record_check("electronics_enclosure", "R001", passed=False)
    reloaded.merge(other)
    merged = reloaded.get("electronics_enclosure", "R001")
    check(merged is not None and merged.failure_count == 9
          and merged.total_checks == 13, "merge adds counts")
    big = ObservationLedger()
    for _ in range(2 * MAX_CONFIDENCE):
        big.record_check("f", "X", passed=False)
    x = big.get("f", "X")
    check(x is not None and x.confidence == MAX_CONFIDENCE, "confidence saturates")

    # A real line from AgentSCAD's shipped ledger parses.
    agentscad_line = json.dumps({
        "observation_type": "validation_failure", "family": "unknown",
        "rule_id": "C001", "rule_name": "OpenSCAD Compile",
        "failure_count": 1, "total_checks": 1, "failure_rate": 1,
        "repair_success_rate": 0, "source": "validation_pattern",
        "confidence": 6, "ts": "2026-05-03T00:09:22.313Z"})
    shipped = ObservationLedger.from_jsonl(agentscad_line)
    obs = shipped.get("unknown", "C001")
    check(obs is not None and obs.confidence == 6 and obs.failure_count == 1,
          "AgentSCAD's shipped record ingests")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("observation_ledger selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validation-failure observation ledger (AgentSCAD)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
