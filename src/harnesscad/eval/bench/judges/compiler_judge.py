"""Stage-aware compiler judge with explicit verification levels and caching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from math import dist


class VerificationLevel(IntEnum):
    NONE = 0
    STRUCTURAL = 1
    VALIDITY = 2
    MORPHOLOGY = 3
    REQUIREMENTS = 4


@dataclass(frozen=True)
class JudgeResult:
    label: bool
    level: VerificationLevel
    stage: str
    distance: float | None
    threshold: float
    diagnostics: tuple[str, ...]
    provenance: dict
    cache_hit: bool = False

    @property
    def morphology_verified(self):
        return self.level >= VerificationLevel.MORPHOLOGY

    @property
    def requirements_verified(self):
        return self.level >= VerificationLevel.REQUIREMENTS


def symmetric_chamfer(left, right):
    a, b = tuple(left), tuple(right)
    if not a or not b:
        raise ValueError("point clouds must be non-empty")
    directed = lambda x, y: sum(min(dist(p, q)**2 for q in y) for p in x) / len(x)
    return directed(a, b) + directed(b, a)


def _digest(value):
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)
    except TypeError:
        text = repr(value)
    return hashlib.sha256(text.encode()).hexdigest()


class CompilerJudge:
    def __init__(self, compile_sequence, sample_shape, *, threshold,
                 sample_count=1024, seed=0, provenance=None, cache=None):
        if threshold < 0 or sample_count <= 0:
            raise ValueError("invalid judge configuration")
        self.compile = compile_sequence
        self.sample = sample_shape
        self.threshold = float(threshold)
        self.sample_count = sample_count
        self.seed = seed
        self.provenance = dict(provenance or {})
        self.cache = cache if cache is not None else {}

    def judge(self, candidate, reference):
        key = _digest({"candidate": candidate, "reference": reference,
                       "threshold": self.threshold, "count": self.sample_count,
                       "seed": self.seed, "provenance": self.provenance})
        if key in self.cache:
            prior = self.cache[key]
            return JudgeResult(**{**prior.__dict__, "cache_hit": True})
        base = {**self.provenance, "cache_key": key,
                "candidate_digest": _digest(candidate),
                "reference_digest": _digest(reference),
                "sample_count": self.sample_count, "seed": self.seed}
        try:
            predicted = self.compile(candidate)
        except Exception as exc:
            result = JudgeResult(False, VerificationLevel.STRUCTURAL, "candidate_compile",
                                 None, self.threshold,
                                 (f"{type(exc).__name__}: {exc}",), base)
            self.cache[key] = result
            return result
        try:
            expected = self.compile(reference)
        except Exception as exc:
            result = JudgeResult(False, VerificationLevel.NONE, "reference_compile",
                                 None, self.threshold,
                                 (f"{type(exc).__name__}: {exc}",), base)
            self.cache[key] = result
            return result
        try:
            left = self.sample(predicted, self.sample_count, self.seed)
            right = self.sample(expected, self.sample_count, self.seed)
            distance = symmetric_chamfer(left, right)
        except Exception as exc:
            result = JudgeResult(False, VerificationLevel.VALIDITY, "sample",
                                 None, self.threshold,
                                 (f"{type(exc).__name__}: {exc}",), base)
            self.cache[key] = result
            return result
        accepted = distance <= self.threshold
        result = JudgeResult(accepted, VerificationLevel.MORPHOLOGY, "distance",
                             distance, self.threshold, (), base)
        self.cache[key] = result
        return result


def component_scorecard(judge_result, *, command_fidelity=None,
                        requirements_verified=False):
    """Keep component evidence separate; validity is a hard gate."""
    return {
        "accepted": bool(judge_result.label and
                         judge_result.level >= VerificationLevel.MORPHOLOGY),
        "compile_valid": judge_result.level >= VerificationLevel.VALIDITY,
        "morphology_verified": judge_result.morphology_verified,
        "requirements_verified": bool(requirements_verified),
        "distance": judge_result.distance,
        "command_fidelity": command_fidelity,
    }


def pareto_scorecards(cards):
    cards = tuple(cards)
    def dominates(a, b):
        if not a["compile_valid"]:
            return False
        av = (-(a["distance"] if a["distance"] is not None else float("inf")),
              a["command_fidelity"] if a["command_fidelity"] is not None else -1,
              int(a["requirements_verified"]))
        bv = (-(b["distance"] if b["distance"] is not None else float("inf")),
              b["command_fidelity"] if b["command_fidelity"] is not None else -1,
              int(b["requirements_verified"]))
        return all(x >= y for x, y in zip(av, bv)) and any(x > y for x, y in zip(av, bv))
    return tuple(card for card in cards if card["compile_valid"] and
                 not any(dominates(other, card) for other in cards if other is not card))
