"""clarify_metrics -- ProCAD clarifier evaluation (Efficiency + Resolution).

Appendix G of the paper defines two metrics for the clarifying agent, both
"cast as a set-matching problem" that the paper delegates to an LLM-as-judge.
Because the clarification questions and issues in :mod:`clarify_ambiguity` carry
stable feature *keys*, the same matching can be computed deterministically:

  * **Efficiency** -- align generated questions to ground-truth questions;
    ``precision``/``recall`` over the match, reported as the ``F1``. A generated
    question with no ground-truth match is *redundant/hallucinated*.

  * **Resolution** -- a discrete score in ``{0, 0.5, 1}`` measuring whether the
    clarified specification resolves the ground-truth ambiguities: ``1`` fully,
    ``0.5`` partially (a subset of multiple issues fixed), ``0`` unresolved.

The paper's special-case rules on the ``is_misleading`` flag are also
implemented (Appendix G):
  * unambiguous prompt flagged ambiguous -> both scores 0;
  * unambiguous prompt correctly accepted -> both scores 1;
  * ambiguous prompt flagged unambiguous -> both scores 0.

Deterministic, stdlib-only. No LLM judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from clarify_ambiguity import CADSpec, AmbiguityDetector
from clarify_dialogue import _lookup


# --------------------------------------------------------------------------- #
# Efficiency: F1 over question-key set matching
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EfficiencyScore:
    precision: float
    recall: float
    f1: float
    matched: Tuple[str, ...]
    redundant: Tuple[str, ...]   # generated keys with no ground-truth match
    missed: Tuple[str, ...]      # ground-truth keys never asked


def efficiency(generated_keys: Iterable[str],
               ground_truth_keys: Iterable[str]) -> EfficiencyScore:
    """Efficiency F1 by aligning generated question keys to ground-truth keys."""
    gen = _dedup(generated_keys)
    gt = _dedup(ground_truth_keys)
    gt_set = set(gt)
    gen_set = set(gen)

    matched = [k for k in gen if k in gt_set]
    redundant = [k for k in gen if k not in gt_set]
    missed = [k for k in gt if k not in gen_set]

    tp = len(set(matched))
    precision = tp / len(gen_set) if gen_set else (1.0 if not gt_set else 0.0)
    recall = tp / len(gt_set) if gt_set else (1.0 if not gen_set else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return EfficiencyScore(precision, recall, f1,
                           tuple(sorted(set(matched))),
                           tuple(sorted(set(redundant))),
                           tuple(sorted(set(missed))))


# --------------------------------------------------------------------------- #
# Resolution: {0, 0.5, 1} over the target spec
# --------------------------------------------------------------------------- #

def resolution(clarified: CADSpec, target: CADSpec,
               issue_keys: Sequence[str]) -> float:
    """Discrete resolution score comparing ``clarified`` against ``target``.

    ``issue_keys`` are the ground-truth ambiguous slots. Returns 1.0 if every
    one now matches the target, 0.5 if some (but not all) do, else 0.0. Any
    remaining detectable ambiguity in the clarified spec caps the score at 0.
    """
    if AmbiguityDetector().detect(clarified):
        # New conflicts / still-missing dims => unresolved or invalid.
        remaining = {i.key for i in AmbiguityDetector().detect(clarified)}
        if remaining & set(issue_keys):
            fixed_any = _count_fixed(clarified, target, issue_keys) > 0
            return 0.5 if (fixed_any and len(issue_keys) > 1
                           and remaining != set(issue_keys)) else 0.0
        # ambiguity elsewhere -> treat as introduced conflict / invalid
        return 0.0
    fixed = _count_fixed(clarified, target, issue_keys)
    total = len(issue_keys)
    if total == 0:
        return 1.0
    if fixed == total:
        return 1.0
    if fixed == 0:
        return 0.0
    return 0.5


def _count_fixed(clarified: CADSpec, target: CADSpec,
                 keys: Sequence[str]) -> int:
    n = 0
    for key in keys:
        if _values_match(_lookup(clarified, key), _lookup(target, key)):
            n += 1
    return n


def _values_match(a: object, b: object) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_scalar_eq(x, y) for x, y in zip(a, b))
    return _scalar_eq(a, b)


def _scalar_eq(a: object, b: object) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9
    return a == b


# --------------------------------------------------------------------------- #
# Combined judgement with the is_misleading special cases (Appendix G)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClarifierScore:
    efficiency: float
    resolution: float


def score_interaction(*, prompt_is_ambiguous: bool, agent_flagged: bool,
                      generated_keys: Optional[Iterable[str]] = None,
                      ground_truth_keys: Optional[Iterable[str]] = None,
                      clarified: Optional[CADSpec] = None,
                      target: Optional[CADSpec] = None,
                      issue_keys: Optional[Sequence[str]] = None
                      ) -> ClarifierScore:
    """Full Appendix-G scoring, including the flag special cases.

    * unambiguous prompt, agent did not flag  -> (1, 1)
    * unambiguous prompt, agent flagged        -> (0, 0)
    * ambiguous prompt, agent did not flag     -> (0, 0)
    * otherwise -> Efficiency F1 and Resolution over the supplied specs.
    """
    if not prompt_is_ambiguous:
        return ClarifierScore(0.0, 0.0) if agent_flagged else ClarifierScore(1.0, 1.0)
    if not agent_flagged:
        return ClarifierScore(0.0, 0.0)
    eff = efficiency(generated_keys or (), ground_truth_keys or ()).f1
    if clarified is not None and target is not None:
        res = resolution(clarified, target, issue_keys or ())
    else:
        res = 0.0
    return ClarifierScore(eff, res)


# --------------------------------------------------------------------------- #
# Benchmark aggregation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BenchmarkReport:
    n: int
    mean_efficiency: float
    mean_resolution: float


def aggregate(scores: Sequence[ClarifierScore]) -> BenchmarkReport:
    """Mean Efficiency and Resolution over a benchmark of interactions."""
    if not scores:
        return BenchmarkReport(0, 0.0, 0.0)
    n = len(scores)
    me = sum(s.efficiency for s in scores) / n
    mr = sum(s.resolution for s in scores) / n
    return BenchmarkReport(n, me, mr)


def _dedup(keys: Iterable[str]) -> List[str]:
    out: List[str] = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out
