"""CReFT-CAD curriculum data-engine: deterministic negative/masked sampling.

TriView2CAD's data engine (Sec. 4.1, "Data engine") builds instruction pairs for
the three curriculum tasks from a set of ground-truth parameter key-value pairs.
The *construction* of these training samples is a deterministic, seedable data
transformation — no model, no GPU — so it is buildable and testable here:

  * **Balanced split** — 50% of responses are fully correct, 50% contain errors.
  * **Task 1 (dichotomous)** — each negative corrupts ``n`` parameter values,
    with ``n ~ Uniform(1, 15)`` over the parameter count.
  * **Task 2 (multiple choice)** — ``p`` parameter values are masked out; an
    incorrect candidate list keeps all unmasked values correct except ``q``
    randomly chosen erroneous entries.
  * **Task 3 (parameterization / CoT)** — emit the composite-parameter reasoning
    steps: identify factor params, state the formula, compute the result.

All randomness flows through ``random.Random(seed)`` so a fixed seed reproduces
the exact sample stream. Corruption perturbs a value to a distinct value from a
supplied domain (or by a deterministic offset for numbers), never the same value.

Pure stdlib, deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _corrupt_value(value: object, rng: random.Random,
                   domain: Optional[Sequence[object]] = None) -> object:
    """Return a value distinct from ``value``.

    Prefers a random pick from ``domain`` (excluding the true value); falls back to
    a deterministic numeric offset or a string tag when no domain is given.
    """
    if domain:
        choices = [d for d in domain if d != value]
        if choices:
            return rng.choice(choices)
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        # Offset by a nonzero amount scaled off the value's magnitude.
        step = rng.randint(1, 9)
        delta = step if rng.random() < 0.5 else -step
        result = value + delta
        return int(result) if isinstance(value, int) else float(result)
    return str(value) + "_x"


def corrupt_parameters(params: Mapping[str, object], n: int,
                       rng: random.Random,
                       domains: Optional[Mapping[str, Sequence[object]]] = None
                       ) -> Dict[str, object]:
    """Corrupt exactly ``n`` distinct parameters of ``params`` (deterministic)."""
    keys = list(params.keys())
    if not (0 <= n <= len(keys)):
        raise ValueError("n must be in [0, %d], got %d" % (len(keys), n))
    domains = domains or {}
    chosen = rng.sample(keys, n) if n else []
    out = dict(params)
    for k in chosen:
        out[k] = _corrupt_value(params[k], rng, domains.get(k))
    return out


@dataclass(frozen=True)
class DichotomousSample:
    """A Task-1 instruction pair: parameters + the yes/no label."""

    parameters: Dict[str, object]
    label: bool  # True == "yes" (all correct); False == "no"
    num_errors: int = 0


def make_dichotomous_samples(ground_truth: Mapping[str, object],
                             count: int,
                             seed: int,
                             domains: Optional[Mapping[str, Sequence[object]]] = None
                             ) -> List[DichotomousSample]:
    """Build ``count`` Task-1 samples, 50% correct / 50% with n~Uniform(1,N) errors."""
    rng = random.Random(seed)
    keys = list(ground_truth.keys())
    samples: List[DichotomousSample] = []
    for i in range(count):
        positive = (i % 2 == 0)  # deterministic 50/50 split
        if positive:
            samples.append(DichotomousSample(dict(ground_truth), True, 0))
        else:
            n = rng.randint(1, len(keys)) if keys else 0
            corrupted = corrupt_parameters(ground_truth, n, rng, domains)
            samples.append(DichotomousSample(corrupted, False, n))
    return samples


@dataclass(frozen=True)
class MultipleChoiceSample:
    """A Task-2 instruction pair: a masked prompt + candidate value lists.

    ``masked`` are the parameter names hidden in the prompt. ``candidates`` maps a
    candidate id to a value list; ``correct_ids`` marks which candidates keep every
    unmasked value correct.
    """

    masked: Tuple[str, ...]
    candidates: Dict[str, Dict[str, object]]
    correct_ids: Tuple[str, ...]


def make_multiple_choice_sample(ground_truth: Mapping[str, object],
                                p: int, q: int, num_candidates: int,
                                seed: int,
                                domains: Optional[Mapping[str, Sequence[object]]] = None
                                ) -> MultipleChoiceSample:
    """Build one Task-2 sample: ``p`` masked params, incorrect candidates flip ``q``.

    Half of the (non-first) candidates are correct (unmasked values all right); the
    rest are incorrect with ``q`` erroneous unmasked entries. Candidate ``c0`` is
    always the fully-correct reference.
    """
    rng = random.Random(seed)
    keys = list(ground_truth.keys())
    if not (0 <= p <= len(keys)):
        raise ValueError("p must be in [0, %d]" % len(keys))
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")
    masked = tuple(rng.sample(keys, p)) if p else ()
    unmasked = [k for k in keys if k not in masked]
    if not (0 <= q <= len(unmasked)):
        raise ValueError("q must be in [0, %d]" % len(unmasked))
    domains = domains or {}
    candidates: Dict[str, Dict[str, object]] = {}
    correct_ids: List[str] = []
    for idx in range(num_candidates):
        cid = "c%d" % idx
        cand = {k: ground_truth[k] for k in unmasked}
        is_correct = (idx == 0) or (idx % 2 == 0)
        if not is_correct and q > 0:
            flip = rng.sample(unmasked, q)
            for k in flip:
                cand[k] = _corrupt_value(ground_truth[k], rng, domains.get(k))
        else:
            is_correct = True
        candidates[cid] = cand
        if is_correct:
            correct_ids.append(cid)
    return MultipleChoiceSample(masked=masked, candidates=candidates,
                                correct_ids=tuple(correct_ids))


@dataclass(frozen=True)
class CoTStep:
    kind: str      # "identify" | "formula" | "compute"
    detail: str


def build_cot_steps(name: str, op: str, factors: Sequence[str],
                    params: Mapping[str, float]) -> List[CoTStep]:
    """Chain-of-Thought steps for a composite parameter (Sec. 4.1, Task 3).

    1) identify the factor parameters, 2) state the formula, 3) compute the result.
    """
    factor_list = list(factors)
    joiner = " %s " % op
    formula = joiner.join(factor_list)
    acc = float(params[factor_list[0]])
    for f in factor_list[1:]:
        if op == "+":
            acc += float(params[f])
        elif op == "-":
            acc -= float(params[f])
        elif op == "*":
            acc *= float(params[f])
        else:
            raise ValueError("unsupported op %r" % (op,))
    return [
        CoTStep("identify", "factors: " + ", ".join(factor_list)),
        CoTStep("formula", "%s = %s" % (name, formula)),
        CoTStep("compute", "%s = %s" % (name, _fmt(acc))),
    ]


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return ("%.4f" % x).rstrip("0").rstrip(".")
