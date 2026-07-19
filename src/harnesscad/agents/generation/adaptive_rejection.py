"""Adaptive rejection-sampling control policy with rollback regeneration.

The harness carries bounds, collision, and stability verifiers; this control
policy turns a per-step verifier into a reliable whole-sequence generator.
Three mechanisms:

1. **Rejection sampling with temperature escalation** (source:
   ``generate_brick_with_rejection_sampling``): sample a step, verify it,
   and on rejection resample within a per-step rejection budget
   (``max_step_rejections``, source default 500). When the sampler re-emits
   a step that was ALREADY rejected, the sampling temperature is raised by
   ``temperature_increase`` (+0.01) per repeat, capped at ``max_temperature``
   (2.0) -- heat is added only when the sampler is stuck in a loop, not on
   every failure.

2. **Per-step budget exhaustion is non-fatal** (source: the loop logs a
   warning and keeps the last sample): the step is emitted flagged rather
   than aborting the whole generation.

3. **Rollback-to-last-verified-prefix regeneration** (source: ``__call__`` +
   ``_remove_all_bricks_after_first_unstable_brick``): after a full sequence
   is generated, a WHOLE-SEQUENCE verifier is consulted; if it names a bad
   index, everything from the first bad element onward is truncated and
   generation resumes from that verified prefix, up to ``max_regenerations``
   (source default 100) times.

No model dependency: both the sampler and the verifiers are injected
callables, so the policy drives an LLM, a grammar sampler, or a synthetic
test generator identically.

Attribution: BrickGPT (models/brickgpt.py). Pure stdlib, deterministic given
a deterministic sampler; no kernel, no model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

#: sampler(prefix, temperature, step_index) -> step or None (None = done/EOS).
Sampler = Callable[[Sequence[object], float, int], Optional[object]]

#: step_verifier(step, prefix) -> "success" or a rejection-reason string.
StepVerifier = Callable[[object, Sequence[object]], str]

#: sequence_verifier(steps) -> index of first bad element, or None if stable.
SequenceVerifier = Callable[[Sequence[object]], Optional[int]]

SUCCESS = "success"
ALREADY_REJECTED = "already_rejected"


@dataclass
class RejectionConfig:
    """Knobs of the control policy; defaults mirror BrickGPTConfig."""

    max_steps: int = 2000                 # source: max_bricks
    max_step_rejections: int = 500        # source: max_brick_rejections
    max_regenerations: int = 100          # source: max_regenerations
    temperature: float = 0.6              # source: temperature
    temperature_increase: float = 0.01    # source: temperature_increase
    max_temperature: float = 2.0          # source: max_temperature

    def validate(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_step_rejections < 0:
            raise ValueError("max_step_rejections must be >= 0")
        if self.max_regenerations < 0:
            raise ValueError("max_regenerations must be >= 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.max_temperature < self.temperature:
            raise ValueError("max_temperature must be >= temperature")


@dataclass
class GenerationResult:
    """Outcome of one full adaptive-rejection generation."""

    steps: List[object] = field(default_factory=list)
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    n_regenerations: int = 0
    stable: bool = True
    exhausted_steps: List[int] = field(default_factory=list)  # budget ran out

    def to_dict(self) -> dict:
        return {
            "n_steps": len(self.steps),
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "n_regenerations": self.n_regenerations,
            "stable": self.stable,
            "exhausted_steps": list(self.exhausted_steps),
        }


def _bump(counter: Dict[str, int], reason: str) -> None:
    counter[reason] = counter.get(reason, 0) + 1


def sample_step_with_rejection(
    sampler: Sampler,
    step_verifier: StepVerifier,
    prefix: Sequence[object],
    step_index: int,
    cfg: RejectionConfig,
) -> tuple:
    """Sample one verified step (BrickGPT ``generate_brick_with_rejection_sampling``).

    Returns ``(step_or_None, reasons_dict, exhausted_bool)``. ``None`` means
    the sampler signalled end-of-sequence. Temperature escalates by
    ``cfg.temperature_increase`` (capped at ``cfg.max_temperature``) each time
    an already-rejected step is re-sampled -- the source's exact policy.
    """
    reasons: Dict[str, int] = {}
    rejected: set = set()
    temperature = cfg.temperature
    step: Optional[object] = None

    for generation_num in range(cfg.max_step_rejections + 1):
        step = sampler(prefix, temperature, step_index)
        if step is None:  # EOS
            return None, reasons, False
        if cfg.max_step_rejections == 0:  # rejection sampling disabled
            return step, reasons, False

        key = repr(step)
        if key in rejected:
            verdict = ALREADY_REJECTED
        else:
            verdict = step_verifier(step, prefix)
        if verdict == SUCCESS:
            return step, reasons, False
        if generation_num == cfg.max_step_rejections:
            # Budget exhausted: keep the last sample, flagged (source logs a
            # warning and proceeds with the invalid brick).
            _bump(reasons, verdict)
            return step, reasons, True

        _bump(reasons, verdict)
        rejected.add(key)
        if verdict == ALREADY_REJECTED:
            temperature = min(cfg.max_temperature,
                              temperature + cfg.temperature_increase)

    return step, reasons, True  # not reached; loop always returns


def generate_sequence(
    sampler: Sampler,
    step_verifier: StepVerifier,
    cfg: RejectionConfig,
    starting_steps: Sequence[object] = (),
) -> tuple:
    """One pass of stepwise generation from a prefix (source ``_generate_structure``).

    Returns ``(steps_list, reasons_dict, exhausted_indices)``.
    """
    steps: List[object] = list(starting_steps)
    reasons: Dict[str, int] = {}
    exhausted: List[int] = []
    for step_index in range(len(steps), cfg.max_steps):
        step, step_reasons, was_exhausted = sample_step_with_rejection(
            sampler, step_verifier, steps, step_index, cfg)
        for k, v in step_reasons.items():
            reasons[k] = reasons.get(k, 0) + v
        if step is None:
            break
        if was_exhausted:
            exhausted.append(step_index)
        steps.append(step)
    return steps, reasons, exhausted


def rollback_to_verified_prefix(
    steps: Sequence[object],
    sequence_verifier: SequenceVerifier,
) -> List[object]:
    """Truncate at the first bad element, repeatedly, until the prefix verifies.

    Source: ``_remove_all_bricks_after_first_unstable_brick`` -- removal can
    expose a new first-unstable element, so the check loops to a fixed point.
    Always terminates: each truncation strictly shortens the sequence.
    """
    current = list(steps)
    while current:
        bad = sequence_verifier(current)
        if bad is None:
            return current
        current = current[:max(0, bad)]
    return current


def adaptive_rejection_generate(
    sampler: Sampler,
    step_verifier: StepVerifier,
    sequence_verifier: Optional[SequenceVerifier] = None,
    cfg: Optional[RejectionConfig] = None,
) -> GenerationResult:
    """Full BrickGPT control policy: rejection-sample, then rollback-regenerate.

    ``sequence_verifier`` is the whole-sequence (physics/feasibility) check;
    ``None`` (or ``cfg.max_regenerations == 0``) disables rollback, exactly
    like the source's ``max_regenerations=0``.
    """
    cfg = cfg or RejectionConfig()
    cfg.validate()

    result = GenerationResult()
    starting: List[object] = []
    regeneration_num = 0
    steps: List[object] = []

    for regeneration_num in range(cfg.max_regenerations + 1):
        steps, reasons, exhausted = generate_sequence(
            sampler, step_verifier, cfg, starting_steps=starting)
        for k, v in reasons.items():
            _bump_n(result.rejection_reasons, k, v)
        result.exhausted_steps.extend(exhausted)

        if sequence_verifier is None or cfg.max_regenerations == 0:
            result.stable = (sequence_verifier is None
                             or sequence_verifier(steps) is None)
            break
        if sequence_verifier(steps) is None:
            result.stable = True
            break
        if regeneration_num == cfg.max_regenerations:
            result.stable = False
            break
        starting = rollback_to_verified_prefix(steps, sequence_verifier)

    result.steps = steps
    result.n_regenerations = regeneration_num
    return result


def _bump_n(counter: Dict[str, int], reason: str, n: int) -> None:
    counter[reason] = counter.get(reason, 0) + n


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adaptive rejection-sampling control policy with "
                    "temperature escalation and rollback regeneration "
                    "(BrickGPT models/brickgpt.py port).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="drive the policy with synthetic samplers and "
                             "verifiers; assert escalation, budget, and "
                             "rollback behavior.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    cfg = RejectionConfig(max_steps=6, max_step_rejections=10,
                          max_regenerations=5, temperature=0.6)

    # 1. A sampler that keeps re-emitting the same invalid step until the
    #    temperature has been escalated enough, then emits valid steps.
    seen_temps: List[float] = []

    def stuck_sampler(prefix, temperature, step_index):
        seen_temps.append(temperature)
        if step_index >= 3:
            return None
        if step_index == 0 and temperature < 0.63:
            return ("bad", step_index)
        return ("ok", step_index)

    def verifier(step, prefix):
        return SUCCESS if step[0] == "ok" else "out_of_bounds"

    r = adaptive_rejection_generate(stuck_sampler, verifier, None, cfg)
    assert [s[0] for s in r.steps] == ["ok", "ok", "ok"], r.to_dict()
    assert r.rejection_reasons.get("out_of_bounds") == 1
    assert r.rejection_reasons.get(ALREADY_REJECTED, 0) >= 2
    assert max(seen_temps) > 0.6 and max(seen_temps) <= cfg.max_temperature
    assert not r.exhausted_steps
    print(f"[selfcheck] temperature escalated to {max(seen_temps):.2f}, "
          f"reasons={r.rejection_reasons}")

    # 2. Budget exhaustion is flagged, not fatal.
    def hopeless_sampler(prefix, temperature, step_index):
        return None if step_index >= 1 else ("bad", step_index, temperature)

    def reject_all(step, prefix):
        return "collision"

    r = adaptive_rejection_generate(hopeless_sampler, reject_all, None,
                                    RejectionConfig(max_steps=2,
                                                    max_step_rejections=3,
                                                    max_regenerations=0))
    assert r.exhausted_steps == [0] and len(r.steps) == 1, r.to_dict()
    print(f"[selfcheck] budget exhaustion flagged: {r.exhausted_steps}")

    # 3. Rollback: sequence verifier calls index 2 unstable on the first full
    #    build; the regenerated suffix must differ and pass.
    attempt = {"n": 0}

    def phased_sampler(prefix, temperature, step_index):
        if step_index >= 4:
            return None
        # First full pass emits "wobbly" at index 2; after rollback the
        # regeneration emits "firm".
        if step_index == 2 and attempt["n"] == 0:
            attempt["n"] = 1
            return ("wobbly", step_index)
        return ("firm", step_index)

    def always_ok(step, prefix):
        return SUCCESS

    def seq_verifier(steps):
        for i, s in enumerate(steps):
            if s[0] == "wobbly":
                return i
        return None

    r = adaptive_rejection_generate(phased_sampler, always_ok, seq_verifier, cfg)
    assert r.stable and r.n_regenerations == 1, r.to_dict()
    assert all(s[0] == "firm" for s in r.steps), r.steps
    print(f"[selfcheck] rollback regenerated from verified prefix "
          f"(n_regenerations={r.n_regenerations})")

    # 4. rollback_to_verified_prefix truncates to fixed point.
    seq = [("firm", 0), ("wobbly", 1), ("firm", 2)]
    assert rollback_to_verified_prefix(seq, seq_verifier) == [("firm", 0)]
    print("[selfcheck] prefix truncation reaches a verified fixed point")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
