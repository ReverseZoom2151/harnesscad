"""Self-training: turn the oracle's verdicts into a training corpus.

The audit's gap #9: HarnessCAD owns the scarcest asset in applied RL -- a
deterministic, human-label-free correctness oracle -- generates ~220 graded
trajectories per pressure run, and throws every one of them away.

This package keeps them. It does NOT train. It produces:

``ledger``      the ORACLE CAPABILITY LEDGER. What each instrument can and
                cannot see, stated in code, machine-readable, and asserted by a
                test. **Read this first.** A model trained against a blind
                oracle learns to exploit the blindness; the blindness is
                therefore enumerated before a single training pair is emitted.
``trajectory``  the versioned on-disk format for a graded rollout, plus the
                capture interface the loop needs to call.
``divergence``  the per-step (process) reward and the FIRST-DIVERGENCE detector.
                Computed, not guessed: the geometry after every op is fully
                determined, so the step at which a plan stopped being repairable
                is a fact, not an LLM's opinion.
``rft``         rejection-sampling fine-tuning: keep only oracle-certified
                candidates, emit SFT records.
``preference``  DPO / KTO records with GROUND-TRUTH labels, adjudicated by the
                oracle and NEVER by the verifier fleet.
``recipe``      the training recipe and its cost. Stated, not run.
``corpus``      the driver: read a pressure results.json, emit the datasets.

Stdlib only, deterministic, no wall clock.
"""

from __future__ import annotations

SCHEMA_VERSION = "selftrain/1"

__all__ = ["SCHEMA_VERSION"]
