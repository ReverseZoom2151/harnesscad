"""Deterministic CAD-program crossover & mutation operators for EvoCAD.

EvoCAD (Preintner et al., 2025) performs crossover and mutation with an LLM,
prompting it to "combine two CAD codes in a complementary way" (crossover) and
to "refine and improve the code" (mutation). Those prompt-driven LM operators
are research-heavy / external. This module supplies a concrete, deterministic,
program-level realisation of the *same variation semantics* so the evolutionary
loop in ``exploration.evocad_evolution`` can be exercised and tested without any
model in the loop, and so a CAD-program mutation/recombination scheme exists in
the codebase.

A CAD program is modelled as an ordered tuple of :class:`CadOp` steps (mirroring
a CADQuery/OpenSCAD-style op sequence: sketch -> extrude -> hole -> fillet ...).
Operators:

  * ``crossover`` -- single-cut recombination: prefix of parent A + suffix of
    parent B ("mating two objects to generate an offspring", Sec. III-B).
  * ``mutate``    -- one of {parameter jitter, op duplication, op deletion,
    op insertion} chosen by the seeded RNG ("refine and improve", exploring
    "alternative novel solution strategies").

Determinism: every choice flows through the ``random.Random`` handed in by the
evolutionary loop.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

# Jitter multipliers applied to numeric parameters during mutation (fixed set
# so mutation is deterministic given the RNG).
_JITTER_FACTORS = (0.5, 0.8, 1.25, 2.0)
# Op kinds that may be inserted during mutation.
_INSERTABLE = ("hole", "fillet", "chamfer", "shell")


@dataclass(frozen=True)
class CadOp:
    """One CAD program step: an op name plus numeric parameters."""

    name: str
    params: Tuple[Tuple[str, float], ...] = ()

    @staticmethod
    def make(name: str, **params: float) -> "CadOp":
        return CadOp(name, tuple(sorted((k, float(v)) for k, v in params.items())))

    def with_params(self, params: Dict[str, float]) -> "CadOp":
        return CadOp(self.name, tuple(sorted((k, float(v)) for k, v in params.items())))

    def param_dict(self) -> Dict[str, float]:
        return dict(self.params)


@dataclass(frozen=True)
class CadProgram:
    """An ordered sequence of :class:`CadOp` steps."""

    ops: Tuple[CadOp, ...] = ()

    @staticmethod
    def of(*ops: CadOp) -> "CadProgram":
        return CadProgram(tuple(ops))

    def __len__(self) -> int:
        return len(self.ops)


def crossover(parent_a: CadProgram, parent_b: CadProgram, rng: random.Random) -> CadProgram:
    """Single-cut recombination: A[:cut_a] + B[cut_b:] (complementary mating).

    Cut points are drawn from the RNG. Empty parents degrade gracefully to the
    non-empty parent. The child is always a valid op sequence.
    """
    a, b = parent_a.ops, parent_b.ops
    if not a:
        return parent_b
    if not b:
        return parent_a
    cut_a = rng.randint(0, len(a))
    cut_b = rng.randint(0, len(b))
    child = a[:cut_a] + b[cut_b:]
    if not child:
        # Never emit an empty program; keep at least the head of A.
        child = a[:1]
    return CadProgram(child)


def _mutate_parameter(prog: CadProgram, rng: random.Random) -> CadProgram:
    if not prog.ops:
        return prog
    idx = rng.randrange(len(prog.ops))
    op = prog.ops[idx]
    params = op.param_dict()
    if not params:
        return prog
    key = rng.choice(sorted(params))
    params[key] = params[key] * rng.choice(_JITTER_FACTORS)
    ops = list(prog.ops)
    ops[idx] = op.with_params(params)
    return CadProgram(tuple(ops))


def _mutate_duplicate(prog: CadProgram, rng: random.Random) -> CadProgram:
    if not prog.ops:
        return prog
    idx = rng.randrange(len(prog.ops))
    ops = list(prog.ops)
    ops.insert(idx + 1, prog.ops[idx])
    return CadProgram(tuple(ops))


def _mutate_delete(prog: CadProgram, rng: random.Random) -> CadProgram:
    if len(prog.ops) <= 1:
        return prog
    idx = rng.randrange(len(prog.ops))
    ops = list(prog.ops)
    del ops[idx]
    return CadProgram(tuple(ops))


def _mutate_insert(prog: CadProgram, rng: random.Random) -> CadProgram:
    name = rng.choice(_INSERTABLE)
    new_op = CadOp.make(name, size=float(rng.randint(1, 9)))
    idx = rng.randint(0, len(prog.ops))
    ops = list(prog.ops)
    ops.insert(idx, new_op)
    return CadProgram(tuple(ops))


_MUTATORS = (_mutate_parameter, _mutate_duplicate, _mutate_delete, _mutate_insert)


def mutate(prog: CadProgram, rng: random.Random) -> CadProgram:
    """Apply one randomly-chosen mutation operator (refine/improve the code)."""
    return rng.choice(_MUTATORS)(prog, rng)


def program_signature(prog: CadProgram) -> Tuple:
    """A hashable canonical signature (op names + params) for de-duplication."""
    return tuple((op.name, op.params) for op in prog.ops)
