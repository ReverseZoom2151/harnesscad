"""Solver-in-the-loop dataset pipeline (docs/blueprint.md sec.21 data engine).

The generators (datagen/generators.py) manufacture candidate (NL brief -> CISP ops)
pairs cheaply. This module is the *ground-truth* half: every candidate is run through
a real :class:`~loop.HarnessSession` — the applyOps -> regen -> verify -> checkpoint
spine — and **only the parts that build ok are kept**. The verifier (the same plural
verifier the agent is judged by) is the label: a kept sample is verified ground truth,
tagged with its deterministic state digest and backend summary.

This is the blueprint's precedent (a topology-opt team trained on 122k MIT designs with
RL + custom verifiers): the solver, not a human, produces the labels. The cheap human
layer is :func:`verifiers_as_labor` — it decomposes a sample into *binary, non-expert*
check questions ("is this wall human-scale?") so labelling is decomposed into work a
verifier or a non-expert crowd can do (the "verifiers-as-cheap-labor" / Scale-AI play).

CAVEAT (stated up front, per the blueprint): synthetic data has a **synthetic-vs-real
distribution gap**. A 100% build-yield on the stub backend measures *validity*, not
*realism* — these parametric plates/brackets are not drawn from the true distribution of
engineered parts. CADBench must cover real-part distributions, and the flywheel metric
that actually matters ("human corrections per plan") comes from real sessions, not this
bootstrap. Treat this generator as a cold-start seed, not the target distribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from harnesscad.core.cisp.ops import Op, parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.data.datagen.generators import (
    DEFAULT_GENERATORS, Generator, ParametricSampler,
)

# A backend_factory returns a fresh GeometryBackend per sample (isolated state).
BackendFactory = Callable[[], object]
# A session_factory wraps a backend into a HarnessSession (override for tracing).
SessionFactory = Callable[[object], HarnessSession]


@dataclass
class Sample:
    """One verified (NL brief -> ops) training/eval example.

    ``ops`` is stored as a list of dicts (each an ``Op.to_dict()``) so a Sample is
    JSON-serialisable; :meth:`reference_ops` rehydrates real ``cisp.ops.Op`` objects.
    ``digest``/``summary`` are the solver-in-the-loop ground-truth tags: the digest is
    the deterministic model hash (identical replay -> identical digest).
    """

    brief: str
    generator: str
    params: dict
    ops: List[dict]
    digest: str
    summary: dict = field(default_factory=dict)

    def reference_ops(self) -> List[Op]:
        return [parse_op(d) for d in self.ops]

    def to_dict(self) -> dict:
        return {
            "brief": self.brief,
            "generator": self.generator,
            "params": self.params,
            "ops": self.ops,
            "digest": self.digest,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            brief=d["brief"],
            generator=d["generator"],
            params=dict(d.get("params", {})),
            ops=list(d.get("ops", [])),
            digest=d["digest"],
            summary=dict(d.get("summary", {})),
        )


@dataclass
class DatasetReport:
    """Result of a generation run, carrying the yield (kept/total).

    ``yield_rate`` is the fraction of candidates that survived the solver-in-the-loop
    filter — the headline health metric of the data engine on a given backend.
    """

    total: int
    kept: int
    yield_rate: float
    samples: List[Sample] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "kept": self.kept,
            "yield_rate": self.yield_rate,
            "samples": [s.to_dict() for s in self.samples],
        }


def generate_dataset_report(
    n: int,
    seed: int,
    backend_factory: BackendFactory,
    session_factory: Optional[SessionFactory] = None,
    generators: Optional[List[Generator]] = None,
) -> DatasetReport:
    """Generate ``n`` candidate parts, verify each in the loop, and report the yield.

    Cycles the generator mix round-robin off a single seeded
    :class:`ParametricSampler`, so the whole run is reproducible from ``seed``.
    """
    gens = generators if generators is not None else DEFAULT_GENERATORS
    if not gens:
        raise ValueError("no generators provided")
    sampler = ParametricSampler(seed)
    samples: List[Sample] = []
    total = 0
    for i in range(n):
        gen = gens[i % len(gens)]
        brief, ops, params = gen(sampler)
        total += 1
        backend = backend_factory()
        session = session_factory(backend) if session_factory else HarnessSession(backend)
        result = session.apply_ops(ops)
        if not result.ok:
            continue  # solver-in-the-loop: drop parts that don't build (no ground truth)
        samples.append(Sample(
            brief=brief,
            generator=params.get("generator", getattr(gen, "__name__", "?")),
            params=params,
            ops=[op.to_dict() for op in ops],
            digest=result.digest,
            summary=session.summary(),
        ))
    kept = len(samples)
    yield_rate = (kept / total) if total else 0.0
    return DatasetReport(total=total, kept=kept, yield_rate=yield_rate,
                         samples=samples)


def generate_dataset(
    n: int,
    seed: int,
    backend_factory: BackendFactory,
    session_factory: Optional[SessionFactory] = None,
    generators: Optional[List[Generator]] = None,
) -> List[Sample]:
    """Generate a verified dataset (the kept samples only).

    Thin wrapper over :func:`generate_dataset_report`; use the report form when you
    also want the yield (kept/total). Keeps ONLY samples that build ok — every
    returned Sample is verified ground truth.
    """
    return generate_dataset_report(
        n, seed, backend_factory, session_factory, generators).samples


# --- persistence -----------------------------------------------------------
def to_jsonl(path: str, samples: List[Sample]) -> None:
    """Write samples as JSON Lines (one Sample.to_dict() per line)."""
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s.to_dict(), sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[Sample]:
    """Read a JSONL file back into Samples (the inverse of :func:`to_jsonl`)."""
    out: List[Sample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Sample.from_dict(json.loads(line)))
    return out


# --- verifiers-as-cheap-labor ---------------------------------------------
def verifiers_as_labor(sample: Sample) -> List[dict]:
    """Decompose a sample into binary, non-expert check questions.

    Per the blueprint's "verifiers-as-cheap-labor": expert labelling is broken into
    simple yes/no checks a non-expert (or a cheap verifier) can answer, e.g. "is this
    wall human-scale?". Each returned item is ``{"question": str, "answer": bool}`` with
    the answer computed deterministically from the sample's recorded params + build
    result, so the same decomposition can be shown to a human OR auto-graded.
    """
    p = sample.params
    questions: List[dict] = []

    def ask(question: str, condition) -> None:
        questions.append({"question": question, "answer": bool(condition)})

    w = p.get("w")
    h = p.get("h")
    t = p.get("thickness")
    hole_r = p.get("hole_r")
    holes = p.get("holes")

    if t is not None:
        ask(f"Is the {t} mm wall/plate thickness human-scale (0.5-100 mm)?",
            0.5 <= t <= 100.0)
    if w is not None and h is not None:
        ask(f"Does the {w} x {h} mm footprint fit on a benchtop (<= 1000 mm/side)?",
            w <= 1000.0 and h <= 1000.0)
        ask("Are all outer dimensions strictly positive?",
            w > 0 and h > 0 and (t is None or t > 0))
        longest, shortest = max(w, h), min(w, h)
        ask("Is the plan aspect ratio moderate (longest <= 20x shortest)?",
            shortest > 0 and longest <= 20.0 * shortest)
    if hole_r is not None and w is not None and h is not None:
        ask(f"Does a hole of radius {hole_r} mm fit inside the plate?",
            2.0 * hole_r < min(w, h))
    if holes and hole_r is not None and w is not None and h is not None:
        inside = all(
            (hh["cx"] - hole_r) >= 0 and (hh["cx"] + hole_r) <= w and
            (hh["cy"] - hole_r) >= 0 and (hh["cy"] + hole_r) <= h
            for hh in holes)
        ask("Does every hole stay fully within the plate material?", inside)

    # Always at least one question: the ground-truth build check.
    ask("Did the part build and verify without kernel errors (ground truth)?",
        bool(sample.digest) and sample.summary.get("solid_present", False))
    return questions
