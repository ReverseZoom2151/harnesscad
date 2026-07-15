"""The selector for the Best-of-N arm: differential oracle + output gate.

*The Hitchhiker's Guide to Agentic AI*, section 8.5.4, in a callout box:

    **Always compare your RL method against Best-of-N with the same compute
    budget.**

v1 never ran that arm. It compared typed diagnostics against blind resampling
with NO selection -- the weakest baseline available -- while carrying, unused,
the strongest selector in the repository.

This module is that selector. It scores one candidate op stream with two
reference-free instruments:

``the differential oracle`` (:mod:`harnesscad.eval.selftest.differential`)
    the same plan on six independently-implemented engines (F-rep, CadQuery,
    FreeCAD, OpenSCAD, Blender, plus the stub). Where they disagree, at least one
    is wrong -- and you did not need to know the right answer to find that out.
    An engine that CRASHES is a finding. An engine that refuses is a capability
    gap and is not held against the candidate.

``the output gate`` (:mod:`harnesscad.io.gate`)
    closed, 2-manifold, consistently wound, positive volume, outward normals --
    and the DECLARED intent checks: a shell did not grow the envelope and left a
    wall of the declared thickness, a cut did not add volume, the first extrude
    produced a part of the declared height.

WHAT THIS ORACLE IS NOT
-----------------------
It is **not** a perfect reward model for the brief, and the audit that demanded
this arm overstates it when it calls it "exact correctness". Both instruments are
REFERENCE-FREE: neither has ever read the brief. They can prove that a part is
internally coherent, well-defined across six kernels, and honours the intent its
own op stream declared. They cannot know that the brief asked for four holes and
the model cut one, or that the plate should have been 60 mm and is 50.

So the book's scaling law -- "perfect reward model, p=0.3, N=10 -> 97%" -- does
not transfer. This selector's ceiling is the rate at which a *self-consistent,
gate-passing* candidate is also the *right* candidate. Where the model's failure
mode is a misread brief rather than broken geometry, this oracle is blind and
Best-of-N degrades to picking arbitrarily among N wrong answers. That is a
prediction, made before the run, and the result either bears it out or does not.

The score is a lexicographic tuple, fixed in advance, and ties break on the
lowest sample index -- so the arm is deterministic given the samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["OracleScore", "score_ops", "rank"]


def _pinned_factory(name: str):
    """Resolve an engine for the differential oracle, PINNING F-rep's mesher.

    The other five engines are exact kernels and sample no grid, so they have no
    mesher to pin. F-rep does, and if its default flips under this experiment the
    oracle's own consensus cluster moves with it. See ``session.py``.
    """
    from harnesscad.eval.pressure.session import pin
    from harnesscad.eval.selftest.probe import resolve

    backend, _skip = resolve(name)        # factory=None: the normal resolution
    if backend is not None and name == "frep":
        pin(backend)
    return backend


@dataclass
class OracleScore:
    """What the reference-free instruments say about one candidate."""

    built: bool = False
    apply_ok: bool = False
    gate_ok: bool = False
    gate_failures: List[str] = field(default_factory=list)
    engines_agreeing: int = 0
    engines_disagreeing: int = 0
    engines_crashed: int = 0
    engines_refused: int = 0
    volume_spread: Optional[float] = None
    error: str = ""

    @property
    def key(self) -> Tuple:
        """The ranking key. Bigger is better. Lexicographic, decided a priori.

        1. it builds at all
        2. the core verifiers accepted the plan
        3. the output gate passes (measured + declared intent)
        4. no engine disagrees with the consensus
        5. no engine crashed on it
        6. the largest consensus cluster
        """
        return (
            int(self.built),
            int(self.apply_ok),
            int(self.gate_ok),
            -self.engines_disagreeing,
            -self.engines_crashed,
            self.engines_agreeing,
        )

    def to_dict(self) -> dict:
        return {
            "built": self.built, "apply_ok": self.apply_ok,
            "gate_ok": self.gate_ok, "gate_failures": list(self.gate_failures),
            "engines_agreeing": self.engines_agreeing,
            "engines_disagreeing": self.engines_disagreeing,
            "engines_crashed": self.engines_crashed,
            "engines_refused": self.engines_refused,
            "volume_spread": self.volume_spread,
            "error": self.error,
            "key": list(self.key),
        }


def score_ops(ops: Sequence[dict], name: str = "candidate",
              backends: Optional[Sequence[str]] = None) -> OracleScore:
    """Score one candidate op stream. Reference-free. Never raises."""
    from harnesscad.core.cisp.ops import parse_op
    from harnesscad.eval.pressure.session import frep_server
    from harnesscad.eval.selftest import differential
    from harnesscad.io import gate

    s = OracleScore()
    if not ops:
        s.error = "no operations"
        return s

    # --- the gate, on the harness's own backend ----------------------------- #
    server = frep_server("core")          # PINNED mesher -- see session.py
    try:
        result = server.applyOps([dict(o) for o in ops])
        s.apply_ok = bool(result.get("ok"))
    except Exception as exc:                              # noqa: BLE001
        s.error = f"apply raised {type(exc).__name__}: {exc}"
        return s

    try:
        report = gate.check(server.backend, source=server)
        s.gate_ok = bool(report.ok)
        s.gate_failures = [f.check for f in report.failures]
        s.built = bool(report.measurement.get("triangle_count"))
    except Exception as exc:                              # noqa: BLE001
        s.error = f"gate raised {type(exc).__name__}: {exc}"
        return s

    # --- the six engines ---------------------------------------------------- #
    try:
        parsed = tuple(parse_op(dict(o)) for o in ops)
    except Exception as exc:                              # noqa: BLE001
        s.error = f"ops do not parse: {exc}"
        return s
    try:
        case = differential.compare(name, parsed, backends=backends,
                                    factory=_pinned_factory)
    except Exception as exc:                              # noqa: BLE001
        s.error = f"differential raised {type(exc).__name__}: {exc}"
        return s

    s.engines_agreeing = len(case.consensus)
    s.engines_disagreeing = len({d.backend for d in case.disagreements})
    s.engines_crashed = len(case.crashed)
    s.engines_refused = len(case.refused)
    s.volume_spread = case.volume_spread()
    return s


def rank(candidates: Sequence[Sequence[dict]],
         name: str = "candidate",
         backends: Optional[Sequence[str]] = None
         ) -> Tuple[int, List[OracleScore]]:
    """Score every candidate; return (index of the best, all scores).

    Ties break on the LOWEST index, so the arm degrades gracefully to "take the
    first sample" when the oracle cannot separate the candidates -- which is the
    honest behaviour, and it is what will happen whenever every candidate builds
    cleanly and only the brief could tell them apart.
    """
    scores = [score_ops(c, name=f"{name}#{i}", backends=backends)
              for i, c in enumerate(candidates)]
    best = 0
    for i in range(1, len(scores)):
        if scores[i].key > scores[best].key:
            best = i
    return best, scores
