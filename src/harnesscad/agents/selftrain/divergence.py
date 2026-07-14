"""The process reward, and the FIRST-DIVERGENCE detector -- computed, not guessed.

The book, H2 sec. 12.8.3 ("Multi-Turn Trajectory Slicing"):

    "If an agent executes 20 tool actions and finally fails a unit test, a
    terminal reward of 0 punishes all 20 actions equally. ... 1. Replay the
    successful prefix (steps 1-k) 2. Identify the first divergence point
    3. Assign negative reward only to that specific step 4. Assign
    neutral/positive rewards to correct prefix steps."

Every other system that wants this has to ESTIMATE it. Math-Shepherd samples M
completions from each prefix and calls step k correct if any completion reaches
the right answer -- a Monte-Carlo estimate. Fara has an LLM *guess* which step
broke. Neither is necessary here, and both would be worse.

**After every CAD op, the geometry is fully determined.** There is nothing to
sample and nothing to guess. The state at prefix k is a solid, we can measure it,
and we can compare it against the brief's reference solid exactly. So the
divergence point is a fact.

WHAT MAKES A STEP DIVERGENT
===========================
Three tests, in order. The first that fires wins, and it fires at the smallest k.

**1. The kernel refused the op.** The op stream stopped being executable at k.
   Hard, exact, no measurement.

**2. A step-local monotone law was broken.** These are ``eval/selftest/properties``'
   laws, applied per-op instead of per-stream: a ``cut``/``hole`` must not add
   volume; a ``shell`` must not grow the bounding box (the bug that shipped); a
   ``union`` must not remove volume. A law violation is the KERNEL's divergence,
   not the model's -- it is recorded with ``blame="kernel"`` and it does not
   condemn the model's op. This distinction matters: training a model to avoid
   ops that trip a backend bug is teaching it our bugs.

**3. THE MATERIAL IS UNRECOVERABLE.** This is the one that does the work, and it
   is exact because CISP ops are monotone in material:

       Let ``R`` be the brief's reference solid and ``C_k`` the candidate's solid
       after op k. Write ``missing = |R \\ C_k|`` (material the answer needs that
       the candidate does not have) and ``excess = |C_k \\ R|``.

       If ``missing > tol`` and NO op after k can ADD material, that material can
       never appear: the plan is already wrong at k.
       If ``excess  > tol`` and NO op after k can REMOVE material, that material
       can never leave: the plan is already wrong at k.

   Both quantities are integrals of the two exact signed-distance fields over the
   union of their bounding boxes, estimated by a seeded quasi-uniform sample. The
   ADD/REMOVE classification of an op is a static property of the op tag
   (:data:`ADDITIVE`, :data:`SUBTRACTIVE`) and is therefore free.

   Note what this does NOT do: it does not ask whether a *better* suffix existed.
   It asks whether ANY suffix drawn from the ops the model actually emitted could
   have recovered. That is a strictly weaker and strictly sound claim -- if it
   says "divergent at k", the plan as written was doomed at k. A prefix it calls
   sound may still be a prefix of a doomed plan; that is the correct asymmetry for
   a *negative* reward signal, which is what the book asks for.

VALIDATED AGAINST THE KNOWN REGRESSION
======================================
The 14b's ``trap_hole_oversize`` regression. Attempt 1 is correct (40x40x10 plate,
d=12 through hole). The fleet says ``infeasible-plan: hole diameter 12 mm >= plate
wall 10 mm`` -- a false statement, the bug that cost 8 briefs -- and the 14b, which
follows instructions flawlessly, changes exactly one field: ``diameter 12 -> 8``.

Ops 0, 1, 2 (``new_sketch``, ``add_rectangle``, ``extrude``) are identical to the
correct attempt. Op **3** is the hole, and it is the only thing that changed.

The detector must say 3. Ops 0-2 leave a solid plate: it has EXCESS material
(the un-bored hole) but a subtractive op remains, so it is recoverable and they
are not divergent. Op 3 bores 8 mm where 12 mm was required: the 4 mm annulus is
excess, nothing subtractive remains, and it is therefore unrecoverable AT op 3.
``tests/agents/selftrain/test_divergence.py`` asserts exactly this, on the real
op streams lifted from ``assets/pressure/results.json``.

For scale: that annulus is 628 mm^3 of a ~15,000 mm^3 part. The SHAPE metric scores
it **IoU 0.963** and calls it a match. The divergence detector catches what IoU
cannot, because it measures the defect against what could still be *undone*, not
against the size of the part.

Deterministic. Seeded. No wall clock.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.agents.selftrain.trajectory import StepReward

__all__ = [
    "ADDITIVE",
    "SUBTRACTIVE",
    "NEUTRAL",
    "StepAnalysis",
    "DivergenceReport",
    "analyse",
    "step_rewards",
]

Vec3 = Tuple[float, float, float]

#: Ops that can ADD material to the solid.
ADDITIVE = frozenset({"extrude", "revolve", "loft", "sweep", "linear_pattern",
                      "circular_pattern", "mirror", "add_instance"})

#: Ops that can REMOVE material from the solid.
SUBTRACTIVE = frozenset({"hole", "shell", "fillet", "chamfer", "draft"})

#: Ops that touch no solid at all (sketch-domain, parameters, assembly metadata).
NEUTRAL = frozenset({"new_sketch", "add_point", "add_line", "add_circle",
                     "add_rectangle", "constrain", "set_param", "mate"})

#: ``boolean`` is both, depending on ``kind`` -- resolved by :func:`_polarity`.

#: Sample count for the set-difference integrals. Lower than shape.SAMPLES
#: (20,000) because this runs once per PREFIX rather than once per stream; at
#: 8,000 the standard error on a volume fraction near 0.05 is 0.0024, an order of
#: magnitude below TOL.
SAMPLES = 8000

#: The seed. Fixed, so a divergence index is a function of the geometry alone.
SEED = 20260714

#: Padding on the sampling box, mm.
PAD = 1.0

#: A set-difference is *material* when it exceeds this fraction of the reference
#: solid's volume. 2% is ~8x the sampling standard error and is below the 4.2%
#: defect of the 14b regression this detector exists to catch. Fixed a priori.
TOL = 0.02


def _polarity(op: Dict[str, Any]) -> str:
    """"add" | "remove" | "both" | "none" for one op dict."""
    tag = str(op.get("op") or "")
    if tag == "boolean":
        kind = str(op.get("kind") or "union")
        if kind == "union":
            return "add"
        return "remove"          # cut, intersect
    if tag in ADDITIVE:
        return "add"
    if tag in SUBTRACTIVE:
        return "remove"
    if tag in NEUTRAL:
        return "none"
    return "both"                # unknown op: assume it can do anything (sound)


def _can_add(ops: Sequence[dict]) -> bool:
    return any(_polarity(o) in ("add", "both") for o in ops)


def _can_remove(ops: Sequence[dict]) -> bool:
    return any(_polarity(o) in ("remove", "both") for o in ops)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _field(backend: Any):
    fn = getattr(backend, "field", None)
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception:                                     # noqa: BLE001
        return None


def _bounds(backend: Any) -> Optional[Tuple[Vec3, Vec3]]:
    mesh = getattr(backend, "mesh", None)
    if not callable(mesh):
        return None
    try:
        verts, faces = mesh()
    except Exception:                                     # noqa: BLE001
        return None
    if not verts or not faces:
        return None
    lo = tuple(min(float(v[i]) for v in verts) for i in range(3))
    hi = tuple(max(float(v[i]) for v in verts) for i in range(3))
    return lo, hi


def _volume(backend: Any) -> Optional[float]:
    try:
        m = backend.query("measure")
    except Exception:                                     # noqa: BLE001
        return None
    v = m.get("volume")
    return float(v) if v is not None else None


def _bbox(backend: Any) -> Optional[Vec3]:
    try:
        m = backend.query("measure")
    except Exception:                                     # noqa: BLE001
        return None
    b = m.get("bbox")
    return tuple(float(x) for x in b) if b else None


@dataclass
class _SetDiff:
    """|R \\ C| and |C \\ R| as fractions of |R|, plus whether R was measurable."""

    ok: bool = False
    missing: float = 0.0      # of the reference's volume, absent from candidate
    excess: float = 0.0       # of the reference's volume, present and unwanted
    reason: str = ""


def _set_difference(candidate: Any, reference: Any,
                    samples: int = SAMPLES, seed: int = SEED) -> _SetDiff:
    """Monte-Carlo |R\\C| and |C\\R|, normalised by |R|. Deterministic."""
    fc, fr = _field(candidate), _field(reference)
    if fr is None:
        return _SetDiff(reason="the reference exposes no signed-distance field")
    br = _bounds(reference)
    if br is None:
        return _SetDiff(reason="the reference produced no solid")

    if fc is None:
        # No candidate solid at all: everything the reference needs is missing.
        return _SetDiff(ok=True, missing=1.0, excess=0.0,
                        reason="the candidate has no solid yet")
    bc = _bounds(candidate)
    if bc is None:
        return _SetDiff(ok=True, missing=1.0, excess=0.0,
                        reason="the candidate has no solid yet")

    lo = tuple(min(bc[0][i], br[0][i]) - PAD for i in range(3))
    hi = tuple(max(bc[1][i], br[1][i]) + PAD for i in range(3))
    if any(hi[i] <= lo[i] for i in range(3)):
        return _SetDiff(reason="degenerate sampling box")

    rnd = random.Random(seed)
    n_ref = n_missing = n_excess = 0
    for _ in range(samples):
        p = (rnd.uniform(lo[0], hi[0]),
             rnd.uniform(lo[1], hi[1]),
             rnd.uniform(lo[2], hi[2]))
        try:
            in_c = fc(p) <= 0.0
            in_r = fr(p) <= 0.0
        except Exception:                                 # noqa: BLE001
            continue
        if in_r:
            n_ref += 1
            if not in_c:
                n_missing += 1
        elif in_c:
            n_excess += 1
    if n_ref == 0:
        return _SetDiff(reason="the reference does not occupy the sampled box")
    return _SetDiff(ok=True,
                    missing=n_missing / n_ref,
                    excess=n_excess / n_ref,
                    reason="")


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #
@dataclass
class StepAnalysis:
    """What happened at one op."""

    index: int
    op: str
    applied: bool = False
    missing: Optional[float] = None
    excess: Optional[float] = None
    can_add_later: bool = False
    can_remove_later: bool = False
    law_violation: str = ""
    blame: str = ""            # "" | "model" | "kernel"
    divergent: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index, "op": self.op, "applied": self.applied,
            "missing": self.missing, "excess": self.excess,
            "can_add_later": self.can_add_later,
            "can_remove_later": self.can_remove_later,
            "law_violation": self.law_violation, "blame": self.blame,
            "divergent": self.divergent, "detail": self.detail,
        }


@dataclass
class DivergenceReport:
    """The per-op verdict on one op stream."""

    ok: bool = False
    steps: List[StepAnalysis] = field(default_factory=list)
    first_divergence: Optional[int] = None      # 0-based op index
    blame: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok,
                "steps": [s.to_dict() for s in self.steps],
                "first_divergence": self.first_divergence,
                "blame": self.blame, "detail": self.detail}


def _law_violation(op: Dict[str, Any],
                   before: Tuple[Optional[float], Optional[Vec3]],
                   after: Tuple[Optional[float], Optional[Vec3]]) -> str:
    """Step-local monotone laws. A violation here is the KERNEL's, not the model's."""
    v0, b0 = before
    v1, b1 = after
    pol = _polarity(op)
    tag = str(op.get("op") or "")
    if v0 is None or v1 is None:
        return ""
    eps = max(1e-6, 0.01 * abs(v0))
    if pol == "remove" and v1 > v0 + eps:
        return "'%s' is subtractive but volume rose %.0f -> %.0f" % (tag, v0, v1)
    if pol == "add" and v1 < v0 - eps:
        return "'%s' is additive but volume fell %.0f -> %.0f" % (tag, v0, v1)
    if tag == "shell" and b0 and b1:
        grew = [i for i in range(3) if b1[i] > b0[i] + 1e-3]
        if grew:
            return ("'shell' GREW the bounding box on axis %s (%s -> %s): this is "
                    "the dilation bug, and it is the kernel's fault, not the "
                    "model's" % ("".join("xyz"[i] for i in grew),
                                 [round(x, 2) for x in b0],
                                 [round(x, 2) for x in b1]))
    return ""


def analyse(brief: Any, ops: Sequence[dict], *,
            samples: int = SAMPLES, seed: int = SEED,
            tol: float = TOL) -> DivergenceReport:
    """Replay ``ops`` one at a time and find the first op that doomed the plan."""
    from harnesscad.eval.pressure import shape as shape_mod
    from harnesscad.io.surfaces.server import CISPServer

    report = DivergenceReport()
    ops = [dict(o) for o in ops]
    if not ops:
        report.detail = "no operations"
        return report

    reference = shape_mod.reference_backend(brief)
    if reference is None:
        report.detail = "the brief's reference stream does not build"
        return report

    server = CISPServer(backend="frep", verify_level="core")
    prev_v: Optional[float] = None
    prev_b: Optional[Vec3] = None

    for k, op in enumerate(ops):
        step = StepAnalysis(index=k, op=str(op.get("op") or "?"))
        rest = ops[k + 1:]
        step.can_add_later = _can_add(rest)
        step.can_remove_later = _can_remove(rest)

        # --- 1. did the kernel take it? ------------------------------------ #
        try:
            result = server.applyOps([dict(op)])
            step.applied = bool(result.get("ok"))
            if not step.applied:
                step.divergent = True
                step.blame = "model"
                step.detail = ("the kernel refused this op after %s applied"
                               % result.get("applied"))
        except Exception as exc:                          # noqa: BLE001
            step.applied = False
            step.divergent = True
            step.blame = "model"
            step.detail = "the op raised %s: %s" % (type(exc).__name__, exc)

        if step.divergent:
            report.steps.append(step)
            break

        v, b = _volume(server.backend), _bbox(server.backend)

        # --- 2. step-local monotone laws ----------------------------------- #
        violation = _law_violation(op, (prev_v, prev_b), (v, b))
        if violation:
            step.law_violation = violation
            step.blame = "kernel"
            step.detail = violation
            # A kernel bug is NOT the model's divergence. Record it, do not
            # condemn the op. Training a model to avoid our bugs is a way of
            # shipping our bugs into the weights.

        # --- 3. is the material still recoverable? ------------------------- #
        diff = _set_difference(server.backend, reference,
                               samples=samples, seed=seed)
        if diff.ok:
            step.missing = round(diff.missing, 5)
            step.excess = round(diff.excess, 5)
            unrecoverable = []
            if diff.missing > tol and not step.can_add_later:
                unrecoverable.append(
                    "%.1f%% of the required material is ABSENT and no later op "
                    "can add material" % (100.0 * diff.missing))
            if diff.excess > tol and not step.can_remove_later:
                unrecoverable.append(
                    "%.1f%% excess material is PRESENT and no later op can "
                    "remove material" % (100.0 * diff.excess))
            if unrecoverable:
                step.divergent = True
                step.blame = step.blame or "model"
                step.detail = "; ".join(unrecoverable)
        elif not step.law_violation:
            step.detail = diff.reason

        prev_v, prev_b = v, b
        report.steps.append(step)
        if step.divergent:
            break

    report.ok = True
    for s in report.steps:
        if s.divergent:
            report.first_divergence = s.index
            report.blame = s.blame
            report.detail = s.detail
            break
    if report.first_divergence is None:
        report.detail = "no op made the plan unrecoverable"
    return report


def step_rewards(report: DivergenceReport,
                 ops: Sequence[dict]) -> List[StepReward]:
    """The book's assignment, sec. 12.8.3, verbatim.

    +1 to every op in the correct prefix, **-1 to the divergent op and to it
    alone**, 0 to everything after it (the model was already lost; punishing the
    tail is the sparse-reward mistake this whole module exists to avoid).

    When nothing diverged, every op scores +1 -- including the ops of a stream the
    ENVELOPE grader failed, which happens when the plan is recoverable and simply
    was not recovered, and when the failure is invisible to the reference (a brief
    whose reference is one of several correct answers). That is a real limit and
    it is why the aggregate reward keeps the outcome term.
    """
    d = report.first_divergence
    out: List[StepReward] = []
    for k, op in enumerate(ops):
        analysis = next((s for s in report.steps if s.index == k), None)
        applied = bool(analysis.applied) if analysis else False
        if d is None:
            reward, divergent = 1.0, False
        elif k < d:
            reward, divergent = 1.0, False
        elif k == d:
            reward, divergent = -1.0, True
        else:
            reward, divergent = 0.0, False
        out.append(StepReward(
            index=k, op=str(op.get("op") or "?"), applied=applied,
            reward=reward, divergent=divergent,
            detail=(analysis.detail if analysis else "not reached"),
        ))
    return out
