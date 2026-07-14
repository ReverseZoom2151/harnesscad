"""The referee. It never asks the verifier fleet anything.

That sentence is the whole design. The fleet is the system under test. A grader
that built an op stream at ``verify_level="full"`` and treated a clean fleet as
part of "solved" would be scoring a model on its ability to please a rule -- and
when the rule is wrong (``preflight-RADIUS_TOO_LARGE`` fired at r = 3.1 and stayed
silent at r = 3.0, the true degenerate limit) the model is scored on its ability
to please a BUG. That is precisely how the pressure corpus came to reward obeying
the harness. So: build at ``verify_level="core"``, measure the SOLID, compare
against arithmetic, and let the fleet's opinion be measured somewhere else
(``eval/redteam``), by something that is not also the scoreboard.

FOUR FAMILIES, SCORED AND REPORTED SEPARATELY
---------------------------------------------
``bbox``    the envelope. MANDATORY on every brief (see ``spec.py``). This is the
            check the pressure corpus did not have, and its absence is why a shell
            that dilated a 60x40x20 box into 63x43x23 passed.
``volume``  the exact closed form, within the MEASURING ENGINE's own physical
            tolerance (``selftest.probe.tolerance``: machine precision for a B-rep
            kernel, a polygonisation error for a mesher, a grid term for a sampled
            field). Not a hand-picked band -- a band is a knob, and a knob is how a
            corpus gets tuned until it agrees with the code it is judging.
``genus``   topology. N through holes give genus N. A part with the right volume,
            the right box and no holes at all fails here and nowhere else.
``probes``  points, in world coordinates, derived from the part's own dimensions:
            the mid-plane of a wall, the axis of a bore. They pin a feature to a
            LOCATION. Evaluated against the exact signed-distance field, so they
            carry no meshing error.

and then, separately again:

``shape``   volumetric IoU against the brief's reference solid (``shape.py``).
            The four families above are all ENVELOPE families and are many-to-one:
            they can all pass on a part whose holes are in the wrong places. IoU
            is the only metric here that integrates over the part. It is reported
            ALONGSIDE the envelope verdict. ``solved`` is the envelope verdict;
            ``solved_shape`` is the conjunction. Neither replaces the other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from harnesscad.core.cisp.ops import Op
from harnesscad.eval.corpus import shape as shape_mod
from harnesscad.eval.corpus.spec import Brief
from harnesscad.eval.selftest.probe import Observation, observe, tolerance

__all__ = ["Score", "grade", "grade_reference", "resolvable", "CELLS_PER_FEATURE"]

#: How many grid cells a SAMPLED engine needs across the thinnest feature of a
#: part before its measurement of that part means anything. Two: below Nyquist a
#: sampled field cannot represent the feature at all, and the engine does not
#: refuse -- it quietly builds a smaller, different part (the exact failure
#: ``properties.shell_does_not_shrink`` was written to catch).
#:
#: This is a PRE-REGISTERED constant, not a tuned one. It is the sampling theorem,
#: and it was chosen before any brief was scored. It exists so that "this engine
#: cannot measure this part" is a THIRD ANSWER, distinct from "the part is wrong":
#: charging a 2.5 mm wall in an 80 mm box (1.5 cells) to the MODEL would be
#: charging the model for the grader's physics, and loosening the tolerance until
#: it passed would hide the 32% shell bug that this corpus exists to catch.
CELLS_PER_FEATURE = 2.0


def resolvable(brief: Brief, backend: str = "frep") -> Optional[str]:
    """``None`` when the engine can measure this part; otherwise WHY it cannot.

    Only sampled engines (``Tolerance.cells > 0`` -- today, ``frep``) can fail
    this. An exact B-rep kernel has no grid and resolves everything.
    """
    tol = tolerance(backend)
    if tol.cells <= 0:
        return None
    cell = tol.cell(brief.extent)
    if brief.feature >= CELLS_PER_FEATURE * cell:
        return None
    return ("%s samples a field on %d cells across the part's largest extent "
            "(%.1f mm), so a cell is %.2f mm; the thinnest feature of this part "
            "is %.2f mm, which is %.1f cells. Below %g cells the field cannot "
            "represent the feature and the engine builds a different, smaller "
            "part without saying so. THIS PART IS NOT MEASURABLE ON THIS ENGINE "
            "-- it is not a wrong part, and it is not scored here."
            % (backend, tol.cells, brief.extent, cell, brief.feature,
               brief.feature / cell, CELLS_PER_FEATURE))


@dataclass
class Score:
    """The verdict on one op stream against one brief.

    THREE outcomes, not two. ``unmeasurable`` is the third: the engine doing the
    measuring physically cannot resolve this part, so neither "solved" nor "failed"
    is an honest thing to write down. A grader with only two outcomes has to call
    one of them, and whichever it calls is a lie.
    """

    brief: str = ""
    backend: str = "frep"
    built: bool = False
    solved: bool = False              # every ENVELOPE family passed
    solved_shape: bool = False        # ...and the SHAPE matched too
    unmeasurable: bool = False        # the ENGINE cannot resolve this part
    unmeasurable_why: str = ""
    bbox_ok: bool = False
    volume_ok: bool = False
    genus_ok: bool = True             # vacuously true when the brief states none
    probes_ok: bool = False
    reasons: List[str] = field(default_factory=list)
    measured: Dict[str, Any] = field(default_factory=dict)
    shape: Dict[str, Any] = field(default_factory=dict)

    @property
    def iou(self) -> Optional[float]:
        return self.shape.get("iou")

    @property
    def scored(self) -> bool:
        """Did this brief contribute a verdict at all?"""
        return not self.unmeasurable

    def to_dict(self) -> dict:
        d = asdict(self)
        d["iou"] = self.iou
        d["scored"] = self.scored
        return d


def _probe_tolerance(brief: Brief, backend: str) -> float:
    """How far from a surface a probe point must be to mean anything.

    Derived, not chosen: it is the measuring engine's own bbox tolerance at this
    part's size. A sampled field on a 48-cell grid across a 100 mm part cannot
    resolve a surface to better than a fraction of a 2 mm cell, so demanding more
    would charge the engine for physics rather than for a bug.
    """
    return max(tolerance(backend).bbox_tol(brief.extent), 0.05)


def grade(brief: Brief, ops: Sequence[Op], backend: str = "frep",
          with_shape: bool = True) -> Score:
    """Rebuild ``ops`` from scratch and score them against the brief's truth."""
    s = Score(brief=brief.id, backend=backend)
    why = resolvable(brief, backend)
    if why is not None:
        s.unmeasurable = True
        s.unmeasurable_why = why
        s.reasons.append(why)
        return s
    if not ops:
        s.reasons.append("no operations were produced")
        return s

    obs: Observation = observe(backend, list(ops), verify_level="core")
    if not obs.available:
        s.reasons.append("engine %r is not available here: %s"
                         % (backend, obs.skip_reason))
        return s
    if obs.error:
        s.reasons.append("the engine raised: %s" % obs.error)
        return s
    if not obs.ok:
        s.reasons.append("the engine refused the plan at %r (%s)"
                         % (obs.rejected or "?", ",".join(obs.codes) or "no code"))
        return s
    if not obs.geometric:
        s.reasons.append("the plan produced no measurable solid")
        return s

    s.built = True
    s.measured = {"volume": obs.volume, "bbox": list(obs.bbox or ()),
                  "genus": obs.genus, "watertight": obs.watertight,
                  "manifold": obs.manifold}

    tol = tolerance(backend)

    # -- bbox: the check the pressure corpus did not have ------------------- #
    btol = tol.bbox_tol(brief.extent)
    bbox_bad = []
    for axis, want, got in zip("xyz", brief.bbox, obs.bbox or (0, 0, 0)):
        if abs(got - want) > btol:
            bbox_bad.append("%s: %.3f mm, brief says %g mm (tol %.3f)"
                            % (axis, got, want, btol))
    s.bbox_ok = not bbox_bad
    for b in bbox_bad:
        s.reasons.append("bounding box wrong on " + b)

    # -- volume against the closed form ------------------------------------- #
    # NOTE: the second argument is the THINNEST FEATURE, not the smallest bbox
    # extent. ``probe.Tolerance.volume_tol`` documents its own parameter as "a
    # sampled backend loses a fraction of a part whose THINNEST FEATURE is only a
    # few cells thick", and ``golden.check_part`` passes ``min(part.bbox)`` --
    # which is the same number for a solid plate and ten times too large for a
    # shelled box, where the thin thing is the WALL and the wall is nowhere in the
    # bounding box. That is a real bug in golden.py/probe.py; it is REPORTED, not
    # fixed here (selftest/ belongs to another agent).
    vtol = tol.volume_tol(brief.extent, brief.feature)
    rel = abs((obs.volume or 0.0) - brief.volume) / max(brief.volume, 1e-9)
    s.volume_ok = rel <= vtol
    if not s.volume_ok:
        s.reasons.append(
            "volume %.1f mm3, the closed form says %.1f mm3 (off by %+.1f%%, "
            "tolerance %.1f%%)"
            % (obs.volume or 0.0, brief.volume,
               100.0 * ((obs.volume or 0.0) - brief.volume) / brief.volume,
               100.0 * vtol))

    # -- genus --------------------------------------------------------------- #
    if brief.genus is not None and obs.genus is not None:
        s.genus_ok = obs.genus == brief.genus
        if not s.genus_ok:
            s.reasons.append("genus %d, the brief's topology says %d (the right "
                             "size with the wrong holes)" % (obs.genus, brief.genus))

    if obs.watertight is False or obs.manifold is False:
        s.reasons.append("the solid is not closed (watertight=%s manifold=%s)"
                         % (obs.watertight, obs.manifold))

    # -- probes: the only envelope check that pins a LOCATION ---------------- #
    s.probes_ok = _probe(brief, ops, backend, s)

    s.solved = bool(s.built and s.bbox_ok and s.volume_ok and s.genus_ok
                    and s.probes_ok and not s.reasons)

    # -- shape: reported ALONGSIDE, never instead of ------------------------- #
    if with_shape:
        sc = shape_mod.iou_of_ops(list(ops), list(brief.reference), backend=backend)
        s.shape = sc.to_dict()
        s.solved_shape = bool(s.solved and sc.matched)
    return s


def _probe(brief: Brief, ops: Sequence[Op], backend: str, s: Score) -> bool:
    """Evaluate the brief's probe points on the built solid's exact SDF."""
    if not brief.inside and not brief.outside:
        return True
    solid = shape_mod.build(list(ops), backend)
    if solid is None:
        s.reasons.append("could not rebuild the solid to probe it")
        return False
    f = getattr(solid, "field", None)
    f = f() if callable(f) else None
    if f is None:
        # No SDF on this engine (a B-rep kernel has none). Not a failure of the
        # part: the check simply cannot be made here, and saying so is honest.
        return True
    tol = _probe_tolerance(brief, backend)
    ok = True
    for p in brief.inside:
        d = f(p)
        if d > -tol:
            ok = False
            s.reasons.append(
                "point %s must be SOLID MATERIAL (it is the middle of a wall, or "
                "the body of the part) but the surface is %+.2f mm away; positive "
                "means the point is outside the part" % (list(p), d))
    for p in brief.outside:
        d = f(p)
        if d < tol:
            ok = False
            s.reasons.append(
                "point %s must be EMPTY (it is on a hole's axis, or inside a "
                "cavity) but it is %+.2f mm from the surface; negative means the "
                "point is buried in material -- the feature is not there"
                % (list(p), d))
    return ok


def grade_reference(brief: Brief, backend: str = "frep") -> Score:
    """Grade a brief against ITS OWN reference stream.

    A corpus whose reference solution does not pass its own grader is a corpus
    that is measuring the engine's bugs and calling them the model's. That is not
    hypothetical: with the F-rep shell fixed to hollow inward,
    ``grade(shell_box_3mm, shell_box_3mm.reference)`` returns solved=False in the
    pressure corpus, because its probe sits on the outer face. Every brief here is
    run through this in the test suite, so the same failure cannot ship twice.
    """
    return grade(brief, list(brief.reference), backend=backend)
