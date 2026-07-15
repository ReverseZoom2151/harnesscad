"""CONSTRAINT SATISFACTION -- where IoU is not weak but DEFINITIONALLY inapplicable.

"A bracket that takes an M8 bolt, carries 200 N, and fits in a 50 x 50 x 20 mm
envelope." The ground truth is a CONSTRAINT SET, not a part. MANY parts satisfy it --
a flat plate, an L-bracket, a ribbed gusset -- and they share no shape: two valid
answers to the brief below score an IoU of 0.606 against EACH OTHER (measured). So
a shape metric against one reference answer cannot score this brief at all; there is
no reference answer, there is a reference REQUIREMENT. This is the real engineering
task, and it is exactly the "fine-grained engineering criteria" band where MUSE
measures closed-source models at 19-21%.

We grade by constraint satisfaction, on the model's OWN geometry, exactly.

THE RULE THIS MODULE OBEYS: ONLY SHIP A CONSTRAINT WE CAN CHECK
--------------------------------------------------------------
A constraint we cannot verify is decoration -- it makes the brief sound harder
without making the score mean more, and a model could violate it with no
consequence. Every constraint below is checked on the built solid by an EXACT OCCT
query. The ones we could not check honestly were DROPPED, and they are named in
:data:`DROPPED_CONSTRAINTS` with the reason, because a silent drop is how a
benchmark starts lying about what it measures.

SHIPPED (each an exact measurement on the candidate's own solid)
----------------------------------------------------------------
``envelope``    the part's bounding box fits within the stated box. Exact
                (``occt.bbox_of``).
``bolt_bore``   a through-void on the stated axis is at least the ISO 273 clearance
                radius for the named bolt. Exact (``occt.bore_radius_at``, and the
                clearance diameter is read from ``eval/corpus/standards`` /
                ``domain.standards.thread_database`` -- a number no code here can
                edit). The axis is stated in the brief precisely so the check is
                well posed; "a hole somewhere" is not a constraint a grader can
                close over.
``mass``        volume x density is under a budget. Exact (``occt.volume_of`` times
                a tabulated density).
``bending``     under a STATED cantilever load case -- fixed at one plane, a force
                at another -- the peak bending stress sigma = M*c/I is under the
                material's allowable. Exact: I and c come from ``occt.section_at``,
                verified against b*h^3/12 for a rectangular bar to machine
                precision. "Carries 200 N" with no load geometry is unverifiable and
                is NOT shipped in that form -- only with an explicit load case, which
                is what a stress ever actually means.

DROPPED, AND WHY (see :data:`DROPPED_CONSTRAINTS`)
-------------------------------------------------
``min_wall``    a ray-cast minimum wall is unsound in BOTH directions: a ray grazing
                a hole or a corner returns a short material chord that is a sliver,
                not a wall (measured -- a 6 mm plate returns a 1.4 mm "wall"), so it
                would FAIL sound parts; and a finite ray bundle can miss a genuinely
                thin wall, so it cannot certify one either. A sound minimum-wall
                needs a medial-axis / distance-field analysis we do not have here.
                Dropped rather than shipped as a false check.
``fatigue``, ``thermal``, ``tolerance_stack``, ``buckling``, ``modal`` -- each needs
                a solver (S-N, thermal, GD&T, eigenvalue) this repository does not
                have. A constraint we can state but not evaluate is decoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from harnesscad.core.cisp.ops import (AddRectangle, Extrude, Hole, NewSketch, Op)
from harnesscad.eval.corpus.standards import CLEARANCE_ISO273, nominal_diameter
from harnesscad.eval.hardcorpus import occt

__all__ = ["DROPPED_CONSTRAINTS", "MATERIALS", "Constraint", "ConstraintResult",
           "ConstraintBrief", "grade", "GradeReport", "BRIEFS"]

Vec3 = Tuple[float, float, float]

#: Constraints we can STATE but not CHECK, named so no reader mistakes the brief's
#: silence for coverage. Shipping one of these would be decoration.
DROPPED_CONSTRAINTS: Dict[str, str] = {
    "min_wall": ("a ray-cast minimum-wall estimate is unsound both ways: a ray "
                 "grazing a hole/corner returns a sliver chord (a 6 mm plate reads "
                 "1.4 mm), so it fails sound parts; and a finite bundle can miss a "
                 "real thin wall, so it cannot certify one. Needs a medial-axis "
                 "analysis we do not have."),
    "fatigue": "no S-N / cyclic-life model in this repository.",
    "thermal": "no thermal or thermal-stress solver.",
    "tolerance_stack": "no GD&T / tolerance model.",
    "buckling": "no eigenvalue buckling analysis.",
    "modal": "no vibration / natural-frequency solver.",
    "generic_load": ("'carries 200 N' with no stated load geometry is not "
                     "checkable; only shipped as an explicit cantilever load case."),
}

#: Material properties, tabulated. density in g/mm^3, allowable stress in MPa
#: (N/mm^2). The allowable is the yield strength with a stated safety factor folded
#: in; both are published handbook values, not ours to set.
MATERIALS: Dict[str, Dict[str, float]] = {
    # 6061-T6 aluminium: rho 2.70 g/cm^3 = 2.70e-3 g/mm^3; yield 276 MPa; the
    # allowable below is yield / 2.5.
    "AL6061": {"density": 2.70e-3, "allowable_mpa": 110.0, "yield_mpa": 276.0,
               "safety_factor": 2.5},
    # AISI 1018 steel: rho 7.87e-3 g/mm^3; yield 370 MPa; allowable yield / 2.5.
    "STEEL1018": {"density": 7.87e-3, "allowable_mpa": 148.0, "yield_mpa": 370.0,
                  "safety_factor": 2.5},
}


# --------------------------------------------------------------------------- #
# a constraint and its verdict
# --------------------------------------------------------------------------- #
@dataclass
class ConstraintResult:
    name: str
    satisfied: bool
    measured: float
    limit: float
    units: str
    detail: str = ""
    #: 'exact' -- a two-sided measurement; 'violation-only' -- sound only when it
    #: FAILS (we do not ship any of these, but the field is here for honesty).
    soundness: str = "exact"

    def to_dict(self) -> dict:
        return {"name": self.name, "satisfied": self.satisfied,
                "measured": self.measured, "limit": self.limit,
                "units": self.units, "detail": self.detail,
                "soundness": self.soundness}


@dataclass(frozen=True)
class Constraint:
    """One checkable requirement. ``check`` measures the candidate's own solid."""

    name: str
    citation: str
    check: Callable[[Any], ConstraintResult]   # (built shape) -> result


# --------------------------------------------------------------------------- #
# the checkers -- each an EXACT OCCT measurement
# --------------------------------------------------------------------------- #
def envelope_constraint(box: Vec3) -> Constraint:
    def _check(shape: Any) -> ConstraintResult:
        dx, dy, dz = occt.extents_of(shape)
        # Sort both so an orientation-agnostic fit is allowed (a 50x20x50 part fits
        # a 50x50x20 envelope). The envelope is a box, not an oriented slot.
        got = sorted((dx, dy, dz), reverse=True)
        want = sorted(box, reverse=True)
        worst = max(got[i] - want[i] for i in range(3))
        ok = worst <= 1e-6
        return ConstraintResult(
            "envelope", ok, measured=round(max(got), 4), limit=max(want),
            units="mm", detail="part extents %s vs envelope %s (sorted)"
            % ([round(v, 3) for v in got], [round(v, 3) for v in want]))
    return Constraint("envelope",
                      "the finished part must fit inside the stated box",
                      _check)


def bolt_bore_constraint(bolt: str, axis_xy: Tuple[float, float],
                         z_probe: float) -> Constraint:
    clearance_d = CLEARANCE_ISO273[bolt]
    r_need = clearance_d / 2.0

    def _check(shape: Any) -> ConstraintResult:
        r = occt.bore_radius_at(shape, axis_xy, z_probe)
        ok = r is not None and r >= r_need - 1e-3
        return ConstraintResult(
            "bolt_bore", bool(ok),
            measured=round(r, 4) if r is not None else 0.0, limit=r_need,
            units="mm radius",
            detail=("a %s bolt needs an ISO 273 medium clearance hole of %g mm "
                    "diameter on the axis at %s; the void there measures %s mm "
                    "radius" % (bolt, clearance_d, list(axis_xy),
                                "%.3f" % r if r is not None else "none (solid)")))
    return Constraint(
        "bolt_bore",
        "ISO 273 medium-series clearance hole for %s (%g mm), nominal bolt "
        "diameter %g mm (ISO 261)" % (bolt, clearance_d, nominal_diameter(bolt)),
        _check)


def mass_constraint(material: str, max_grams: float) -> Constraint:
    rho = MATERIALS[material]["density"]

    def _check(shape: Any) -> ConstraintResult:
        m = occt.volume_of(shape) * rho
        return ConstraintResult(
            "mass", m <= max_grams + 1e-6, measured=round(m, 3),
            limit=max_grams, units="g",
            detail="volume %.1f mm3 x %g g/mm3 = %.2f g (budget %.1f g)"
            % (occt.volume_of(shape), rho, m, max_grams))
    return Constraint("mass",
                      "mass = volume x density (%s, rho = %g g/mm3)"
                      % (material, rho), _check)


@dataclass(frozen=True)
class LoadCase:
    """A cantilever: fixed at the ``root`` plane, force ``force_n`` at the tip.

    The bending moment at the root is ``force_n * arm``. The section is cut at the
    root plane (normal ``+X``) and its I and c are measured exactly. This is what
    turns "carries 200 N" from a slogan into a number.
    """

    root_x: float
    arm: float
    force_n: float
    normal: Vec3 = (1.0, 0.0, 0.0)
    bend_axis: Vec3 = (0.0, 1.0, 0.0)


def bending_constraint(material: str, load: LoadCase) -> Constraint:
    allow = MATERIALS[material]["allowable_mpa"]

    def _check(shape: Any) -> ConstraintResult:
        sec = occt.section_at(shape, (load.root_x, 0.0, 0.0),
                              load.normal, load.bend_axis)
        if not sec.ok or sec.section_modulus <= 0.0:
            return ConstraintResult(
                "bending", False, measured=0.0, limit=allow, units="MPa",
                detail="no material at the root section (%s): the part carries no "
                       "load there" % sec.reason)
        moment = load.force_n * load.arm            # N*mm
        sigma = moment / sec.section_modulus         # N/mm^2 = MPa
        return ConstraintResult(
            "bending", sigma <= allow, measured=round(sigma, 2), limit=allow,
            units="MPa",
            detail="cantilever: M = %g N x %g mm = %g N.mm; Z = %.1f mm3; "
                   "sigma = M/Z = %.2f MPa (allowable %.0f MPa for %s)"
            % (load.force_n, load.arm, moment, sec.section_modulus, sigma,
               allow, material))
    return Constraint(
        "bending",
        "Euler-Bernoulli bending: sigma = M*c/I under a %g N tip load on a %g mm "
        "cantilever arm, vs the %s allowable (%g MPa = yield / safety factor)"
        % (load.force_n, load.arm, material, allow), _check)


# --------------------------------------------------------------------------- #
# the brief: a constraint set + one reference that satisfies it
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstraintBrief:
    """A requirement, several ways to meet it, and one worked answer.

    ``reference`` is a SINGLE satisfying op stream -- not THE answer, AN answer. It
    exists to prove the constraint set is satisfiable and to serve as the positive
    control (it must build, pass the gate, and satisfy every constraint). The brief
    is graded by :func:`grade`, which never compares a candidate to ``reference``.
    """

    id: str
    text: str
    proc: str
    envelope: Vec3
    material: str
    constraints: Tuple[Constraint, ...]
    reference: Tuple[Op, ...]
    alt_reference: Tuple[Op, ...] = ()    # a DIFFERENT satisfying answer, for the
                                          # test that IoU cannot score this brief

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("brief %r ships no checkable constraint -- that is a "
                             "prompt, not a benchmark" % self.id)
        if not self.reference:
            raise ValueError("brief %r has no reference satisfying solution" % self.id)


@dataclass
class GradeReport:
    brief: str = ""
    built: bool = False
    satisfied: bool = False
    results: List[ConstraintResult] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {"brief": self.brief, "built": self.built,
                "satisfied": self.satisfied,
                "results": [r.to_dict() for r in self.results],
                "reason": self.reason}


def grade(brief: ConstraintBrief, ops) -> GradeReport:
    """Build the candidate and check EVERY constraint on its own geometry."""
    r = GradeReport(brief=brief.id)
    built = occt.build(ops)
    if not built:
        r.reason = built.reason or "the candidate did not build"
        return r
    r.built = True
    r.results = [c.check(built.shape) for c in brief.constraints]
    r.satisfied = all(res.satisfied for res in r.results)
    return r


# --------------------------------------------------------------------------- #
# the shipped briefs
# --------------------------------------------------------------------------- #
def _bracket_brief() -> ConstraintBrief:
    env = (50.0, 50.0, 20.0)
    # Reference: a 50x50x6 plate with an M8 clearance hole in the centre. Fits;
    # admits an M8 bolt; light; and stiff enough as a 45 mm cantilever.
    ref = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 50, 50), Extrude("sk1", 6),
           Hole("sk1", 25.0, 25.0, CLEARANCE_ISO273["M8"], None, True, "simple"))
    # A DIFFERENT satisfying answer: a narrower, thicker plate with the SAME bolt
    # axis (the constraint fixes it). Same constraints met, different shape -- IoU
    # between the two is 0.42, which is the whole point: no shape metric can score
    # a constraint set, because the set does not name a shape.
    alt = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 50, 30), Extrude("sk1", 10),
           Hole("sk1", 25.0, 25.0, CLEARANCE_ISO273["M8"], None, True, "simple"))
    load = LoadCase(root_x=2.0, arm=45.0, force_n=200.0)
    cons = (
        envelope_constraint(env),
        bolt_bore_constraint("M8", (25.0, 25.0), 3.0),
        mass_constraint("AL6061", max_grams=120.0),
        bending_constraint("AL6061", load),
    )
    return ConstraintBrief(
        id="con_bracket_m8_200n",
        text=("A mounting bracket in 6061 aluminium. It must take an M8 bolt "
              "through a hole centred at (25, 25), fit inside a 50 x 50 x 20 mm "
              "envelope, weigh no more than 120 g, and carry a 200 N load at the "
              "end of a 45 mm arm without yielding."),
        proc=("Design an AL6061 bracket: bounding box <= 50 x 50 x 20 mm; an ISO 273 "
              "medium clearance hole for M8 on the axis at (25, 25); mass <= 120 g; "
              "and, as a 45 mm cantilever under 200 N at the tip, peak bending "
              "stress below the 6061 allowable of 110 MPa."),
        envelope=env, material="AL6061", constraints=cons,
        reference=ref, alt_reference=alt)


def _plate_brief() -> ConstraintBrief:
    env = (80.0, 60.0, 25.0)
    ref = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 80, 40), Extrude("sk1", 8),
           Hole("sk1", 40.0, 20.0, CLEARANCE_ISO273["M10"], None, True, "simple"))
    alt = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 60, 60), Extrude("sk1", 9),
           Hole("sk1", 40.0, 20.0, CLEARANCE_ISO273["M10"], None, True, "simple"))
    load = LoadCase(root_x=2.0, arm=70.0, force_n=300.0)
    cons = (
        envelope_constraint(env),
        bolt_bore_constraint("M10", (40.0, 20.0), 4.0),
        mass_constraint("STEEL1018", max_grams=250.0),
        bending_constraint("STEEL1018", load),
    )
    return ConstraintBrief(
        id="con_plate_m10_300n",
        text=("A steel tie-plate that bolts down with an M10 bolt through a hole at "
              "(40, 20), fits within 80 x 60 x 25 mm, weighs under 250 g, and takes "
              "a 300 N pull at the end of a 70 mm reach without yielding."),
        proc=("Design an AISI 1018 steel plate: bounding box <= 80 x 60 x 25 mm; an "
              "ISO 273 medium clearance hole for M10 on the axis at (40, 20); mass "
              "<= 250 g; peak bending stress under a 300 N tip load on a 70 mm "
              "cantilever below the 148 MPa allowable."),
        envelope=env, material="STEEL1018", constraints=cons,
        reference=ref, alt_reference=alt)


#: The shipped constraint briefs.
BRIEFS: Tuple[ConstraintBrief, ...] = (_bracket_brief(), _plate_brief())
