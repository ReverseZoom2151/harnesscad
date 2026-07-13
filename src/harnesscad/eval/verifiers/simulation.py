"""Simulation critic — an analytic (closed-form) mechanical-stress verifier.

The corpus names *"simulate before you build"* as the top value. Full FEA needs an
external solver we do not ship, so this module delivers the **analytic** tier now:
real, cited mechanical-engineering closed forms that catch the obvious failures
(overstress, excessive deflection, buckling, stress raisers) long before a mesh is
ever built — and a clean protocol seam (:class:`FEASolver`) where a real CalculiX /
Elmer backend drops in later.

Design commitments (mirroring :mod:`verifiers.dfm` and :mod:`contract`):

  * **Standalone verifier.** :class:`SimulationCheck` is a :class:`verify.Verifier`
    (``name='simulation'``) added explicitly by a caller (see :func:`with_simulation`);
    it is NOT wired into :func:`verify.default_verifiers` (that would be a circular
    import, and simulation is an opt-in stage).
  * **Never fabricate.** Where a case does not reduce to a standard closed form
    (arbitrary 3-D geometry, thick-wall vessels, contact, plasticity, fatigue,
    modal/thermal), the check emits a typed INFO ``needs-fea`` naming exactly what a
    real solver would compute — never a made-up number.
  * **Degrade gracefully.** No load case, or geometry that cannot be reduced to a
    standard case and is not supplied explicitly, yields an INFO skip — never an
    ERROR and never a crash. A missing geometry backend (the stub answers no
    ``'metrics'`` / ``'measure'``) simply means the load case must carry its own
    dimensions.
  * **Deterministic & stdlib-only** (``math``): the same load case + geometry always
    yields the same diagnostics.

Units follow :mod:`sizing.calc`: N, mm, MPa = N/mm^2, N*mm for torque, E in MPa. So
stresses come out in MPa and deflections in mm directly.

Every formula below is the standard closed form with an inline citation so a reviewer
can audit it (Shigley, Gere & Goodno, Hibbeler, Peterson, Inglis, Euler/Johnson).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport


# =========================================================================== #
# Pure closed-form formulas (each cited; each independently testable)
# =========================================================================== #

# -- section properties ----------------------------------------------------- #
def rectangular_section(b: float, h: float) -> Tuple[float, float, float, float]:
    """Second moment of area, extreme-fibre distance, section modulus and area of
    a solid rectangle ``b`` (width) x ``h`` (depth, in the bending plane).

        I = b h^3 / 12,   c = h / 2,   Z = I / c = b h^2 / 6,   A = b h

    Ref: Gere & Goodno, *Mechanics of Materials*, table of section properties.
    Returns ``(I, c, Z, A)`` in (mm^4, mm, mm^3, mm^2).
    """
    b = float(b)
    h = float(h)
    I = b * h ** 3 / 12.0
    c = h / 2.0
    Z = I / c
    A = b * h
    return I, c, Z, A


def circular_section(d: float) -> Tuple[float, float, float, float]:
    """Section properties of a solid circular section of diameter ``d``.

        I = pi d^4 / 64,   c = d / 2,   Z = pi d^3 / 32,   A = pi d^2 / 4

    Ref: Shigley, *Mechanical Engineering Design*, appendix section properties.
    Returns ``(I, c, Z, A)``.
    """
    d = float(d)
    I = math.pi * d ** 4 / 64.0
    c = d / 2.0
    Z = math.pi * d ** 3 / 32.0
    A = math.pi * d ** 2 / 4.0
    return I, c, Z, A


# -- beam bending ----------------------------------------------------------- #
# Supported standard cases (concentrated load P, span L):
#   cantilever        : end load        -> M_max = P L,      delta = P L^3 / (3 E I)
#   simply_supported  : central load    -> M_max = P L / 4,  delta = P L^3 / (48 E I)
# Ref: Gere & Goodno / Roark, standard beam-deflection tables.
_BEAM_CASES = {
    "cantilever": {"moment": 1.0, "defl": 1.0 / 3.0},
    "simply_supported": {"moment": 0.25, "defl": 1.0 / 48.0},
}


def beam_max_moment(force: float, span: float, support: str) -> float:
    """Maximum bending moment [N*mm] for a supported standard case (see
    ``_BEAM_CASES``). Ref: Gere & Goodno, beam bending-moment diagrams."""
    coeff = _BEAM_CASES[support]["moment"]
    return coeff * float(force) * float(span)


def beam_bending_stress(force: float, span: float, I: float, c: float,
                        support: str) -> float:
    """Maximum bending (flexural) stress ``sigma = M c / I`` [MPa].

    Ref: Euler-Bernoulli flexure formula sigma = M y / I (Gere & Goodno, Hibbeler).
    """
    M = beam_max_moment(force, span, support)
    return M * float(c) / float(I)


def beam_max_deflection(force: float, span: float, E: float, I: float,
                        support: str) -> float:
    """Maximum transverse deflection [mm] for a supported standard case.

    Ref: standard beam-deflection tables (Roark, Gere & Goodno):
    cantilever end load  delta = P L^3 / (3 E I); simply-supported central load
    delta = P L^3 / (48 E I).
    """
    coeff = _BEAM_CASES[support]["defl"]
    return coeff * float(force) * float(span) ** 3 / (float(E) * float(I))


# -- thin-wall pressure vessel ---------------------------------------------- #
def hoop_stress(pressure: float, radius: float, thickness: float) -> float:
    """Circumferential (hoop) stress in a thin-wall cylinder ``sigma_h = p r / t``
    [MPa]. Ref: Shigley / Hibbeler, thin-wall pressure-vessel theory (valid for
    r/t >= ~10)."""
    return float(pressure) * float(radius) / float(thickness)


def longitudinal_stress(pressure: float, radius: float, thickness: float) -> float:
    """Longitudinal (axial) stress in a thin-wall cylinder ``sigma_l = p r / (2 t)``
    [MPa] — half the hoop stress. Ref: Shigley, thin-wall pressure vessels."""
    return float(pressure) * float(radius) / (2.0 * float(thickness))


# -- stress-concentration factors ------------------------------------------- #
def kt_hole(d_over_w: float) -> float:
    """Elastic stress-concentration factor Kt for a transverse circular hole in a
    finite-width plate under uniaxial tension, referenced to the NET section.

    Peterson polynomial fit (R. E. Peterson, *Stress Concentration Factors*),
    widely reproduced (e.g. Shigley fig. A-15-1):

        Kt = 3.00 - 3.13 (d/w) + 3.66 (d/w)^2 - 1.53 (d/w)^3

    As d/w -> 0 this tends to the classical Kt = 3.0 for a small hole in an
    infinite plate. Valid for 0 <= d/w < 1.
    """
    x = float(d_over_w)
    return 3.00 - 3.13 * x + 3.66 * x ** 2 - 1.53 * x ** 3


def kt_fillet(width_large: float, width_small: float, radius: float) -> float:
    """Approximate stress-concentration factor Kt at a shoulder fillet, from the
    Inglis elliptical-flaw relation used as a closed-form approximation:

        Kt = 1 + 2 sqrt(t / rho)

    with notch depth ``t = (D - d)/2`` (the shoulder step) and root radius
    ``rho = r`` (the fillet radius). Ref: C. E. Inglis (1913); Peterson tabulates
    refined values for specific D/d — this closed form is the conservative
    engineering approximation. Requires radius > 0 and D >= d.
    """
    D = float(width_large)
    d = float(width_small)
    r = float(radius)
    depth = max(0.0, (D - d) / 2.0)
    return 1.0 + 2.0 * math.sqrt(depth / r)


# -- columns: Euler / Johnson buckling -------------------------------------- #
# Effective-length factor K by end condition (theoretical values, Shigley table).
_END_CONDITION_K = {
    "pinned-pinned": 1.0,
    "fixed-fixed": 0.5,
    "fixed-free": 2.0,
    "fixed-pinned": 0.699,
}


def effective_length_factor(end_condition: str) -> float:
    """Column effective-length factor K for a named end condition.
    Ref: Shigley, column end-condition constants."""
    return _END_CONDITION_K[end_condition]


def radius_of_gyration(I: float, A: float) -> float:
    """Radius of gyration ``k = sqrt(I / A)`` [mm]. Ref: Shigley, columns."""
    return math.sqrt(float(I) / float(A))


def slenderness_ratio(length: float, I: float, A: float,
                      end_condition: str = "pinned-pinned") -> float:
    """Effective slenderness ratio ``(K L) / k`` (dimensionless).
    Ref: Shigley, column buckling."""
    k = radius_of_gyration(I, A)
    return effective_length_factor(end_condition) * float(length) / k


def euler_critical_load(E: float, I: float, length: float,
                        end_condition: str = "pinned-pinned") -> float:
    """Euler critical (buckling) load ``P_cr = pi^2 E I / (K L)^2`` [N].

    Ref: L. Euler; Shigley, long-column buckling. Valid only for slender columns
    (slenderness >= the tangent point with yielding — see
    :func:`buckling_transition_slenderness`); for shorter columns use
    :func:`johnson_critical_stress`.
    """
    Le = effective_length_factor(end_condition) * float(length)
    return math.pi ** 2 * float(E) * float(I) / Le ** 2


def buckling_transition_slenderness(E: float, yield_strength: float) -> float:
    """Slenderness at the Euler/Johnson tangent point ``(K L / k)_1 =
    sqrt(2 pi^2 E / Sy)``. Above it Euler governs; below it the J. B. Johnson
    parabola governs. Ref: Shigley, intermediate-length columns."""
    return math.sqrt(2.0 * math.pi ** 2 * float(E) / float(yield_strength))


def johnson_critical_stress(yield_strength: float, slenderness: float,
                            E: float) -> float:
    """J. B. Johnson parabolic critical stress for an intermediate column:

        sigma_cr = Sy - (1/E) ( Sy (K L / k) / (2 pi) )^2

    Ref: Shigley, J. B. Johnson formula. Use when slenderness is below the
    transition value; it correctly reduces to Sy for a stub column.
    """
    Sy = float(yield_strength)
    return Sy - (Sy * float(slenderness) / (2.0 * math.pi)) ** 2 / float(E)


# =========================================================================== #
# Rules (configurable thresholds; JSON round-trips like DFMRules)
# =========================================================================== #
@dataclass
class SimRules:
    """Tunable thresholds for the simulation critic.

    * ``marginal_band`` — an achieved factor of safety at or above the required
      SF but below ``required_sf * marginal_band`` is flagged WARNING (marginal),
      not ERROR. Default 1.10 (within 10 %).
    * ``thin_wall_ratio`` — minimum r/t for the thin-wall vessel formula to apply;
      below it the pr/t estimate is unconservative, so the check emits a
      ``needs-fea`` (thick-wall Lame / FEA) rather than a pass.
    """

    marginal_band: float = 1.10
    thin_wall_ratio: float = 10.0

    def to_dict(self) -> dict:
        return {"marginal_band": self.marginal_band,
                "thin_wall_ratio": self.thin_wall_ratio}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "SimRules":
        d = d or {}
        defaults = cls()
        return cls(
            marginal_band=float(d.get("marginal_band", defaults.marginal_band)),
            thin_wall_ratio=float(d.get("thin_wall_ratio", defaults.thin_wall_ratio)),
        )


# =========================================================================== #
# LoadCase — the applied loads + material + acceptance criteria
# =========================================================================== #
_ANALYSES = ("beam_bending", "pressure_vessel", "stress_concentration", "buckling")


@dataclass
class LoadCase:
    """One mechanical load case to verify against.

    Loads (any subset; canonical units):
      * ``force``    — concentrated force [N] (transverse for a beam, axial for a
        column / a net-section tension for stress concentration).
      * ``pressure`` — internal gauge pressure [MPa] for a vessel.
      * ``torque``   — applied torque [N*mm] (reserved; general torsion of a
        non-round section is a ``needs-fea`` case).

    Material / acceptance:
      * ``yield_strength`` — material yield Sy [MPa] (allowable stress).
      * ``youngs_modulus`` — E [MPa] (deflection & buckling). Default steel 200 GPa.
      * ``safety_factor``  — required factor of safety (achieved FoS below this is
        an ERROR). Default 2.0.
      * ``deflection_limit`` — optional serviceability limit [mm] for beam bending.

    Case selection / geometry:
      * ``where``   — free-text location ("free end", "mid-span", "at hole", ...).
      * ``support`` — beam case: ``"cantilever"`` | ``"simply_supported"``.
      * ``end_condition`` — column end fixity (see ``_END_CONDITION_K``).
      * ``analysis`` — which analytic check(s) to run: an item of ``_ANALYSES``, a
        list of them, or ``None``/``"auto"`` to infer (pressure -> vessel;
        force -> beam bending). ``stress_concentration`` and ``buckling`` are only
        run when named explicitly (a bare force is ambiguous otherwise).
      * ``geometry`` — explicit standard-case dimensions that OVERRIDE anything
        derived from the backend bbox. Recognised keys: ``span``, ``length``,
        ``section_b``, ``section_h``, ``diameter``, ``moment_of_inertia``, ``c``,
        ``area``, ``radius``, ``wall_thickness``, ``hole_diameter``,
        ``plate_width``, ``thickness``, ``width_large``, ``width_small``,
        ``fillet_radius``.
    """

    force: Optional[float] = None
    pressure: Optional[float] = None
    torque: Optional[float] = None

    yield_strength: float = 250.0
    youngs_modulus: float = 200000.0
    safety_factor: float = 2.0
    deflection_limit: Optional[float] = None

    where: str = ""
    support: str = "cantilever"
    end_condition: str = "pinned-pinned"
    analysis: Optional[object] = None  # str | List[str] | None
    geometry: Dict[str, float] = field(default_factory=dict)

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        d: dict = {
            "yield_strength": self.yield_strength,
            "youngs_modulus": self.youngs_modulus,
            "safety_factor": self.safety_factor,
            "where": self.where,
            "support": self.support,
            "end_condition": self.end_condition,
        }
        if self.force is not None:
            d["force"] = self.force
        if self.pressure is not None:
            d["pressure"] = self.pressure
        if self.torque is not None:
            d["torque"] = self.torque
        if self.deflection_limit is not None:
            d["deflection_limit"] = self.deflection_limit
        if self.analysis is not None:
            d["analysis"] = self.analysis
        if self.geometry:
            d["geometry"] = dict(self.geometry)
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "LoadCase":
        d = d or {}
        defaults = cls()

        def _opt(key):
            v = d.get(key)
            return None if v is None else float(v)

        return cls(
            force=_opt("force"),
            pressure=_opt("pressure"),
            torque=_opt("torque"),
            yield_strength=float(d.get("yield_strength", defaults.yield_strength)),
            youngs_modulus=float(d.get("youngs_modulus", defaults.youngs_modulus)),
            safety_factor=float(d.get("safety_factor", defaults.safety_factor)),
            deflection_limit=_opt("deflection_limit"),
            where=str(d.get("where", defaults.where)),
            support=str(d.get("support", defaults.support)),
            end_condition=str(d.get("end_condition", defaults.end_condition)),
            analysis=d.get("analysis"),
            geometry={k: float(v) for k, v in (d.get("geometry") or {}).items()},
        )

    def analyses(self) -> List[str]:
        """Resolve the ``analysis`` field to a concrete, ordered list of checks.

        Explicit selection wins; ``None``/``"auto"`` infers: pressure -> vessel,
        force -> beam bending. (``stress_concentration`` / ``buckling`` must be
        named explicitly — a bare force does not disambiguate them from bending.)
        """
        a = self.analysis
        if isinstance(a, str) and a != "auto":
            return [a] if a in _ANALYSES else []
        if isinstance(a, (list, tuple)):
            return [x for x in a if x in _ANALYSES]
        # auto
        out: List[str] = []
        if self.pressure is not None:
            out.append("pressure_vessel")
        if self.force is not None:
            out.append("beam_bending")
        return out


# =========================================================================== #
# FEASolver protocol — the real-solver drop-in seam (NO fake solver shipped)
# =========================================================================== #
@runtime_checkable
class FEASolver(Protocol):
    """The seam a real finite-element backend (CalculiX, Elmer, code_aster, ...)
    implements to replace the ``needs-fea`` INFO stubs with computed fields.

    Deliberately tiny and kernel-agnostic — exactly the ``mesh`` then ``solve``
    two-step every FE solver exposes:

      * ``mesh(shape) -> mesh``   — discretise a backend solid (e.g. the OCCT shape
        from ``query('measure')`` / ``_combined()``) into a volume mesh.
      * ``solve(mesh, load_case) -> fields`` — apply the :class:`LoadCase` loads and
        boundary conditions and return result fields. The result SHOULD expose at
        least ``max_von_mises`` [MPa] and ``max_displacement`` [mm] so
        :class:`SimulationCheck` can apply the same factor-of-safety logic it uses
        for the analytic cases.

    No implementation ships here on purpose: a fake solver would fabricate results,
    which this module must never do. When no solver is attached, non-analytic cases
    return a typed ``needs-fea`` INFO naming what a solver would compute.
    """

    name: str

    def mesh(self, shape) -> object: ...

    def solve(self, mesh, load_case: "LoadCase") -> object: ...


# =========================================================================== #
# The verifier
# =========================================================================== #
class SimulationCheck:
    """A :class:`verify.Verifier` (``name='simulation'``) that runs analytic
    mechanical checks for a :class:`LoadCase` and reports factor-of-safety findings.

    Severity policy (per check):
      * ERROR   ``overstressed`` / ``over-deflected`` / ``buckling`` — achieved
        factor of safety < required ``load_case.safety_factor`` (or deflection >
        limit).
      * WARNING ``marginal`` — achieved FoS in ``[required, required*marginal_band)``.
      * INFO    ``sim-pass`` — achieved FoS >= required*marginal_band (with the
        computed numbers, for transparency).
      * INFO    ``needs-fea`` — the case is not analytically reducible; names what a
        real solver would compute.
      * INFO    ``simulation-skipped`` — no load case, no applicable check, or the
        geometry is neither measurable nor supplied.

    Reads ``query('metrics')`` (preferred) then ``query('measure')`` for a bbox to
    reduce simple parts to a standard case; explicit ``load_case.geometry`` always
    overrides. Never raises: any per-check failure degrades to an INFO skip.
    """

    name = "simulation"

    def __init__(self, load_case: Optional[LoadCase] = None,
                 rules: Optional[SimRules] = None,
                 solver: Optional[FEASolver] = None) -> None:
        self.load_case = load_case
        self.rules = rules or SimRules()
        self.solver = solver

    # -- entry point -------------------------------------------------------- #
    def check(self, backend, opdag) -> VerifyReport:
        diags: List[Diagnostic] = []
        lc = self.load_case
        if lc is None:
            diags.append(_info(
                "simulation-skipped",
                "no load case supplied; add a LoadCase (force/pressure/torque + "
                "material yield) to run analytic stress checks."))
            return VerifyReport(diags)

        geom = self._resolve_geometry(backend, lc)
        analyses = lc.analyses()
        if not analyses:
            diags.append(_info(
                "simulation-skipped",
                "no applicable analytic check: load case names no analysis and "
                "carries no pressure (-> vessel) or force (-> beam bending)."))
            return VerifyReport(diags)

        dispatch = {
            "beam_bending": self._beam_bending,
            "pressure_vessel": self._pressure_vessel,
            "stress_concentration": self._stress_concentration,
            "buckling": self._buckling,
        }
        for name in analyses:
            fn = dispatch.get(name)
            if fn is None:
                diags.append(_info("simulation-skipped",
                                   f"unknown analysis '{name}'."))
                continue
            try:
                fn(lc, geom, diags)
            except Exception as exc:  # noqa: BLE001 - a bad case must skip, not crash
                diags.append(_info(
                    "simulation-skipped",
                    f"{name} not evaluated ({type(exc).__name__}: {exc}); supply the "
                    f"standard-case dimensions in load_case.geometry."))
        return VerifyReport(diags)

    # -- geometry resolution ------------------------------------------------ #
    def _resolve_geometry(self, backend, lc: LoadCase) -> Dict[str, float]:
        """Merge explicit ``load_case.geometry`` (wins) with dims derived from the
        backend bounding box (longest edge = span/length; the other two = a
        conservative weak-axis rectangular section)."""
        geom: Dict[str, float] = dict(lc.geometry)
        bbox = _bbox(backend)
        if bbox:
            dims = sorted((d for d in bbox if d > 0.0), reverse=True)
            if len(dims) == 3:
                span, a, b = dims[0], dims[1], dims[2]
                # weak-axis rectangular section (depth = smallest edge) is the
                # conservative reduction: it maximises stress and deflection.
                I_weak = a * b ** 3 / 12.0
                geom.setdefault("span", span)
                geom.setdefault("length", span)
                geom.setdefault("_bbox_I", I_weak)
                geom.setdefault("_bbox_c", b / 2.0)
                geom.setdefault("_bbox_area", a * b)
        return geom

    # -- individual analytic checks ---------------------------------------- #
    def _beam_bending(self, lc: LoadCase, geom: Dict[str, float],
                      diags: List[Diagnostic]) -> None:
        if lc.force is None:
            diags.append(_info("simulation-skipped",
                               "beam bending needs a transverse 'force'."))
            return
        if lc.support not in _BEAM_CASES:
            diags.append(_needs_fea(
                f"beam support '{lc.support}' is not a supported standard case "
                f"({'/'.join(_BEAM_CASES)}); a solver would compute the moment and "
                f"deflection field for arbitrary supports.", lc.where))
            return
        span = geom.get("span")
        if span is None:
            diags.append(_info("simulation-skipped",
                               "beam bending needs a 'span' (in load_case.geometry) "
                               "or a measurable bbox."))
            return
        props = _section_props(geom)
        if props is None:
            diags.append(_info("simulation-skipped",
                               "beam bending needs a cross-section (section_b+section_h, "
                               "diameter, moment_of_inertia+c, or a measurable bbox)."))
            return
        I, c, _Z, _A = props

        sigma = beam_bending_stress(lc.force, span, I, c, lc.support)
        detail = (f"{lc.support} beam, P={lc.force:g} N, L={span:g} mm, "
                  f"I={I:.4g} mm^4, c={c:g} mm -> sigma=M c/I={sigma:.4g} MPa")
        self._fos_diag(sigma, lc, diags, code_ok="sim-pass",
                       code_over="overstressed", where=lc.where, detail=detail)

        # Optional serviceability (deflection) check.
        if lc.deflection_limit is not None:
            delta = beam_max_deflection(lc.force, span, lc.youngs_modulus, I,
                                        lc.support)
            ddet = (f"{lc.support} beam deflection delta={delta:.4g} mm vs limit "
                    f"{lc.deflection_limit:g} mm")
            if delta > lc.deflection_limit:
                diags.append(_err("over-deflected",
                                  f"deflection exceeds limit: {ddet}.", lc.where))
            elif delta > lc.deflection_limit / self.rules.marginal_band:
                diags.append(_warn("marginal-deflection",
                                   f"deflection near limit: {ddet}.", lc.where))
            else:
                diags.append(_info("sim-pass", f"deflection ok: {ddet}.", lc.where))

    def _pressure_vessel(self, lc: LoadCase, geom: Dict[str, float],
                         diags: List[Diagnostic]) -> None:
        if lc.pressure is None:
            diags.append(_info("simulation-skipped",
                               "pressure-vessel check needs a 'pressure'."))
            return
        r = geom.get("radius")
        t = geom.get("wall_thickness")
        if r is None or t is None:
            diags.append(_needs_fea(
                "thin-wall hoop stress needs an explicit 'radius' and "
                "'wall_thickness' (a solid bbox cannot reveal wall thickness); a "
                "solver would compute the membrane + bending stress field of the "
                "actual shell.", lc.where))
            return
        ratio = float(r) / float(t)
        if ratio < self.rules.thin_wall_ratio:
            diags.append(_needs_fea(
                f"vessel r/t={ratio:.2f} < {self.rules.thin_wall_ratio:g}: the thin-wall "
                f"sigma=pr/t assumption is unconservative here; a thick-wall Lame "
                f"solution or FEA is required for the true through-thickness stress.",
                lc.where))
            return
        sigma_h = hoop_stress(lc.pressure, r, t)
        sigma_l = longitudinal_stress(lc.pressure, r, t)
        detail = (f"thin-wall cylinder p={lc.pressure:g} MPa, r={float(r):g} mm, "
                  f"t={float(t):g} mm (r/t={ratio:.1f}) -> hoop sigma_h=pr/t="
                  f"{sigma_h:.4g} MPa, long sigma_l=pr/2t={sigma_l:.4g} MPa")
        # Hoop stress governs (it is the larger of the two).
        self._fos_diag(sigma_h, lc, diags, code_ok="sim-pass",
                       code_over="overstressed", where=lc.where, detail=detail)

    def _stress_concentration(self, lc: LoadCase, geom: Dict[str, float],
                              diags: List[Diagnostic]) -> None:
        if lc.force is None:
            diags.append(_info("simulation-skipped",
                               "stress-concentration check needs a tension 'force'."))
            return
        th = geom.get("thickness")
        # Hole in a finite-width plate (net-section Kt).
        if "hole_diameter" in geom and "plate_width" in geom and th:
            d = float(geom["hole_diameter"])
            w = float(geom["plate_width"])
            net_area = (w - d) * float(th)
            if net_area <= 0:
                diags.append(_info("simulation-skipped",
                                   "hole diameter >= plate width: no net section."))
                return
            kt = kt_hole(d / w)
            nominal = float(lc.force) / net_area
            peak = kt * nominal
            detail = (f"hole in plate d/w={d / w:.3f} -> Kt={kt:.3f} (Peterson); "
                      f"net sigma_nom={nominal:.4g} MPa -> peak={peak:.4g} MPa")
            self._fos_diag(peak, lc, diags, code_ok="sim-pass",
                           code_over="overstressed", where=lc.where, detail=detail)
            return
        # Shoulder fillet in a stepped flat bar.
        if all(k in geom for k in ("width_large", "width_small", "fillet_radius")) and th:
            D = float(geom["width_large"])
            d = float(geom["width_small"])
            r = float(geom["fillet_radius"])
            kt = kt_fillet(D, d, r)
            nominal = float(lc.force) / (d * float(th))
            peak = kt * nominal
            detail = (f"shoulder fillet D={D:g}/d={d:g}, r={r:g} -> Kt={kt:.3f} "
                      f"(Inglis approx); sigma_nom={nominal:.4g} MPa -> "
                      f"peak={peak:.4g} MPa")
            self._fos_diag(peak, lc, diags, code_ok="sim-pass",
                           code_over="overstressed", where=lc.where, detail=detail)
            return
        diags.append(_needs_fea(
            "stress-concentration needs either (hole_diameter, plate_width, "
            "thickness) or (width_large, width_small, fillet_radius, thickness) in "
            "load_case.geometry; a solver would extract the true peak stress from "
            "the mesh.", lc.where))

    def _buckling(self, lc: LoadCase, geom: Dict[str, float],
                  diags: List[Diagnostic]) -> None:
        if lc.force is None:
            diags.append(_info("simulation-skipped",
                               "buckling check needs an axial 'force'."))
            return
        if lc.end_condition not in _END_CONDITION_K:
            diags.append(_needs_fea(
                f"column end condition '{lc.end_condition}' unknown "
                f"({'/'.join(_END_CONDITION_K)}); a solver would run an eigenvalue "
                f"(linear buckling) analysis.", lc.where))
            return
        length = geom.get("length")
        props = _section_props(geom)
        if length is None or props is None:
            diags.append(_info("simulation-skipped",
                               "buckling needs a 'length' and a cross-section "
                               "(section_b+section_h, diameter, or a measurable bbox)."))
            return
        I, _c, _Z, A = props
        lam = slenderness_ratio(length, I, A, lc.end_condition)
        transition = buckling_transition_slenderness(lc.youngs_modulus,
                                                      lc.yield_strength)
        if lam >= transition:
            p_cr = euler_critical_load(lc.youngs_modulus, I, length, lc.end_condition)
            mode = "Euler"
        else:
            sigma_cr = johnson_critical_stress(lc.yield_strength, lam,
                                               lc.youngs_modulus)
            p_cr = sigma_cr * A
            mode = "J.B.Johnson (intermediate column)"
        fos = p_cr / float(lc.force) if lc.force else float("inf")
        detail = (f"column K={effective_length_factor(lc.end_condition):g} "
                  f"({lc.end_condition}), L={float(length):g} mm, slenderness="
                  f"{lam:.1f} (transition {transition:.1f}) -> {mode}; "
                  f"P_cr={p_cr:.4g} N vs applied {lc.force:g} N; FoS={fos:.3g}")
        req = lc.safety_factor
        if fos < req:
            diags.append(_err("buckling",
                              f"column buckles below required FoS {req:g}: {detail}.",
                              lc.where))
        elif fos < req * self.rules.marginal_band:
            diags.append(_warn("marginal",
                               f"buckling FoS marginal: {detail}.", lc.where))
        else:
            diags.append(_info("sim-pass",
                               f"buckling ok: {detail}.", lc.where))

    # -- shared factor-of-safety -> diagnostic ------------------------------ #
    def _fos_diag(self, sigma: float, lc: LoadCase, diags: List[Diagnostic],
                  code_ok: str, code_over: str, where: str, detail: str) -> None:
        if sigma <= 0.0:
            diags.append(_info("sim-pass",
                               f"non-positive stress ({sigma:.4g} MPa); {detail}.",
                               where))
            return
        fos = lc.yield_strength / sigma
        req = lc.safety_factor
        tail = (f"{detail}; Sy={lc.yield_strength:g} MPa -> FoS={fos:.3g} "
                f"(required {req:g})")
        if fos < req:
            diags.append(_err(code_over,
                              f"stress exceeds allowable: {tail}.", where))
        elif fos < req * self.rules.marginal_band:
            diags.append(_warn("marginal",
                               f"factor of safety marginal: {tail}.", where))
        else:
            diags.append(_info(code_ok, f"{tail}.", where))


# =========================================================================== #
# Wiring helper
# =========================================================================== #
def with_simulation(verifiers, load_case: Optional[LoadCase] = None,
                    rules: Optional[SimRules] = None,
                    solver: Optional[FEASolver] = None) -> List:
    """Return a new verifier list with a :class:`SimulationCheck` appended.

    Mirrors :func:`verifiers.dfm.with_dfm` / adding the geometry check to the
    default set without editing ``verify.py``::

        from harnesscad.eval.verifiers.verify import default_verifiers
        from harnesscad.eval.verifiers.simulation import with_simulation, LoadCase
        verifiers = with_simulation(default_verifiers(), LoadCase(force=1000, ...))
    """
    return list(verifiers) + [SimulationCheck(load_case, rules, solver)]


# =========================================================================== #
# Helpers
# =========================================================================== #
def _section_props(geom: Dict[str, float]) -> Optional[Tuple[float, float, float, float]]:
    """Resolve ``(I, c, Z, A)`` from an explicit section, or fall back to the
    bbox-derived weak-axis section. Explicit definitions win, in order:
    moment_of_inertia+c, section_b+section_h, diameter, then bbox."""
    if "moment_of_inertia" in geom and "c" in geom:
        I = float(geom["moment_of_inertia"])
        c = float(geom["c"])
        A = float(geom.get("area", geom.get("_bbox_area", 0.0)))
        Z = I / c if c else 0.0
        return I, c, Z, A
    if "section_b" in geom and "section_h" in geom:
        return rectangular_section(geom["section_b"], geom["section_h"])
    if "diameter" in geom:
        return circular_section(geom["diameter"])
    if "_bbox_I" in geom and "_bbox_c" in geom:
        I = float(geom["_bbox_I"])
        c = float(geom["_bbox_c"])
        A = float(geom.get("_bbox_area", 0.0))
        Z = I / c if c else 0.0
        return I, c, Z, A
    return None


def _bbox(backend) -> Optional[List[float]]:
    """The three bbox edge lengths from ``query('metrics')`` then
    ``query('measure')``, or None when unavailable (stub / no solid)."""
    for q in ("metrics", "measure"):
        data = _query(backend, q)
        if data:
            bbox = data.get("bbox")
            if bbox and len(bbox) >= 3:
                dims = [float(v) for v in bbox[:3]]
                if any(d > 0.0 for d in dims):
                    return dims
    return None


def _query(backend, q: str) -> Optional[dict]:
    """Read a backend query, returning None when unanswered (backends return {}
    for unknown queries) so callers degrade gracefully."""
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade, not crash
        return None
    return result or None


def _err(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg, where or None)


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where or None)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where or None)


def _needs_fea(msg: str, where: Optional[str] = None) -> Diagnostic:
    """A typed INFO marking a case that is not analytically reducible — names what
    a real :class:`FEASolver` would compute, never a fabricated number."""
    return Diagnostic(Severity.INFO, "needs-fea", msg, where or None)
