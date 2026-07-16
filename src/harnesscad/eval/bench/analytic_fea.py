"""Closed-form structural FEA oracles: an answer key, not a second opinion.

Every other benchmark in ``eval/bench`` grades a model against something a
human wrote down. This module grades against ARITHMETIC. A cantilever's tip
deflection under an end load is ``P L^3 / (3 E I)`` -- not because a solver
said so, not because the harness usually returns that, but because
Euler-Bernoulli beam theory says so and has said so since 1750. A backend that
misses it is the backend with the bug. There is no appeal.

This is the discipline :mod:`harnesscad.eval.corpus.analytic` uses for volumes,
carried into structural mechanics: the closed form is stated, cited, and
COMPUTED here; nothing is measured and nothing is stored as a magic constant.

WHERE THE FORMULAS COME FROM. They are NOT re-derived here. Every one that the
harness already owns is IMPORTED from :mod:`harnesscad.eval.verifiers.simulation`
-- ``rectangular_section`` (I, c, Z, A), ``beam_max_deflection`` and
``beam_bending_stress`` (cantilever end load), ``euler_critical_load``. That
module is the harness's cited closed-form library and it is a VERIFIER: it asks
"does this design pass?". This module is a BENCHMARK: it asks "does this
engine compute the right number?". One derivation, two consumers, and a fix to
a formula cannot desynchronise them. Only the four standard cases ``simulation``
has no notion of -- distributed load, fixed-fixed span, off-end point load, and
the modal fundamental (which ``simulation`` explicitly refuses as ``needs-fea``)
-- are added below, each with its own citation.

THE CROSS-CHECK, AND WHAT IT FOUND. cad-cae-copilot ships the same nine cases
with STORED reference numbers (MIT; vendored under ``cad_cae_copilot/`` with a
SHA-256 manifest). Those files are read here for exactly one purpose: to check
an independent party's arithmetic against ours. They are not the truth and are
never adopted as it. Running the check today:

  * THIRTEEN of the fifteen (case, metric) rows agree with their own formula.
    Nine agree exactly; four (modal, buckling, midspan, lateral) agree only to
    about 2e-8 relative, because upstream evaluated ``I = 1666.667`` /
    ``6666.667`` rather than the exact fraction ``5000/3`` / ``20000/3``. That
    is rounding in the 8th significant figure, it passes
    :data:`AGREEMENT_RTOL`, and it is named here rather than left to be
    rediscovered.
  * TWO DO NOT, and by a wide margin:
      - ``fixed_fixed_udl.max_displacement``      stored 2.63e-4 mm vs
        ``w L^4 / (384 E I)`` = 1.8601e-4 mm  -> +41.4%
      - ``fixed_fixed_center_load.max_displacement`` stored 6.57e-4 mm vs
        ``P L^3 / (192 E I)`` = 3.7202e-4 mm  -> +76.6%
    Upstream is HONEST about this in its own notes: both are coarse-mesh
    real-solver goldens for a max-SURFACE-displacement field, not neutral-axis
    closed forms, and it says so in the file. So they are loaded, flagged
    ``is_oracle=False``, and EXCLUDED from :func:`oracles`. A number that
    disagrees with the formula it cites is not an oracle no matter who
    committed it -- and a corpus that quietly averaged these two in would be
    shipping a 41% error as ground truth.

UNITS follow ``verifiers/simulation`` and the upstream fixtures: mm, N,
MPa = N/mm^2, E in MPa, and for the modal case the consistent mm-tonne-second
system (density in t/mm^3) so the frequency comes out in Hz.

Stdlib only, deterministic, ASCII only. No solver, no kernel, no model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from harnesscad.eval.corpus.fixtures import (FixtureEntry, Manifest,
                                             resources_root, sha256_of)
# The derivations the harness already owns live in verifiers/simulation.py.
# They are imported, not copied.
from harnesscad.eval.verifiers.simulation import (beam_bending_stress,
                                                  beam_max_deflection,
                                                  euler_critical_load,
                                                  rectangular_section)

__all__ = [
    "Beam",
    "STEEL_E",
    "STEEL_RHO",
    "FeaOracle",
    "CrossCheck",
    "cases",
    "oracles",
    "case",
    "manifest",
    "upstream_reference",
    "crosscheck",
    "disagreements",
    "main",
]

_SOURCE = "cad_cae_copilot"

#: Young's modulus of the fixtures' steel [MPa = N/mm^2].
STEEL_E = 210000.0
#: Density in the consistent mm-tonne-second system [t/mm^3]; 7850 kg/m^3.
STEEL_RHO = 7.85e-9

#: First clamped-free eigenvalue of the Euler-Bernoulli beam: the smallest root
#: of ``cos(bL) cosh(bL) + 1 = 0``. Ref: Blevins, *Formulas for Natural
#: Frequency and Mode Shape*, table 8-1; Rao, *Mechanical Vibrations*.
BETA1_L_CLAMPED_FREE = 1.875104

#: Relative tolerance at which a stored number is judged to AGREE with its own
#: formula. 1e-6 is far tighter than any tolerance band in the corpus and far
#: looser than float noise, so it separates "upstream rounded I" from
#: "upstream stored a different quantity".
AGREEMENT_RTOL = 1e-6

_CITE_AXIAL = ("Hooke's law for a prismatic bar: delta = F L / (A E), "
               "sigma = F / A (Gere & Goodno, axially loaded members)")
_CITE_CANT_END = ("standard beam-deflection table: cantilever, end load P -> "
                  "delta = P L^3 / (3 E I), M_max = P L "
                  "(verifiers.simulation.beam_max_deflection / "
                  "beam_bending_stress)")
_CITE_CANT_UDL = ("standard beam-deflection table: cantilever, uniform load w "
                  "-> delta = w L^4 / (8 E I), M_max = w L^2 / 2 at the root "
                  "(Roark; Gere & Goodno)")
_CITE_CANT_MID = ("cantilever with a point load P at distance a from the root: "
                  "free-tip delta = P a^2 (3 L - a) / (6 E I), M_root = P a "
                  "(Roark, cantilever load tables)")
_CITE_FF_UDL = ("fixed-fixed beam, uniform load w -> mid-span neutral-axis "
                "delta = w L^4 / (384 E I), M_max = w L^2 / 12 at the ends "
                "(Roark; Gere & Goodno)")
_CITE_FF_CTR = ("fixed-fixed beam, central point load P -> mid-span "
                "neutral-axis delta = P L^3 / (192 E I), M = P L / 8 "
                "(Roark; Gere & Goodno)")
_CITE_MODAL = ("Euler-Bernoulli clamped-free fundamental: "
               "f1 = (beta1^2 / 2 pi) sqrt(E I / (rho A L^4)) with "
               "beta1 L = 1.875104 (Blevins table 8-1; Rao)")
_CITE_EULER = ("Euler long-column buckling: P_cr = pi^2 E I / (K L)^2, K = 2 "
               "for fixed-free; the reported factor is P_cr / P_ref "
               "(verifiers.simulation.euler_critical_load)")

_CITE_FLEXURE = "Euler-Bernoulli flexure formula sigma = M c / I"


# --------------------------------------------------------------------------- #
# the beam
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Beam:
    """The prismatic bar every case in this corpus is cut from.

    All nine upstream fixtures are one rectangular prism meshed as coarse C3D8
    hexes: ``length`` along X, section ``width`` (Y) by ``depth`` (Z). A load in
    -Z bends about the STRONG axis (``I = width * depth^3 / 12``); a load in -Y,
    and the fundamental mode, bend about the WEAK axis
    (``I = depth * width^3 / 12``). Getting that pair the wrong way round is the
    single easiest way to be confidently wrong by a factor of four here, so the
    two are named rather than inlined.
    """

    length: float
    width: float
    depth: float
    E: float = STEEL_E
    rho: float = STEEL_RHO

    @property
    def strong(self) -> Tuple[float, float, float, float]:
        """(I, c, Z, A) for bending about the strong axis (load along -Z)."""
        return rectangular_section(self.width, self.depth)

    @property
    def weak(self) -> Tuple[float, float, float, float]:
        """(I, c, Z, A) for bending about the weak axis (load along -Y)."""
        return rectangular_section(self.depth, self.width)

    @property
    def area(self) -> float:
        return self.width * self.depth


#: The upstream fixture beam: L = 100 mm, 10 x 20 mm section, steel.
BEAM = Beam(length=100.0, width=10.0, depth=20.0)
#: The tension_rod fixture is a slimmer bar: L = 100 mm, 10 x 10 mm section.
ROD = Beam(length=100.0, width=10.0, depth=10.0)


# --------------------------------------------------------------------------- #
# the four closed forms the harness did not already own
# --------------------------------------------------------------------------- #
def axial_extension(force: float, length: float, area: float, E: float) -> float:
    """Extension of an axially loaded prismatic bar [mm]. See ``_CITE_AXIAL``."""
    return float(force) * float(length) / (float(area) * float(E))


def cantilever_udl_deflection(w: float, length: float, E: float, I: float) -> float:
    """Tip deflection of a cantilever under uniform load ``w`` [N/mm], in mm.

    ``delta = w L^4 / (8 E I)``. See ``_CITE_CANT_UDL``.
    """
    return float(w) * float(length) ** 4 / (8.0 * float(E) * float(I))


def cantilever_point_load_tip_deflection(force: float, a: float, length: float,
                                         E: float, I: float) -> float:
    """Tip deflection with a point load ``force`` at distance ``a`` from the root.

    ``delta = P a^2 (3 L - a) / (6 E I)`` [mm]. At ``a = L`` this reduces to the
    familiar ``P L^3 / (3 E I)``, which is the property the selfcheck asserts
    against the imported :func:`beam_max_deflection`. See ``_CITE_CANT_MID``.
    """
    a = float(a)
    return (float(force) * a ** 2 * (3.0 * float(length) - a)
            / (6.0 * float(E) * float(I)))


def fixed_fixed_udl_deflection(w: float, length: float, E: float, I: float) -> float:
    """Mid-span NEUTRAL-AXIS deflection, fixed-fixed, uniform load [mm].

    ``delta = w L^4 / (384 E I)``. See ``_CITE_FF_UDL``. This is the deflection
    of the beam AXIS; a 3-D solid model reports the max over its SURFACE, which
    is a different and larger quantity -- the distinction that makes upstream's
    stored 2.63e-4 a golden and not this oracle.
    """
    return float(w) * float(length) ** 4 / (384.0 * float(E) * float(I))


def fixed_fixed_center_load_deflection(force: float, length: float, E: float,
                                       I: float) -> float:
    """Mid-span NEUTRAL-AXIS deflection, fixed-fixed, central point load [mm].

    ``delta = P L^3 / (192 E I)``. See ``_CITE_FF_CTR``.
    """
    return float(force) * float(length) ** 3 / (192.0 * float(E) * float(I))


def clamped_free_first_frequency(E: float, I: float, rho: float, area: float,
                                 length: float) -> float:
    """First bending natural frequency of a clamped-free beam [Hz].

    ``f1 = (beta1^2 / 2 pi) sqrt(E I / (rho A L^4))``. See ``_CITE_MODAL``.
    Units must be consistent: with E in N/mm^2, I in mm^4, rho in t/mm^3, A in
    mm^2 and L in mm the result is in Hz.
    """
    return ((BETA1_L_CLAMPED_FREE ** 2 / (2.0 * math.pi))
            * math.sqrt(float(E) * float(I)
                        / (float(rho) * float(area) * float(length) ** 4)))


def bending_stress_from_moment(moment: float, c: float, I: float) -> float:
    """``sigma = M c / I`` [MPa]. See ``_CITE_FLEXURE``."""
    return float(moment) * float(c) / float(I)


# --------------------------------------------------------------------------- #
# the oracle records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeaOracle:
    """One closed-form answer: what to compute, what it must equal, and why.

    ``value`` is computed by :func:`cases` from the formula in ``formula``; it
    is never a literal. ``tolerance_percent`` is the band a DISCRETISED solver
    is allowed to land inside -- it is upstream's engineering judgement about
    mesh error, carried over as-is, and it is emphatically not a licence for the
    closed form itself to be approximate.

    ``is_oracle`` is the honesty switch. When ``False`` this row is a stored
    solver golden that does NOT equal its own closed form, ``deviation_percent``
    says by how much, and :func:`oracles` refuses to hand it out as truth.
    """

    case_id: str
    metric: str
    description: str
    value: float
    unit: str
    formula: str
    citation: str
    tolerance_percent: float
    analysis_type: str
    #: False when this metric must not gate a verdict (upstream's own call:
    #: peak stress in a coarse hex mesh is not a converged quantity).
    gating: bool = True
    is_oracle: bool = True
    note: str = ""

    def within_tolerance(self, computed: float) -> bool:
        """Is a solver's ``computed`` number inside this case's band?

        The comparison an FEA backend is graded by. Never call this on a
        ``is_oracle=False`` row expecting a truth verdict -- it answers "does
        this match the stored number", which for those two rows is a
        regression question, not a correctness one.
        """
        if self.value == 0.0:
            return computed == 0.0
        return abs(computed - self.value) / abs(self.value) <= (
            self.tolerance_percent / 100.0)


def _strong_beam_cases() -> List[FeaOracle]:
    b = BEAM
    I, c, _Z, _A = b.strong
    L = b.length
    P = 100.0                      # total applied force [N], all beam fixtures
    w = P / L                      # the UDL fixtures spread the same 100 N
    out: List[FeaOracle] = []

    # -- cantilever, end load: BOTH formulas imported from verifiers.simulation
    out.append(FeaOracle(
        "cantilever_end_load", "max_displacement",
        "End-loaded cantilever beam",
        beam_max_deflection(P, L, b.E, I, "cantilever"), "mm",
        "delta = P L^3 / (3 E I)", _CITE_CANT_END, 10.0, "static"))
    out.append(FeaOracle(
        "cantilever_end_load", "max_von_mises_stress",
        "End-loaded cantilever beam",
        beam_bending_stress(P, L, I, c, "cantilever"), "MPa",
        "sigma = M c / I with M = P L", _CITE_CANT_END, 10.0, "static",
        gating=False,
        note="coarse C3D8 bending under-predicts peak stress; upstream gates "
             "on displacement. The closed form is exact regardless."))

    # -- cantilever, uniformly distributed load
    out.append(FeaOracle(
        "cantilever_udl", "max_displacement",
        "Uniformly distributed downward load on cantilever",
        cantilever_udl_deflection(w, L, b.E, I), "mm",
        "delta = w L^4 / (8 E I), w = 1 N/mm", _CITE_CANT_UDL, 10.0, "static"))
    out.append(FeaOracle(
        "cantilever_udl", "max_von_mises_stress",
        "Uniformly distributed downward load on cantilever",
        bending_stress_from_moment(w * L ** 2 / 2.0, c, I), "MPa",
        "sigma = (w L^2 / 2) c / I", _CITE_CANT_UDL, 10.0, "static",
        gating=False,
        note="upstream records that an earlier proposal stated 15 MPa here by "
             "using M = w L^2 (not w L^2 / 2) -- a factor-of-two slip it "
             "caught and corrected to 7.5 MPa. The formula is why it could."))

    # -- cantilever, point load at mid-span
    a = L / 2.0
    out.append(FeaOracle(
        "cantilever_midspan_load", "max_displacement",
        "Cantilever with a point load at mid-span (X=L/2)",
        cantilever_point_load_tip_deflection(P, a, L, b.E, I), "mm",
        "delta_tip = P a^2 (3 L - a) / (6 E I), a = L/2", _CITE_CANT_MID,
        10.0, "static"))
    out.append(FeaOracle(
        "cantilever_midspan_load", "max_von_mises_stress",
        "Cantilever with a point load at mid-span (X=L/2)",
        bending_stress_from_moment(P * a, c, I), "MPa",
        "sigma_root = (P a) c / I", _CITE_CANT_MID, 10.0, "static",
        gating=False,
        note="max stress is at the ROOT, max displacement at the TIP: one "
             "load case, two different places, and a backend can get either "
             "one right while placing the other wrong."))

    # -- fixed-fixed: the two rows whose stored numbers are NOT their formula
    out.append(FeaOracle(
        "fixed_fixed_udl", "max_displacement",
        "Fixed-fixed beam under uniformly distributed downward load",
        fixed_fixed_udl_deflection(w, L, b.E, I), "mm",
        "delta = w L^4 / (384 E I)", _CITE_FF_UDL, 15.0, "static",
        is_oracle=False,
        note="NOT AN ORACLE. This is the neutral-axis closed form; upstream's "
             "stored 2.63e-4 mm is a coarse-mesh solver golden for max "
             "SURFACE displacement and disagrees with it by +41.4%. Upstream "
             "says so in its own note. Two different quantities, so neither "
             "number is 'wrong' -- but only one of them is closed form, and "
             "the stored one must never be graded as ground truth."))
    out.append(FeaOracle(
        "fixed_fixed_udl", "max_von_mises_stress",
        "Fixed-fixed beam under uniformly distributed downward load",
        bending_stress_from_moment(w * L ** 2 / 12.0, c, I), "MPa",
        "sigma = (w L^2 / 12) c / I at the fixed ends", _CITE_FF_UDL,
        10.0, "static", gating=False))
    out.append(FeaOracle(
        "fixed_fixed_center_load", "max_displacement",
        "Fixed-fixed beam with center point load on top face",
        fixed_fixed_center_load_deflection(P, L, b.E, I), "mm",
        "delta = P L^3 / (192 E I)", _CITE_FF_CTR, 15.0, "static",
        is_oracle=False,
        note="NOT AN ORACLE. Same split as fixed_fixed_udl and wider: "
             "upstream's stored 6.57e-4 mm is a point-load solver golden "
             "(local 3-D compliance under the loaded node) and disagrees with "
             "the neutral-axis closed form by +76.6%. Upstream says so."))
    return out


def _weak_axis_cases() -> List[FeaOracle]:
    b = BEAM
    Iw, cw, _Z, _A = b.weak
    L, P = b.length, 100.0
    return [
        FeaOracle(
            "cantilever_end_load_lateral", "max_displacement",
            "End-loaded cantilever bending about the weak (Z) axis (load in -Y)",
            beam_max_deflection(P, L, b.E, Iw, "cantilever"), "mm",
            "delta = P L^3 / (3 E I), I = depth * width^3 / 12 (WEAK axis)",
            _CITE_CANT_END, 12.0, "static",
            note="same beam and same load magnitude as cantilever_end_load, "
                 "turned 90 degrees: I falls 4x and the deflection rises 4x. "
                 "A backend that hard-codes one section property passes one of "
                 "these two and fails the other."),
        FeaOracle(
            "cantilever_end_load_lateral", "max_von_mises_stress",
            "End-loaded cantilever bending about the weak (Z) axis (load in -Y)",
            beam_bending_stress(P, L, Iw, cw, "cantilever"), "MPa",
            "sigma = M c / I with M = P L, weak-axis I and c", _CITE_CANT_END,
            10.0, "static", gating=False),
        FeaOracle(
            "cantilever_modal", "first_natural_frequency_hz",
            "Clamped-free cantilever first bending natural frequency (modal)",
            clamped_free_first_frequency(b.E, Iw, b.rho, b.area, L), "Hz",
            "f1 = (beta1^2 / 2 pi) sqrt(E I / (rho A L^4)), beta1 L = 1.875104",
            _CITE_MODAL, 20.0, "modal",
            note="the fundamental is WEAK-axis bending: the beam is softest "
                 "where I is smallest. verifiers/simulation refuses modal as "
                 "'needs-fea', so this closed form is new to the harness."),
        FeaOracle(
            "column_buckling", "lowest_buckling_factor",
            "Clamped-free slender column linear (Euler) buckling factor",
            euler_critical_load(b.E, Iw, L, "fixed-free") / 1000.0,
            "dimensionless",
            "lambda1 = P_cr / P_ref = pi^2 E I / (K L)^2 / 1000 N, K = 2",
            _CITE_EULER, 20.0, "buckling",
            note="the reference load is 1000 N compressive, so the factor is "
                 "P_cr / 1000. Lowest mode buckles about the WEAK axis."),
    ]


def _rod_cases() -> List[FeaOracle]:
    r = ROD
    A = r.area
    F = 1000.0
    return [
        FeaOracle(
            "tension_rod", "max_displacement", "Axial tension of a square steel rod",
            axial_extension(F, r.length, A, r.E), "mm",
            "delta = F L / (A E)", _CITE_AXIAL, 10.0, "static"),
        FeaOracle(
            "tension_rod", "max_von_mises_stress",
            "Axial tension of a square steel rod",
            F / A, "MPa", "sigma = F / A", _CITE_AXIAL, 10.0, "static",
            gating=False,
            note="upstream degates this: a fully fixed end face creates a "
                 "local triaxial concentration a uniaxial closed form does "
                 "not model. sigma = F/A remains exactly right for the bar."),
    ]


def cases() -> List[FeaOracle]:
    """Every case, oracles and non-oracles alike, in corpus order.

    Pure arithmetic over :data:`BEAM` / :data:`ROD` -- no file is read, so this
    works on a bare wheel with no resources tree and no vendored data.
    """
    order = [c["case_id"] for c in _corpus_order()]
    rows = _rod_cases() + _strong_beam_cases() + _weak_axis_cases()
    index = {cid: i for i, cid in enumerate(order)}
    return sorted(rows, key=lambda r: (index.get(r.case_id, 99), r.metric))


def oracles() -> List[FeaOracle]:
    """Only the rows that ARE their own closed form. The answer key.

    The two ``fixed_fixed_*`` displacement rows are excluded: see the module
    docstring. Grade against this, not against :func:`cases`.
    """
    return [c for c in cases() if c.is_oracle]


def case(case_id: str, metric: str) -> FeaOracle:
    for c in cases():
        if c.case_id == case_id and c.metric == metric:
            return c
    raise KeyError("no such case/metric: %s.%s" % (case_id, metric))


def _corpus_order() -> List[Dict[str, str]]:
    """The upstream corpus's case order, or a hard-coded fallback.

    Reading it keeps this module's ordering pinned to the source it cites; the
    fallback keeps :func:`cases` working when the vendored data is absent (a
    trimmed wheel), because the ORACLES do not depend on any file.
    """
    try:
        raw = json.loads(
            (_data_dir() / "corpus.json").read_text(encoding="utf-8"))
        return list(raw["cases"])
    except (OSError, ValueError, KeyError):
        return [{"case_id": cid} for cid in (
            "tension_rod", "cantilever_end_load", "cantilever_udl",
            "fixed_fixed_udl", "fixed_fixed_center_load",
            "cantilever_midspan_load", "cantilever_end_load_lateral",
            "cantilever_modal", "column_buckling")]


# --------------------------------------------------------------------------- #
# the upstream cross-check
# --------------------------------------------------------------------------- #
def _data_dir() -> Path:
    return Path(__file__).resolve().parent / _SOURCE


def manifest() -> Manifest:
    """The vendored cad-cae-copilot data's SHA-256 manifest."""
    raw = json.loads((_data_dir() / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = tuple(
        FixtureEntry(name=e["name"], role=e["role"], vendored=e.get("vendored"),
                     resource=e.get("resource"), sha256=e["sha256"],
                     bytes=int(e["bytes"]), format=e.get("format", ""))
        for e in raw["entries"])
    return Manifest(source_repo=raw.get("source_repo", ""),
                    source_path=raw.get("source_path", ""),
                    license=raw.get("license", ""),
                    attribution=raw.get("attribution", ""),
                    entries=entries, data_dir=_data_dir())


def upstream_reference(case_id: str) -> Optional[Dict[str, object]]:
    """The stored ``reference.json`` for one case, or ``None`` when absent.

    Degrades to ``None`` rather than raising: the cross-check is a bonus, the
    oracles are the point, and a wheel without the data dir must still compute
    every closed form.
    """
    m = manifest()
    entry = m.by_name(case_id)
    if entry is None:
        return None
    path = m.resolve(entry)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class CrossCheck:
    """One (case, metric) compared against the number upstream stored."""

    case_id: str
    metric: str
    ours: float
    upstream: float
    #: signed (upstream - ours) / ours, in percent.
    deviation_percent: float
    agrees: bool

    def __str__(self) -> str:
        return ("%-28s %-26s ours=%-14.8g upstream=%-14.8g dev=%+8.4f%% %s"
                % (self.case_id, self.metric, self.ours, self.upstream,
                   self.deviation_percent, "OK" if self.agrees else "DISAGREES"))


def crosscheck() -> List[CrossCheck]:
    """Compare every computed oracle against upstream's stored number.

    Empty when the vendored data cannot be read -- and an empty list means "not
    checked", never "checked and clean".
    """
    out: List[CrossCheck] = []
    for c in cases():
        ref = upstream_reference(c.case_id)
        if not isinstance(ref, dict):
            continue
        metrics = ref.get("metrics")
        if not isinstance(metrics, dict) or c.metric not in metrics:
            continue
        stored = float(metrics[c.metric]["value"])
        dev = ((stored - c.value) / c.value * 100.0) if c.value else 0.0
        out.append(CrossCheck(
            c.case_id, c.metric, c.value, stored, dev,
            math.isclose(stored, c.value, rel_tol=AGREEMENT_RTOL)))
    return out


def disagreements() -> List[CrossCheck]:
    """The cross-checks where a stored number is NOT its own formula.

    The whole reason this module reads upstream at all. Expected today: exactly
    the two ``fixed_fixed_*`` displacements, which upstream itself documents as
    coarse-mesh solver goldens rather than closed forms. The four rows where
    upstream rounded ``I`` to 7 significant figures do NOT appear here -- they
    agree to ~2e-8, well inside :data:`AGREEMENT_RTOL`.
    """
    return [x for x in crosscheck() if not x.agrees]


# --------------------------------------------------------------------------- #
# selfcheck: prove the formulas, do not echo the table
# --------------------------------------------------------------------------- #
def _prove_formulas() -> None:
    """Hand-computed cases, worked independently of the code under test.

    Each assertion below is a number a person can check with a calculator from
    the formula in the docstring. This is the part that makes the module an
    oracle: if these fail, the module is wrong, and no amount of agreement with
    a stored table would rescue it.
    """
    # (1) SECTION. b=10, h=20 -> I = 10*20^3/12 = 80000/12 = 6666.666...,
    #     c = 10, A = 200. Weak: I = 20*10^3/12 = 20000/12 = 1666.666...
    I, c, _Z, A = BEAM.strong
    assert math.isclose(I, 20000.0 / 3.0), I          # 80000/12 = 20000/3
    assert c == 10.0 and A == 200.0, (c, A)
    Iw, cw, _Zw, _Aw = BEAM.weak
    assert math.isclose(Iw, 5000.0 / 3.0), Iw          # 20000/12 = 5000/3
    assert cw == 5.0, cw

    # (2) AXIAL. F=1000, L=100, A=100, E=210000.
    #     delta = 1000*100 / (100*210000) = 100000 / 21000000 = 1/210 mm.
    assert math.isclose(axial_extension(1000.0, 100.0, 100.0, 210000.0),
                        1.0 / 210.0), "axial extension != F L / (A E)"

    # (3) CANTILEVER END LOAD. P=100, L=100, E=210000, I=20000/3.
    #     delta = 100 * 1e6 / (3 * 210000 * 20000/3) = 1e8 / 4.2e9 = 1/42 mm.
    #     sigma = (100*100) * 10 / (20000/3) = 10000*30/20000 = 15 MPa exactly.
    assert math.isclose(beam_max_deflection(100.0, 100.0, 210000.0, I,
                                            "cantilever"), 1.0 / 42.0)
    assert math.isclose(beam_bending_stress(100.0, 100.0, I, c, "cantilever"),
                        15.0)

    # (4) UDL. w=1 N/mm, L=100 -> delta = 1e8 / (8*210000*20000/3)
    #     = 1e8 / 1.12e10 = 1/112 mm. sigma = (1*10000/2)*10/(20000/3)
    #     = 5000*30/20000 = 7.5 MPa exactly.
    assert math.isclose(cantilever_udl_deflection(1.0, 100.0, 210000.0, I),
                        1.0 / 112.0)
    assert math.isclose(bending_stress_from_moment(5000.0, c, I), 7.5)

    # (5) MIDSPAN reduces to the end-load form at a = L. That is a PROPERTY, and
    #     it ties the new formula to the one the harness already trusts:
    #     P L^2 (3L - L) / (6 E I) = 2 P L^3 / (6 E I) = P L^3 / (3 E I).
    assert math.isclose(
        cantilever_point_load_tip_deflection(100.0, 100.0, 100.0, 210000.0, I),
        beam_max_deflection(100.0, 100.0, 210000.0, I, "cantilever")), \
        "midspan formula must reduce to the imported end-load formula at a = L"
    #     And at a = L/2: 100 * 2500 * 250 / (6*210000*20000/3)
    #     = 62500000 / 8.4e9 = 0.007440476190...
    assert math.isclose(
        cantilever_point_load_tip_deflection(100.0, 50.0, 100.0, 210000.0, I),
        62500000.0 / 8.4e9)

    # (6) FIXED-FIXED. UDL: 1e8 / (384*210000*20000/3) = 1e8 / 5.376e11.
    #     Centre: 1e8 / (192*210000*20000/3) = 1e8 / 2.688e11. The centre-load
    #     case is exactly TWICE the UDL case for the same total force -- a ratio
    #     384/192 that is independent of E, I and L, so it holds by algebra.
    ff_u = fixed_fixed_udl_deflection(1.0, 100.0, 210000.0, I)
    ff_c = fixed_fixed_center_load_deflection(100.0, 100.0, 210000.0, I)
    assert math.isclose(ff_u, 1e8 / 5.376e11), ff_u
    assert math.isclose(ff_c, 1e8 / 2.688e11), ff_c
    assert math.isclose(ff_c / ff_u, 2.0), "384/192 = 2, independent of E,I,L"

    # (7) MODAL. Hand-worked: E*I = 210000 * 5000/3 = 3.5e8. rho*A*L^4 =
    #     7.85e-9 * 200 * 1e8 = 157.0. ratio = 3.5e8/157 = 2229299.363...,
    #     sqrt = 1493.0838..., beta1^2/(2 pi) = 3.5160150.../6.2831853... =
    #     0.5595919..., product = 835.5165... Hz.
    f1 = clamped_free_first_frequency(210000.0, 5000.0 / 3.0, 7.85e-9, 200.0,
                                      100.0)
    assert math.isclose(f1, (1.875104 ** 2 / (2.0 * math.pi))
                        * math.sqrt(3.5e8 / 157.0)), f1
    assert 835.0 < f1 < 836.0, f1
    #     PROPERTY: f1 scales as 1/L^2. Double the length, quarter the frequency.
    f1_long = clamped_free_first_frequency(210000.0, 5000.0 / 3.0, 7.85e-9,
                                           200.0, 200.0)
    assert math.isclose(f1 / f1_long, 4.0), "f1 must scale as 1 / L^2"

    # (8) EULER. P_cr = pi^2 * 210000 * 5000/3 / 200^2 = pi^2 * 3.5e8 / 40000
    #     = pi^2 * 8750. lambda = P_cr / 1000 = pi^2 * 8.75 = 86.359...
    p_cr = euler_critical_load(210000.0, 5000.0 / 3.0, 100.0, "fixed-free")
    assert math.isclose(p_cr, math.pi ** 2 * 8750.0), p_cr
    assert math.isclose(p_cr / 1000.0, math.pi ** 2 * 8.75)
    assert 86.3 < p_cr / 1000.0 < 86.4, p_cr


def _selfcheck() -> int:
    _prove_formulas()
    print("SELFCHECK: 8 hand-computed proofs pass (section props, axial, "
          "cantilever end/UDL/midspan, fixed-fixed pair + the 384/192 = 2 "
          "identity, modal + its 1/L^2 scaling, Euler)")

    rows = cases()
    assert len(rows) == 15, len(rows)
    oks = oracles()
    assert len(oks) == 13, len(oks)
    for r in rows:
        assert r.value > 0.0, "%s.%s is not positive" % (r.case_id, r.metric)
        assert r.formula and r.citation, "%s.%s is uncited" % (r.case_id, r.metric)
        assert r.tolerance_percent > 0.0
    non = sorted((r.case_id, r.metric) for r in rows if not r.is_oracle)
    assert non == [("fixed_fixed_center_load", "max_displacement"),
                   ("fixed_fixed_udl", "max_displacement")], non
    # within_tolerance must actually bind: an oracle's own value passes, and a
    # value just outside its band fails.
    probe = case("cantilever_end_load", "max_displacement")
    assert probe.within_tolerance(probe.value)
    assert not probe.within_tolerance(probe.value * 1.11)
    assert probe.within_tolerance(probe.value * 1.09)
    print("SELFCHECK: %d cases, %d oracles, 2 non-oracles correctly excluded; "
          "every row cited and positive; tolerance bands bind" % (len(rows), len(oks)))

    # Manifest + vendored bytes.
    m = manifest()
    assert m.license == "MIT", m.license
    problems = m.verify_vendored()
    assert not problems, problems
    avail = m.availability()
    assert avail["total"] == 10, avail
    print("SELFCHECK: MANIFEST %s, %d/%d vendored files present and SHA-256 "
          "verified" % (m.license, avail["vendored"], avail["total"]))

    # The cross-check against upstream's stored numbers.
    checks = crosscheck()
    assert len(checks) == 15, len(checks)
    bad = disagreements()
    for x in checks:
        print("  " + str(x))
    material = [x for x in bad if abs(x.deviation_percent) > 0.001]
    assert len(material) == 2, [str(x) for x in material]
    assert {x.case_id for x in material} == {"fixed_fixed_udl",
                                             "fixed_fixed_center_load"}
    for x in material:
        assert not case(x.case_id, x.metric).is_oracle, (
            "a stored number that disagrees with its own formula by %.2f%% is "
            "flagged is_oracle=True: %s" % (x.deviation_percent, x))
    print("SELFCHECK: cross-check ran on all %d rows. %d rounding-level "
          "disagreements (upstream evaluated I to 7 s.f.); %d MATERIAL "
          "disagreements, both already excluded from oracles()."
          % (len(checks), len(bad) - len(material), len(material)))
    for x in material:
        print("  FLAG: %s.%s stored %.6g vs formula %.6g -> %+.1f%% "
              "(upstream documents it as a coarse-mesh solver golden, not a "
              "closed form)" % (x.case_id, x.metric, x.upstream, x.ours,
                                x.deviation_percent))

    # resources/ is optional: the oracles never touch it.
    root = resources_root()
    print("SELFCHECK OK: resources/ %s; oracles are pure arithmetic and do not "
          "need it" % ("present" if root else "absent"))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Closed-form structural FEA oracles (cantilever, "
                    "fixed-fixed, modal, Euler buckling), cross-checked "
                    "against cad-cae-copilot's stored reference numbers.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove every formula against a hand-computed "
                             "case, verify the vendored manifest, and report "
                             "any stored number that disagrees with its own "
                             "formula.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
