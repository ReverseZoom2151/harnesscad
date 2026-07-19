"""ISO 286 limits-and-fits, ISO 2768 general tolerances, and process capability.

The module provides IT-grade widths, fundamental deviations, zone limits,
hole/shaft fit resolution, general linear and angular tolerances, and
process-capability screening through embedded factual tables.

What this gives the harness
---------------------------

An ISO fit designation splits into two halves: a fundamental-deviation letter
that fixes where the tolerance zone sits relative to the basic size (the ``H``
in ``H7``) and an IT grade number that fixes how wide the zone is (the ``7``).

* :func:`standard_tolerance` resolves the grade half -- the zone width -- for
  IT5 through IT16 over basic sizes up to 500 mm.
* :func:`zone_limits` resolves a full zone (``H7``, ``g6``, ``js6``, ``p6``,
  ``K7``, ...) into signed upper/lower deviations from the basic size.  The
  encoded letters are h/H (basis, zero deviation), the clearance letters
  d/e/f/g (holes via the general rule EI = -es), js/JS (symmetric, +/-IT/2),
  the transition/interference letters k, m, n, p (holes via the special rule
  ES = -ei + delta with delta = IT_n - IT_(n-1)), and the finer-stepped
  interference shafts r/s (exact up to 50 mm) and u (exact up to 18 mm).
* :func:`fit` mates a hole zone with a shaft zone (``"H7/g6"``) into a signed
  clearance range and classifies it clearance / transition / interference.
* :func:`iso2768_linear` and :func:`iso2768_angular` resolve ISO 2768-1
  general tolerances by class (f fine, m medium, c coarse, v very coarse).
* :func:`process_capability`, :func:`tolerance_is_achievable` and
  :func:`processes_that_can_hold` screen a demanded tolerance band against
  per-process finest-achievable floors (DFM screening estimates).

All lengths are millimetres unless a name says otherwise (``*_um`` values are
micrometres, angular deviations are arcminutes).  Pure stdlib, deterministic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ToleranceRangeError",
    "StandardTolerance",
    "standard_tolerance",
    "ZoneLimits",
    "zone_limits",
    "Fit",
    "fit",
    "LinearTolerance",
    "iso2768_linear",
    "AngularTolerance",
    "iso2768_angular",
    "ProcessCapability",
    "process_capability",
    "tolerance_is_achievable",
    "processes_that_can_hold",
    "process_names",
    "main",
]


class ToleranceRangeError(ValueError):
    """A size, grade, zone letter, class or process is outside the encoded tables."""


# ---------------------------------------------------------------------------
# ISO 286-1 standard tolerance grades (IT grades).
#
# Embedded from anvilate iso286_grades.yaml.  Each row covers basic sizes over
# the previous row's bound up to and including up_to_mm; the dict maps the IT
# grade number to the standard tolerance (total zone width) in micrometres.
# Exact tabulated integers, IT5-IT16, over 0 up to and including 500 mm.
# ---------------------------------------------------------------------------

ISO286_MAX_NOMINAL_MM = 500.0

_IT_GRADES: Tuple[Tuple[float, Dict[int, int]], ...] = (
    (3, {5: 4, 6: 6, 7: 10, 8: 14, 9: 25, 10: 40, 11: 60, 12: 100, 13: 140, 14: 250, 15: 400, 16: 600}),
    (6, {5: 5, 6: 8, 7: 12, 8: 18, 9: 30, 10: 48, 11: 75, 12: 120, 13: 180, 14: 300, 15: 480, 16: 750}),
    (10, {5: 6, 6: 9, 7: 15, 8: 22, 9: 36, 10: 58, 11: 90, 12: 150, 13: 220, 14: 360, 15: 580, 16: 900}),
    (18, {5: 8, 6: 11, 7: 18, 8: 27, 9: 43, 10: 70, 11: 110, 12: 180, 13: 270, 14: 430, 15: 700, 16: 1100}),
    (30, {5: 9, 6: 13, 7: 21, 8: 33, 9: 52, 10: 84, 11: 130, 12: 210, 13: 330, 14: 520, 15: 840, 16: 1300}),
    (50, {5: 11, 6: 16, 7: 25, 8: 39, 9: 62, 10: 100, 11: 160, 12: 250, 13: 390, 14: 620, 15: 1000, 16: 1600}),
    (80, {5: 13, 6: 19, 7: 30, 8: 46, 9: 74, 10: 120, 11: 190, 12: 300, 13: 460, 14: 740, 15: 1200, 16: 1900}),
    (120, {5: 15, 6: 22, 7: 35, 8: 54, 9: 87, 10: 140, 11: 220, 12: 350, 13: 540, 14: 870, 15: 1400, 16: 2200}),
    (180, {5: 18, 6: 25, 7: 40, 8: 63, 9: 100, 10: 160, 11: 250, 12: 400, 13: 630, 14: 1000, 15: 1600, 16: 2500}),
    (250, {5: 20, 6: 29, 7: 46, 8: 72, 9: 115, 10: 185, 11: 290, 12: 460, 13: 720, 14: 1150, 15: 1850, 16: 2900}),
    (315, {5: 23, 6: 32, 7: 52, 8: 81, 9: 130, 10: 210, 11: 320, 12: 520, 13: 810, 14: 1300, 15: 2100, 16: 3200}),
    (400, {5: 25, 6: 36, 7: 57, 8: 89, 9: 140, 10: 230, 11: 360, 12: 570, 13: 890, 14: 1400, 15: 2300, 16: 3600}),
    (500, {5: 27, 6: 40, 7: 63, 8: 97, 9: 155, 10: 250, 11: 400, 12: 630, 13: 970, 14: 1550, 15: 2500, 16: 4000}),
)

# ---------------------------------------------------------------------------
# ISO 286-1 fundamental deviations (grade-independent shaft letters).
#
# Embedded from anvilate iso286_deviations.yaml.  Ranges match the IT table.
# "es" maps a clearance shaft letter (d/e/f/g) to its upper deviation in
# micrometres (negative); the uppercase holes D..G mirror it via the general
# rule EI = -es.  "ei" maps a transition/interference shaft letter to its
# lower deviation in micrometres (positive): m/n/p everywhere, k (nonzero
# only for grades IT4-IT7; the code enforces the band), and the finer-stepped
# interference shafts r/s (exact only up to 50 mm) and u (exact only up to
# 18 mm) -- above those bounds the letter is absent and the lookup rejects
# rather than reading a neighbouring step.
# ---------------------------------------------------------------------------

_DEVIATIONS: Tuple[Tuple[float, Dict[str, int], Dict[str, int]], ...] = (
    # (up_to_mm, es, ei)
    (3, {"d": -20, "e": -14, "f": -6, "g": -2}, {"m": 2, "n": 4, "p": 6, "k": 0, "r": 10, "s": 14, "u": 18}),
    (6, {"d": -30, "e": -20, "f": -10, "g": -4}, {"m": 4, "n": 8, "p": 12, "k": 1, "r": 15, "s": 19, "u": 23}),
    (10, {"d": -40, "e": -25, "f": -13, "g": -5}, {"m": 6, "n": 10, "p": 15, "k": 1, "r": 19, "s": 23, "u": 28}),
    (18, {"d": -50, "e": -32, "f": -16, "g": -6}, {"m": 7, "n": 12, "p": 18, "k": 1, "r": 23, "s": 28, "u": 33}),
    (30, {"d": -65, "e": -40, "f": -20, "g": -7}, {"m": 8, "n": 15, "p": 22, "k": 2, "r": 28, "s": 35}),
    (50, {"d": -80, "e": -50, "f": -25, "g": -9}, {"m": 9, "n": 17, "p": 26, "k": 2, "r": 34, "s": 43}),
    (80, {"d": -100, "e": -60, "f": -30, "g": -10}, {"m": 11, "n": 20, "p": 32, "k": 2}),
    (120, {"d": -120, "e": -72, "f": -36, "g": -12}, {"m": 13, "n": 23, "p": 37, "k": 3}),
    (180, {"d": -145, "e": -85, "f": -43, "g": -14}, {"m": 15, "n": 27, "p": 43, "k": 3}),
    (250, {"d": -170, "e": -100, "f": -50, "g": -15}, {"m": 17, "n": 31, "p": 50, "k": 4}),
    (315, {"d": -190, "e": -110, "f": -56, "g": -17}, {"m": 20, "n": 34, "p": 56, "k": 4}),
    (400, {"d": -210, "e": -125, "f": -62, "g": -18}, {"m": 21, "n": 37, "p": 62, "k": 4}),
    (500, {"d": -230, "e": -135, "f": -68, "g": -20}, {"m": 23, "n": 40, "p": 68, "k": 5}),
)

# Letter classification (mirrors anvilate iso286.py).
_BASIS_LETTERS = frozenset({"h"})  # zero fundamental deviation
_CLEARANCE_LETTERS = frozenset({"d", "e", "f", "g"})  # negative shaft es
_SYMMETRIC_LETTERS = frozenset({"js"})  # zone centered on basic size, +/-IT/2
_INTERFERENCE_LETTERS = frozenset({"m", "n", "p"})  # positive shaft ei
_GRADE_DEP_LETTERS = frozenset({"k"})  # ei nonzero only for IT4-IT7
_FINE_SHAFT_LETTERS = frozenset({"r", "s", "u"})  # exact only up to a bound
_ENCODED_LETTERS = (
    _BASIS_LETTERS
    | _CLEARANCE_LETTERS
    | _SYMMETRIC_LETTERS
    | _INTERFERENCE_LETTERS
    | _GRADE_DEP_LETTERS
    | _FINE_SHAFT_LETTERS
)

# The finer-stepped interference shafts are exact across the coarse diameter
# steps only up to a per-letter bound (r/s to 50 mm; u to 18 mm).
_FINE_SHAFT_MAX_MM: Dict[str, float] = {"r": 50.0, "s": 50.0, "u": 18.0}

# The k shaft carries its tabulated ei only for grades IT4 through IT7.
_K_GRADE_BAND = range(4, 8)

# The ISO 286 special rule ES = -ei + delta holds for the M/N holes up to IT8
# and the K/P/R/S/U holes up to IT7; below IT6 the correction would need
# IT_(n-1) beneath the encoded IT5, so IT6 is the finest hole grade resolved.
_HOLE_MAX_GRADE: Dict[str, int] = {"m": 8, "n": 8, "p": 7, "k": 7, "r": 7, "s": 7, "u": 7}
_HOLE_MIN_GRADE = 6

_ISO286_SOURCE = "ISO 286-1 standard tolerance grades and fundamental deviations (via anvilate)"


# ---------------------------------------------------------------------------
# ISO 2768-1 general tolerances.
#
# Embedded from anvilate iso2768_linear.yaml / iso2768_angular.yaml.  Linear:
# permissible +/- deviations in mm by nominal size range and class (f fine,
# m medium, c coarse, v very coarse); None means the class is undefined for
# that range; dimensions below 0.5 mm need an explicit tolerance.  Angular:
# permissible +/- deviations in arcminutes by the shorter leg length of the
# angle; a None bound is the open top range; f and m share values.
# ---------------------------------------------------------------------------

ISO2768_MIN_NOMINAL_MM = 0.5

_ISO2768_LINEAR: Tuple[Tuple[float, Dict[str, Optional[float]]], ...] = (
    (3, {"f": 0.05, "m": 0.10, "c": 0.20, "v": None}),
    (6, {"f": 0.05, "m": 0.10, "c": 0.30, "v": 0.50}),
    (30, {"f": 0.10, "m": 0.20, "c": 0.50, "v": 1.00}),
    (120, {"f": 0.15, "m": 0.30, "c": 0.80, "v": 1.50}),
    (400, {"f": 0.20, "m": 0.50, "c": 1.20, "v": 2.50}),
    (1000, {"f": 0.30, "m": 0.80, "c": 2.00, "v": 4.00}),
    (2000, {"f": 0.50, "m": 1.20, "c": 3.00, "v": 6.00}),
    (4000, {"f": None, "m": 2.00, "c": 4.00, "v": 8.00}),
)

_ISO2768_ANGULAR: Tuple[Tuple[Optional[float], Dict[str, float]], ...] = (
    (10, {"f": 60, "m": 60, "c": 90, "v": 180}),
    (50, {"f": 30, "m": 30, "c": 60, "v": 120}),
    (120, {"f": 20, "m": 20, "c": 30, "v": 60}),
    (400, {"f": 10, "m": 10, "c": 15, "v": 30}),
    (None, {"f": 5, "m": 5, "c": 10, "v": 20}),
)

_ISO2768_CLASSES = ("f", "m", "c", "v")
_ISO2768_CLASS_WORDS = {"fine": "f", "medium": "m", "coarse": "c", "very_coarse": "v"}
_ISO2768_SOURCE = "ISO 2768-1 general tolerances (via anvilate)"


# ---------------------------------------------------------------------------
# Manufacturing process tolerance capability.
#
# Embedded from anvilate process_capability.yaml.  The finest total tolerance
# band (mm, i.e. twice a symmetric +/- deviation) a process can reasonably
# hold under good conditions -- deliberately conservative SCREENING ESTIMATES
# so a check only fails clearly-unachievable tolerances.
# ---------------------------------------------------------------------------

_PROCESS_CAPABILITY: Dict[str, Tuple[float, str]] = {
    # process: (finest total tolerance band mm, note)
    "cnc_milling": (0.05, "precision CNC milling, general feature; tighter needs grinding/reaming"),
    "cnc_turning": (0.025, "precision CNC turning; diametral tolerances hold tighter than milling"),
    "fdm": (0.20, "fused deposition; strongly layer- and orientation-dependent"),
    "sls": (0.20, "selective laser sintering; powder-process shrinkage limits the floor"),
    "sheet_metal": (0.20, "sheet metal features; bend and form tolerances are coarser still"),
    "grinding": (0.005, "precision surface/cylindrical grinding; a finishing floor below milling"),
    "wire_edm": (0.005, "wire EDM; holds tight profile tolerances independent of material hardness"),
    "reaming": (0.01, "reamed holes; a finishing floor for bores below drilled/milled capability"),
    "injection_molding": (0.10, "molded thermoplastic; tighter than FDM but shrinkage- and tool-limited"),
    "die_casting": (0.10, "as-cast metal features; machined features on the casting hold tighter"),
}

_PROCESS_SOURCE = "DFM screening estimates (typical finest achievable tolerances, via anvilate)"


# ---------------------------------------------------------------------------
# ISO 286: standard tolerance (IT-grade width)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardTolerance:
    """A resolved ISO 286-1 standard tolerance: the zone width and its range."""

    nominal_mm: float
    grade: int  # the IT grade number, e.g. 7 for IT7
    width_um: int  # the standard tolerance (total zone width), micrometres
    size_range: str  # the ISO 286-1 range applied, e.g. "over 18 up to 30 mm"

    @property
    def width_mm(self) -> float:
        """The standard tolerance (total zone width) in millimetres."""
        return self.width_um / 1000.0

    @property
    def designation(self) -> str:
        """The IT-grade designation, e.g. ``IT7``."""
        return "IT%d" % self.grade

    def __str__(self) -> str:
        return "%g mm IT%d = %d um" % (self.nominal_mm, self.grade, self.width_um)


def _parse_grade(grade: object) -> int:
    """Parse an IT grade from an int (``7``) or a string (``"IT7"`` / ``"7"``)."""
    if isinstance(grade, int):
        return grade
    token = str(grade).strip().upper()
    if token.startswith("IT"):
        token = token[2:]
    try:
        return int(token)
    except ValueError:
        raise ValueError(
            "unrecognized IT grade %r; expected e.g. 7 or 'IT7'" % (grade,)
        ) from None


def _grade_range_label(low: float, up_to: float, first: bool) -> str:
    # The first range starts above 0; the rest are "over X".
    if first:
        return "up to %g mm" % up_to
    return "over %g up to %g mm" % (low, up_to)


def standard_tolerance(size_mm: float, grade: object) -> StandardTolerance:
    """Resolve the ISO 286-1 standard tolerance (IT grade width) at ``size_mm``.

    ``grade`` is an IT grade as an int (``7``) or a string (``"IT7"`` / ``"7"``).
    Raises :class:`ToleranceRangeError` if the size is at or below zero, beyond
    the table's 500 mm maximum, or the grade is not encoded (only IT5-IT16 are);
    :class:`ValueError` for a malformed grade string.
    """
    it = _parse_grade(grade)
    magnitude = abs(float(size_mm))
    if magnitude <= 0:
        raise ToleranceRangeError("basic size must be greater than 0 mm")
    low = 0.0
    for index, (up_to, widths) in enumerate(_IT_GRADES):
        if magnitude <= up_to:
            if it not in widths:
                grades = ", ".join("IT%d" % g for g in sorted(widths))
                raise ToleranceRangeError(
                    "IT%d is not in the encoded ISO 286-1 table (have %s)" % (it, grades)
                )
            return StandardTolerance(
                nominal_mm=magnitude,
                grade=it,
                width_um=widths[it],
                size_range=_grade_range_label(low, up_to, first=index == 0),
            )
        low = up_to
    raise ToleranceRangeError(
        "%g mm exceeds ISO 286-1's %g mm maximum for this table; "
        "needs an explicit tolerance" % (magnitude, ISO286_MAX_NOMINAL_MM)
    )


# ---------------------------------------------------------------------------
# ISO 286: limit deviations for a tolerance zone
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneLimits:
    """Resolved limit deviations for an ISO 286 tolerance zone.

    ``upper_mm`` and ``lower_mm`` are the signed deviations from the basic size
    (ES/EI for a hole, es/ei for a shaft), in millimetres.  The permitted
    feature size runs from ``nominal + lower`` to ``nominal + upper``.
    """

    nominal_mm: float
    designation: str  # the zone, e.g. "H7" or "g6"
    hole: bool  # True for a hole (uppercase letter), False for a shaft
    grade: int
    lower_mm: float  # EI (hole) / ei (shaft), signed
    upper_mm: float  # ES (hole) / es (shaft), signed
    size_range: str

    @property
    def limits(self) -> Tuple[float, float]:
        """The ``(lower, upper)`` deviation pair in millimetres."""
        return (self.lower_mm, self.upper_mm)

    @property
    def width_mm(self) -> float:
        """The total width of the tolerance zone (``upper - lower``)."""
        return self.upper_mm - self.lower_mm

    @property
    def min_size_mm(self) -> float:
        """The smallest permitted feature size (``nominal + lower``)."""
        return self.nominal_mm + self.lower_mm

    @property
    def max_size_mm(self) -> float:
        """The largest permitted feature size (``nominal + upper``)."""
        return self.nominal_mm + self.upper_mm

    def __str__(self) -> str:
        return "%g mm %s (%+.3f / %+.3f mm)" % (
            self.nominal_mm,
            self.designation,
            self.upper_mm,
            self.lower_mm,
        )


def _parse_designation(designation: str) -> Tuple[str, str]:
    """Split a zone designation into letter(s) and grade, e.g. ``"H7"`` into
    ``("H", "7")``.  Raises :class:`ValueError` if either part is missing."""
    token = designation.strip()
    cut = 0
    while cut < len(token) and token[cut].isalpha():
        cut += 1
    letter, grade = token[:cut], token[cut:]
    if not letter or not grade:
        raise ValueError(
            "malformed ISO 286 zone %r; expected a letter and grade, e.g. 'H7'"
            % (designation,)
        )
    return letter, grade


def _fundamental_dev(key: str, letter: str, nominal_mm: float) -> float:
    """The tabulated fundamental deviation (mm) for a shaft letter.

    ``key`` selects the deviation kind: ``"es"`` (clearance letters, <= 0) or
    ``"ei"`` (transition/interference letters, >= 0).  ``nominal_mm`` is the
    basic size, already validated in range by :func:`standard_tolerance`.
    """
    for up_to, es, ei in _DEVIATIONS:
        if nominal_mm <= up_to:
            table = es if key == "es" else ei
            return table[letter] / 1000.0
    raise AssertionError("nominal beyond deviation table")


def _delta_correction(letter: str, grade: int, nominal_mm: float) -> float:
    """The ISO 286 delta correction (mm), delta = IT_n - IT_(n-1), for a
    special-rule hole.

    The uppercase transition/interference holes K/M/N/P/R/S/U take their
    fundamental deviation from the shaft's ``ei`` by the rule ES = -ei + delta.
    That rule holds for M/N up to IT8 and K/P/R/S/U up to IT7; finer than IT6
    the correction would need IT_(n-1) below the encoded IT5.  Both bounds
    raise :class:`ToleranceRangeError`.
    """
    cap = _HOLE_MAX_GRADE[letter]
    if grade < _HOLE_MIN_GRADE or grade > cap:
        up = letter.upper()
        raise ToleranceRangeError(
            "the delta-corrected hole '%s%d' is out of range; the encoded "
            "ISO 286 special rule covers %s%d through %s%d"
            % (up, grade, up, _HOLE_MIN_GRADE, up, cap)
        )
    it_n = standard_tolerance(nominal_mm, grade).width_mm
    it_prev = standard_tolerance(nominal_mm, grade - 1).width_mm
    return it_n - it_prev


def zone_limits(size_mm: float, designation: str) -> ZoneLimits:
    """Resolve the limit deviations for an ISO 286 tolerance zone at ``size_mm``.

    ``designation`` is a fundamental-deviation letter plus an IT grade, e.g.
    ``"H7"`` (hole), ``"h6"`` (shaft), ``"g6"``, ``"js6"``, or ``"p6"``.  The
    H/h basis zones, the clearance letters d/e/f/g and js/JS resolve in both
    cases; the transition/interference letters m/n/p resolve on the shaft side
    at any grade and on the hole side (M/N/P) via the ISO 286 delta rule for
    M/N up to IT8 and P up to IT7.  The k transition shaft resolves at any
    grade (its grade-banded deviation collapsing to zero outside IT4-IT7), and
    its K hole via the delta rule up to IT7.  The r/s interference zones
    resolve up to 50 mm nominal and the heavier u zone up to 18 mm -- the shaft
    at any grade, the R/S/U hole via the delta rule up to IT7.  Any other
    letter raises :class:`ToleranceRangeError`, as does a delta-corrected hole
    outside its grade band or an r/s/u zone above its exact bound.  Raises
    :class:`ValueError` for a malformed designation.
    """
    letter, grade = _parse_designation(designation)
    base = letter.lower()
    if base not in _ENCODED_LETTERS:
        encoded = ", ".join(sorted(_ENCODED_LETTERS))
        raise ToleranceRangeError(
            "fundamental deviation for zone '%s' is not encoded; the encoded "
            "letters are %s (each with its uppercase hole form)" % (letter, encoded)
        )
    grade_tol = standard_tolerance(size_mm, grade)
    hole = letter.isupper()
    it = grade_tol.width_mm
    nominal_mm = abs(float(size_mm))
    if base in _SYMMETRIC_LETTERS:
        # js/JS: the zone straddles the basic size, es = +IT/2, ei = -IT/2.
        upper_mm, lower_mm = it / 2.0, -it / 2.0
    elif base in _INTERFERENCE_LETTERS:
        ei = _fundamental_dev("ei", base, nominal_mm)
        if hole:
            # Hole (ISO 286 special rule): ES = -ei + delta, EI = ES - IT.
            delta = _delta_correction(base, grade_tol.grade, nominal_mm)
            es = -ei + delta
            upper_mm, lower_mm = es, es - it
        else:
            # Shaft: ei is the lower deviation, es = ei + IT.
            upper_mm, lower_mm = ei + it, ei
    elif base in _GRADE_DEP_LETTERS:
        # k shaft: ei is tabulated only for grades IT4-IT7, else zero.
        if grade_tol.grade in _K_GRADE_BAND:
            ei = _fundamental_dev("ei", base, nominal_mm)
        else:
            ei = 0.0
        if hole:
            # Hole (ISO 286 special rule): ES = -ei + delta, EI = ES - IT.
            # Capped at IT7, so the k shaft's ei is always in its nonzero band.
            delta = _delta_correction(base, grade_tol.grade, nominal_mm)
            es = -ei + delta
            upper_mm, lower_mm = es, es - it
        else:
            upper_mm, lower_mm = ei + it, ei
    elif base in _FINE_SHAFT_LETTERS:
        max_mm = _FINE_SHAFT_MAX_MM[base]
        if nominal_mm > max_mm:
            raise ToleranceRangeError(
                "the %s zone is encoded only up to %g mm, where the coarse "
                "diameter steps are exact; %g mm needs the finer-stepped table"
                % (base, max_mm, nominal_mm)
            )
        ei = _fundamental_dev("ei", base, nominal_mm)
        if hole:
            # Hole (ISO 286 special rule): ES = -ei + delta, EI = ES - IT.
            delta = _delta_correction(base, grade_tol.grade, nominal_mm)
            es = -ei + delta
            upper_mm, lower_mm = es, es - it
        else:
            upper_mm, lower_mm = ei + it, ei
    else:
        if base in _BASIS_LETTERS:
            es = 0.0
        else:
            es = _fundamental_dev("es", base, nominal_mm)
        # Shaft: es is the upper deviation, ei = es - IT.  Hole (general rule):
        # the fundamental deviation EI = -es, and ES = EI + IT.
        if hole:
            upper_mm, lower_mm = -es + it, -es
        else:
            upper_mm, lower_mm = es, es - it
    # Adding 0.0 collapses the -0.0 that -es yields when es == 0 (the H/h
    # basis), so a zero deviation renders "+0.000", not "-0.000".
    return ZoneLimits(
        nominal_mm=nominal_mm,
        designation="%s%d" % (letter, grade_tol.grade),
        hole=hole,
        grade=grade_tol.grade,
        lower_mm=lower_mm + 0.0,
        upper_mm=upper_mm + 0.0,
        size_range=grade_tol.size_range,
    )


# ---------------------------------------------------------------------------
# ISO 286: fits (a hole zone mated with a shaft zone)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    """A resolved ISO 286 fit: a hole zone mated with a shaft zone.

    Clearance is measured as hole size minus shaft size, so a positive value
    is a gap and a negative value is interference.  ``min_clearance_mm`` pairs
    the smallest hole with the largest shaft; ``max_clearance_mm`` the largest
    hole with the smallest shaft.  ``kind`` is ``"clearance"`` when even the
    tightest pairing leaves a gap, ``"interference"`` when even the loosest
    pairing interferes, and ``"transition"`` otherwise.
    """

    nominal_mm: float
    designation: str  # e.g. "H7/g6"
    hole: ZoneLimits
    shaft: ZoneLimits
    min_clearance_mm: float  # signed; negative means interference
    max_clearance_mm: float  # signed
    kind: str  # "clearance" | "transition" | "interference"

    def satisfies_clearance(self, min_required_mm: float, max_required_mm: float) -> bool:
        """Whether the fit's whole clearance range falls within a required band.

        Passes only when the worst cases both fit: the tightest clearance
        (``min_clearance_mm``) is at least ``min_required_mm`` and the loosest
        (``max_clearance_mm``) is at most ``max_required_mm``.
        """
        return (
            min_required_mm <= self.min_clearance_mm
            and self.max_clearance_mm <= max_required_mm
        )

    def __str__(self) -> str:
        return "%g mm %s %s (%+.3f to %+.3f mm)" % (
            self.nominal_mm,
            self.designation,
            self.kind,
            self.min_clearance_mm,
            self.max_clearance_mm,
        )


def fit(designation: str, size_mm: float) -> Fit:
    """Resolve an ISO 286 fit (e.g. ``"H7/g6"``) at ``size_mm`` into its
    clearance range and kind.

    ``designation`` is a hole zone and a shaft zone separated by ``/``.  Both
    zones must resolve (see :func:`zone_limits`).  Raises :class:`ValueError`
    if the designation is malformed or the hole/shaft roles are swapped, and
    propagates :class:`ToleranceRangeError` from the zone lookups.
    """
    parts = [p for p in designation.split("/") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            "malformed fit %r; expected a hole and shaft zone, e.g. 'H7/h6'"
            % (designation,)
        )
    hole = zone_limits(size_mm, parts[0].strip())
    shaft = zone_limits(size_mm, parts[1].strip())
    if not hole.hole or shaft.hole:
        raise ValueError(
            "fit %r must be hole/shaft (uppercase then lowercase), e.g. 'H7/h6'"
            % (designation,)
        )
    min_clearance = hole.lower_mm - shaft.upper_mm
    max_clearance = hole.upper_mm - shaft.lower_mm
    if min_clearance >= 0:
        kind = "clearance"
    elif max_clearance <= 0:
        kind = "interference"
    else:
        kind = "transition"
    return Fit(
        nominal_mm=abs(float(size_mm)),
        designation="%s/%s" % (hole.designation, shaft.designation),
        hole=hole,
        shaft=shaft,
        min_clearance_mm=min_clearance + 0.0,
        max_clearance_mm=max_clearance + 0.0,
        kind=kind,
    )


# ---------------------------------------------------------------------------
# ISO 2768-1 general tolerances
# ---------------------------------------------------------------------------


def _parse_class(tolerance_class: str) -> str:
    """Normalise a class token: a letter (``"m"``) or word (``"medium"``)."""
    token = tolerance_class.strip().lower()
    if token in _ISO2768_CLASSES:
        return token
    if token in _ISO2768_CLASS_WORDS:
        return _ISO2768_CLASS_WORDS[token]
    raise ValueError(
        "unrecognized ISO 2768 tolerance class %r; expected one of f, m, c, v "
        "(or fine, medium, coarse, very_coarse)" % (tolerance_class,)
    )


@dataclass(frozen=True)
class LinearTolerance:
    """A resolved ISO 2768-1 general linear tolerance: the permissible
    +/- deviation (mm) and the size range applied."""

    nominal_mm: float
    tolerance_class: str  # the class letter: f, m, c or v
    deviation_mm: float  # the permissible +/- deviation
    size_range: str  # e.g. "over 30 up to 120 mm"

    @property
    def min_size_mm(self) -> float:
        """The smallest permitted feature size (``nominal - deviation``)."""
        return self.nominal_mm - self.deviation_mm

    @property
    def max_size_mm(self) -> float:
        """The largest permitted feature size (``nominal + deviation``)."""
        return self.nominal_mm + self.deviation_mm

    def __str__(self) -> str:
        return "%g mm +/-%g mm (ISO 2768 %s)" % (
            self.nominal_mm,
            self.deviation_mm,
            self.tolerance_class,
        )


def iso2768_linear(size_mm: float, tolerance_class: str = "m") -> LinearTolerance:
    """Resolve the ISO 2768-1 general tolerance for a linear dimension.

    ``tolerance_class`` is a letter (``"f"``/``"m"``/``"c"``/``"v"``) or word
    (``"fine"``/``"medium"``/``"coarse"``/``"very_coarse"``); the ISO 2768
    default is medium.  Raises :class:`ToleranceRangeError` if the dimension is
    below the table's 0.5 mm minimum (needs an explicit tolerance), beyond its
    4000 mm maximum, or the class is undefined for the matched range; and
    :class:`ValueError` for an unrecognized class.
    """
    cls = _parse_class(tolerance_class)
    magnitude = abs(float(size_mm))
    if magnitude < ISO2768_MIN_NOMINAL_MM:
        raise ToleranceRangeError(
            "%g mm is below ISO 2768-1's %g mm minimum; dimensions this small "
            "need an explicit tolerance" % (magnitude, ISO2768_MIN_NOMINAL_MM)
        )
    low = ISO2768_MIN_NOMINAL_MM
    for index, (up_to, deviations) in enumerate(_ISO2768_LINEAR):
        if magnitude <= up_to:
            if index == 0:
                label = "%g up to %g mm" % (low, up_to)
            else:
                label = "over %g up to %g mm" % (low, up_to)
            deviation = deviations[cls]
            if deviation is None:
                raise ToleranceRangeError(
                    "ISO 2768-1 class %s is not defined for the %s range" % (cls, label)
                )
            return LinearTolerance(
                nominal_mm=magnitude,
                tolerance_class=cls,
                deviation_mm=deviation,
                size_range=label,
            )
        low = up_to
    raise ToleranceRangeError(
        "%g mm exceeds ISO 2768-1's %g mm maximum; needs an explicit tolerance"
        % (magnitude, low)
    )


@dataclass(frozen=True)
class AngularTolerance:
    """A resolved ISO 2768-1 general angular tolerance: the permissible
    +/- deviation in arcminutes, keyed by the shorter leg length."""

    shorter_leg_mm: float
    tolerance_class: str  # the class letter: f, m, c or v
    deviation_arcmin: float  # the permissible +/- deviation, arcminutes
    leg_range: str  # the shorter-leg length range applied

    @property
    def deviation_degrees(self) -> float:
        """The permissible +/- deviation in decimal degrees."""
        return self.deviation_arcmin / 60.0

    def __str__(self) -> str:
        return "+/-%g arcmin (ISO 2768 %s)" % (self.deviation_arcmin, self.tolerance_class)


def iso2768_angular(shorter_leg_mm: float, tolerance_class: str = "m") -> AngularTolerance:
    """Resolve the ISO 2768-1 general angular tolerance for an angle whose
    shorter leg is ``shorter_leg_mm`` long, under a class (default medium).

    The deviation is returned in arcminutes (fine and medium share the same
    angular values in ISO 2768-1).  Raises :class:`ValueError` for an
    unrecognized class.  The table's top range is open, so every leg length
    resolves.
    """
    cls = _parse_class(tolerance_class)
    magnitude = abs(float(shorter_leg_mm))
    low = 0.0
    for index, (up_to, deviations) in enumerate(_ISO2768_ANGULAR):
        if up_to is None or magnitude <= up_to:
            if up_to is None:
                label = "over %g mm shorter leg" % low
            elif index == 0:
                label = "up to %g mm shorter leg" % up_to
            else:
                label = "over %g up to %g mm shorter leg" % (low, up_to)
            return AngularTolerance(
                shorter_leg_mm=magnitude,
                tolerance_class=cls,
                deviation_arcmin=float(deviations[cls]),
                leg_range=label,
            )
        low = up_to
    raise ToleranceRangeError("no ISO 2768-1 angular range matched")  # unreachable


# ---------------------------------------------------------------------------
# Process capability screening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessCapability:
    """The finest total tolerance band a process can reasonably hold (a
    screening estimate), with a caveat note."""

    process: str
    finest_tolerance_mm: float  # finest achievable total tolerance band
    note: str

    def __str__(self) -> str:
        return "%s: %.3f mm finest band (%s)" % (
            self.process,
            self.finest_tolerance_mm,
            self.note,
        )


def process_capability(process: str) -> ProcessCapability:
    """The finest-achievable tolerance floor for ``process`` (a screening
    estimate).

    ``process`` is a manufacturing-process name, e.g. ``"cnc_milling"``.
    Raises :class:`ToleranceRangeError` if the process has no capability
    record.
    """
    row = _PROCESS_CAPABILITY.get(process)
    if row is None:
        known = ", ".join(sorted(_PROCESS_CAPABILITY))
        raise ToleranceRangeError(
            "no tolerance-capability record for process %r; have %s" % (process, known)
        )
    finest, note = row
    return ProcessCapability(process=process, finest_tolerance_mm=finest, note=note)


def tolerance_is_achievable(process: str, demanded_width_mm: float) -> bool:
    """Screen a demanded total tolerance band against ``process``'s floor.

    ``demanded_width_mm`` is the total tolerance band (a symmetric +/-0.05 mm
    is a 0.1 mm band).  True when the band is at least the process's finest
    achievable band.  The floor is a screening estimate that varies by machine
    and setup, so False means "clearly unachievable", not a hard limit.
    """
    cap = process_capability(process)
    return demanded_width_mm >= cap.finest_tolerance_mm


def processes_that_can_hold(demanded_width_mm: float) -> List[str]:
    """The known processes whose finest-achievable floor can hold a band.

    Ordered coarsest-capable first -- the least-precise (typically most
    economical) process whose floor still meets the band comes first.  Empty
    when no process can hold the band, i.e. the tolerance must be relaxed.
    """
    holders = [
        p for p, (finest, _note) in _PROCESS_CAPABILITY.items()
        if finest <= demanded_width_mm
    ]
    return sorted(holders, key=lambda p: _PROCESS_CAPABILITY[p][0], reverse=True)


def process_names() -> List[str]:
    """Sorted list of known process-capability record names."""
    return sorted(_PROCESS_CAPABILITY)


# ---------------------------------------------------------------------------
# CLI / selfcheck
# ---------------------------------------------------------------------------


def _isclose(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _selfcheck() -> int:
    # IT-grade widths (exact tabulated values).
    assert standard_tolerance(25.0, 7).width_um == 21
    assert standard_tolerance(25.0, "IT7").width_um == 21
    assert standard_tolerance(0.5, 5).width_um == 4
    assert standard_tolerance(3.0, 6).width_um == 6  # boundary is inclusive
    assert standard_tolerance(3.001, 6).width_um == 8
    assert standard_tolerance(500.0, 16).width_um == 4000
    assert standard_tolerance(100.0, 11).width_um == 220

    # Basis zones: H7 hole and h6 shaft at 25 mm.
    h7 = zone_limits(25.0, "H7")
    assert h7.hole and h7.limits == (0.0, 0.021)
    h6 = zone_limits(25.0, "h6")
    assert not h6.hole and _isclose(h6.lower_mm, -0.013) and h6.upper_mm == 0.0

    # Clearance shaft g6 at 25 mm: es = -7 um, ei = -20 um.
    g6 = zone_limits(25.0, "g6")
    assert _isclose(g6.upper_mm, -0.007) and _isclose(g6.lower_mm, -0.020)
    # Uppercase mirror (general rule EI = -es): G7 at 25 mm is +7/+28 um.
    g7_hole = zone_limits(25.0, "G7")
    assert _isclose(g7_hole.lower_mm, 0.007) and _isclose(g7_hole.upper_mm, 0.028)

    # Symmetric js6 at 25 mm: +/- IT6/2 = +/- 6.5 um.
    js6 = zone_limits(25.0, "js6")
    assert _isclose(js6.upper_mm, 0.0065) and _isclose(js6.lower_mm, -0.0065)

    # Delta rule holes at 25 mm (delta = IT7 - IT6 = 21 - 13 = 8 um):
    # K7 = +6/-15 um, N7 = -7/-28 um (textbook values).
    k7 = zone_limits(25.0, "K7")
    assert _isclose(k7.upper_mm, 0.006) and _isclose(k7.lower_mm, -0.015)
    n7 = zone_limits(25.0, "N7")
    assert _isclose(n7.upper_mm, -0.007) and _isclose(n7.lower_mm, -0.028)

    # k6 shaft at 25 mm (grade in the IT4-IT7 band): +2/+15 um.
    k6 = zone_limits(25.0, "k6")
    assert _isclose(k6.lower_mm, 0.002) and _isclose(k6.upper_mm, 0.015)
    # k9 shaft: outside the band, deviation collapses to zero: 0/+52 um.
    k9 = zone_limits(25.0, "k9")
    assert k9.lower_mm == 0.0 and _isclose(k9.upper_mm, 0.052)

    # Fine-stepped interference shaft s6 at 25 mm: +35/+48 um.
    s6 = zone_limits(25.0, "s6")
    assert _isclose(s6.lower_mm, 0.035) and _isclose(s6.upper_mm, 0.048)
    # ...and rejected above its 50 mm exact bound.
    try:
        zone_limits(60.0, "s6")
    except ToleranceRangeError:
        pass
    else:
        raise AssertionError("s6 above 50 mm should be rejected")

    # H7/g6 at 25 mm: the textbook close-running clearance fit, +7 to +41 um.
    running = fit("H7/g6", 25.0)
    assert running.kind == "clearance"
    assert _isclose(running.min_clearance_mm, 0.007)
    assert _isclose(running.max_clearance_mm, 0.041)

    # H7/p6 at 25 mm: press fit, -35 to -1 um interference.
    press = fit("H7/p6", 25.0)
    assert press.kind == "interference"
    assert _isclose(press.min_clearance_mm, -0.035)
    assert _isclose(press.max_clearance_mm, -0.001)

    # H7/k6 at 25 mm: transition fit, -15 to +19 um.
    trans = fit("H7/k6", 25.0)
    assert trans.kind == "transition"
    assert _isclose(trans.min_clearance_mm, -0.015)
    assert _isclose(trans.max_clearance_mm, 0.019)
    assert trans.satisfies_clearance(-0.02, 0.02)
    assert not trans.satisfies_clearance(0.0, 0.02)

    # ISO 2768-1 linear: medium class at 50 mm is +/-0.3 mm.
    lin = iso2768_linear(50.0, "m")
    assert _isclose(lin.deviation_mm, 0.30)
    assert _isclose(iso2768_linear(10.0, "fine").deviation_mm, 0.10)
    assert _isclose(iso2768_linear(3000.0, "v").deviation_mm, 8.00)
    # f is undefined above 2000 mm; below 0.5 mm needs an explicit tolerance.
    try:
        iso2768_linear(3000.0, "f")
    except ToleranceRangeError:
        pass
    else:
        raise AssertionError("2768-f above 2000 mm should be rejected")
    try:
        iso2768_linear(0.3, "m")
    except ToleranceRangeError:
        pass
    else:
        raise AssertionError("2768 below 0.5 mm should be rejected")

    # ISO 2768-1 angular: medium at a 25 mm shorter leg is +/-30 arcmin.
    ang = iso2768_angular(25.0, "m")
    assert _isclose(ang.deviation_arcmin, 30.0)
    assert _isclose(ang.deviation_degrees, 0.5)
    assert _isclose(iso2768_angular(5000.0, "c").deviation_arcmin, 10.0)  # open top

    # Process capability screening.
    cap = process_capability("cnc_milling")
    assert _isclose(cap.finest_tolerance_mm, 0.05)
    assert tolerance_is_achievable("cnc_milling", 0.1)
    assert not tolerance_is_achievable("fdm", 0.1)
    holders = processes_that_can_hold(0.05)
    assert holders[-1] in ("grinding", "wire_edm")  # finest-capable last
    assert "fdm" not in holders
    assert len(process_names()) == 10

    print("iso286_fits selfcheck OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="iso286_fits",
        description="ISO 286 limits-and-fits, ISO 2768 general tolerances and "
        "process-capability lookups.",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run the built-in assertions against known-good values and exit",
    )
    parser.add_argument(
        "--fit",
        metavar="DESIGNATION",
        help="resolve a fit, e.g. H7/g6 (requires --size)",
    )
    parser.add_argument(
        "--zone",
        metavar="ZONE",
        help="resolve a tolerance zone, e.g. H7 or g6 (requires --size)",
    )
    parser.add_argument(
        "--size",
        type=float,
        metavar="MM",
        help="basic size in mm for --fit / --zone",
    )
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    if args.fit or args.zone:
        if args.size is None:
            parser.error("--fit / --zone require --size MM")
        if args.fit:
            print(fit(args.fit, args.size))
        if args.zone:
            print(zone_limits(args.size, args.zone))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
