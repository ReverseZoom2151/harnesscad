"""Engineering-standards critic — a standalone advisory verifier.

The blueprint's Contract / plural-verifier machinery (sec.6, sec.12, sec.21) is
not only about geometric validity: a part can be perfectly valid B-rep and still
be *needlessly hard to source* because its dimensions ignore the standard values
real shops actually stock. This verifier encodes that engineering-standards
layer:

  * **Hole diameters** should match a standard drill or tap/clearance size — an
    arbitrary Ø7.3 hole means a custom reamer where a Ø7.5 (or a tapped M8) would
    have done.
  * **Key linear dimensions and fillet radii** should fall on an ISO
    preferred-number series (Renard R10/R20) or a standard fillet-radius / sheet
    thickness — arbitrary dimensions inflate tooling and inspection cost.

Like :class:`checks_dfm.DFMCheck`, every finding here is *advisory*: a
non-standard dimension is still buildable, just less economical. So every finding
is a WARNING (with the nearest standard value suggested) and every unmeasurable
case is an INFO skip — this verifier **never** emits an ERROR and can never flip
a :class:`verify.VerifyReport` to ``ok == False``.

Standalone by design, exactly like :class:`checks_geometry.BRepValidityCheck` and
:class:`checks_dfm.DFMCheck`: it is NOT wired into
:func:`verify.default_verifiers` (that would be a circular import, and this is an
opt-in stage). A caller adds it explicitly via :func:`with_standards`.

What is inspectable *today*: hole diameters and key dimensions are read straight
off the op stream (``AddCircle`` carries a radius, ``Fillet`` a radius,
``Extrude`` a distance, ``AddRectangle`` w/h) via the ops-DAG, and — when a
backend exposes it — an optional ``query('metrics')`` may report a ``holes`` list
of measured diameters. When neither the ops nor a metrics query are available the
relevant check INFO-skips rather than crashing or erroring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Standard-value tables
# --------------------------------------------------------------------------- #
def _renard(factor_root: int, lo: float = 1.0, hi: float = 1000.0) -> List[float]:
    """Generate an ISO 3 / Renard preferred-number series across decades.

    ``factor_root`` is the series index (10 -> R10, 20 -> R20): step is the
    ``factor_root``-th root of 10, values rounded to the conventional preferred
    numbers. Spans ``[lo, hi]`` (mm) so it is directly usable for dimensions.
    """
    values: List[float] = []
    n = factor_root
    # Preferred numbers are decade-repeating; build one decade then scale.
    base_decade = [round(10.0 ** (i / n), 4) for i in range(n)]
    decade = lo
    while decade <= hi + 1e-9:
        for b in base_decade:
            v = round(b * decade, 4)
            if lo - 1e-9 <= v <= hi + 1e-9:
                values.append(v)
        decade *= 10.0
    # Deduplicate + sort.
    return sorted(set(values))


# Conventional rounded Renard values for a single decade (the canonical ISO
# preferred numbers) — used verbatim for the 1..10 decade so the series matches
# the textbook values (e.g. 1.25, 3.15, 6.3) rather than raw roots.
_R10_DECADE = [1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0]
_R20_DECADE = [1.0, 1.12, 1.25, 1.4, 1.6, 1.8, 2.0, 2.24, 2.5, 2.8,
               3.15, 3.55, 4.0, 4.5, 5.0, 5.6, 6.3, 7.1, 8.0, 9.0]


def _preferred_series(decade_values: Sequence[float],
                      lo: float = 0.1, hi: float = 1000.0) -> List[float]:
    """Scale a canonical single-decade preferred-number list across decades."""
    values: List[float] = []
    scale = 0.1
    while scale <= hi + 1e-9:
        for b in decade_values:
            v = round(b * scale, 4)
            if lo - 1e-9 <= v <= hi + 1e-9:
                values.append(v)
        scale *= 10.0
    return sorted(set(values))


# Standard metric jobber-drill diameters (mm), a common preferred subset. Note
# the deliberate 7.0 -> 7.5 -> 8.0 spacing (no 7.3): an arbitrary Ø7.3 is *not*
# a standard drill and should be flagged toward Ø7.5.
_STD_DRILL_MM = [
    1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
    7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 13.0,
    14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 25.0,
]

# Standard metric coarse tap / nominal thread diameters (mm).
_STD_TAP_MM = [1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0,
               16.0, 20.0, 24.0]

# Standard sheet-metal thicknesses (mm).
_STD_SHEET_MM = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]

# Standard fillet / rounding radii (mm).
_STD_FILLET_MM = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


# --------------------------------------------------------------------------- #
# nearest-standard helper
# --------------------------------------------------------------------------- #
def nearest_standard(value: float, series: Sequence[float]) -> Optional[float]:
    """Return the entry in ``series`` closest to ``value`` (None if empty)."""
    best: Optional[float] = None
    best_d = None
    for s in series:
        d = abs(float(s) - float(value))
        if best_d is None or d < best_d:
            best_d = d
            best = float(s)
    return best


def _matches(value: float, series: Sequence[float],
             abs_tol: float, rel_tol: float) -> Optional[float]:
    """Return the nearest standard when ``value`` is within tolerance, else None.

    Tolerance is ``max(abs_tol, rel_tol * value)`` so both tiny and large
    dimensions get a sensible band.
    """
    near = nearest_standard(value, series)
    if near is None:
        return None
    tol = max(abs_tol, rel_tol * abs(float(value)))
    return near if abs(near - float(value)) <= tol else None


# --------------------------------------------------------------------------- #
# Rules (configurable standard-value tables + match tolerance)
# --------------------------------------------------------------------------- #
@dataclass
class StandardsRules:
    """Configurable standard-value tables for the standards critic.

    Every table is a list of allowed values in millimetres; a dimension is
    "standard" when it lands within ``abs_tol``/``rel_tol`` of a table entry.
    Defaults are common metric engineering values — a caller overrides them per
    shop / material / process. All findings are advisory (WARNING), never ERROR.
    """

    drill_sizes: List[float] = field(default_factory=lambda: list(_STD_DRILL_MM))
    tap_sizes: List[float] = field(default_factory=lambda: list(_STD_TAP_MM))
    preferred_r10: List[float] = field(default_factory=lambda: _preferred_series(_R10_DECADE))
    preferred_r20: List[float] = field(default_factory=lambda: _preferred_series(_R20_DECADE))
    sheet_thicknesses: List[float] = field(default_factory=lambda: list(_STD_SHEET_MM))
    fillet_radii: List[float] = field(default_factory=lambda: list(_STD_FILLET_MM))

    # Match tolerance: a value counts as standard within max(abs_tol, rel_tol*v).
    abs_tol: float = 0.05        # mm
    rel_tol: float = 0.01        # fraction of the value

    # -- derived views ------------------------------------------------------
    def standard_hole_sizes(self) -> List[float]:
        """Union of drill and tap diameters — the acceptable hole diameters."""
        return sorted(set(list(self.drill_sizes) + list(self.tap_sizes)))

    def preferred_numbers(self) -> List[float]:
        """The finer R20 series (a superset of R10) for linear dimensions."""
        return sorted(set(list(self.preferred_r20)))

    def to_dict(self) -> dict:
        return {
            "drill_sizes": list(self.drill_sizes),
            "tap_sizes": list(self.tap_sizes),
            "preferred_r10": list(self.preferred_r10),
            "preferred_r20": list(self.preferred_r20),
            "sheet_thicknesses": list(self.sheet_thicknesses),
            "fillet_radii": list(self.fillet_radii),
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "StandardsRules":
        d = d or {}
        defaults = cls()

        def _floats(key, fallback):
            v = d.get(key)
            if v is None:
                return list(fallback)
            return [float(x) for x in v]

        return cls(
            drill_sizes=_floats("drill_sizes", defaults.drill_sizes),
            tap_sizes=_floats("tap_sizes", defaults.tap_sizes),
            preferred_r10=_floats("preferred_r10", defaults.preferred_r10),
            preferred_r20=_floats("preferred_r20", defaults.preferred_r20),
            sheet_thicknesses=_floats("sheet_thicknesses", defaults.sheet_thicknesses),
            fillet_radii=_floats("fillet_radii", defaults.fillet_radii),
            abs_tol=float(d.get("abs_tol", defaults.abs_tol)),
            rel_tol=float(d.get("rel_tol", defaults.rel_tol)),
        )


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class StandardsCheck:
    """A :class:`verify.Verifier` that flags non-standard sizes (advisory only).

    ``check(backend, opdag)`` reads hole diameters and key dimensions from the
    op stream (and, when present, ``query('metrics')['holes']``) and returns a
    :class:`verify.VerifyReport`. Each non-standard value is a WARNING carrying
    the nearest standard suggestion; each unmeasurable case is an INFO skip — so
    this verifier NEVER emits an ERROR.

    Codes emitted:
      * WARNING ``non-standard-hole``      — hole Ø not a standard drill/tap size.
      * WARNING ``non-standard-fillet``    — fillet radius not a standard radius.
      * WARNING ``non-preferred-dimension``— linear dim off the preferred-number series.
      * INFO    ``standards-skipped``      — nothing inspectable (no ops/metrics).
    """

    name = "standards"

    def __init__(self, rules: Optional[StandardsRules] = None) -> None:
        self.rules = rules or StandardsRules()

    def check(self, backend, opdag) -> VerifyReport:
        diags: List[Diagnostic] = []
        ops = _iter_ops(opdag)
        metric_holes = _metric_holes(backend)

        if not ops and not metric_holes:
            diags.append(_info(
                "standards-skipped",
                "no ops or 'metrics' holes available to inspect for standard "
                "sizes (backend/op-DAG did not expose measurable dimensions)."))
            return VerifyReport(diags)

        self._check_holes(ops, metric_holes, diags)
        self._check_dimensions(ops, diags)
        return VerifyReport(diags)

    # -- hole diameters -----------------------------------------------------
    def _check_holes(self, ops, metric_holes, diags: List[Diagnostic]) -> None:
        r = self.rules
        std = r.standard_hole_sizes()
        holes: List[tuple] = []
        for idx, op in enumerate(ops):
            if type(op).__name__ == "AddCircle":
                dia = 2.0 * float(getattr(op, "r", 0.0))
                if dia > 0:
                    holes.append((dia, f"circle#{idx}"))
        for i, dia in enumerate(metric_holes):
            holes.append((float(dia), f"metrics.holes[{i}]"))

        for dia, where in holes:
            if _matches(dia, std, r.abs_tol, r.rel_tol) is None:
                near = nearest_standard(dia, std)
                diags.append(_warn(
                    "non-standard-hole",
                    f"hole diameter {dia:g} mm is not a standard drill/tap size; "
                    f"nearest standard is {near:g} mm.",
                    where=where))

    # -- key dimensions -----------------------------------------------------
    def _check_dimensions(self, ops, diags: List[Diagnostic]) -> None:
        r = self.rules
        preferred = r.preferred_numbers()
        for idx, op in enumerate(ops):
            name = type(op).__name__
            if name == "Fillet":
                rad = float(getattr(op, "radius", 0.0))
                if rad > 0 and _matches(rad, r.fillet_radii, r.abs_tol, r.rel_tol) is None:
                    near = nearest_standard(rad, r.fillet_radii)
                    diags.append(_warn(
                        "non-standard-fillet",
                        f"fillet radius {rad:g} mm is not a standard radius; "
                        f"nearest standard is {near:g} mm.",
                        where=f"fillet#{idx}"))
            elif name == "Extrude":
                dist = abs(float(getattr(op, "distance", 0.0)))
                self._flag_dim(dist, preferred, f"extrude#{idx}", diags)
            elif name == "AddRectangle":
                for axis, key in (("w", "w"), ("h", "h")):
                    val = abs(float(getattr(op, key, 0.0)))
                    self._flag_dim(val, preferred, f"rect#{idx}.{axis}", diags)

    def _flag_dim(self, value: float, preferred, where: str,
                  diags: List[Diagnostic]) -> None:
        r = self.rules
        if value <= 0:
            return
        if _matches(value, preferred, r.abs_tol, r.rel_tol) is None:
            near = nearest_standard(value, preferred)
            diags.append(_warn(
                "non-preferred-dimension",
                f"dimension {value:g} mm is not on the ISO preferred-number "
                f"series; nearest preferred value is {near:g} mm.",
                where=where))


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_standards(verifiers, rules: Optional[StandardsRules] = None) -> List:
    """Return a new verifier list with a :class:`StandardsCheck` appended.

    Mirrors :func:`checks_dfm.with_dfm` / how the standalone geometry check is
    added to the default set without editing ``verify.py``::

        from harnesscad.eval.verifiers.verify import default_verifiers
        from harnesscad.eval.verifiers.standards import with_standards
        verifiers = with_standards(default_verifiers())
    """
    return list(verifiers) + [StandardsCheck(rules)]


# --------------------------------------------------------------------------- #
# Helpers (mirror checks_dfm's graceful-degradation conventions)
# --------------------------------------------------------------------------- #
def _iter_ops(opdag) -> list:
    """Best-effort extraction of the op list from whatever ``opdag`` is.

    Accepts an :class:`state.opdag.OpDAG` (has ``ops()``), a plain list/tuple of
    ops, or None — always returns a list, never raises.
    """
    if opdag is None:
        return []
    ops_attr = getattr(opdag, "ops", None)
    if callable(ops_attr):
        try:
            return list(ops_attr())
        except Exception:  # noqa: BLE001 - degrade, never crash the verifier
            return []
    if isinstance(opdag, (list, tuple)):
        return list(opdag)
    return []


def _metric_holes(backend) -> list:
    """Read measured hole diameters from an optional ``query('metrics')``.

    A backend that does not answer 'metrics' (the stub) returns {}, so this
    yields [] and the ops path alone drives the check.
    """
    try:
        m = backend.query("metrics")
    except Exception:  # noqa: BLE001 - unsupported query must degrade
        return []
    if not m:
        return []
    holes = m.get("holes")
    if not holes:
        return []
    out = []
    for h in holes:
        try:
            out.append(float(h))
        except (TypeError, ValueError):
            continue
    return out


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)
