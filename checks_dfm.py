"""DFM (design-for-manufacturing) critic — a standalone advisory verifier.

The blueprint reserves a **DFM Critic** stage in the plural verifier / multi-agent
pipeline (sec.12: "wall thickness, draft, tool-access, min radii"; sec.6: DFM rules
belong in the machine-verifiable Contract; sec.21: "the verifier is PLURAL, with
recycling"). Unlike the topology and constraint checks, DFM findings are *advisory*
— a part that violates a manufacturing guideline is still geometrically valid, it is
just harder/costlier to make. So every DFM finding is emitted as a WARNING (or INFO),
**never an ERROR**: this verifier can never flip a :class:`verify.VerifyReport` to
``ok == False``.

Standalone by design, exactly like :class:`checks_geometry.BRepValidityCheck`: this is
NOT wired into :func:`verify.default_verifiers` (that would be a circular import, and
DFM is an opt-in stage anyway). A caller adds it explicitly, e.g. via :func:`with_dfm`.

What is computable *today*: the current backends expose ``query('summary')`` (feature
counts), ``query('validity')`` (topology) and — on the real CadQuery/OCCT backend —
``query('measure')`` (overall bounding box + volume). From those we can already check:

  * ``high-aspect-ratio`` — long, thin overall envelope (from the bbox) that warps /
    is hard to fixture.
  * ``thin-envelope`` / ``small-part`` — an overall dimension below a plausible
    minimum-feature floor (a coarse proxy for "nothing in this part can be thinner
    than its own bounding box").
  * ``small-part-detail`` — a solid-volume-to-bbox ratio implying very fine detail
    relative to the min-hole / min-radius rules.

What genuinely needs *face-level* metrics the backends do not expose yet (true local
wall thickness, per-face draft angle relative to a pull direction, tool-access /
reachability, internal fillet radii): those are emitted as INFO
``dfm-not-yet-measurable`` stubs. They are a real hook point — the moment a backend
answers a ``query('faces')`` / ``query('thickness')`` / ``query('draft')`` these turn
into live WARNINGs — not fake results dressed up as measurements.

Degrade gracefully: if a query is unavailable (the stub answers only 'summary'), the
dependent checks INFO-skip rather than crash or error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from verify import Diagnostic, Severity, VerifyReport


# --------------------------------------------------------------------------- #
# Rules (configurable manufacturing limits)
# --------------------------------------------------------------------------- #
@dataclass
class DFMRules:
    """Configurable manufacturing limits for the DFM critic.

    Defaults are generic, conservative guidelines (roughly injection-moulding /
    small-part machining, in millimetres). A caller tunes them per process /
    material. All limits are advisory — violating one yields a WARNING, never an
    ERROR.
    """

    # Face-level limits (used once the backend exposes face metrics; see the
    # INFO stubs in DFMCheck). Kept here so rules round-trip and a real backend
    # can wire them straight in.
    min_wall_thickness: float = 1.0       # thinnest producible wall (mm)
    min_internal_radius: float = 0.5      # tightest concave/internal fillet (mm)
    min_draft_angle: float = 1.0          # minimum draft for mould release (deg)
    min_hole_diameter: float = 1.0        # smallest drillable/mouldable hole (mm)

    # Envelope-level limits (computable now from the bounding box / volume).
    max_aspect_ratio: float = 20.0        # longest/shortest bbox edge before warp/fixturing risk
    min_feature_size: float = 0.5         # any overall dimension below this is implausibly small (mm)
    max_overall_size: float = 1000.0      # any overall dimension above this exceeds a typical build volume (mm)

    def to_dict(self) -> dict:
        return {
            "min_wall_thickness": self.min_wall_thickness,
            "min_internal_radius": self.min_internal_radius,
            "min_draft_angle": self.min_draft_angle,
            "min_hole_diameter": self.min_hole_diameter,
            "max_aspect_ratio": self.max_aspect_ratio,
            "min_feature_size": self.min_feature_size,
            "max_overall_size": self.max_overall_size,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "DFMRules":
        d = d or {}
        defaults = cls()
        return cls(
            min_wall_thickness=float(d.get("min_wall_thickness", defaults.min_wall_thickness)),
            min_internal_radius=float(d.get("min_internal_radius", defaults.min_internal_radius)),
            min_draft_angle=float(d.get("min_draft_angle", defaults.min_draft_angle)),
            min_hole_diameter=float(d.get("min_hole_diameter", defaults.min_hole_diameter)),
            max_aspect_ratio=float(d.get("max_aspect_ratio", defaults.max_aspect_ratio)),
            min_feature_size=float(d.get("min_feature_size", defaults.min_feature_size)),
            max_overall_size=float(d.get("max_overall_size", defaults.max_overall_size)),
        )


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class DFMCheck:
    """A :class:`verify.Verifier` that flags design-for-manufacturing issues.

    ``check(backend, opdag)`` reads ``query('summary')``, ``query('validity')``
    and ``query('measure')`` and returns a :class:`verify.VerifyReport`. Every
    manufacturability finding is a WARNING (advisory) and every skipped or
    not-yet-measurable check is an INFO — so this verifier NEVER emits an ERROR
    and can never make a report ``ok == False``.

    Codes emitted:
      * WARNING ``high-aspect-ratio``  — long/thin overall envelope.
      * WARNING ``thin-envelope``      — an overall dimension below min_feature_size.
      * WARNING ``oversized``          — an overall dimension above max_overall_size.
      * INFO    ``thin-wall``, ``small-radius``, ``no-draft``, ``small-hole``,
                ``tool-access`` (as ``dfm-not-yet-measurable`` stubs) — real hook
                points that go live once a backend answers face-level queries.
      * INFO    ``*-skipped``          — a backing query was unavailable.
    """

    name = "dfm"

    def __init__(self, rules: Optional[DFMRules] = None) -> None:
        self.rules = rules or DFMRules()

    def check(self, backend, opdag) -> VerifyReport:
        diags: List[Diagnostic] = []
        self._check_envelope(backend, diags)
        self._check_face_level_stubs(backend, diags)
        return VerifyReport(diags)

    # -- envelope-level checks (computable from bbox / volume today) ---------
    def _check_envelope(self, backend, diags: List[Diagnostic]) -> None:
        measure = _query(backend, "measure")
        if measure is None:
            diags.append(_info(
                "envelope-skipped",
                "aspect-ratio / envelope-size checks skipped: backend has no "
                "'measure' query (only a real geometry kernel exposes bbox)."))
            return
        bbox = measure.get("bbox")
        if not bbox or len(bbox) < 3:
            diags.append(_info(
                "envelope-skipped",
                "aspect-ratio / envelope-size checks skipped: 'measure' "
                "returned no usable bbox."))
            return

        dims = [float(v) for v in bbox[:3]]
        positive = [d for d in dims if d > 0.0]
        if not positive:
            diags.append(_info(
                "envelope-skipped",
                "aspect-ratio / envelope-size checks skipped: bbox has no "
                "positive dimension (no solid yet)."))
            return

        r = self.rules

        # Aspect ratio: longest / shortest positive edge.
        longest = max(positive)
        shortest = min(positive)
        aspect = longest / shortest
        if aspect > r.max_aspect_ratio:
            diags.append(_warn(
                "high-aspect-ratio",
                f"overall aspect ratio {aspect:.1f} exceeds max "
                f"{r.max_aspect_ratio:.1f} (bbox {_fmt(dims)}); long/thin parts "
                f"warp and are hard to fixture."))

        # Thin envelope: an overall dimension below the plausible feature floor.
        for axis, d in zip(("x", "y", "z"), dims):
            if 0.0 < d < r.min_feature_size:
                diags.append(_warn(
                    "thin-envelope",
                    f"overall {axis} dimension {d:g} mm is below min feature "
                    f"size {r.min_feature_size:g} mm; likely unmanufacturable "
                    f"as a distinct feature.",
                    where=axis))

        # Oversized: beyond a typical build/machine envelope.
        for axis, d in zip(("x", "y", "z"), dims):
            if d > r.max_overall_size:
                diags.append(_warn(
                    "oversized",
                    f"overall {axis} dimension {d:g} mm exceeds max overall "
                    f"size {r.max_overall_size:g} mm; may not fit the target "
                    f"machine/build envelope.",
                    where=axis))

    # -- face-level checks (need metrics the backends do not expose yet) -----
    def _check_face_level_stubs(self, backend, diags: List[Diagnostic]) -> None:
        """Emit INFO stubs for the DFM checks the blueprint names but that need
        per-face geometry the current backends do not compute.

        These are a REAL hook point, not fake results: each names the backend
        query it is waiting on and the rule it would apply. The moment a backend
        answers those queries, promote the matching stub to a live WARNING (e.g.
        ``thin-wall`` when ``min(thickness) < rules.min_wall_thickness``).
        """
        validity = _query(backend, "validity")
        solid = bool(validity and validity.get("solid_present"))
        # Only advertise the hook points once there is actually a solid to
        # measure; otherwise there is nothing to say.
        if not solid:
            diags.append(_info(
                "dfm-not-yet-measurable",
                "face-level DFM checks (thin-wall, small-radius, no-draft, "
                "small-hole, tool-access) not evaluated: no solid present to "
                "measure."))
            return

        r = self.rules
        pending = [
            ("thin-wall",
             f"true wall thickness not measured (needs query('thickness'); "
             f"would warn when min wall < {r.min_wall_thickness:g} mm)."),
            ("small-radius",
             f"internal fillet radii not measured (needs face-level radii; "
             f"would warn when min internal radius < {r.min_internal_radius:g} mm)."),
            ("no-draft",
             f"per-face draft angle not measured (needs query('draft') vs a pull "
             f"direction; would warn when draft < {r.min_draft_angle:g} deg)."),
            ("small-hole",
             f"hole diameters not measured (needs cylindrical-face detection; "
             f"would warn when min hole diameter < {r.min_hole_diameter:g} mm)."),
            ("tool-access",
             "tool-access / reachability not evaluated (needs a face-visibility "
             "or accessibility query)."),
        ]
        for code, detail in pending:
            diags.append(_info(
                "dfm-not-yet-measurable",
                f"{code}: {detail}", where=code))


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_dfm(verifiers, rules: Optional[DFMRules] = None) -> List:
    """Return a new verifier list with a :class:`DFMCheck` appended.

    Mirrors how a caller adds the standalone geometry check to the default set
    without editing ``verify.py``::

        from verify import default_verifiers
        from checks_dfm import with_dfm
        verifiers = with_dfm(default_verifiers())
    """
    return list(verifiers) + [DFMCheck(rules)]


# --------------------------------------------------------------------------- #
# Helpers (mirror contract.py's graceful-degradation conventions)
# --------------------------------------------------------------------------- #
def _query(backend, q: str) -> Optional[dict]:
    """Read a backend query, returning None when the backend does not answer it
    (backends return {} for unknown queries) so callers can INFO-skip."""
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade, not crash
        return None
    return result or None


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)


def _fmt(dims: List[float]) -> str:
    return "[" + ", ".join(f"{d:g}" for d in dims) + "]"
