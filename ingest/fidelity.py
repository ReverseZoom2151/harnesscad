"""ingest.fidelity — round-trip fidelity checking for imported / rebuilt solids.

When a reference solid is ingested (``ingest.import_brep.import_solid``) it can be
*decompiled* into a best-effort CISP op tree (``ingest.decompile.decompile``) and
*rebuilt* through a geometry backend, or exported and re-imported. Either path is
a round trip, and a round trip is only trustworthy if the rebuilt geometry still
measures the same as the source. This module scores that:

  * :func:`roundtrip_fidelity` compares a *source* solid against its
    *rebuilt* version, measuring the delta in volume / surface area / topology
    (face/edge/solid counts) / bounding box. It reuses the ``quality.diff``
    metrics-delta primitive (:func:`quality.diff.geom_diff`) on the two sides —
    a self-comparison in the sense that both sides describe *the same intended
    part* — and reports whether they matched, the per-metric deltas, any metadata
    the round trip dropped, and a human note.

  * :func:`import_fidelity` verifies a freshly imported part actually measured
    *non-degenerate* geometry (positive volume, real bounding box), so a silent
    "loaded but empty" import is caught before it is trusted downstream.

Degradation contract (mirrors the rest of ``ingest``): OCCT is only ever touched
indirectly through the objects handed in. With no kernel — or when no rebuilt
side can be produced — the check degrades to a metrics-only comparison, or to a
clean ``matched=None`` "unavailable" report. It NEVER raises. Deterministic; no
wall-clock; stdlib + the in-repo ``quality.diff`` primitive only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from quality.diff import geom_diff


# Metrics compared between the two sides. Scalars use a relative tolerance; the
# bounding box is compared component-wise with the same relative tolerance.
_SCALAR_KEYS = ("volume", "surface_area")
_COUNT_KEYS = ("faces", "edges", "solids")

_DEFAULT_REL_TOL = 0.02   # 2% — round-trip meshing/export noise stays under this
_EPS = 1e-9


@dataclass
class FidelityReport:
    """The verdict of a round-trip / import fidelity check.

    - ``matched``      : True when every measured metric is within tolerance;
                         False when at least one diverged; None when the check
                         could not run (no measurable geometry / no rebuilt side).
    - ``deltas``       : per-metric ``{source, rebuilt, delta, rel, within}`` (a
                         reused metrics-delta view). Bounding box is reported as
                         ``bbox`` with per-axis deltas.
    - ``lost_metadata``: metric keys measurable on the source but missing after
                         the round trip (metadata the rebuild dropped).
    - ``note``         : human-readable status / degradation note.
    - ``mode``         : "boolean" | "metrics" | "unavailable" — how the compare
                         ran (mirrors ``quality.diff.GeomDiff.mode``).
    - ``available``    : True only when a real comparison (or measurement) ran.
    """

    matched: Optional[bool]
    deltas: Dict[str, Any] = field(default_factory=dict)
    lost_metadata: List[str] = field(default_factory=list)
    note: str = ""
    mode: str = "unavailable"
    available: bool = False

    @property
    def ok(self) -> bool:
        return self.matched is True

    def to_dict(self) -> dict:
        return {
            "matched": self.matched,
            "deltas": self.deltas,
            "lost_metadata": list(self.lost_metadata),
            "note": self.note,
            "mode": self.mode,
            "available": self.available,
        }


# --------------------------------------------------------------------------- #
# Input adaptation — pull metrics / shape from ImportedPart or a backend
# --------------------------------------------------------------------------- #
def _extract_metrics(obj) -> Dict[str, Any]:
    """Best-available metrics dict from an ImportedPart, a backend, or a mapping.

    Mirrors ``ingest.decompile._get_metrics`` / ``quality.diff._volume_of``:
    prefer a populated ``.metrics`` attribute, else ``query('metrics')`` /
    ``query('measure')``, else an object that *is* a metrics mapping. ``{}`` when
    nothing is measurable (never raises).
    """
    if obj is None:
        return {}
    metrics = getattr(obj, "metrics", None)
    if isinstance(metrics, dict) and metrics:
        return dict(metrics)
    query = getattr(obj, "query", None)
    if callable(query):
        for q in ("metrics", "measure"):
            try:
                res = query(q)
            except Exception:  # noqa: BLE001 - a query hiccup must not crash
                res = None
            if isinstance(res, dict) and res:
                return dict(res)
    if isinstance(obj, dict) and obj:
        return dict(obj)
    return {}


def _extract_shape(obj):
    """A cq/OCP shape from an ImportedPart (``.shape``) or backend (``_combined``)."""
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return shape
    combined = getattr(obj, "_combined", None)
    if callable(combined):
        try:
            return combined()
        except Exception:  # noqa: BLE001
            return None
    return None


class _MetricsBackend:
    """Adapter exposing the ``quality.diff`` backend protocol over static metrics.

    ``geom_diff`` expects backends with ``query('metrics'|'measure')`` and an
    optional ``_combined()`` returning a solid. Wrapping the two sides this way
    lets us reuse ``geom_diff`` verbatim as the metrics-delta primitive.
    """

    def __init__(self, metrics: Dict[str, Any], shape=None) -> None:
        self._metrics = dict(metrics)
        self._shape = shape

    def query(self, q: str) -> Dict[str, Any]:
        if q in ("metrics", "measure"):
            return dict(self._metrics)
        return {}

    def _combined(self):
        return self._shape


# --------------------------------------------------------------------------- #
# Delta computation
# --------------------------------------------------------------------------- #
def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _scalar_delta(src, reb, rel_tol: float) -> Optional[Dict[str, Any]]:
    a, b = _num(src), _num(reb)
    if a is None or b is None:
        return None
    delta = b - a
    denom = max(abs(a), abs(b), _EPS)
    rel = abs(delta) / denom
    return {"source": a, "rebuilt": b, "delta": delta, "rel": rel,
            "within": rel <= rel_tol}


def _bbox_delta(src, reb, rel_tol: float) -> Optional[Dict[str, Any]]:
    if not (isinstance(src, (list, tuple)) and isinstance(reb, (list, tuple))):
        return None
    if len(src) != len(reb) or not src:
        return None
    axes = []
    within = True
    for a, b in zip(src, reb):
        fa, fb = _num(a), _num(b)
        if fa is None or fb is None:
            return None
        d = fb - fa
        rel = abs(d) / max(abs(fa), abs(fb), _EPS)
        ok = rel <= rel_tol
        within = within and ok
        axes.append({"source": fa, "rebuilt": fb, "delta": d, "rel": rel,
                     "within": ok})
    return {"axes": axes, "within": within}


def _compute_deltas(src_m: Dict[str, Any], reb_m: Dict[str, Any],
                    rel_tol: float) -> Dict[str, Any]:
    deltas: Dict[str, Any] = {}
    for key in _SCALAR_KEYS:
        d = _scalar_delta(src_m.get(key), reb_m.get(key), rel_tol)
        if d is not None:
            deltas[key] = d
    for key in _COUNT_KEYS:
        a, b = src_m.get(key), reb_m.get(key)
        if isinstance(a, int) and isinstance(b, int):
            deltas[key] = {"source": a, "rebuilt": b, "delta": b - a,
                           "within": a == b}
    bb = _bbox_delta(src_m.get("bbox"), reb_m.get("bbox"), rel_tol)
    if bb is not None:
        deltas["bbox"] = bb
    return deltas


def _lost_metadata(src_m: Dict[str, Any], reb_m: Dict[str, Any]) -> List[str]:
    """Keys the source measured that the rebuilt side no longer carries."""
    lost = []
    for key in sorted(src_m):
        if src_m.get(key) in (None, [], {}, ""):
            continue
        if key not in reb_m or reb_m.get(key) in (None, [], {}, ""):
            lost.append(key)
    return lost


def _all_within(deltas: Dict[str, Any]) -> bool:
    for key, d in deltas.items():
        if key == "bbox":
            if not d.get("within", False):
                return False
        elif not d.get("within", False):
            return False
    return True


# --------------------------------------------------------------------------- #
# Optional rebuild (import -> decompile -> replay), guarded on OCCT
# --------------------------------------------------------------------------- #
def _rebuild_backend(imported_or_backend):
    """Decompile the source and replay its ops onto a fresh geometry backend.

    Returns a measurable backend, or None when no kernel is available / the ops
    do not build / decompilation recovered nothing. Fully guarded — the absence
    of OCCT simply yields None and the caller degrades to an unavailable report.
    """
    try:
        from ingest.decompile import decompile
        result = decompile(imported_or_backend)
    except Exception:  # noqa: BLE001
        return None
    ops = list(getattr(result, "ops", []) or [])
    if not ops:
        return None
    try:
        from backends.cadquery_backend import CadQueryBackend
        backend = CadQueryBackend()
    except Exception:  # noqa: BLE001
        return None
    try:
        for op in ops:
            r = backend.apply(op)
            if not getattr(r, "ok", False):
                return None
    except Exception:  # noqa: BLE001 - no kernel / kernel failure -> degrade
        return None
    if not _extract_metrics(backend):
        return None
    return backend


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def roundtrip_fidelity(imported_or_backend, rebuilt_backend=None,
                       rel_tol: float = _DEFAULT_REL_TOL) -> FidelityReport:
    """Score a source solid against its decompiled/rebuilt (round-tripped) version.

    ``imported_or_backend`` is the *source* — an ``ImportedPart``, a geometry
    backend, or any object exposing ``.metrics`` / ``query('metrics')``.
    ``rebuilt_backend`` is the *rebuilt* side; when omitted this attempts an
    import -> decompile -> replay rebuild (needs OCCT) and, failing that, degrades
    to a clean ``matched=None`` "unavailable" report.

    Reuses ``quality.diff.geom_diff`` as the metrics-delta primitive over the two
    sides, then layers surface / topology / bbox deltas on top. Never raises.
    """
    src_m = _extract_metrics(imported_or_backend)

    if rebuilt_backend is None:
        rebuilt_backend = _rebuild_backend(imported_or_backend)
        rebuilt_source = "decompile+replay"
    else:
        rebuilt_source = "provided"

    if rebuilt_backend is None:
        return FidelityReport(
            matched=None, deltas={}, lost_metadata=[],
            note="unavailable: no rebuilt side to compare against "
                 "(no OCCT to rebuild, and none provided)",
            mode="unavailable", available=False)

    reb_m = _extract_metrics(rebuilt_backend)
    if not src_m or not reb_m:
        return FidelityReport(
            matched=None, deltas={}, lost_metadata=_lost_metadata(src_m, reb_m),
            note="unavailable: one side has no measurable metrics "
                 f"(source={bool(src_m)}, rebuilt={bool(reb_m)})",
            mode="unavailable", available=False)

    # Reuse the quality.diff metrics-delta primitive for the volume (and, with a
    # kernel on both sides, face-count) delta.
    gd = geom_diff(
        _MetricsBackend(src_m, _extract_shape(imported_or_backend)),
        _MetricsBackend(reb_m, _extract_shape(rebuilt_backend)))

    deltas = _compute_deltas(src_m, reb_m, rel_tol)
    lost = _lost_metadata(src_m, reb_m)
    matched = _all_within(deltas) if deltas else False

    note = (f"round-trip via {rebuilt_source}: "
            f"{'match' if matched else 'MISMATCH'} "
            f"({gd.render()})")
    if not deltas:
        note = (f"round-trip via {rebuilt_source}: no comparable metrics "
                f"between the two sides ({gd.render()})")
    return FidelityReport(
        matched=matched, deltas=deltas, lost_metadata=lost, note=note,
        mode=gd.mode, available=True)


def import_fidelity(imported, rel_tol: float = _DEFAULT_REL_TOL) -> FidelityReport:
    """Verify a freshly imported part measured *non-degenerate* geometry.

    Non-degenerate == a positive volume and a real (all-positive) bounding box;
    face count, when reported, must be >= 1. When the part carries no measurable
    metrics (e.g. OCCT absent, so ``import_solid`` returned an unavailable part)
    the check degrades to ``matched=None`` "unavailable" rather than failing it.
    Never raises.
    """
    m = _extract_metrics(imported)
    available = bool(getattr(imported, "available", True))
    if not m:
        return FidelityReport(
            matched=None, deltas={}, lost_metadata=[],
            note="unavailable: imported part carries no measurable geometry "
                 "(OCCT absent or file unmeasurable)",
            mode="unavailable", available=False)

    checks: Dict[str, Any] = {}
    reasons: List[str] = []

    vol = _num(m.get("volume"))
    if vol is not None:
        ok = vol > _EPS
        checks["volume"] = {"value": vol, "ok": ok}
        if not ok:
            reasons.append("non-positive volume")

    bbox = m.get("bbox")
    if isinstance(bbox, (list, tuple)) and bbox:
        vals = [_num(v) for v in bbox]
        ok = all(v is not None and v > _EPS for v in vals)
        checks["bbox"] = {"value": [v for v in vals], "ok": ok}
        if not ok:
            reasons.append("degenerate bounding box")

    faces = m.get("faces")
    if isinstance(faces, int):
        ok = faces >= 1
        checks["faces"] = {"value": faces, "ok": ok}
        if not ok:
            reasons.append("no faces")

    if not checks:
        return FidelityReport(
            matched=None, deltas={}, lost_metadata=[],
            note="unavailable: metrics present but none were dimensional "
                 "(no volume/bbox/faces to judge)",
            mode="unavailable", available=False)

    non_degenerate = all(c["ok"] for c in checks.values())
    note = ("import measured non-degenerate geometry"
            if non_degenerate
            else "DEGENERATE import: " + ", ".join(reasons))
    if not available and non_degenerate:
        note += " (note: part flagged unavailable but metrics look valid)"
    return FidelityReport(
        matched=non_degenerate, deltas=checks, lost_metadata=[], note=note,
        mode="metrics", available=True)
