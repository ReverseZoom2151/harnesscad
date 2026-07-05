"""Interference / collision gate — the *collision* half of the assembly solver.

Sibling to :mod:`checks_assembly`: ``verify.py`` reserves an assembly solver for
*mates / DOF / collision*; the mate/DOF accounting lives in
:mod:`checks_assembly`, and this module detects the collisions — solid-solid
overlap between the placed parts of a multi-part assembly.

Two stages, cheapest first (the classic broad-phase / narrow-phase split):

  1. **Broad phase — bounding-box sweep-and-prune** (an "R-tree-lite"): sort the
     parts' axis-aligned bounding boxes by their minimum x, sweep a line along x
     keeping only boxes whose x-interval is still open, and test the surviving
     candidate pairs for a full 3-axis AABB overlap. This is O(n log n + k) and
     throws away the quadratic majority of disjoint pairs before any expensive
     kernel call.

  2. **Narrow phase** on each surviving pair:
       * If CadQuery/OCCT is available *and* both parts carry a real shape, the
         exact overlap is the volume of the boolean *common* (intersection) of
         the two solids (``BRepAlgoAPI_Common``). A positive common volume is a
         definite clash -> ERROR ``interference``.
       * Otherwise, if the parts carry (or can produce) an AABB, fall back to a
         pure-python bounding-box overlap. A box overlap does not prove the
         solids touch, so it is reported as an *approximate* WARNING
         ``interference-approx`` with the overlap-box volume — never a hard
         ERROR.

Clashes are emitted ranked by descending overlap volume (worst first).

Degrades gracefully, like :mod:`contract` / :mod:`checks_dfm` /
:mod:`checks_assembly`:

  * INFO ``interference-skipped`` — no ``'assembly'`` query, or it is empty.
  * INFO ``interference-trivial`` — fewer than two parts (nothing can clash).
  * INFO ``interference-not-measurable`` — a pair carries neither a shape (with
    OCCT available) nor an AABB, so it cannot be tested.

Standalone by design: not wired into :func:`verify.default_verifiers`. Add it
explicitly via :func:`with_interference`.

The part records read from ``query('assembly')['parts']`` are plain dicts::

    {"id": "p1", "bbox": [xmin, ymin, zmin, xmax, ymax, zmax], "shape": <opt>}

``bbox`` (a 6-list) drives the broad phase and the pure-python fallback;
``shape`` (an optional cq/OCP solid) enables the exact OCCT narrow phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from verify import Diagnostic, Severity, VerifyReport


BBox = Tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
@dataclass
class Clash:
    """A detected interference between two parts.

    ``exact`` is True when the volume came from an OCCT boolean common; False
    when it is the approximate bounding-box overlap volume.
    """

    id_a: str
    id_b: str
    volume: float
    exact: bool
    overlap_dims: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class InterferenceCheck:
    """A :class:`verify.Verifier` (``name='interference'``) collision gate.

    ``check(backend, opdag)`` reads ``query('assembly')`` and returns a
    :class:`verify.VerifyReport`. Exact (OCCT) clashes are ERRORs; approximate
    (bounding-box) clashes are WARNINGs; everything skipped is INFO.

    Args:
        min_volume: smallest overlap volume (model units^3) worth reporting;
            filters away kernel/AABB noise. Default ``1e-9``.
    """

    name = "interference"

    def __init__(self, min_volume: float = 1e-9) -> None:
        self.min_volume = float(min_volume)

    def check(self, backend, opdag) -> VerifyReport:
        raw = _query(backend, "assembly")
        if not raw:
            return VerifyReport([_info(
                "interference-skipped",
                "interference check skipped: backend has no 'assembly' query "
                "(only an assembly-aware backend exposes placed parts).")])
        parts = list(raw.get("parts", []) or [])
        return VerifyReport(self._diagnose(parts))

    def check_parts(self, parts: List[dict]) -> VerifyReport:
        """Run the collision gate on a list of part records directly."""
        return VerifyReport(self._diagnose(list(parts or [])))

    # -- core --------------------------------------------------------------- #
    def _diagnose(self, parts: List[dict]) -> List[Diagnostic]:
        if len(parts) < 2:
            return [_info(
                "interference-trivial",
                f"interference check skipped: {len(parts)} part(s) — at least "
                "two placed bodies are needed for a collision.")]

        diags: List[Diagnostic] = []
        cq_ok = _cadquery_available()

        # Precompute a bbox per part (from the record, or lazily from its shape).
        boxes: List[Optional[BBox]] = []
        for p in parts:
            boxes.append(_part_bbox(p, cq_ok))

        candidate_pairs = _sweep_and_prune(boxes)

        clashes: List[Clash] = []
        not_measurable: List[Tuple[str, str]] = []
        for i, j in candidate_pairs:
            pa, pb = parts[i], parts[j]
            ida, idb = _part_id(pa, i), _part_id(pb, j)
            clash = self._narrow_phase(pa, pb, boxes[i], boxes[j],
                                       ida, idb, cq_ok)
            if clash is None:
                # Overlapping AABBs but nothing measurable to confirm/deny.
                not_measurable.append((ida, idb))
            elif clash.volume > self.min_volume:
                clashes.append(clash)

        # Rank worst-first.
        clashes.sort(key=lambda c: c.volume, reverse=True)
        for c in clashes:
            if c.exact:
                diags.append(_err(
                    "interference",
                    f"parts '{c.id_a}' and '{c.id_b}' interfere: solid overlap "
                    f"volume {c.volume:.6g} (exact boolean common)."))
            else:
                diags.append(_warn(
                    "interference-approx",
                    f"parts '{c.id_a}' and '{c.id_b}' may interfere: bounding "
                    f"boxes overlap by ~{c.volume:.6g} "
                    f"(dims {_fmt(c.overlap_dims)}); approximate — no exact "
                    "solid test available (install cadquery / provide shapes)."))

        for ida, idb in not_measurable:
            diags.append(_info(
                "interference-not-measurable",
                f"parts '{ida}' and '{idb}' could not be tested: no exact shape "
                "(OCCT unavailable) and no bounding box to fall back on."))

        if not diags:
            diags.append(_info(
                "interference-clear",
                f"no interference among {len(parts)} parts "
                f"({len(candidate_pairs)} candidate pair(s) after the bbox "
                "sweep passed the narrow phase)."))
        return diags

    def _narrow_phase(self, pa: dict, pb: dict,
                      bb_a: Optional[BBox], bb_b: Optional[BBox],
                      ida: str, idb: str, cq_ok: bool) -> Optional[Clash]:
        """Return a :class:`Clash` (exact or approximate) or ``None`` if the
        pair cannot be measured at all."""
        sa, sb = pa.get("shape"), pb.get("shape")
        if cq_ok and sa is not None and sb is not None:
            vol = _common_volume(sa, sb)
            if vol is not None:
                if vol > self.min_volume:
                    return Clash(ida, idb, vol, exact=True,
                                 overlap_dims=_overlap_dims(bb_a, bb_b))
                # A clean, measured zero-overlap is a real "no clash" result.
                return Clash(ida, idb, 0.0, exact=True)
            # OCCT choked -> fall through to the approximate test below.
        if bb_a is not None and bb_b is not None:
            dims = _overlap_dims(bb_a, bb_b)
            vol = dims[0] * dims[1] * dims[2]
            return Clash(ida, idb, vol, exact=False, overlap_dims=dims)
        return None


# --------------------------------------------------------------------------- #
# Broad phase — sweep and prune on the x axis (R-tree-lite)
# --------------------------------------------------------------------------- #
def _sweep_and_prune(boxes: List[Optional[BBox]]) -> List[Tuple[int, int]]:
    """Return index pairs whose AABBs overlap on all three axes.

    Sorts box *starts* along x and sweeps, so only x-overlapping boxes are ever
    compared; the surviving candidates are confirmed with a full 3-axis test.
    Parts without a bbox are excluded from the broad phase (handled as
    not-measurable by the caller only if they would otherwise be candidates —
    here they simply cannot be swept, so they never clash).
    """
    indexed = [(b[0], idx) for idx, b in enumerate(boxes) if b is not None]
    indexed.sort()
    pairs: List[Tuple[int, int]] = []
    active: List[int] = []  # indices whose x-interval is still open
    for _xmin, idx in indexed:
        box = boxes[idx]
        assert box is not None
        # Evict boxes that have closed before this one opened.
        active = [k for k in active if boxes[k][3] >= box[0]]  # type: ignore[index]
        for k in active:
            if _aabb_overlap(boxes[k], box):  # type: ignore[arg-type]
                pairs.append((k, idx) if k < idx else (idx, k))
        active.append(idx)
    return pairs


def _aabb_overlap(a: BBox, b: BBox) -> bool:
    return (a[0] <= b[3] and b[0] <= a[3] and
            a[1] <= b[4] and b[1] <= a[4] and
            a[2] <= b[5] and b[2] <= a[5])


def _overlap_dims(a: Optional[BBox], b: Optional[BBox]) -> Tuple[float, float, float]:
    if a is None or b is None:
        return (0.0, 0.0, 0.0)
    ox = max(0.0, min(a[3], b[3]) - max(a[0], b[0]))
    oy = max(0.0, min(a[4], b[4]) - max(a[1], b[1]))
    oz = max(0.0, min(a[5], b[5]) - max(a[2], b[2]))
    return (ox, oy, oz)


# --------------------------------------------------------------------------- #
# Part geometry helpers
# --------------------------------------------------------------------------- #
def _part_id(part: dict, index: int) -> str:
    return str(part.get("id", part.get("name", f"part{index}")))


def _part_bbox(part: dict, cq_ok: bool) -> Optional[BBox]:
    """The part's AABB from its record, else lazily from its OCCT shape."""
    bb = part.get("bbox")
    if bb is not None and len(bb) >= 6:
        return (float(bb[0]), float(bb[1]), float(bb[2]),
                float(bb[3]), float(bb[4]), float(bb[5]))
    shape = part.get("shape")
    if cq_ok and shape is not None:
        try:
            b = shape.BoundingBox()
            return (float(b.xmin), float(b.ymin), float(b.zmin),
                    float(b.xmax), float(b.ymax), float(b.zmax))
        except Exception:  # noqa: BLE001 - a bad shape must not crash the gate
            return None
    return None


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401, WPS433 (probe only)
        return True
    except Exception:  # noqa: BLE001 - ImportError or a broken OCCT install
        return False


def _common_volume(shape_a, shape_b) -> Optional[float]:
    """Volume of the OCCT boolean common of two solids, or ``None`` if the
    kernel is unavailable / the operation failed (caller degrades)."""
    try:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp

        wa = getattr(shape_a, "wrapped", shape_a)
        wb = getattr(shape_b, "wrapped", shape_b)
        common = BRepAlgoAPI_Common(wa, wb)
        common.Build()
        if not common.IsDone():
            return None
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(common.Shape(), props)
        return abs(float(props.Mass()))
    except Exception:  # noqa: BLE001 - any kernel failure -> approximate fallback
        return None


# --------------------------------------------------------------------------- #
# Wiring helper
# --------------------------------------------------------------------------- #
def with_interference(verifiers, min_volume: float = 1e-9) -> List:
    """Return a new verifier list with an :class:`InterferenceCheck` appended
    (mirrors :func:`checks_assembly.with_assembly`)."""
    return list(verifiers) + [InterferenceCheck(min_volume=min_volume)]


# --------------------------------------------------------------------------- #
# Graceful-degradation helpers (mirror contract.py / checks_dfm.py)
# --------------------------------------------------------------------------- #
def _query(backend, q: str) -> Optional[dict]:
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade, not crash
        return None
    return result or None


def _err(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.ERROR, code, msg, where)


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)


def _fmt(dims) -> str:
    return "[" + ", ".join(f"{d:g}" for d in dims) + "]"
