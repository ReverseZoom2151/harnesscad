"""Tool-access / serviceability clearance gate — the *reachability* verifier.

Where :mod:`interference` answers "do two placed bodies overlap *right now*",
this verifier answers a different, motion-aware question the blueprint reserves
under DFM/serviceability (sec.12 "tool-access", sec.21 "the verifier is PLURAL"):
**can a tool / driver / hand actually reach each hole, fastener or serviceable
feature, and be withdrawn, without hitting another body?**

That is a *swept* test, not a static clash. For every serviceable feature we
sweep a clearance envelope — a cylinder of ``tool_diameter`` (widened to the
feature's own diameter when larger) extruded ``approach_length`` along the
feature's approach / extraction vector (its hole axis, default +Z) — and test it
against every *other* body:

  1. **Exact (OCCT) path** — when cadquery/OCCT is available and the obstructing
     part carries a real shape: the swept tool cylinder is intersected
     (``BRepAlgoAPI_Common``) with the part; a positive common volume means the
     corridor is physically blocked -> WARNING ``no-tool-access``.

  2. **Pure-python bbox-corridor fallback** — always available: the swept
     envelope is bounded by an axis-aligned corridor box (the approach segment
     grown by the tool radius). A *core* corridor (tool radius) that overlaps a
     part is a blockage (``no-tool-access``); a part that clears the core but
     encroaches the corridor grown by ``min_clearance`` is a squeeze
     (``tight-clearance``). Both report the required-vs-available gap.

Findings are advisory serviceability warnings, never hard ERRORs — a part you
cannot easily service is still geometrically valid, just costly to assemble /
maintain. So this verifier can never flip a report to ``ok == False``.

Degrades gracefully, exactly like :mod:`interference` / :mod:`dfm`:

  * INFO ``access-skipped``          — no ``'assembly'``/``'access'`` query.
  * INFO ``no-serviceable-features`` — nothing to reach (nothing to say).
  * INFO ``access-not-measurable``   — an obstruction carries neither a shape
    (OCCT available) nor a bbox, so the corridor cannot be tested against it.
  * INFO ``access-clear``            — every feature is reachable.

Standalone by design: NOT wired into :func:`verify.default_verifiers`; a caller
adds it via :func:`with_access`.

Feature records (from ``query('access')['features']`` or a per-part ``features``
list under ``query('assembly')['parts']``) are plain dicts::

    {"id": "h1", "kind": "hole", "position": [x, y, z],
     "axis": [dx, dy, dz], "diameter": 6.0, "part": "plate"}

``axis`` may be a 3-vector (direction) or a 6-tuple (two points); it defaults to
+Z. ``part``/``owner`` names the body the feature belongs to so the feature is
never reported as blocked by its own host body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from verifiers.verify import Diagnostic, Severity, VerifyReport


BBox = Tuple[float, float, float, float, float, float]  # xmin..zmax
Vec3 = Tuple[float, float, float]

_EPS = 1e-9

# Feature kinds treated as serviceable / tool-accessed when a part advertises
# itself as a feature (an explicit features list is always taken as serviceable).
SERVICEABLE_KINDS = frozenset({
    "hole", "fastener", "screw", "bolt", "nut", "rivet", "pin",
    "port", "connector", "serviceable", "access", "vent", "drain",
})


# --------------------------------------------------------------------------- #
# Rules (configurable tool / clearance envelope)
# --------------------------------------------------------------------------- #
@dataclass
class AccessRules:
    """Configurable tool-access envelope (millimetres).

    ``tool_diameter``  — diameter of the driver / socket / hand clearance the
        swept envelope must guarantee (widened to the feature's own diameter
        when the feature is larger).
    ``approach_length`` — how far along the approach / extraction vector the tool
        must travel unobstructed (the reach / insertion depth).
    ``min_clearance``  — extra radial slack around the tool below which access is
        merely *tight* rather than blocked.
    """

    tool_diameter: float = 8.0
    approach_length: float = 30.0
    min_clearance: float = 2.0

    def to_dict(self) -> dict:
        return {
            "tool_diameter": self.tool_diameter,
            "approach_length": self.approach_length,
            "min_clearance": self.min_clearance,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AccessRules":
        d = d or {}
        defaults = cls()
        return cls(
            tool_diameter=float(d.get("tool_diameter", defaults.tool_diameter)),
            approach_length=float(d.get("approach_length", defaults.approach_length)),
            min_clearance=float(d.get("min_clearance", defaults.min_clearance)),
        )


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #
class AccessCheck:
    """A :class:`verify.Verifier` (``name='access'``) serviceability gate.

    ``check(backend, opdag)`` reads ``query('access')`` (preferred) or
    ``query('assembly')`` and returns a :class:`verify.VerifyReport`. Every
    reachability finding is a WARNING and every skip is an INFO — this verifier
    never emits an ERROR.
    """

    name = "access"

    def __init__(self, rules: Optional[AccessRules] = None) -> None:
        self.rules = rules or AccessRules()

    def check(self, backend, opdag) -> VerifyReport:
        raw = _query(backend, "access")
        if raw is None:
            raw = _query(backend, "assembly")
        if not raw:
            return VerifyReport([_info(
                "access-skipped",
                "tool-access check skipped: backend exposes no 'access'/"
                "'assembly' query (only an assembly-aware backend places bodies "
                "and serviceable features).")])
        parts = list(raw.get("parts", []) or [])
        features = _collect_features(raw)
        return VerifyReport(self._diagnose(parts, features))

    def check_access(self, parts: List[dict], features: List[dict]) -> VerifyReport:
        """Run the gate on explicit part + feature records (backend-free)."""
        norm = [f for f in (_normalize_feature(x) for x in (features or []))
                if f is not None]
        return VerifyReport(self._diagnose(list(parts or []), norm))

    # -- core --------------------------------------------------------------- #
    def _diagnose(self, parts: List[dict], features: List[dict]) -> List[Diagnostic]:
        if not features:
            return [_info(
                "no-serviceable-features",
                "tool-access check skipped: no serviceable features (holes / "
                "fasteners) to reach.")]

        cq_ok = _cadquery_available()
        r = self.rules
        diags: List[Diagnostic] = []
        not_measurable: List[Tuple[str, str]] = []
        found_issue = False

        for feat in features:
            fid = feat["id"]
            kind = feat["kind"]
            owner = feat["owner"]
            radius = max(r.tool_diameter, feat["diameter"] or 0.0) / 2.0
            pos = feat["pos"]
            axis = feat["axis"]
            core = _corridor_bbox(pos, axis, r.approach_length, radius)
            grown = _corridor_bbox(pos, axis, r.approach_length,
                                   radius + r.min_clearance)

            for p in parts:
                pid = _part_id(p)
                if owner is not None and pid == owner:
                    continue  # a feature is never blocked by its own host body

                verdict = self._test_pair(feat, radius, core, grown, p, cq_ok)
                if verdict is None:
                    not_measurable.append((fid, pid))
                    continue
                status, available = verdict
                if status == "blocked":
                    found_issue = True
                    diags.append(_warn(
                        "no-tool-access",
                        f"no tool access to {kind} '{fid}': the approach "
                        f"corridor (tool Ø{r.tool_diameter:g} mm, reach "
                        f"{r.approach_length:g} mm along {_fmt(axis)}) is blocked "
                        f"by body '{pid}'; required clearance "
                        f"{r.min_clearance:g} mm, available {available:g} mm "
                        f"(negative = interference).",
                        where=fid))
                elif status == "tight":
                    found_issue = True
                    diags.append(_warn(
                        "tight-clearance",
                        f"tight clearance for {kind} '{fid}': body '{pid}' "
                        f"encroaches the tool corridor; required "
                        f"{r.min_clearance:g} mm, available {available:g} mm.",
                        where=fid))

        for fid, pid in not_measurable:
            diags.append(_info(
                "access-not-measurable",
                f"tool access to '{fid}' vs body '{pid}' could not be tested: "
                "the body has no shape (OCCT unavailable) and no bounding box."))

        if not found_issue and not diags:
            diags.append(_info(
                "access-clear",
                f"all {len(features)} serviceable feature(s) are reachable "
                f"(tool Ø{r.tool_diameter:g} mm, reach {r.approach_length:g} mm)."))
        elif not found_issue:
            # Only not-measurable notes were produced; still advertise clearance.
            diags.insert(0, _info(
                "access-clear",
                f"no blocked feature among {len(features)} checked "
                "(some obstructions were not measurable)."))
        return diags

    def _test_pair(self, feat: dict, radius: float,
                   core: BBox, grown: BBox, part: dict,
                   cq_ok: bool) -> Optional[Tuple[str, float]]:
        """Classify one feature-vs-body pair.

        Returns ``("blocked"|"tight"|"clear", available_gap)`` or ``None`` when
        the body cannot be measured at all.
        """
        shape = part.get("shape")
        if cq_ok and shape is not None:
            vol = _swept_common_volume(feat, radius, self.rules.approach_length,
                                       shape)
            if vol is not None:
                if vol > _EPS:
                    return ("blocked", -_cube_root(vol))
                return ("clear", radius)  # exact: corridor is free of this body
            # OCCT choked -> fall through to the bbox corridor.

        bb = _part_bbox(part, cq_ok)
        if bb is None:
            return None

        core_ov = _overlap_dims(core, bb)
        if _positive_volume(core_ov) > _EPS:
            encroach = _min_positive(core_ov)
            return ("blocked", -encroach)

        grown_ov = _overlap_dims(grown, bb)
        if _positive_volume(grown_ov) > _EPS:
            penetration = _min_positive(grown_ov)
            available = max(0.0, self.rules.min_clearance - penetration)
            return ("tight", available)

        return ("clear", self.rules.min_clearance)


# --------------------------------------------------------------------------- #
# Feature parsing
# --------------------------------------------------------------------------- #
def _collect_features(raw: dict) -> List[dict]:
    """Gather + normalise serviceable features from an access/assembly view."""
    out: List[dict] = []
    for f in (raw.get("features", []) or []):
        nf = _normalize_feature(f)
        if nf is not None:
            out.append(nf)
    for p in (raw.get("parts", []) or []):
        pid = _part_id(p)
        for f in (p.get("features", []) or []):
            nf = _normalize_feature(f, default_owner=pid)
            if nf is not None:
                out.append(nf)
        # A part may itself be a serviceable feature.
        if str(p.get("kind", "")).lower() in SERVICEABLE_KINDS or p.get("serviceable"):
            nf = _normalize_feature(p)
            if nf is not None:
                out.append(nf)
    return out


def _normalize_feature(f: dict, default_owner: Optional[str] = None) -> Optional[dict]:
    if not isinstance(f, dict):
        return None
    fid = str(f.get("id", f.get("name", "feature")))
    kind = str(f.get("kind", "hole"))
    owner = f.get("part", f.get("owner", default_owner))
    owner = str(owner) if owner is not None else None
    return {
        "id": fid,
        "kind": kind,
        "owner": owner,
        "pos": _feature_pos(f),
        "axis": _feature_axis(f),
        "diameter": _feature_diameter(f),
    }


def _feature_pos(f: dict) -> Vec3:
    p = f.get("position") or f.get("pos") or f.get("center")
    if p is not None and len(p) >= 3:
        return (float(p[0]), float(p[1]), float(p[2]))
    return (float(f.get("x", 0.0)), float(f.get("y", 0.0)), float(f.get("z", 0.0)))


def _feature_axis(f: dict) -> Vec3:
    a = f.get("axis")
    if a is None:
        return (0.0, 0.0, 1.0)
    if len(a) >= 6:  # two points -> direction
        d = (float(a[3]) - float(a[0]),
             float(a[4]) - float(a[1]),
             float(a[5]) - float(a[2]))
    elif len(a) >= 3:
        d = (float(a[0]), float(a[1]), float(a[2]))
    else:
        return (0.0, 0.0, 1.0)
    return _normalize(d)


def _feature_diameter(f: dict) -> Optional[float]:
    for key in ("diameter", "dia", "d"):
        if f.get(key) is not None:
            return abs(float(f[key]))
    if f.get("radius") is not None:
        return abs(float(f["radius"])) * 2.0
    return None


# --------------------------------------------------------------------------- #
# Corridor geometry (pure-python fallback)
# --------------------------------------------------------------------------- #
def _corridor_bbox(pos: Vec3, axis: Vec3, length: float, radius: float) -> BBox:
    """AABB bounding the swept tool cylinder (approach segment grown by radius).

    The segment runs from ``pos`` along ``axis`` for ``length``; its endpoints'
    bounding box, grown by ``radius`` on every side, conservatively contains the
    swept envelope for any approach direction and, when the axis pierces a body,
    reliably overlaps it.
    """
    sx, sy, sz = pos
    ex, ey, ez = (sx + axis[0] * length, sy + axis[1] * length, sz + axis[2] * length)
    return (
        min(sx, ex) - radius, min(sy, ey) - radius, min(sz, ez) - radius,
        max(sx, ex) + radius, max(sy, ey) + radius, max(sz, ez) + radius,
    )


def _overlap_dims(a: BBox, b: BBox) -> Vec3:
    ox = max(0.0, min(a[3], b[3]) - max(a[0], b[0]))
    oy = max(0.0, min(a[4], b[4]) - max(a[1], b[1]))
    oz = max(0.0, min(a[5], b[5]) - max(a[2], b[2]))
    return (ox, oy, oz)


def _positive_volume(dims: Vec3) -> float:
    return dims[0] * dims[1] * dims[2]


def _min_positive(dims: Vec3) -> float:
    pos = [d for d in dims if d > 0.0]
    return min(pos) if pos else 0.0


def _normalize(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < _EPS:
        return (0.0, 0.0, 1.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _cube_root(v: float) -> float:
    return v ** (1.0 / 3.0) if v > 0 else 0.0


# --------------------------------------------------------------------------- #
# Part geometry helpers (mirror interference.py)
# --------------------------------------------------------------------------- #
def _part_id(part: dict, index: Optional[int] = None) -> str:
    if index is None:
        return str(part.get("id", part.get("name", "part")))
    return str(part.get("id", part.get("name", f"part{index}")))


def _part_bbox(part: dict, cq_ok: bool) -> Optional[BBox]:
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


def _swept_common_volume(feat: dict, radius: float, length: float,
                         part_shape) -> Optional[float]:
    """Volume of the OCCT common of the swept tool cylinder and a part, or
    ``None`` when the kernel is unavailable / the operation failed."""
    try:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir

        pos = feat["pos"]
        axis = feat["axis"]
        frame = gp_Ax2(gp_Pnt(pos[0], pos[1], pos[2]),
                       gp_Dir(axis[0], axis[1], axis[2]))
        tool = BRepPrimAPI_MakeCylinder(frame, radius, length).Shape()

        wb = getattr(part_shape, "wrapped", part_shape)
        common = BRepAlgoAPI_Common(tool, wb)
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
def with_access(verifiers, rules: Optional[AccessRules] = None) -> List:
    """Return a new verifier list with an :class:`AccessCheck` appended
    (mirrors :func:`interference.with_interference`)."""
    return list(verifiers) + [AccessCheck(rules)]


# --------------------------------------------------------------------------- #
# Graceful-degradation helpers (mirror interference.py / dfm.py)
# --------------------------------------------------------------------------- #
def _query(backend, q: str) -> Optional[dict]:
    try:
        result = backend.query(q)
    except Exception:  # noqa: BLE001 - an unsupported query must degrade, not crash
        return None
    return result or None


def _warn(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.WARNING, code, msg, where)


def _info(code: str, msg: str, where: Optional[str] = None) -> Diagnostic:
    return Diagnostic(Severity.INFO, code, msg, where)


def _fmt(dims) -> str:
    return "[" + ", ".join(f"{d:g}" for d in dims) + "]"
