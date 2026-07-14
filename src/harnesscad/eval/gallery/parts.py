"""The gallery catalogue: sixteen distinct parts, each exercising a different capability.

Every entry is a :class:`Part`. A part declares, and is TESTED against:

``capability``
    The dotted capability module (``harnesscad.registry``) whose geometry the
    part actually depends on. If the module is not in the static index, the
    catalogue is lying and the test fails.
``operation``
    The :mod:`harnesscad.domain.geometry.services` operation name the part
    dispatches (``None`` when the part is pure CISP). Must be in
    ``services.names()``.
``cisp_ops``
    The CISP op tags the part emits. Must be in ``core.cisp.ops._REGISTRY``.
``backends``
    The backends that can GENUINELY build it, preferred one first.
``unsupported``
    Backends that provably CANNOT build it, with the reason in ``why_not``.
    These are not omissions -- they are the honest capability gaps, and the
    test suite asserts they really do fail.

Two build kinds:

``"ops"``
    ``builder()`` returns a CISP op stream (a list of op dicts). It is applied
    through :class:`harnesscad.io.surfaces.server.CISPServer` on the named
    backend and the resulting session is rasterised.
``"mesh"``
    ``builder()`` returns a ``(vertices, faces)`` triangle mesh straight from
    the geometry services / SDF fleet -- no CISP backend is involved, because
    the op set has no verb for it (helical sweeps, TPMS iso-surfaces, spirals).
    ``backends`` is then ``("services",)`` and the mesh is rendered directly.

Stdlib only. Deterministic: no randomness, no wall clock, fixed sample counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.geometry.features import enclosure, holes, screw_thread, sweep
from harnesscad.domain.geometry.features import thread_profile
from harnesscad.domain.geometry.kinematics import involute_gear
from harnesscad.domain.geometry.sdf import cam_profile
from harnesscad.domain.geometry.sdf import combinators as comb
from harnesscad.domain.geometry.sdf import primitives as prim
from harnesscad.domain.geometry.sdf import spiral as spiral_sdf
from harnesscad.domain.geometry.sdf import tpms
from harnesscad.io.backends import frep as frep_backend

__all__ = [
    "Part",
    "CATALOGUE",
    "names",
    "get",
    "SERVICES_BACKEND",
]

#: The pseudo-backend name for parts meshed straight out of the geometry
#: services (no CISP backend can express them: CISP has no helical-sweep,
#: iso-surface or spiral verb).
SERVICES_BACKEND = "services"

Vec3 = Tuple[float, float, float]
Mesh = Tuple[List[Vec3], List[Tuple[int, int, int]]]


@dataclass(frozen=True)
class Part:
    """One catalogued part: what it is, what it proves, and how to build it."""

    name: str
    summary: str                       # one line
    demonstrates: str                  # the capability, in words
    capability: str                    # dotted module in harnesscad.registry
    kind: str                          # "ops" | "mesh"
    builder: Callable[[], Any]
    backends: Tuple[str, ...]          # can build it; [0] is preferred
    operation: Optional[str] = None    # geometry-services op name
    cisp_ops: Tuple[str, ...] = ()     # CISP op tags emitted
    unsupported: Tuple[str, ...] = ()  # backends that provably cannot
    why_not: str = ""                  # why they cannot
    view: str = "hero"
    drawing: bool = False
    resolution: Optional[int] = None   # frep / SDF grid resolution
    notes: str = ""

    @property
    def backend(self) -> str:
        return self.backends[0]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "summary": self.summary,
            "demonstrates": self.demonstrates,
            "capability": self.capability,
            "operation": self.operation,
            "cisp_ops": list(self.cisp_ops),
            "kind": self.kind,
            "backend": self.backend,
            "backends": list(self.backends),
            "unsupported": list(self.unsupported),
            "why_not": self.why_not,
            "view": self.view,
            "drawing": self.drawing,
            "resolution": self.resolution,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# small helpers shared by the builders
# ---------------------------------------------------------------------------

def _sketch(plane: str = "XY") -> dict:
    return {"op": "new_sketch", "plane": plane}


def _rect(sk: str, x: float, y: float, w: float, h: float) -> dict:
    return {"op": "add_rectangle", "sketch": sk, "x": x, "y": y, "w": w, "h": h}


def _circle(sk: str, cx: float, cy: float, r: float) -> dict:
    return {"op": "add_circle", "sketch": sk, "cx": cx, "cy": cy, "r": r}


def _polyline(sk: str, pts: Sequence[Sequence[float]]) -> List[dict]:
    """Close ``pts`` into a loop of ``add_line`` ops.

    The frep / openscad / blender backends assemble three-or-more line entities
    into a closed polygon profile (``frep._profile_of``); the CadQuery backend's
    ``_build_profile`` ignores line entities entirely, which is exactly why the
    polygon-profile parts below list cadquery under ``unsupported``.
    """
    loop = list(pts) + [pts[0]]
    return [{"op": "add_line", "sketch": sk,
             "x1": float(a[0]), "y1": float(a[1]),
             "x2": float(b[0]), "y2": float(b[1])}
            for a, b in zip(loop[:-1], loop[1:])]


def _extrude(sk: str, d: float) -> dict:
    return {"op": "extrude", "sketch": sk, "distance": float(d)}


def _hole(ref: str, x: float, y: float, dia: float,
          depth: Optional[float] = None) -> dict:
    op = {"op": "hole", "face_or_sketch": ref, "x": float(x), "y": float(y),
          "diameter": float(dia), "kind": "simple"}
    if depth is None:
        op["through"] = True
    else:
        op["through"] = False
        op["depth"] = float(depth)
    return op


def _linear_extrude_mesh(polygon: Sequence[Sequence[float]], height: float,
                         twist_deg: float = 0.0, steps: int = 2,
                         scales: Optional[Sequence[float]] = None) -> Mesh:
    """Extrude a planar polygon to a prism via ``features.sweep.extrude_along_path``.

    A straight path is the degenerate sweep, so the same SolidPython-derived
    sweeper that makes the coil spring also makes every prism here -- one
    implementation, not two.
    """
    path = [(0.0, 0.0, height * i / (steps - 1)) for i in range(steps)]
    rotations = None
    if twist_deg:
        rotations = [twist_deg * i / (steps - 1) for i in range(steps)]
    return sweep.extrude_along_path(polygon, path, rotations=rotations,
                                    scales=list(scales) if scales else None,
                                    cap_ends=True)


def _tessellate(field: Callable[[Sequence[float]], float],
                bounds: Tuple[Vec3, Vec3], resolution: int) -> Mesh:
    """Iso-surface the SDF on the frep grid (marching cubes, welded)."""
    return frep_backend.tessellate(field, bounds, resolution)


def _merge(*meshes: Mesh) -> Mesh:
    """Concatenate meshes into one renderable body (no boolean is performed).

    Used only where the part is honestly an interpenetrating ASSEMBLY of solids
    (the bolt: head + shank + helical thread). The z-buffer resolves the union
    correctly at render time; no claim of a booleaned B-rep is made.
    """
    verts: List[Vec3] = []
    faces: List[Tuple[int, int, int]] = []
    for (v, f) in meshes:
        off = len(verts)
        verts.extend((float(p[0]), float(p[1]), float(p[2])) for p in v)
        faces.extend((a + off, b + off, c + off) for (a, b, c) in f)
    return verts, faces


def _regular_polygon(n: int, radius: float, phase: float = 0.0) -> List[Tuple[float, float]]:
    return [(radius * math.cos(phase + 2.0 * math.pi * i / n),
             radius * math.sin(phase + 2.0 * math.pi * i / n))
            for i in range(n)]


def _circle_pts(radius: float, n: int = 48) -> List[Tuple[float, float]]:
    return [(radius * math.cos(2.0 * math.pi * i / n),
             radius * math.sin(2.0 * math.pi * i / n))
            for i in range(n)]


def _involute(a: float) -> float:
    return math.tan(a) - a


# ---------------------------------------------------------------------------
# 1. bracket-counterbore -- hole features (holes.counterbore_hole)
# ---------------------------------------------------------------------------
#: The counterbore recipe, derived (not invented) by the hole-feature module.
BRACKET_CBORE = holes.counterbore_hole(diameter=6.6, cbore_diameter=11.0,
                                       cbore_depth=6.5, depth=12.0)
BRACKET_HOLES = ((-24.0, -14.0), (24.0, -14.0), (-24.0, 14.0), (24.0, 14.0))


def build_bracket() -> List[dict]:
    """A 70x40x12 mounting plate with four M6 counterbored holes.

    ``holes.counterbore_hole`` returns the stepped sections
    ``((r_cbore, r_cbore, cbore_depth), (r, r, depth - cbore_depth))``; each
    section becomes one CISP ``hole`` op -- a through pilot plus a blind
    enlarged pocket, both cut from the top face. That is a REAL counterbore on
    every backend, whereas ``Hole(kind="counterbore")`` is refused by the
    CadQuery backend as not-yet-realised.
    """
    (cb_r, _, cb_depth), (pilot_r, _, _) = BRACKET_CBORE.sections
    ops = [_sketch("XY"), _rect("sk1", -35.0, -20.0, 70.0, 40.0), _extrude("sk1", 12.0)]
    for (x, y) in BRACKET_HOLES:
        ops.append(_hole("f1", x, y, 2.0 * pilot_r))
        ops.append(_hole("f1", x, y, 2.0 * cb_r, depth=cb_depth))
    return ops


# ---------------------------------------------------------------------------
# 2/3. enclosure-shell + enclosure-lid -- the enclosure recipe
# ---------------------------------------------------------------------------
ENCLOSURE_SPEC = enclosure.EnclosureSpec(
    outer_width=70.0, outer_length=50.0, outer_height=28.0,
    thickness=2.4, side_radius=3.0, lip_height=3.0,
)
ENCLOSURE_PLAN = enclosure.plan_enclosure(ENCLOSURE_SPEC)


def build_enclosure_shell() -> List[dict]:
    """The enclosure BODY: a box hollowed to the plan's wall thickness.

    Every number comes from ``enclosure.plan_enclosure`` -- the outer envelope,
    the wall, the inner cavity -- so the solid and the recipe cannot drift.
    """
    s = ENCLOSURE_SPEC
    return [
        _sketch("XY"),
        _rect("sk1", -s.outer_width / 2.0, -s.outer_length / 2.0,
              s.outer_width, s.outer_length),
        _extrude("sk1", s.outer_height),
        {"op": "shell", "faces": [], "thickness": s.thickness},
    ]


def build_enclosure_lid() -> List[dict]:
    """The matching LID: a cover plate with the plan's locating lip.

    The lip footprint is the plan's inner cavity (``lip_width``/``lip_length``)
    and it stands ``lip_height`` proud of a ``thickness``-thick plate; the two
    extrusions are booleaned into one solid, so the lid drops into the shell.
    """
    s = ENCLOSURE_SPEC
    p = ENCLOSURE_PLAN
    lip_w, lip_l = p.lip_width, p.lip_length
    return [
        _sketch("XY"),
        _rect("sk1", -s.outer_width / 2.0, -s.outer_length / 2.0,
              s.outer_width, s.outer_length),
        _extrude("sk1", s.thickness),
        _sketch("XY"),
        _rect("sk2", -lip_w / 2.0, -lip_l / 2.0, lip_w, lip_l),
        _extrude("sk2", s.thickness + s.lip_height),
        {"op": "boolean", "kind": "union", "target": "", "tool": ""},
    ]


# ---------------------------------------------------------------------------
# 4. gear-spur-involute -- involute_gear
# ---------------------------------------------------------------------------
GEAR = involute_gear.gear_geometry(module=2.0, teeth=18, pressure_angle=20.0)


def gear_polygon(flank_steps: int = 6) -> List[Tuple[float, float]]:
    """The closed outline of one involute spur gear (CCW, root land + tip land).

    The flank is the true involute of the base circle: at radius ``r`` the
    pressure angle is ``acos(r_base / r)`` and the polar offset from the tooth
    centreline is ``half_pitch + inv(alpha) - inv(alpha_r)`` with
    ``inv(x) = tan(x) - x``. ``involute_gear.gear_geometry`` supplies every
    radius; nothing here is a fudge factor.
    """
    g = GEAR
    rb, rt, rr = g.base_radius, g.tip_radius, g.root_radius
    alpha = math.radians(g.pressure_angle)
    half = math.pi / (2.0 * g.teeth)
    r_start = max(rb, rr)

    def phi(r: float) -> float:
        return half + _involute(alpha) - _involute(math.acos(min(1.0, rb / r)))

    ps = phi(r_start)
    pts: List[Tuple[float, float]] = []
    for i in range(g.teeth):
        th = 2.0 * math.pi * i / g.teeth
        pts.append((rr * math.cos(th - ps), rr * math.sin(th - ps)))
        for k in range(flank_steps + 1):
            r = r_start + (rt - r_start) * k / flank_steps
            a = th - phi(r)
            pts.append((r * math.cos(a), r * math.sin(a)))
        for k in range(flank_steps, -1, -1):
            r = r_start + (rt - r_start) * k / flank_steps
            a = th + phi(r)
            pts.append((r * math.cos(a), r * math.sin(a)))
        pts.append((rr * math.cos(th + ps), rr * math.sin(th + ps)))
    return pts


def build_gear() -> List[dict]:
    """m=2, z=18, 20-degree involute spur gear, 10 mm face, 10 mm bore.

    288 ``add_line`` ops close the tooth outline into one polygon profile, then
    ``extrude`` + ``hole``. The OpenSCAD backend cuts this exactly (CGAL); the
    CadQuery backend cannot -- its sketch->profile builder only understands
    rectangles and circles, so a line-loop profile is an empty sketch to it.
    """
    ops = [_sketch("XY")]
    ops += _polyline("sk1", gear_polygon())
    ops.append(_extrude("sk1", 10.0))
    ops.append(_hole("f1", 0.0, 0.0, 10.0))
    return ops


# ---------------------------------------------------------------------------
# 5. bolt-m10-iso -- ISO thread profile + helical sweep
# ---------------------------------------------------------------------------
def build_bolt() -> Mesh:
    """An M10x1.5 hex-head bolt: ISO 60-degree tooth swept into a real helix.

    ``thread_profile.iso_thread`` gives the standardised 60-degree tooth section
    (the 7/8-H truncated V); ``screw_thread.thread`` sweeps it around the shank,
    climbing one pitch per revolution, with neck-in / neck-out ramps so the
    thread emerges from and sinks back into the shaft. Head, shank and thread
    are one renderable body (an interpenetrating assembly -- CISP has no
    helical-sweep op and no backend here can boolean a helix, so no B-rep union
    is claimed).
    """
    pitch, major_r = 1.5, 5.0
    section = thread_profile.iso_thread(radius=major_r, pitch=pitch, external=True)
    # iso_thread lives in (axial, radial); screw_thread.thread wants
    # (radial, axial) about the core radius, so swap and re-origin.
    core_r = min(y for (_, y) in section)
    tooth = [(y - core_r, x - pitch / 2.0) for (x, y) in section]

    thread_mesh = screw_thread.thread(
        tooth, inner_rad=core_r, pitch=pitch, length=26.0,
        segments_per_rot=64, neck_in_degrees=180.0, neck_out_degrees=180.0)
    # The head sits at z=24.5, i.e. BELOW the thread's top (z=26), so it buries
    # the helix's end cap -- exactly where a real bolt's thread run-out
    # disappears under the head. Leave it exposed and the cap's triangle fan
    # reads as a starburst on the shank. The shank likewise runs up into the
    # head so no two caps are ever coplanar (coplanar caps z-fight).
    shank = _linear_extrude_mesh(_circle_pts(core_r + 0.02, n=64), 28.0)
    head = _linear_extrude_mesh(_regular_polygon(6, 8.6, phase=math.pi / 6.0), 7.0)
    head_v = [(x, y, z + 24.5) for (x, y, z) in head[0]]
    return _merge(thread_mesh, shank, (head_v, head[1]))


# ---------------------------------------------------------------------------
# 6. gyroid-lattice -- TPMS (SDF-only)
# ---------------------------------------------------------------------------
GYROID_HALF = 12.0
GYROID_PERIOD = 12.0
GYROID_WALL = 1.1


def gyroid_field(p: Sequence[float]) -> float:
    """A gyroid sheet of wall ``2*GYROID_WALL``, trimmed to a cube."""
    sheet = abs(tpms.gyroid(p, period=GYROID_PERIOD)) - GYROID_WALL
    cube = prim.box_exact(p, (2.0 * GYROID_HALF, 2.0 * GYROID_HALF, 2.0 * GYROID_HALF))
    return comb.intersection(sheet, cube)


def build_gyroid() -> Mesh:
    """A 24 mm gyroid TPMS lattice cube. NO B-rep kernel can express this.

    The gyroid is an implicit triply-periodic minimal surface: it has no
    boundary representation, no sketch, no feature tree. It exists only as a
    field, and only the kernel-free SDF backend can mesh it.
    """
    b = GYROID_HALF + 1.0
    return _tessellate(gyroid_field, ((-b, -b, -b), (b, b, b)), 96)


# ---------------------------------------------------------------------------
# 7. blend-smooth-union -- SDF combinators (SDF-only)
# ---------------------------------------------------------------------------
def blend_field(p: Sequence[float]) -> float:
    """Sphere + cylinder + slab fused by a polynomial smooth-min (k = 7 mm).

    The three primitives genuinely INTERPENETRATE (sphere spans x=-23..3, the
    cylinder x=4..20, and both plunge into the slab): a smooth-min that is only
    bridging a gap produces a thin, non-Lipschitz neck that marching cubes then
    tears, which is an artefact of the setup, not of the operator. Overlapping
    them makes the blend a real fillet-free fusion.

    ``prim.sphere`` / ``prim.cylinder`` take DIAMETERS.
    """
    k = 7.0
    sphere = prim.sphere((p[0] + 10.0, p[1], p[2] - 13.0), 26.0)
    cyl = prim.cylinder((p[0] - 12.0, p[1], p[2] - 12.0), 16.0, 30.0)
    slab = prim.box_exact((p[0], p[1], p[2]), (64.0, 30.0, 10.0))
    return comb.smooth_union(comb.smooth_union(sphere, cyl, k), slab, k)


def build_blend() -> Mesh:
    """Three primitives fused with smooth blends -- not fillets.

    ``combinators.smooth_union`` interpolates the two FIELDS, so the transition
    is a continuous blend surface with no edge to select and no fillet radius to
    fail. A B-rep kernel has no operation with these semantics: it can only
    boolean-then-fillet, which produces a different (and here, unfilletable)
    solid.
    """
    return _tessellate(blend_field, ((-34.0, -18.0, -8.0), (24.0, 18.0, 30.0)), 140)


# ---------------------------------------------------------------------------
# 8. pulley-vgroove -- revolve
# ---------------------------------------------------------------------------
PULLEY_PROFILE = [
    (5.0, 0.0), (26.0, 0.0), (26.0, 4.0), (17.0, 11.0),
    (26.0, 18.0), (26.0, 22.0), (5.0, 22.0),
]


def build_pulley() -> List[dict]:
    """A V-groove belt pulley: a closed half-section revolved 360 degrees.

    The section is a line loop in the XZ plane; ``revolve`` spins it about the
    Y axis of that plane. CadQuery is again out: line entities never reach its
    profile builder.
    """
    ops = [_sketch("XZ")]
    ops += _polyline("sk1", PULLEY_PROFILE)
    ops.append({"op": "revolve", "sketch": "sk1",
                "axis": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "angle": 360.0})
    return ops


# ---------------------------------------------------------------------------
# 9. sweep-taper-duct -- extrude_along_path with per-station scales
# ---------------------------------------------------------------------------
def build_sweep_duct() -> Mesh:
    """A rounded-square section swept along a 90-degree bend and lofted down 40%.

    ``features.sweep.extrude_along_path`` (the SolidPython ``extrude_along_path``
    port) frames the section against the path tangent at every station and
    applies a per-station scale, so the sweep is also a loft. CISP has no sweep
    op that any backend realises -- the stub, cadquery and frep backends all
    return a typed 'not-yet-supported' for ``Sweep`` -- so this is built through
    the geometry service directly.
    """
    section = [(x * 1.0, y * 1.0) for (x, y) in _rounded_square(11.0, 3.5, 32)]
    stations = 36
    radius = 42.0
    path = []
    scales = []
    for i in range(stations):
        t = i / (stations - 1)
        a = math.radians(90.0) * t
        path.append((radius * math.sin(a), 0.0, radius * (1.0 - math.cos(a))))
        scales.append(1.0 - 0.4 * t)
    return sweep.extrude_along_path(section, path, scales=scales, cap_ends=True)


def _rounded_square(half: float, radius: float, n: int) -> List[Tuple[float, float]]:
    """A square of half-width ``half`` with corner radius ``radius``, CCW."""
    c = half - radius
    pts: List[Tuple[float, float]] = []
    per = max(2, n // 4)
    for k, (cx, cy, a0) in enumerate(((c, c, 0.0), (-c, c, math.pi / 2.0),
                                      (-c, -c, math.pi), (c, -c, 1.5 * math.pi))):
        for j in range(per):
            a = a0 + (math.pi / 2.0) * j / (per - 1)
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    return pts


# ---------------------------------------------------------------------------
# 10. coil-spring -- extrude_along_path along a helix
# ---------------------------------------------------------------------------
def build_spring() -> Mesh:
    """A 6-coil compression spring: a circular wire section swept along a helix.

    Same sweeper as the duct; the path is the helix. This is what a "spring" is:
    there is no primitive for it in any op set here.
    """
    wire_r, coil_r, pitch, coils = 2.2, 14.0, 9.0, 6
    steps_per_coil = 44
    n = coils * steps_per_coil + 1
    path = []
    for i in range(n):
        t = i / steps_per_coil
        a = 2.0 * math.pi * t
        path.append((coil_r * math.cos(a), coil_r * math.sin(a), pitch * t))
    return sweep.extrude_along_path(_circle_pts(wire_r, n=20), path, cap_ends=True)


# ---------------------------------------------------------------------------
# 11. pattern-heatsink -- LinearPattern
# ---------------------------------------------------------------------------
def build_heatsink() -> List[dict]:
    """Nine fins from ONE fin body via ``linear_pattern``, united onto a base.

    The pattern op replicates the BODY (not a face-level feature): the fin is
    extruded, patterned nine times at 8 mm pitch, and the base plate is unioned
    under the resulting comb.
    """
    return [
        _sketch("XY"), _rect("sk1", -25.0, -2.0, 50.0, 4.0), _extrude("sk1", 30.0),
        {"op": "linear_pattern", "feature": "f1",
         "direction": [0.0, 1.0, 0.0], "count": 9, "spacing": 8.0},
        _sketch("XY"), _rect("sk2", -25.0, -6.0, 50.0, 76.0), _extrude("sk2", 6.0),
        {"op": "boolean", "kind": "union", "target": "", "tool": ""},
    ]


# ---------------------------------------------------------------------------
# 12. pattern-flange -- CircularPattern
# ---------------------------------------------------------------------------
def build_flange() -> List[dict]:
    """A six-lug bored flange: ONE lug, ``circular_pattern`` 6x360, hub, bore.

    The circular counterpart of the heatsink: same body-level replication, about
    the Z axis, and each lug then carries a through hole on the bolt circle.
    """
    ops = [
        _sketch("XY"), _rect("sk1", 15.0, -6.5, 22.0, 13.0), _extrude("sk1", 9.0),
        {"op": "circular_pattern", "feature": "f1",
         "axis": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "count": 6, "angle": 360.0},
        _sketch("XY"), _circle("sk2", 0.0, 0.0, 21.0), _extrude("sk2", 18.0),
        {"op": "boolean", "kind": "union", "target": "", "tool": ""},
        _hole("f4", 0.0, 0.0, 18.0),
    ]
    for i in range(6):
        a = 2.0 * math.pi * i / 6.0
        ops.append(_hole("f4", 31.0 * math.cos(a), 31.0 * math.sin(a), 6.6))
    return ops


# ---------------------------------------------------------------------------
# 13/14. edge-fillet vs edge-chamfer -- the same block, one op apart
# ---------------------------------------------------------------------------
_EDGE_BLOCK = [_sketch("XY"), _rect("sk1", -25.0, -18.0, 50.0, 36.0), _extrude("sk1", 16.0)]


def build_edge_fillet() -> List[dict]:
    """The block with every edge ROUNDED (OCCT ``BRepFilletAPI_MakeFillet``, r=5)."""
    return list(_EDGE_BLOCK) + [{"op": "fillet", "edges": [], "radius": 5.0}]


def build_edge_chamfer() -> List[dict]:
    """The same block with every edge CUT BACK (OCCT chamfer, setback 5).

    Rendered from the same camera as ``edge-fillet``: side by side the two
    images are the difference between a rounded and a straight-cut edge -- two
    genuinely different solids, not a shading trick.
    """
    return list(_EDGE_BLOCK) + [{"op": "chamfer", "edges": [], "distance": 5.0}]


# ---------------------------------------------------------------------------
# 15. cam-three-arc -- the cam SDF (SDF-only)
# ---------------------------------------------------------------------------
CAM = cam_profile.make_three_arc_cam(lift=9.0, duration=math.radians(130.0),
                                     max_diameter=46.0, k=1.06)


def cam_field(p: Sequence[float]) -> float:
    """The three-arc cam, extruded 10 mm and bored 14 mm, with a keyway."""
    disc = prim.extrude(CAM.evaluate((p[0], p[1])), p[2], 10.0)
    bore = prim.cylinder((p[0], p[1], p[2]), 14.0, 40.0)
    key = prim.box_exact((p[0], p[1] - 8.2, p[2]), (4.0, 4.0, 40.0))
    return comb.difference(disc, comb.union(bore, key))


def build_cam() -> Mesh:
    """A three-arc automotive cam lobe from ``sdf.cam_profile``.

    The lobe is defined as an exact signed distance -- base circle, nose circle
    and the two flank arcs tangent to both. There is no sketch and no spline:
    the profile IS the field, so only the SDF backend can mesh it.
    """
    return _tessellate(cam_field, ((-25.0, -25.0, -7.0), (25.0, 34.0, 7.0)), 128)


# ---------------------------------------------------------------------------
# 16. spiral-flexure -- the exact spiral SDF (SDF-only)
# ---------------------------------------------------------------------------
SPIRAL = spiral_sdf.ArcSpiral(a=2.1, k=6.0, start=0.0, end=4.0 * math.pi, d=1.6)


def spiral_field(p: Sequence[float]) -> float:
    """A planar spiral flexure: the spiral band, a hub and an outer rim.

    ``prim.circle`` / ``prim.cylinder`` are parameterised by DIAMETER, so every
    literal here is a diameter: hub d=17, rim d=70 outer / d=64 inner, bore d=8.
    """
    band = prim.extrude(SPIRAL.evaluate((p[0], p[1])), p[2], 6.0)
    hub = prim.extrude(prim.circle((p[0], p[1]), 17.0), p[2], 6.0)
    rim_outer = prim.circle((p[0], p[1]), 70.0)
    rim_inner = -prim.circle((p[0], p[1]), 64.0)
    rim = prim.extrude(comb.intersection(rim_outer, rim_inner), p[2], 6.0)
    bore = prim.cylinder((p[0], p[1], p[2]), 8.0, 30.0)
    return comb.difference(comb.union_all([band, hub, rim]), bore)


def build_spiral() -> Mesh:
    """An Archimedean spiral flexure from the EXACT spiral SDF.

    ``sdf.spiral.ArcSpiral`` evaluates the true distance to r = a*theta + k by
    inverting the spiral in polar coordinates -- not by sampling a polyline.
    Offsetting that distance by ``d`` thickens the curve into a band; that is a
    one-line operation in a field and has no B-rep equivalent.
    """
    return _tessellate(spiral_field, ((-36.0, -36.0, -4.0), (36.0, 36.0, 4.0)), 160)


# ---------------------------------------------------------------------------
# the catalogue
# ---------------------------------------------------------------------------
_GEO = "harnesscad.domain.geometry."

CATALOGUE: Tuple[Part, ...] = (
    Part(
        name="bracket-counterbore",
        summary="70x40x12 mounting plate with four M6 counterbored holes.",
        demonstrates="Semantic hole features: the counterbore recipe (pilot + "
                     "flat-bottomed pocket) derived by holes.counterbore_hole.",
        capability=_GEO + "features.holes",
        operation="hole.counterbore",
        cisp_ops=("new_sketch", "add_rectangle", "extrude", "hole"),
        kind="ops",
        builder=build_bracket,
        backends=("cadquery", "freecad", "openscad", "blender", "frep"),
        view="hero",
        drawing=True,
        notes="The cross-backend comparison subject: built on all four kernels.",
    ),
    Part(
        name="enclosure-shell",
        summary="Hollowed 70x50x28 enclosure body, 2.4 mm wall.",
        demonstrates="The parametric enclosure recipe (enclosure.plan_enclosure) "
                     "driving a real OCCT shell/MakeThickSolid.",
        capability=_GEO + "features.enclosure",
        operation="feature.enclosure.plan",
        cisp_ops=("new_sketch", "add_rectangle", "extrude", "shell"),
        kind="ops",
        builder=build_enclosure_shell,
        backends=("cadquery", "freecad", "blender", "frep"),
        unsupported=("openscad",),
        why_not="the OpenSCAD backend has no shell op (OpenSCAD/CGAL has no "
                "MakeThickSolid; it returns unsupported-op).",
        view="hero",
        drawing=True,
    ),
    Part(
        name="enclosure-lid",
        summary="Matching lid: 2.4 mm cover plate with a 3 mm locating lip.",
        demonstrates="The other half of the enclosure recipe: the lip footprint "
                     "is the plan's cavity, unioned onto the cover plate.",
        capability=_GEO + "features.enclosure",
        operation="feature.enclosure.plan",
        cisp_ops=("new_sketch", "add_rectangle", "extrude", "boolean"),
        kind="ops",
        builder=build_enclosure_lid,
        backends=("cadquery", "freecad", "openscad", "blender", "frep"),
        view="hero",
    ),
    Part(
        name="gear-spur-involute",
        summary="m=2, z=18, 20-degree involute spur gear, 10 mm face, 10 mm bore.",
        demonstrates="A true involute tooth flank from involute_gear.gear_geometry, "
                     "cut exactly by CGAL as a 288-segment polygon profile.",
        capability=_GEO + "kinematics.involute_gear",
        operation="gear.involute.geometry",
        cisp_ops=("new_sketch", "add_line", "extrude", "hole"),
        kind="ops",
        builder=build_gear,
        backends=("openscad", "blender", "frep"),
        unsupported=("cadquery",),
        why_not="the CadQuery backend's _build_profile only realises rectangle "
                "and circle entities; a line-loop profile is an empty sketch to "
                "it (empty-sketch).",
        view="hero",
        drawing=True,
    ),
    Part(
        name="bolt-m10-iso",
        summary="M10x1.5 hex-head flange bolt with a real helical ISO thread.",
        demonstrates="thread_profile.iso_thread (the 60-degree truncated-V tooth) "
                     "swept by screw_thread.thread into an actual helix.",
        capability=_GEO + "features.screw_thread",
        operation="thread.helix",
        kind="mesh",
        builder=build_bolt,
        backends=(SERVICES_BACKEND,),
        unsupported=("cadquery", "frep", "openscad", "blender", "freecad"),
        why_not="CISP has no helical-sweep op: every backend refuses Sweep as "
                "not-yet-supported, so the helix can only come from the geometry "
                "service directly.",
        view="hero",
        notes="head + shank + thread are an interpenetrating assembly; no boolean "
              "union is claimed, the z-buffer resolves it.",
    ),
    Part(
        name="gyroid-lattice",
        summary="24 mm gyroid TPMS lattice cube, 2.2 mm sheet.",
        demonstrates="sdf.tpms.gyroid -- a triply-periodic minimal surface. It has "
                     "no B-rep, no sketch and no feature tree: it exists only as a field.",
        capability=_GEO + "sdf.tpms",
        operation="sdf.infill.gyroid",
        kind="mesh",
        builder=build_gyroid,
        backends=(SERVICES_BACKEND,),
        unsupported=("cadquery", "openscad", "blender", "freecad"),
        why_not="SDF-ONLY. No kernel in the harness can express an implicit "
                "minimal surface; there is no CISP op that names one.",
        view="hero",
        resolution=96,
    ),
    Part(
        name="blend-smooth-union",
        summary="Sphere + cylinder + slab fused by a k=6 polynomial smooth-min.",
        demonstrates="sdf.combinators.smooth_union -- a field-level blend, not a "
                     "boolean-then-fillet. There is no edge to select.",
        capability=_GEO + "sdf.combinators",
        kind="mesh",
        builder=build_blend,
        backends=(SERVICES_BACKEND,),
        unsupported=("cadquery", "openscad", "blender", "freecad"),
        why_not="SDF-ONLY. A B-rep kernel can only union then fillet; that is a "
                "different (and here unfilletable) solid.",
        view="hero",
        resolution=140,
    ),
    Part(
        name="pulley-vgroove",
        summary="52 mm V-groove belt pulley, revolved from a 7-point half-section.",
        demonstrates="revolve: a closed line-loop half-section spun 360 degrees "
                     "about the sketch plane's Y axis.",
        capability=_GEO + "features.revolve",
        operation="feature.revolve.pappus_volume",
        cisp_ops=("new_sketch", "add_line", "revolve"),
        kind="ops",
        builder=build_pulley,
        backends=("openscad", "blender", "frep"),
        unsupported=("cadquery",),
        why_not="same line-profile gap as the gear: CadQuery's profile builder "
                "ignores line entities (empty-sketch).",
        view="hero",
        drawing=True,
    ),
    Part(
        name="sweep-taper-duct",
        summary="Rounded-square section swept round a 90-degree bend, lofted to 60%.",
        demonstrates="features.sweep.extrude_along_path (the SolidPython port): "
                     "frame the section on the path tangent, scale per station.",
        capability=_GEO + "features.sweep",
        kind="mesh",
        builder=build_sweep_duct,
        backends=(SERVICES_BACKEND,),
        unsupported=("cadquery", "frep", "openscad", "blender", "freecad"),
        why_not="every backend returns not-yet-supported for the CISP Sweep and "
                "Loft ops; the sweeper is only reachable as a geometry service.",
        view="hero",
    ),
    Part(
        name="coil-spring",
        summary="6-coil compression spring, 2.2 mm wire on a 28 mm coil.",
        demonstrates="The same sweeper on a helical path -- what a spring actually "
                     "is. No op set in the harness has a 'spring' primitive.",
        capability=_GEO + "features.sweep",
        kind="mesh",
        builder=build_spring,
        backends=(SERVICES_BACKEND,),
        unsupported=("cadquery", "frep", "openscad", "blender", "freecad"),
        why_not="as sweep-taper-duct: the Sweep op is unsupported on every backend.",
        view="hero",
    ),
    Part(
        name="pattern-heatsink",
        summary="Nine-fin heatsink: one fin, linear_pattern x9, base plate unioned.",
        demonstrates="linear_pattern -- body-level replication along a direction, "
                     "then a boolean union with the base.",
        capability="harnesscad.io.backends.frep",
        cisp_ops=("new_sketch", "add_rectangle", "extrude", "linear_pattern", "boolean"),
        kind="ops",
        builder=build_heatsink,
        backends=("cadquery", "openscad", "blender", "frep"),
        view="hero",
    ),
    Part(
        name="pattern-flange",
        summary="Six-lug bored flange: one lug, circular_pattern 6x360, hub, bolt circle.",
        demonstrates="circular_pattern -- body-level replication about an axis, "
                     "plus a bolt-circle of through holes.",
        capability="harnesscad.io.backends.frep",
        cisp_ops=("new_sketch", "add_rectangle", "add_circle", "extrude",
                  "circular_pattern", "boolean", "hole"),
        kind="ops",
        builder=build_flange,
        backends=("cadquery", "openscad", "blender", "frep"),
        view="hero",
        drawing=True,
    ),
    Part(
        name="edge-fillet",
        summary="50x36x16 block, every edge rounded r=5 (OCCT fillet).",
        demonstrates="fillet -- BRepFilletAPI_MakeFillet on a real B-rep kernel. "
                     "Compare with edge-chamfer: same block, same camera.",
        capability="harnesscad.io.backends.cadquery",
        cisp_ops=("new_sketch", "add_rectangle", "extrude", "fillet"),
        kind="ops",
        builder=build_edge_fillet,
        backends=("cadquery", "freecad", "frep"),
        view="hero",
    ),
    Part(
        name="edge-chamfer",
        summary="The same block, every edge cut back 5 mm (OCCT chamfer).",
        demonstrates="chamfer -- a straight setback, NOT a rounded edge. The pair "
                     "(edge-fillet, edge-chamfer) is the visual difference.",
        capability="harnesscad.io.backends.cadquery",
        cisp_ops=("new_sketch", "add_rectangle", "extrude", "chamfer"),
        kind="ops",
        builder=build_edge_chamfer,
        backends=("cadquery", "freecad", "frep"),
        view="hero",
    ),
    Part(
        name="cam-three-arc",
        summary="Three-arc cam lobe, 9 mm lift, 14 mm bore with a keyway.",
        demonstrates="sdf.cam_profile.ThreeArcCam -- base circle, nose circle and "
                     "tangent flank arcs, as an EXACT signed distance.",
        capability=_GEO + "sdf.cam_profile",
        operation="sdf.cam.three_arc",
        kind="mesh",
        builder=build_cam,
        backends=(SERVICES_BACKEND,),
        unsupported=("cadquery", "openscad", "blender", "freecad"),
        why_not="SDF-ONLY: the profile is a distance function, not a sketch; "
                "there is no CISP op that can carry it to a kernel.",
        view="hero",
        resolution=128,
    ),
    Part(
        name="spiral-flexure",
        summary="Archimedean spiral flexure, hub + 2-turn band + rim, 6 mm thick.",
        demonstrates="sdf.spiral.ArcSpiral -- the EXACT distance to r = a*theta + k, "
                     "offset by d to thicken the curve into a band.",
        capability=_GEO + "sdf.spiral",
        operation="sdf.spiral",
        kind="mesh",
        builder=build_spiral,
        backends=(SERVICES_BACKEND,),
        unsupported=("cadquery", "openscad", "blender", "freecad"),
        why_not="SDF-ONLY: thickening a transcendental curve by an offset is one "
                "line in a field and has no B-rep equivalent here.",
        view="hero",
        resolution=160,
    ),
)

#: The part whose build is repeated on every kernel for the comparison strip.
COMPARE_PART = "bracket-counterbore"
COMPARE_BACKENDS: Tuple[str, ...] = ("frep", "cadquery", "openscad", "blender")

_BY_NAME: Dict[str, Part] = {p.name: p for p in CATALOGUE}


def names() -> List[str]:
    return [p.name for p in CATALOGUE]


def get(name: str) -> Part:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError("no gallery part named %r (%d catalogued: %s)"
                       % (name, len(CATALOGUE), ", ".join(names()))) from None
