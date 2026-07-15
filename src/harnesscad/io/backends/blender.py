"""BlenderBackend — CISP ops driven through a headless Blender.

Blender is a real mesh kernel. Its boolean modifier (``solver='EXACT'``, the
Mesh Arrangements / exact-predicate solver) computes genuine mesh set operations,
its bevel modifier is a genuine edge blend, and its solidify modifier is a
genuine shell. So unlike the SDF backend -- whose booleans are exact as *fields*
but whose exported mesh is limited by the marching-cubes grid -- this backend's
union / cut / intersect land on the true intersection curves.

How it runs
-----------
Blender is driven **headless and out of process**::

    blender --background --factory-startup --python <script> -- <spec> <out> <fmt>

``--factory-startup`` is what makes it deterministic: no user preferences, no
add-ons, no saved scene. The script is generated from the model, never typed by
hand, and it is content-addressed: the same op stream produces the same script,
which produces the same directory, which is a cache hit rather than a re-run.

What the script receives
------------------------
Not the op stream, and not the F-rep tree either -- a *lowered* build plan
(JSON), wrapped in an envelope that also stamps the digest of the script itself
(``{"script": <sha>, "plan": <node>}``) so the result cache is keyed on the
kernel recipe as well as on the model:

    {"t": "mesh",      "verts": [...], "faces": [...]}    a solid leaf
    {"t": "bool",      "op": union|cut|intersect, "a": .., "b": ..}
    {"t": "union_all", "children": [...]}
    {"t": "shell",     "child": .., "thickness": t}
    {"t": "bevel",     "child": .., "width": w, "segments": n}

Every leaf carries explicit **world-space** vertices: the sketch-plane mapping,
the mirror reflections and the pattern transforms are all baked into the vertex
coordinates here in Python (:func:`_lower`), where they are exact and testable,
instead of being re-derived from matrices inside Blender. What is left for
Blender is exactly the part only a kernel can do: the booleans, the bevel and
the solidify. Curved profiles are faceted with OpenSCAD's own $fn law
(:func:`external.segments_for`), so this backend and the OpenSCAD backend
tessellate a circle into the identical polygon and their volumes compare
directly.

Absent Blender, the constructor raises
:class:`~harnesscad.io.backends.base.BackendUnavailable` -- the CISP server falls
back to the stub with a note and the tests skip.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Chamfer, Fillet, Op
from harnesscad.domain.geometry.topology import selector_dsl
from harnesscad.domain.geometry.topology.selector_dsl import SelectorError
from harnesscad.domain.geometry.topology.selector_dsl import parse as parse_selector
from harnesscad.domain.programs.validate import bpy_script
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.external import (
    DEFAULT_SEGMENTS, ExternalToolBackend, _err,
    ccw, plane_axes, plane_normal, profile_loops, segments_for, slab, to_world,
)
from harnesscad.io.backends.frep import Node

#: The selector DSL, by path: the build script loads this very file into Blender's
#: own interpreter so that ``Fillet.edges`` means the same thing on both kernels.
SELECTOR_DSL_PATH = os.path.abspath(selector_dsl.__file__)

Vec3 = Tuple[float, float, float]
Point = Tuple[float, float, float]
Xform = Callable[[Vec3], Vec3]

#: Where a Windows/macOS/Linux installer puts blender. Globbed, never hard-coded
#: to a version: the newest matching install wins.
BLENDER_PATTERNS = (
    r"C:\Program Files\Blender Foundation\*\blender.exe",
    r"C:\Program Files (x86)\Blender Foundation\*\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/usr/share/blender/*/blender",
    "/opt/blender*/blender",
)

BLENDER_ENV = "HARNESSCAD_BLENDER"

#: exe path -> version string. One subprocess per executable per process.
_VERSIONS: Dict[str, str] = {}


def blender_version(executable: str) -> str:
    """``bpy.app.version_string`` of the Blender that will run the build.

    Part of the cache key, because :data:`BLENDER_PATTERNS` deliberately picks the
    NEWEST install: without this, upgrading Blender swaps the geometry kernel under
    an unchanged content-addressed key and every cached result silently belongs to
    the old kernel.
    """
    cached = _VERSIONS.get(executable)
    if cached is not None:
        return cached
    version = "unknown"
    try:
        proc = subprocess.run([executable, "--version"], capture_output=True,
                              text=True, timeout=60)
        for line in (proc.stdout or "").splitlines():
            if line.strip().lower().startswith("blender"):
                version = line.strip()
                break
    except (OSError, subprocess.SubprocessError):
        pass
    _VERSIONS[executable] = version
    return version

#: Segments Blender's bevel modifier uses for a fillet (a chamfer is 1 segment).
BEVEL_SEGMENTS = 8


# --------------------------------------------------------------------------
# lowering: F-rep tree -> a Blender build plan with world-space leaf meshes
# --------------------------------------------------------------------------
def _identity(p: Vec3) -> Vec3:
    return p


def _compose(outer: Xform, inner: Xform) -> Xform:
    return lambda p: outer(inner(p))


def _reflector(plane: str) -> Xform:
    nx, ny, nz = plane_normal(plane)

    def reflect(p: Vec3) -> Vec3:
        return (-p[0] if nx else p[0], -p[1] if ny else p[1], -p[2] if nz else p[2])

    return reflect


def _patterner(tr: Sequence[float]) -> Xform:
    """A pattern instance's rigid 3x4 matrix, as a point transform.

    It used to take ``(dx, dy, dz, angle_about_Z)``, which is why
    ``CircularPattern.axis`` was dead here: the tree could not carry any axis but Z.
    frep now emits a full rigid matrix (``frep.rigid_transform``), so the axis the
    op names is the axis the mesh is rotated about.
    """
    m = [float(v) for v in tr]

    def place(p: Vec3) -> Vec3:
        x, y, z = p
        return (m[0] * x + m[1] * y + m[2] * z + m[3],
                m[4] * x + m[5] * y + m[6] * z + m[7],
                m[8] * x + m[9] * y + m[10] * z + m[11])

    return place




def prism(loop: Sequence[Tuple[float, float]], plane: str, w0: float, w1: float,
          xform: Xform) -> Tuple[List[Vec3], List[List[int]]]:
    """A closed prism: a CCW loop swept from ``w0`` to ``w1`` along the plane normal.

    Bottom cap, top cap and one quad per loop edge -- a watertight, 2-manifold
    polyhedron. Blender re-derives the face normals (``recalc_face_normals``), so
    a sketch-plane mapping with a negative determinant (frep's XZ frame is one)
    cannot leave the solid inside-out.
    """
    lo, hi = slab(float(w0), float(w1))
    n = len(loop)
    verts: List[Vec3] = [xform(to_world(plane, u, v, lo)) for (u, v) in loop]
    verts += [xform(to_world(plane, u, v, hi)) for (u, v) in loop]
    faces: List[List[int]] = [list(reversed(range(n))), list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])
    return verts, faces


def frustum(cu: float, cv: float, r0: float, r1: float, plane: str,
            w0: float, w1: float, segments: int,
            xform: Xform) -> Tuple[List[Vec3], List[List[int]]]:
    """A truncated cone: radius ``r0`` at ``w0`` opening to ``r1`` at ``w1``.

    This is a COUNTERSINK. It is the same topology as :func:`prism` -- two caps and
    a quad per segment -- with a different radius at each end, which is the whole
    difference between a countersink and the plain cylinder Blender used to build
    for it. The facet count comes from the larger rim, so the cone and the bore it
    opens out of are tessellated compatibly.
    """
    lo, hi = slab(float(w0), float(w1))
    ra, rb = (r0, r1) if float(w0) <= float(w1) else (r1, r0)
    pts = _circle_pts(max(abs(ra), abs(rb)), segments)
    n = len(pts)
    unit = [(px / max(abs(ra), abs(rb)), py / max(abs(ra), abs(rb))) for (px, py) in pts]
    verts: List[Vec3] = [xform(to_world(plane, cu + ux * ra, cv + uy * ra, lo))
                         for (ux, uy) in unit]
    verts += [xform(to_world(plane, cu + ux * rb, cv + uy * rb, hi))
              for (ux, uy) in unit]
    faces: List[List[int]] = [list(reversed(range(n))), list(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])
    return verts, faces


def revolve_mesh(loop: Sequence[Tuple[float, float]], plane: str,
                 axis: Sequence[float], angle: float, segments: int,
                 xform: Xform) -> Tuple[List[Vec3], List[List[int]]]:
    """A solid of revolution: ``loop`` swept about the in-plane axis.

    The local frame matches frep's revolve exactly: ``d`` is the axis direction
    in the sketch plane, ``n`` the in-plane radial direction, and the plane's own
    normal is the second radial direction -- so a point at profile coordinate
    ``(s, perp)`` sweeps the circle of radius ``perp`` about the axis line, which
    is precisely the field frep evaluates.
    """
    au, av, du, dv, nu, nv = (float(x) for x in axis)
    origin = to_world(plane, au, av, 0.0)
    ex = to_world(plane, nu, nv, 0.0)          # radial direction 1 (in-plane)
    ey = to_world(plane, 0.0, 0.0, 1.0)        # radial direction 2 (plane normal)
    ez = to_world(plane, du, dv, 0.0)          # the axis
    # profile in (radius, height-along-axis) coordinates
    prof = [(((u - au) * nu + (v - av) * nv), ((u - au) * du + (v - av) * dv))
            for (u, v) in loop]
    sweep = abs(float(angle))
    full = sweep >= 360.0
    span = 360.0 if full else sweep
    rings = max(3, int(math.ceil(segments * span / 360.0)))
    steps = rings if full else rings + 1

    def at(rad: float, s: float, theta: float) -> Vec3:
        c, sn = math.cos(theta), math.sin(theta)
        return xform((
            origin[0] + ex[0] * rad * c + ey[0] * rad * sn + ez[0] * s,
            origin[1] + ex[1] * rad * c + ey[1] * rad * sn + ez[1] * s,
            origin[2] + ex[2] * rad * c + ey[2] * rad * sn + ez[2] * s,
        ))

    m = len(prof)
    verts: List[Vec3] = []
    for i in range(steps):
        theta = math.radians(span * i / float(rings))
        for (rad, s) in prof:
            verts.append(at(rad, s, theta))
    faces: List[List[int]] = []
    ring_count = rings if full else rings
    for i in range(ring_count):
        a0 = i * m
        a1 = ((i + 1) % steps if full else i + 1) * m
        for j in range(m):
            k = (j + 1) % m
            faces.append([a0 + j, a0 + k, a1 + k, a1 + j])
    if not full:
        faces.append(list(reversed(range(m))))                       # start cap
        last = (steps - 1) * m
        faces.append(list(range(last, last + m)))                    # end cap
    return verts, faces


def _leaf(name: str, verts: Sequence[Vec3], faces: Sequence[Sequence[int]]) -> dict:
    return {"t": "mesh", "name": name,
            "verts": [[float(v[0]), float(v[1]), float(v[2])] for v in verts],
            "faces": [[int(i) for i in f] for f in faces]}


class _Lowering:
    """Walks the F-rep tree and emits the Blender build plan."""

    def __init__(self, segments: int) -> None:
        self.segments = int(segments)
        self._n = 0

    def _name(self, kind: str) -> str:
        self._n += 1
        return "%s_%03d" % (kind, self._n)

    def node(self, node: Node, xform: Xform) -> dict:
        t = node.t
        if t == "extrude":
            plane = node.d["plane"]
            loops = profile_loops(node.d["profile"], self.segments)
            parts = [_leaf(self._name("extrude"),
                           *prism(loop, plane, node.d["w0"], node.d["w1"], xform))
                     for loop in loops]
            return self._union(parts)
        if t == "cyl":
            plane = node.d["plane"]
            r = float(node.d["r"])
            loop = ccw([(node.d["cu"] + px, node.d["cv"] + py)
                        for (px, py) in _circle_pts(r, self.segments)])
            return _leaf(self._name("cyl"),
                         *prism(loop, plane, node.d["w0"], node.d["w1"], xform))
        if t == "cone":
            return _leaf(self._name("cone"),
                         *frustum(float(node.d["cu"]), float(node.d["cv"]),
                                  float(node.d["r0"]), float(node.d["r1"]),
                                  node.d["plane"], node.d["w0"], node.d["w1"],
                                  self.segments, xform))
        if t == "revolve":
            plane = node.d["plane"]
            loops = profile_loops(node.d["profile"], self.segments)
            parts = [_leaf(self._name("revolve"),
                           *revolve_mesh(loop, plane, node.d["axis"],
                                         node.d["angle"], self.segments, xform))
                     for loop in loops]
            return self._union(parts)
        if t == "bool":
            return {"t": "bool", "op": node.d["op"],
                    "a": self.node(node.d["a"], xform),
                    "b": self.node(node.d["b"], xform)}
        if t == "blend":
            # The fillet/chamfer AT ITS POSITION IN THE HISTORY -- see _lower.
            width = float(node.d.get("value") or 0.0)
            return {"t": "bevel", "child": self.node(node.d["child"], xform),
                    "width": width,
                    "segments": (BEVEL_SEGMENTS
                                 if node.d.get("kind") == "fillet" else 1),
                    "selector": join_selectors(node.d.get("selectors") or ())}
        if t == "shell":
            return {"t": "shell", "child": self.node(node.d["child"], xform),
                    "thickness": float(node.d["thickness"]),
                    "faces": list(node.d.get("faces") or ())}
        if t == "mirror":
            child = node.d["child"]
            flip = _compose(xform, _reflector(node.d["plane"]))
            return self._union([self.node(child, xform), self.node(child, flip)])
        if t == "pattern":
            kids = [self.node(node.d["child"], _compose(xform, _patterner(tr)))
                    for tr in node.d["transforms"]]
            return self._union(kids)
        raise ValueError("blender backend: unknown F-rep node kind %r" % t)

    @staticmethod
    def _union(parts: List[dict]) -> dict:
        if not parts:
            raise ValueError("blender backend: an empty profile reached the lowering")
        if len(parts) == 1:
            return parts[0]
        return {"t": "union_all", "children": parts}


def _circle_pts(r: float, segments: int):
    from harnesscad.domain.geometry.parametric import facets
    return facets.circle_fragment_points(abs(float(r)), fn=float(segments))


def join_selectors(selectors) -> str:
    """The CISP selector *tuple* as ONE selector string, or "" for "every edge".

    CISP carries a tuple (``Fillet.edges``); the selector DSL evaluates one
    expression. Per CadQuery's own selectors doc ("Combining Selectors") string
    selectors compose with ``or``, so a tuple is a union -- which is what the
    CadQuery backend does with the same field, so both kernels select the same
    edges from the same op.
    """
    parts = [str(s).strip() for s in (selectors or ()) if str(s).strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return " or ".join("(%s)" % p for p in parts)


def _blends(features) -> List[Tuple[str, float, str]]:
    """The (kind, width, selector) of every Fillet/Chamfer in the op stream, in order.

    ``Fillet``/``Chamfer`` are not F-rep nodes -- they are a rewrite of the tree
    (``frep.blend_tree``) that stamps a radius onto every leaf, and reading the
    radius back off the tree (the old ``blend_radius``) loses two things a CAD
    fillet cannot do without: WHICH edges, and the fact that there may be SEVERAL
    fillets at different radii. The frep model records each blend as a feature with
    its ``edges`` and ``value``, so that is where the blender lowering reads them.
    """
    out: List[Tuple[str, float, str]] = []
    for f in features or ():
        kind = f.get("type")
        if kind in ("fillet", "chamfer"):
            out.append((kind, float(f.get("value", 0.0)),
                        join_selectors(f.get("edges", ()))))
    return out


def _lower(root: Node, segments: int, blends=()) -> dict:
    """The whole model as a Blender build plan (world-space leaves + kernel ops).

    ``blends`` is a LEGACY tail: fillets and chamfers are now ``blend`` NODES in the
    F-rep tree (frep's ``_blend``), lowered at their own position in the feature
    history by :meth:`_Lowering.node`. They used to be collected from the op log and
    wrapped around the finished root, which meant every fillet went on LAST -- so
    ``LinearPattern(feature="f1")`` (pattern the pad) and ``feature="f2"`` (pattern
    the pad AND its fillet) built the same object. It is kept only so a plan built
    from a tree that carries none still applies them.
    """
    plan = _Lowering(segments).node(root, _identity)
    for (kind, width, selector) in blends:
        if width <= 0.0:
            continue
        plan = {"t": "bevel", "child": plan, "width": width,
                "segments": BEVEL_SEGMENTS if kind == "fillet" else 1,
                "selector": selector}
    return plan


# --------------------------------------------------------------------------
# the bpy script (constant text; the model travels as JSON beside it)
# --------------------------------------------------------------------------
BUILD_SCRIPT = r'''"""Generated by harnesscad BlenderBackend. Do not edit.

Reads a build plan (world-space leaf meshes + kernel operations), realises it
with Blender's exact boolean / bevel / solidify modifiers, and writes an STL.
"""
import importlib.util
import json
import math
import struct
import sys

import bmesh
import bpy

_argv = sys.argv[sys.argv.index("--") + 1:]
SPEC_PATH, OUT_PATH, OUT_FMT, SELECTOR_PATH = _argv[0], _argv[1], _argv[2], _argv[3]


def _load_module(name, path):
    """Load one harnesscad module into Blender's interpreter, by path.

    The selector DSL is stdlib-only and is the SAME code the CadQuery backend
    parses ``Fillet.edges`` with, so both kernels select the same edges from the
    same op. It is loaded by file path rather than by ``import harnesscad...``
    so that Blender's Python never has to import the package (and its
    dependencies) at all.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: @dataclass resolves its annotations through
    # sys.modules[cls.__module__], so a module that is not registered blows up in
    # dataclasses itself.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


selector_dsl = _load_module("harnesscad_selector_dsl", SELECTOR_PATH)

BOOL_OPS = {"union": "UNION", "cut": "DIFFERENCE", "intersect": "INTERSECT"}

#: An edge of the MODEL is one whose adjacent faces turn by at least this much: it
#: keeps a box's 90-degree edges and a hole's rim and drops the 360/n-degree seams a
#: faceted cylinder only has because it was faceted. 30 degrees is Blender's own
#: BevelModifier.angle_limit default.
BEVEL_ANGLE_LIMIT = math.radians(30.0)

#: Weld / degenerate tolerance for a leaf mesh. Well below any feature size we can
#: express, well above float noise in the lowering.
WELD_DIST = 1e-6


def clear_scene():
    """Empty the scene AND the datablocks. --factory-startup already gives a fresh
    process; this keeps a single process deterministic if it ever builds twice."""
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for me in list(bpy.data.meshes):
        bpy.data.meshes.remove(me, do_unlink=True)
    # Blender's unit system can silently scale exported geometry; factory startup
    # is 1.0 (METRIC), and we pin it so one model unit is one exported unit.
    bpy.context.scene.unit_settings.system = "NONE"
    bpy.context.scene.unit_settings.scale_length = 1.0


def activate(ob):
    bpy.context.view_layer.objects.active = ob
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    ob.select_set(True)


def make_object(name, verts, faces):
    """A leaf solid, cleaned into the manifold mesh the EXACT solver requires.

    from_pydata does NO validation, and the boolean manual is explicit that the
    solver's guarantee has a precondition: "Only Manifold meshes are guaranteed to
    give proper results". So every leaf is welded (remove_doubles), stripped of
    zero-length edges and zero-area faces (dissolve_degenerate), given outward
    normals (recalc_face_normals) and finally run through Mesh.validate() -- which
    "validate geometry, return True when the mesh has had invalid geometry
    corrected/removed" -- before any kernel op sees it.
    """
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [list(f) for f in faces])
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=WELD_DIST)
    bmesh.ops.dissolve_degenerate(bm, dist=WELD_DIST, edges=bm.edges[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.validate(verbose=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def apply_modifiers(ob):
    activate(ob)
    for mod in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def boolean(a, b, op):
    mod = a.modifiers.new(name="bool", type="BOOLEAN")
    mod.operand_type = "OBJECT"
    mod.operation = BOOL_OPS[op]
    mod.object = b
    # BooleanModifier.solver: 'EXACT' -- "Slower solver with the best results for
    # coplanar faces" (bpy.types.BooleanModifier). 'FLOAT' is the fast solver and
    # is explicitly documented as "without support for overlapping geometry", which
    # is exactly the case a CAD cut hits (a hole tool flush with a face). 'MANIFOLD'
    # is faster but only valid on manifold input. EXACT is the only CAD-correct one.
    mod.solver = "EXACT"
    if mod.solver != "EXACT":
        # The enum is FLOAT / EXACT / MANIFOLD. 'FLOAT' is documented as a "Simple
        # solver with good performance, WITHOUT SUPPORT FOR OVERLAPPING GEOMETRY" --
        # which is every CAD case (a pocket bottom flush with a face, a through-hole
        # cap, two abutting pattern instances). Landing on it would silently produce
        # garbage, so a failed assignment is a hard stop, not a fallback.
        raise SystemExit("blender: boolean solver is %r, not EXACT" % mod.solver)
    # "Allow self-intersection in operands" -- a mirror/pattern union can hand the
    # solver two copies that touch, so this must be on.
    mod.use_self = True
    # "Better results when there are holes (slots) in the geometry overlapping the
    # cutting object" (BooleanModifier.use_hole_tolerant). This was FALSE, on the
    # premise "our operands are watertight". That premise is a category error: the
    # operands being manifold does NOT make the RESULT manifold. When a hole's
    # cylindrical wall is TANGENT to a face or edge of the target (a bore whose rim
    # just kisses a box corner -- Hole(x, diameter, depth) placed so d/2 reaches the
    # side), the EXACT solver's intersection curve degenerates to a single shared
    # edge/vertex and it emits a NON-MANIFOLD intermediate -- watertight-looking,
    # 2-manifold-failing, and (chained into the next boolean, whose manual demands a
    # manifold input) the reason those holes came back as an unexplained REJ. The
    # tolerant path resolves exactly that near-degenerate contact. It is slower;
    # correctness on a tangent cut is worth more than the speed, the same trade this
    # function already makes choosing EXACT over FLOAT.
    mod.use_hole_tolerant = True
    apply_modifiers(a)
    # The boolean RESULT, not just the leaves, must be handed on manifold: a chained
    # cut (a.cut(b).cut(c)) feeds this intermediate straight back into the solver,
    # whose guarantee holds only on manifold input. Clean it the same way a leaf is
    # cleaned in make_object -- weld coincident verts, drop the zero-area slivers a
    # tangent cut can leave, re-derive outward normals -- and validate().
    _heal(a)
    bpy.data.objects.remove(b, do_unlink=True)
    return a


def _heal(ob):
    """Weld/dissolve/recalc/validate a boolean result into a manifold mesh.

    The same conservative pipeline make_object runs on every leaf, applied to the
    intermediate so the next kernel op sees a manifold input the EXACT solver can
    honour. Every step only REMOVES invalid geometry (coincident verts, zero-area
    faces) or reorients normals; none moves a real surface, so a clean result is
    left byte-for-byte unchanged.
    """
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=WELD_DIST)
    bmesh.ops.dissolve_degenerate(bm, dist=WELD_DIST, edges=bm.edges[:])
    # SEAL the boundary the EXACT solver leaves open on a near-degenerate cut. A
    # BLIND hole whose tool straddles a face boundary (its flat pocket floor cuts
    # ACROSS the material edge -- Hole(x, y=0, depth<through) on a box) makes EXACT
    # drop the floor/wall seam: the result comes back with open boundary edges and
    # scores non-manifold, where cadquery/openscad/frep all build the same part
    # watertight. The true solid IS closed, so any boundary edge remaining after
    # the weld is an artefact of that dropped seam; holes_fill re-caps the loops it
    # bounds. On a clean cut there are NO boundary edges, so this is a no-op -- it
    # only ever acts where the solver already failed.
    boundary = [e for e in bm.edges if e.is_boundary]
    if boundary:
        bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    me.validate(verbose=False)
    me.update()


def build(node):
    kind = node["t"]
    if kind == "mesh":
        return make_object(node["name"], node["verts"], node["faces"])
    if kind == "bool":
        return boolean(build(node["a"]), build(node["b"]), node["op"])
    if kind == "union_all":
        kids = [build(c) for c in node["children"]]
        out = kids[0]
        for other in kids[1:]:
            out = boolean(out, other, "union")
        return out
    if kind == "shell":
        ob = build(node["child"])
        # OPEN THE NAMED FACES FIRST. Shell.faces was dropped entirely here, so
        # Shell(faces=(">Z",)) and Shell(faces=("<Z",)) built the SAME closed,
        # hollow box -- the field was accepted and ignored. Solidify has no notion
        # of an opening; the way you shell with one in Blender is to DELETE the
        # faces and then solidify, and use_rim closes the wall around the hole. So
        # the named faces are deleted from the mesh before the modifier runs.
        open_faces(ob, node.get("faces") or [])
        mod = ob.modifiers.new(name="solidify", type="SOLIDIFY")
        mod.solidify_mode = "EXTRUDE"          # Simple: our input is manifold
        mod.thickness = node["thickness"]
        # THE SHELL RULE. Manual, Solidify > Offset: "A value between (-1 to 1) to
        # locate the solidified output inside or outside the original mesh. The
        # inside and outside is determined by the face normals. Set to 0.0, the
        # solidified output will be CENTERED on the original mesh."
        # Centred (the old offset=0.0) pushes half the thickness OUTWARD, which grew
        # every shelled part -- a 60x40x20 box at t=3 measured 61.732 x 41.732 x
        # 21.732 (the vertices ride the averaged corner normal, so each side gained
        # (t/2)/sqrt(3) = 0.866). A CAD shell hollows INWARD and must not move the
        # outer surface, so the offset is -1: the whole thickness goes to the inside
        # of the face normals (which recalc_face_normals has made point outward).
        mod.offset = -1.0
        # "Maintain thickness by adjusting for sharp corners" (use_even_offset) and
        # "Calculate normals which result in more even thickness" (use_quality_normals).
        # Without them the inner surface is offset along the *vertex* normal, so a box
        # corner walks the diagonal and the wall comes out t/sqrt(3) thin. With them
        # the 60x40x20/t=3 wall volume is exactly 48000 - 54*34*14 = 22296.
        mod.use_even_offset = True
        mod.use_quality_normals = True
        mod.use_rim = True                     # close the shell on boundary edges
        mod.use_rim_only = False               # keep the walls, not just the rim
        mod.thickness_clamp = 0.0              # never rescale the requested wall
        mod.use_flip_normals = False
        apply_modifiers(ob)
        return ob
    if kind == "bevel":
        return bevel(build(node["child"]), node["width"], int(node["segments"]),
                     node.get("selector", ""))
    raise ValueError("unknown build-plan node %r" % kind)


def model_edges(bm):
    """The MODEL's edges: the ones a B-rep kernel would call edges.

    A mesh has an edge between every pair of adjacent faces, including the seams
    that only exist because a cylinder was faceted. Those are not edges of the
    part -- OCCT's cylinder has no seams to fillet -- so the candidate set is the
    edges whose adjacent faces actually turn: BevelModifier.limit_method 'ANGLE',
    "Only bevel edges with sharp enough angles between faces". At the default 30
    degrees this keeps a box's 90-degree edges and a hole's rim, and drops a
    360/n-degree facet seam. Selectors are then evaluated over THIS set, so a
    selector can never pick a seam that is not a real edge of the part.
    """
    out = []
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        if e.calc_face_angle(0.0) >= BEVEL_ANGLE_LIMIT:
            out.append(e)
    return out


def select_edges(bm, selector):
    """The edges a CISP Fillet/Chamfer names, or every model edge if it names none."""
    edges = model_edges(bm)
    if not selector or not edges:
        return edges
    entities = []
    for i, e in enumerate(edges):
        a, b = e.verts[0].co, e.verts[1].co
        center = ((a.x + b.x) * 0.5, (a.y + b.y) * 0.5, (a.z + b.z) * 0.5)
        tangent = (b - a)
        axis = (0.0, 0.0, 0.0)
        if tangent.length > 1e-12:
            tangent = tangent.normalized()
            axis = (tangent.x, tangent.y, tangent.z)
        entities.append(selector_dsl.Entity(center=center, axis=axis,
                                            geom_type="LINE", name=str(i)))
    chosen = selector_dsl.select(selector, entities)
    picked = [edges[int(e.name)] for e in chosen]
    if not picked:
        raise SystemExit("blender: edge selector %r selected no edges" % selector)
    return picked


#: The face names frep's shell node carries, as (axis, +1 max / -1 min). Kept in
#: step with frep.SHELL_FACES -- frep canonicalises every selector down to these
#: six spellings, so this table only ever sees "+x".."-z".
FACE_AXIS = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1),
             "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}


def open_faces(ob, faces):
    """Delete the named faces, so the solidify below leaves an OPENING there.

    A face is "the +z face" if its normal points that way AND it sits at the
    extreme of the mesh along that axis -- both tests, because a hole's inner wall
    can have an upward-facing ring that is nowhere near the top of the part.
    """
    if not faces:
        return
    wanted = [FACE_AXIS[str(f).strip().lower()] for f in faces
              if str(f).strip().lower() in FACE_AXIS]
    if not wanted:
        return
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    coords = [v.co for v in bm.verts]
    if not coords:
        bm.free()
        return
    lo = [min(c[i] for c in coords) for i in range(3)]
    hi = [max(c[i] for c in coords) for i in range(3)]
    tol = 1e-4
    doomed = []
    for f in bm.faces:
        n = f.normal
        c = f.calc_center_median()
        for (axis, sign) in wanted:
            if abs(n[axis] - sign) > 1e-3:
                continue
            extreme = hi[axis] if sign > 0 else lo[axis]
            if abs(c[axis] - extreme) <= tol:
                doomed.append(f)
                break
    if not doomed:
        raise SystemExit("blender: shell face selector matched no face")
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def bevel(ob, width, segments, selector):
    """A fillet / chamfer on THE EDGES THE OP NAMED -- via bmesh, not the modifier.

    BevelModifier.limit_method is a global heuristic, not a selection: 'ANGLE' only
    "bevels edges whose angle of adjacent face normals plus the defined Angle is
    less than 180 degrees" and 'NONE' bevels "the entire mesh by a constant amount".
    Neither can express "these four edges", so a modifier stack cannot honour
    ``Fillet.edges`` at all, and cannot carry two different radii in one model.

    bmesh.ops.bevel takes the edge list itself -- ``geom``: "Input edges and
    vertices" -- so the op's selector picks exactly the BMEdges it names and nothing
    else. That is the exact-edge route, and it is per-feature, so two fillets at two
    radii compose.
    """
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.edges.ensure_lookup_table()
    geom = select_edges(bm, selector)
    bmesh.ops.bevel(
        bm,
        geom=geom,
        offset=float(width),
        # 'OFFSET' -- "Amount is offset of new edges from original": on a 90-degree
        # edge the width IS the fillet radius / the chamfer setback.
        offset_type="OFFSET",
        profile_type="SUPERELLIPSE",
        # 1 segment is a chamfer (one flat face); >1 at profile 0.5 is a circular arc
        # ("The profile shape (0.5 = round)").
        segments=int(segments),
        profile=0.5,
        affect="EDGES",                # the bmesh op defaults to VERTICES
        # clamp_overlap would SILENTLY SHRINK a fillet too big for its edge -- a
        # wrong part with no diagnostic. Left off: an impossible fillet produces
        # visibly broken geometry, which regenerate() reports as invalid-mesh.
        clamp_overlap=False,
        # "more even bevel widths" with loop slide off (manual, Bevel > Loop Slide);
        # Blender's default True is the artistic choice, not the CAD one.
        loop_slide=False,
        material=-1,
        miter_outer="SHARP",
        miter_inner="SHARP",
        harden_normals=False,
        mark_seam=False,
        mark_sharp=False,
    )
    bm.to_mesh(me)
    bm.free()
    me.update()
    return ob


def triangles_of(ob):
    """World-space triangles of the evaluated object."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = ob.evaluated_get(depsgraph)
    me = evaluated.to_mesh()
    me.calc_loop_triangles()
    matrix = ob.matrix_world
    out = []
    for tri in me.loop_triangles:
        out.append([tuple(matrix @ me.vertices[i].co) for i in tri.vertices])
    evaluated.to_mesh_clear()
    return out


def write_binary_stl(path, tris):
    with open(path, "wb") as fh:
        fh.write(b"harnesscad-blender".ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(tris)))
        for (a, b, c) in tris:
            ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            length = (nx * nx + ny * ny + nz * nz) ** 0.5
            if length:
                nx, ny, nz = nx / length, ny / length, nz / length
            fh.write(struct.pack("<3f", nx, ny, nz))
            for p in (a, b, c):
                fh.write(struct.pack("<3f", p[0], p[1], p[2]))
            fh.write(struct.pack("<H", 0))


def main():
    with open(SPEC_PATH, "r") as fh:
        spec = json.load(fh)
    clear_scene()
    ob = build(spec["plan"])
    activate(ob)
    if OUT_FMT == "glb":
        # export_yup defaults to TRUE: the glTF exporter converts Blender's Z-up
        # frame to glTF's Y-up one, i.e. it silently ROTATES the part -90 degrees
        # about X. Every other format this backend emits (and every other backend)
        # is Z-up in model units, so the part must not be re-framed on the way out.
        bpy.ops.export_scene.gltf(filepath=OUT_PATH, export_format="GLB",
                                  use_selection=True, export_yup=False,
                                  export_apply=True)
        return
    tris = triangles_of(ob)
    if not tris:
        raise SystemExit("blender: the build plan produced no geometry")
    # Written by hand rather than through an exporter add-on: the STL operator
    # has been renamed twice across Blender releases, and the bytes are the
    # contract, not the operator name.
    write_binary_stl(OUT_PATH, tris)


main()
'''


class BlenderBackend(ExternalToolBackend):
    """A GeometryBackend backed by headless Blender (real mesh booleans)."""

    TOOL = "blender"
    #: Blender honours everything the F-rep tree can express: its booleans are
    #: exact, its bevel is a real fillet/chamfer and its solidify is a real shell.
    #: (draft / loft / sweep are refused upstream by the F-rep op model itself.)
    #: ``thicken`` is refused: an offset-solid node is not in the lowering, and the
    #: Solidify modifier thickens a surface rather than offsetting a closed solid.
    UNSUPPORTED: Dict[str, str] = {
        "thicken": "the blender lowering has no offset-solid node; Solidify "
                   "thickens a sheet, not a closed solid, so growing/shrinking a "
                   "solid by a wall thickness is not expressed here",
    }
    #: box/cylinder/cone lower through the extrude/cyl/cone nodes the hole path
    #: already builds; sphere/torus/wedge have no node in this lowering.
    PRIMITIVE_SHAPES = ("box", "cylinder", "cone")
    FORMATS = ("stl", "stl-ascii", "stl-binary", "glb", "bpy")

    @classmethod
    def locate(cls) -> str:
        from harnesscad.io.backends.external import find_executable

        path, searched = find_executable(BLENDER_ENV, ("blender",), BLENDER_PATTERNS)
        if path is None:
            raise BackendUnavailable(
                "blender",
                "Blender is not installed (or not on PATH). Install it, or point "
                "%s at blender.exe. Searched: %s" % (BLENDER_ENV, ", ".join(searched)),
                searched)
        return path

    # -- op admission ------------------------------------------------------
    def apply(self, op: Op) -> ApplyResult:
        """Refuse, before anything mutates, the two things this kernel cannot honour.

        A silently dropped field is a wrong part with no diagnostic, which is the
        failure mode that let the "fillet everything" bug live: block-and-correct
        means the agent hears about it.
        """
        if isinstance(op, (Fillet, Chamfer)):
            selector = join_selectors(getattr(op, "edges", ()))
            if selector:
                try:
                    parse_selector(selector)
                except SelectorError as exc:
                    return _err("bad-value",
                                "%s edge selector is malformed: %s"
                                % (type(op).OP, exc), None)
        if isinstance(op, Chamfer) and getattr(op, "distance2", None) is not None:
            # bmesh.ops.bevel / BevelModifier carry ONE width; there is no second
            # setback, so an asymmetric chamfer is not expressible. OCCT's
            # BRepFilletAPI_MakeChamfer takes two distances, so CadQuery honours the
            # field -- Blender cannot, and says so rather than quietly building the
            # symmetric chamfer and calling it done.
            return _err("unsupported-op",
                        "the blender backend cannot honour an asymmetric chamfer: "
                        "Blender's bevel has a single width (bmesh.ops.bevel takes "
                        "one 'offset'), so Chamfer.distance2 has no counterpart",
                        "distance2")
        return super().apply(op)

    # -- the program -------------------------------------------------------
    def build_plan(self) -> dict:
        root = self.root()
        if root is None:
            raise ValueError("blender backend: no solid to build")
        # No op-log blends: they are 'blend' NODES in the tree now, lowered at
        # their own position in the feature history. Passing them here as well
        # would bevel the model twice.
        return _lower(root, self.segments)

    def program(self) -> str:
        """The model as JSON, stamped with the digest of the bpy script AND the
        version of the Blender that will run it.

        The on-disk result cache is content-addressed on this text alone.

        * Without the ``script`` stamp, changing a modifier setting inside the script
          (say, fixing the solidify offset) leaves every previously cached STL in
          place -- the backend keeps serving geometry built by the old, wrong script.
        * Without the ``kernel`` stamp it is worse: :data:`BLENDER_PATTERNS` globs for
          the NEWEST install, so upgrading Blender swaps the geometry kernel
          underneath an unchanged key and every cached result silently belongs to a
          kernel that is no longer the one being asked.
        """
        from harnesscad.io.backends.external import program_digest

        return json.dumps({"script": program_digest(BUILD_SCRIPT)[:16],
                           "kernel": blender_version(self.executable),
                           "plan": self.build_plan()},
                          sort_keys=True, separators=(",", ":"))

    def script(self) -> str:
        """The bpy script that will be executed (checked for syntax before it is)."""
        return BUILD_SCRIPT

    def _run(self, source: str, workdir: str, out_path: str) -> None:
        check = bpy_script.check_syntax(BUILD_SCRIPT)
        if not check.ok:
            raise RuntimeError("blender build script is not valid Python: %s (line %s)"
                               % (check.error, check.lineno))
        spec_path = os.path.join(workdir, "plan.json")
        script_path = os.path.join(workdir, "build.py")
        with open(spec_path, "w", encoding="utf-8") as fh:
            fh.write(source)
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(BUILD_SCRIPT)
        fmt = "glb" if out_path.lower().endswith(".glb") else "stl"
        # --python-exit-code: "Set the exit-code in [0..255] to exit if a Python
        # exception is raised (only for scripts executed from the command line),
        # zero disables." Zero is the DEFAULT, so without this a traceback inside the
        # build script exits 0 -- and a build that crashed but left a stale file in
        # the cache directory behind would be reported as a success.
        argv = [self.executable, "--background", "--factory-startup",
                "--python-exit-code", "1",
                "--python", script_path, "--",
                spec_path, out_path, fmt, SELECTOR_DSL_PATH]
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=self.timeout)
        ok_file = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        if proc.returncode != 0 or not ok_file:
            tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:]
            raise RuntimeError(
                "blender failed to build the model (exit %d, output %s)\n%s"
                % (proc.returncode, "written" if ok_file else "missing", tail))

    # -- export ------------------------------------------------------------
    def export(self, fmt: str):
        f = str(fmt).lower()
        if f == "bpy":
            return BUILD_SCRIPT
        if f == "glb" and self.root() is not None:
            # Let Blender itself write the glTF, rather than re-wrapping the STL.
            source = self.program()
            from harnesscad.io.backends.external import cache_dir, program_digest

            workdir = cache_dir(self.TOOL, program_digest(source))
            out_path = os.path.join(workdir, "model.glb")
            if not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0):
                self._run(source, workdir, out_path)
            with open(out_path, "rb") as fh:
                return fh.read()
        return super().export(fmt)
