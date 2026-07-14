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
(JSON):

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

from harnesscad.domain.programs.validate import bpy_script
from harnesscad.io.backends.base import BackendUnavailable
from harnesscad.io.backends.external import (
    DEFAULT_SEGMENTS, ExternalToolBackend,
    ccw, plane_axes, plane_normal, profile_loops, segments_for, slab, to_world,
    blend_radius,
)
from harnesscad.io.backends.frep import Node

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
    """Rotate about Z then translate -- the forward form of frep's ``_untransform``."""
    dx, dy, dz, ang = (float(tr[0]), float(tr[1]), float(tr[2]), float(tr[3]))
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)

    def place(p: Vec3) -> Vec3:
        x, y, z = p
        if ang:
            x, y = ca * x - sa * y, sa * x + ca * y
        return (x + dx, y + dy, z + dz)

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
        if t == "shell":
            return {"t": "shell", "child": self.node(node.d["child"], xform),
                    "thickness": float(node.d["thickness"])}
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


def _lower(root: Node, segments: int) -> dict:
    """The whole model as a Blender build plan (world-space leaves + kernel ops)."""
    plan = _Lowering(segments).node(root, _identity)
    r, c = blend_radius(root)
    if r > 0.0:
        plan = {"t": "bevel", "child": plan, "width": r, "segments": BEVEL_SEGMENTS}
    elif c > 0.0:
        plan = {"t": "bevel", "child": plan, "width": c, "segments": 1}
    return plan


# --------------------------------------------------------------------------
# the bpy script (constant text; the model travels as JSON beside it)
# --------------------------------------------------------------------------
BUILD_SCRIPT = r'''"""Generated by harnesscad BlenderBackend. Do not edit.

Reads a build plan (world-space leaf meshes + kernel operations), realises it
with Blender's exact boolean / bevel / solidify modifiers, and writes an STL.
"""
import json
import struct
import sys

import bmesh
import bpy

_argv = sys.argv[sys.argv.index("--") + 1:]
SPEC_PATH, OUT_PATH, OUT_FMT = _argv[0], _argv[1], _argv[2]

BOOL_OPS = {"union": "UNION", "cut": "DIFFERENCE", "intersect": "INTERSECT"}


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)


def activate(ob):
    bpy.context.view_layer.objects.active = ob
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    ob.select_set(True)


def make_object(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [list(f) for f in faces])
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


def apply_modifiers(ob):
    activate(ob)
    for mod in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def boolean(a, b, op):
    mod = a.modifiers.new(name="bool", type="BOOLEAN")
    mod.operation = BOOL_OPS[op]
    mod.object = b
    try:
        mod.solver = "EXACT"
    except (AttributeError, TypeError):
        pass
    apply_modifiers(a)
    bpy.data.objects.remove(b, do_unlink=True)
    return a


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
        mod = ob.modifiers.new(name="solidify", type="SOLIDIFY")
        mod.thickness = node["thickness"]
        mod.offset = 0.0
        apply_modifiers(ob)
        return ob
    if kind == "bevel":
        ob = build(node["child"])
        mod = ob.modifiers.new(name="bevel", type="BEVEL")
        mod.width = node["width"]
        mod.segments = int(node["segments"])
        mod.limit_method = "ANGLE"
        apply_modifiers(ob)
        return ob
    raise ValueError("unknown build-plan node %r" % kind)


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
    ob = build(spec)
    activate(ob)
    if OUT_FMT == "glb":
        bpy.ops.export_scene.gltf(filepath=OUT_PATH, export_format="GLB",
                                  use_selection=True)
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
    UNSUPPORTED: Dict[str, str] = {}
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

    # -- the program -------------------------------------------------------
    def build_plan(self) -> dict:
        root = self.root()
        if root is None:
            raise ValueError("blender backend: no solid to build")
        return _lower(root, self.segments)

    def program(self) -> str:
        """The model as JSON. The bpy script itself is a constant (see
        :data:`BUILD_SCRIPT`); the model is its input, so the digest of this text
        is the digest of the geometry."""
        return json.dumps(self.build_plan(), sort_keys=True, separators=(",", ":"))

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
        argv = [self.executable, "--background", "--factory-startup",
                "--python", script_path, "--", spec_path, out_path, fmt]
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=self.timeout)
        if not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0):
            tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:]
            raise RuntimeError(
                "blender produced no geometry (exit %d)\n%s" % (proc.returncode, tail))

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
