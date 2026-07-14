"""OpenScadBackend — CISP ops driven through the OpenSCAD binary.

OpenSCAD is a real CSG kernel (CGAL Nef polyhedra). Its booleans are *exact*:
the difference of a box and a cylinder has the true intersection curve on it, to
floating-point precision, not a marching-cubes staircase. That is the whole point
of this backend -- it is the arithmetic ground truth the SDF backend's grid can
be measured against.

The pipeline
------------
::

    CISP ops -> F-rep CSG tree -> OpenSCAD source -> openscad -o out.stl -> mesh

and every stage of it already existed in this repo. Nothing here re-implements a
thing that was already written:

* the CSG tree is the F-rep tree the frep backend builds
  (:class:`~harnesscad.io.backends.frep.FRepBackend`), used as a kernel-neutral
  IR -- see :mod:`harnesscad.io.backends.external`;
* the source is emitted with the SolidPython-style object model in
  :mod:`harnesscad.domain.programs.emit.openscad_emit` (``ScadNode``,
  ``scad_render``), so parameter ordering and float formatting are somebody
  else's solved problem and the output is byte-stable;
* the source is gated *before* the binary is ever spawned by
  :func:`harnesscad.domain.programs.validate.openscad_check.check` -- the
  execution-free compile check;
* the invocation is planned by
  :func:`harnesscad.domain.fabrication.openscad_export.plan_export`, which
  content-addresses the temp files (so an identical model is a cache hit, not a
  fresh ``uuid4``) and validates the format/extension pair;
* the result is classified by
  :func:`~harnesscad.domain.fabrication.openscad_export.classify_result`, which
  catches OpenSCAD's signature failure: **exit code 0 with an empty top-level
  object**. A backend that only checked the exit code would hand the harness a
  0-triangle STL and call it a part.
* curved primitives are faceted through OpenSCAD's own ``$fn/$fa/$fs`` law
  (:mod:`harnesscad.domain.geometry.parametric.facets`), so the volume this
  backend reports is the exact volume of the polygon prism OpenSCAD actually
  built -- a number that can be predicted in closed form and asserted on.

Ops OpenSCAD cannot honour honestly (:data:`OpenScadBackend.UNSUPPORTED`) are
REFUSED with a typed diagnostic rather than approximated: OpenSCAD has no 3D
offset, so it has no shell, no fillet and no chamfer. Faking them with a
``minkowski()`` would silently grow the part.

Absent the binary, the constructor raises
:class:`~harnesscad.io.backends.base.BackendUnavailable`.
"""

from __future__ import annotations

import math
import os
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.domain.fabrication import openscad_export
from harnesscad.domain.programs.emit import openscad_emit as se
from harnesscad.domain.programs.validate import openscad_check
from harnesscad.io.backends.base import BackendUnavailable
from harnesscad.io.backends.external import (
    ExternalToolBackend, ccw, circle_loop, plane_axes, plane_normal,
    profile_loops, rect_loop, segments_for, slab,
)
from harnesscad.io.backends.frep import Node

#: Installer locations for the binary, globbed (never a hard-coded version).
OPENSCAD_PATTERNS = (
    r"C:\Program Files\OpenSCAD\openscad.exe",
    r"C:\Program Files\OpenSCAD*\openscad.exe",
    r"C:\Program Files (x86)\OpenSCAD\openscad.exe",
    "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
    "/usr/bin/openscad",
    "/usr/local/bin/openscad",
)

OPENSCAD_ENV = "HARNESSCAD_OPENSCAD"


class OpenScadError(RuntimeError):
    """OpenSCAD ran but did not produce geometry (including the exit-0 empty case)."""


# --------------------------------------------------------------------------
# lowering: F-rep tree -> ScadNode tree
# --------------------------------------------------------------------------
def _basis(plane: str) -> List[List[float]]:
    """The 4x4 that maps sketch-local (u, v, w) to world.

    Its columns are the world images of the local axes, which is exactly frep's
    ``_to_world``: the two backends therefore place a sketch on a plane the same
    way, by construction, and not by two agreeing conventions.
    """
    iu, iv, iw = plane_axes(plane)
    m = [[0.0] * 4 for _ in range(4)]
    m[iu][0] = 1.0
    m[iv][1] = 1.0
    m[iw][2] = 1.0
    m[3][3] = 1.0
    return m


def _placed(plane: str, child: se.ScadNode) -> se.ScadNode:
    """Put a solid built in sketch-local coordinates onto its sketch plane."""
    if str(plane).upper() == "XY":
        return child  # the identity basis: emit no transform at all
    return se.multmatrix(_basis(plane))(child)


def _profile_2d(profile, segments: int) -> se.ScadNode:
    """The sketch's 2D region as OpenSCAD 2D geometry (a union of its entities)."""
    parts: List[se.ScadNode] = []
    for (x, y, w, h) in profile.rects:
        parts.append(se.translate((x, y))(se.square([w, h])))
    for (cx, cy, r) in profile.circles:
        parts.append(se.translate((cx, cy))(
            se.circle(r=r, segments=segments_for(r, segments))))
    for verts in profile.polys:
        if len(verts) >= 3:
            parts.append(se.polygon([list(p) for p in ccw(verts)]))
    if not parts:
        raise OpenScadError("openscad backend: empty sketch profile reached the emitter")
    if len(parts) == 1:
        return parts[0]
    return se.union()(*parts)


def _revolve_profile(profile, axis: Sequence[float], segments: int) -> se.ScadNode:
    """The profile re-expressed in (radius, axis) coordinates for rotate_extrude.

    ``rotate_extrude`` revolves a 2D region about the local Z axis, reading its X
    as radius and its Y as height. frep's revolve is stated as an axis LINE in
    the sketch plane, so every loop is transformed into that (perp, along) frame
    here; the frame itself is then carried by the enclosing ``multmatrix``.
    """
    au, av, du, dv, nu, nv = (float(x) for x in axis)
    parts: List[se.ScadNode] = []
    for loop in profile_loops(profile, segments):
        pts = [[(u - au) * nu + (v - av) * nv, (u - au) * du + (v - av) * dv]
               for (u, v) in loop]
        parts.append(se.polygon(ccw([(p[0], p[1]) for p in pts])))
    if not parts:
        raise OpenScadError("openscad backend: empty revolve profile")
    if len(parts) == 1:
        return parts[0]
    return se.union()(*parts)


def _revolve_basis(plane: str, axis: Sequence[float]) -> List[List[float]]:
    """The 4x4 taking rotate_extrude's local frame to the world.

    Local X and Y span the plane the profile sweeps through (the sketch's in-plane
    radial direction and the sketch-plane normal); local Z is the revolution axis.
    Identical to the frame ``frep._eval_revolve`` decomposes a world point into.
    """
    from harnesscad.io.backends.external import to_world

    au, av, du, dv, nu, nv = (float(x) for x in axis)
    ex = to_world(plane, nu, nv, 0.0)
    ey = to_world(plane, 0.0, 0.0, 1.0)
    ez = to_world(plane, du, dv, 0.0)
    origin = to_world(plane, au, av, 0.0)
    return [
        [ex[0], ey[0], ez[0], origin[0]],
        [ex[1], ey[1], ez[1], origin[1]],
        [ex[2], ey[2], ez[2], origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def lower(node: Node, segments: int) -> se.ScadNode:
    """One F-rep node as OpenSCAD geometry."""
    t = node.t
    if t == "extrude":
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        body = se.linear_extrude(height=hi - lo)(_profile_2d(node.d["profile"], segments))
        if lo:
            body = se.translate((0.0, 0.0, lo))(body)
        return _placed(node.d["plane"], body)
    if t == "cyl":
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        r = float(node.d["r"])
        body = se.translate((float(node.d["cu"]), float(node.d["cv"]), lo))(
            se.cylinder(r=r, h=hi - lo, segments=segments_for(r, segments)))
        return _placed(node.d["plane"], body)
    if t == "revolve":
        angle = abs(float(node.d.get("angle", 360.0)))
        angle = 360.0 if angle >= 360.0 else angle
        prof = _revolve_profile(node.d["profile"], node.d["axis"], segments)
        body = se.rotate_extrude(angle=angle, segments=segments)(prof)
        return se.multmatrix(_revolve_basis(node.d["plane"], node.d["axis"]))(body)
    if t == "bool":
        a = lower(node.d["a"], segments)
        b = lower(node.d["b"], segments)
        op = node.d["op"]
        if op == "union":
            return se.union()(a, b)
        if op == "intersect":
            return se.intersection()(a, b)
        return se.difference()(a, b)
    if t == "mirror":
        child = lower(node.d["child"], segments)
        normal = plane_normal(node.d["plane"])
        return se.union()(child, se.mirror(list(normal))(lower(node.d["child"], segments)))
    if t == "pattern":
        kids: List[se.ScadNode] = []
        for tr in node.d["transforms"]:
            dx, dy, dz, ang = (float(tr[0]), float(tr[1]), float(tr[2]), float(tr[3]))
            body = lower(node.d["child"], segments)
            if ang:
                body = se.rotate(a=[0.0, 0.0, ang])(body)
            if dx or dy or dz:
                body = se.translate((dx, dy, dz))(body)
            kids.append(body)
        if len(kids) == 1:
            return kids[0]
        return se.union()(*kids)
    raise OpenScadError("openscad backend: unknown F-rep node kind %r" % t)


HEADER = ("// Generated by harnesscad OpenScadBackend. Do not edit.\n"
          "// Curved primitives are faceted by OpenSCAD's own $fn law, so the\n"
          "// volume of this model is predictable in closed form.")


def render(root: Node, segments: int) -> str:
    """The whole model as OpenSCAD source (deterministic, byte-stable)."""
    return se.scad_render(lower(root, segments), header=HEADER)


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
class OpenScadBackend(ExternalToolBackend):
    """A GeometryBackend backed by the OpenSCAD binary (exact CGAL CSG)."""

    TOOL = "openscad"
    #: OpenSCAD has no 3D offset operator, so it cannot express these HONESTLY.
    #: They are refused with a typed diagnostic rather than approximated -- a
    #: ``minkowski()`` "fillet" would grow the part instead of rounding it.
    UNSUPPORTED: Dict[str, str] = {
        "fillet": "OpenSCAD has no 3D offset, so it cannot round solid edges "
                  "(a minkowski() would dilate the part, not fillet it)",
        "chamfer": "OpenSCAD has no 3D offset, so it cannot chamfer solid edges",
        "shell": "OpenSCAD has no 3D offset, so it cannot hollow a solid to a "
                 "wall thickness",
    }
    FORMATS = ("stl", "stl-ascii", "stl-binary", "glb", "scad")

    @classmethod
    def locate(cls) -> str:
        from harnesscad.io.backends.external import find_executable

        path, searched = find_executable(OPENSCAD_ENV, ("openscad",), OPENSCAD_PATTERNS)
        if path is None:
            raise BackendUnavailable(
                "openscad",
                "OpenSCAD is not installed (or not on PATH). Install it "
                "(`winget install --id OpenSCAD.OpenSCAD -e`), or point %s at "
                "openscad.exe. Searched: %s" % (OPENSCAD_ENV, ", ".join(searched)),
                searched)
        return path

    # -- the program -------------------------------------------------------
    def program(self) -> str:
        root = self.root()
        if root is None:
            raise OpenScadError("openscad backend: no solid to render")
        source = render(root, self.segments)
        issues = openscad_check.check(source)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise OpenScadError(
                "the emitted OpenSCAD source does not compile:\n%s"
                % openscad_check.format_report(errors))
        return source

    def _run(self, source: str, workdir: str, out_path: str) -> None:
        """Render the source with the binary, and CLASSIFY the result.

        The plan (argv, content-addressed paths, format/extension check) comes
        from the fabrication export planner; the classifier is what turns
        OpenSCAD's exit-0-but-empty into a real failure.
        """
        plan = openscad_export.plan_export(
            source, export_format="binstl", out_dir=workdir.replace("\\", "/"),
            executable=self.executable)
        with open(plan.scad_path, "w", encoding="utf-8") as fh:
            fh.write(source)
        argv = [self.executable, "-o", out_path, "--export-format", "binstl",
                plan.scad_path]
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=self.timeout)
        status, messages = openscad_export.classify_result(proc.returncode, proc.stderr)
        if status != openscad_export.STATUS_OK:
            raise OpenScadError(
                "openscad %s (exit %d): %s"
                % (status, proc.returncode, "; ".join(messages) or (proc.stderr or "").strip()))
        if not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0):
            raise OpenScadError(
                "openscad exited %d but wrote no geometry to %s"
                % (proc.returncode, out_path))

    # -- export ------------------------------------------------------------
    def export(self, fmt: str):
        if str(fmt).lower() == "scad":
            return self.program()
        return super().export(fmt)
