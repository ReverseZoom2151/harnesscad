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

What OpenSCAD is, and is not
----------------------------
OpenSCAD is a **mesh/CSG language with no topological entities**. Its tree is
built from solids and booleans; there is no B-rep, so there are no persistent
edges and no persistent faces. ``Fillet(edges=("|Z",))`` and
``Shell(faces=(">Z",))`` name things OpenSCAD does not have. That is a
structural limitation, not a bug -- and the only honest responses to it are to
implement the op exactly, or to REFUSE it with a typed diagnostic. Accepting a
field and quietly ignoring it (rounding *every* edge when four were asked for)
produces a different part while reporting success, and is precisely the failure
this backend refuses to commit.

So, per op:

* **Implemented exactly.** ``extrude``, ``revolve``, ``boolean``, ``mirror``,
  ``linear_pattern``, ``circular_pattern`` -- all are direct OpenSCAD
  primitives. ``hole`` is implemented *including* its ``kind`` / ``cbore_*`` /
  ``csk_*`` fields and its face datum: a counterbore is a stacked cylinder and a
  countersink is a ``cylinder(r1=, r2=)`` cone -- a real truncated cone, one
  primitive, exact. (Four of six engines used to collapse a counterbore, a
  countersink and a plain hole into the SAME cylinder. The stepped tool is now
  built in the shared F-rep tree, and lowered faithfully here.)
  ``shell`` is implemented for a **prism** (an extruded profile), where the
  inward erosion IS OpenSCAD's exact 2D ``offset()`` of the profile plus an
  inset of the extrusion slab -- so a 60x40x20 box shelled to t=3 has volume
  exactly ``48000 - 54*34*14 = 22296``, not a number near it. Both of CadQuery's
  join kinds are honoured, because OpenSCAD has both: ``offset(r=)`` is the
  "arc" join and ``offset(delta=)`` is the "intersection" join.
* **Refused, with a typed ``unsupported-op``.** ``fillet``, ``chamfer``,
  ``draft``, ``loft``, ``sweep``, and ``shell`` of a non-prism. The reason is
  always the same one: OpenSCAD has no 3D erosion. ``offset()`` "generates a new
  2d interior or exterior outline from an existing outline" (OpenSCAD User
  Manual, Transformations/offset) -- it is 2D only; and ``minkowski()`` is a
  *sum* (a dilation) with no documented inverse anywhere in the manual or the
  cheatsheet. A constant-radius edge blend needs an erode-then-dilate, so it is
  not expressible. ``minkowski(){ cube([20,10,5]); sphere(1); }`` does not
  fillet the box: it grows it to 22 x 12 x 7, a different part with a different
  bounding box. Shipping that as "fillet" would be the bug, not the feature.
* **Approximated.** Nothing. No op here is approximated.

Tessellation is pinned
----------------------
Every curved entity this backend emits (``circle``, ``cylinder``,
``rotate_extrude``, ``offset``) carries an **explicit ``$fn``**, computed by
OpenSCAD's own ``get_fragments_from_r`` law
(:mod:`harnesscad.domain.geometry.parametric.facets`). Nothing is ever left to
OpenSCAD's ``$fa=12`` / ``$fs=2`` defaults, so a cylinder's volume is a closed
form and not a property of the installed binary's defaults.

And both ``$fn`` **and the OpenSCAD version** are stamped into the program text
(and into :meth:`OpenScadBackend.state_digest`), so they are part of the
content-addressed cache key. Without the version in the key, upgrading OpenSCAD
would silently re-serve the STL that the *previous* kernel built.

Absent the binary, the constructor raises
:class:`~harnesscad.io.backends.base.BackendUnavailable`.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

from harnesscad.core.cisp.ops import Shell
from harnesscad.domain.fabrication import openscad_export
from harnesscad.domain.programs.emit import openscad_emit as se
from harnesscad.domain.programs.validate import openscad_check
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.external import (
    ExternalToolBackend, ccw, circle_loop, plane_axes, plane_normal,
    profile_loops, rect_loop, segments_for, slab,
)
from harnesscad.io.backends.frep import (
    SHELL_FACES, Node, countersink_depth, node_bounds,
)

__all__ = ["OpenScadBackend", "OpenScadError", "lower", "render",
           "countersink_depth"]

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


# --------------------------------------------------------------------------
# the primitives OpenSCAD needs that the shared lowering does not already have
# --------------------------------------------------------------------------
# The F-rep tree is the kernel-neutral CSG DAG every external backend lowers, and
# it now carries the whole stepped hole: a `cyl` shaft, a second `cyl` for a
# counterbore, and a `cone` for a countersink -- each resolved onto the face its
# datum names. So a counterbore, a countersink and a plain hole are three
# different trees, and this backend simply builds what it is given. (They were
# ONE tree until recently: every engine lowered all three to the same bare
# cylinder. The fix is in frep, and lowering it faithfully is the job here.)
#
# What OpenSCAD adds is that `cone` and `shell` have EXACT expressions --
# cylinder(r1=, r2=) is a truncated cone, and the inward erosion of a prism IS
# offset() of its profile -- so nothing below is approximated.

#: Clearance added beyond a face so a cut passes cleanly through it.
CUT_PAD = 1.0


def _frustum(node: Node, segments: int) -> se.ScadNode:
    """A ``cone`` node: radius ``r0`` at ``w0`` tapering to ``r1`` at ``w1``.

    ``cylinder(h, r1|d1, r2|d2, center)`` IS a truncated cone in OpenSCAD (the
    cheatsheet's 3D primitives), so a countersink is one primitive and one call,
    exact to the facet law -- there is nothing here to approximate and no reason
    any engine ever had to collapse a countersink into a cylinder.
    """
    w0, w1 = float(node.d["w0"]), float(node.d["w1"])
    r0, r1 = float(node.d["r0"]), float(node.d["r1"])
    lo, hi = slab(w0, w1)
    a, b = (r0, r1) if w0 <= w1 else (r1, r0)
    seg = segments_for(max(a, b), segments)
    body = se.cylinder(r1=a, r2=b, h=hi - lo, segments=seg)
    body = se.translate((float(node.d["cu"]), float(node.d["cv"]), lo))(body)
    return _placed(node.d["plane"], body)


def _shell_prism(shell: Node, segments: int) -> se.ScadNode:
    """A shelled prism: the solid minus its inward erosion. EXACT.

    The erosion of an extruded profile by a wall thickness t is the extrusion of
    the profile's inward 2D offset by t, over a slab inset by t at each closed
    end. OpenSCAD's ``offset()`` is exactly that 2D offset -- the manual's
    Transformations/offset: "delta ... creates a new outline with sides having a
    fixed distance ... inward (delta < 0) from the original outline", and "r ...
    as if a circle of some radius is rotated around the ... interior (r < 0)".
    So this is a real constant-thickness wall, not a fudge, and the OUTER surface
    is untouched -- a shell must never grow the bounding box.

    ``Shell.kind`` is honoured: CadQuery's "intersection" join extends the offset
    sides to their intersection (a sharp corner), which is ``offset(delta=)``;
    its "arc" join rolls a radius around the corner, which is ``offset(r=)``.
    Those are the two joins OpenSCAD documents, and they are the two CadQuery
    names.
    """
    child = shell.d["child"]
    plane = child.d["plane"]
    t = float(shell.d["thickness"])
    lo, hi = slab(float(child.d["w0"]), float(child.d["w1"]))
    _, _, iw = plane_axes(plane)

    faces = [str(f).strip().lower() for f in (shell.d.get("faces") or ())]
    open_lo = any(SHELL_FACES[f] == (iw, -1) for f in faces)
    open_hi = any(SHELL_FACES[f] == (iw, +1) for f in faces)

    profile = _profile_2d(child.d["profile"], segments)
    outer = se.linear_extrude(height=hi - lo)(profile)
    if lo:
        outer = se.translate((0.0, 0.0, lo))(outer)

    kind = str(shell.d.get("kind", "arc"))
    if kind == "intersection":
        inner_profile = se.offset(delta=-t)(_profile_2d(child.d["profile"], segments))
    else:
        inner_profile = se.offset(r=-t, segments=segments)(
            _profile_2d(child.d["profile"], segments))

    # An opened cap is not inset: the cavity runs out through it (and past it,
    # so the face's wall is actually removed rather than left as a skin).
    ilo = (lo - CUT_PAD) if open_lo else (lo + t)
    ihi = (hi + CUT_PAD) if open_hi else (hi - t)
    inner = se.translate((0.0, 0.0, ilo))(
        se.linear_extrude(height=ihi - ilo)(inner_profile))
    return _placed(plane, se.difference()(outer, inner))


def lower(node: Node, segments: int) -> se.ScadNode:
    """One F-rep node as OpenSCAD geometry."""
    t = node.t
    if t == "extrude":
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        body = se.linear_extrude(height=hi - lo)(_profile_2d(node.d["profile"], segments))
        if lo:
            body = se.translate((0.0, 0.0, lo))(body)
        return _placed(node.d["plane"], body)
    if t == "shell":
        child = node.d["child"]
        if child.t != "extrude":
            raise OpenScadError(
                "openscad backend: shell of a %r is not expressible (OpenSCAD's "
                "offset() is 2D only, so only a prism can be eroded exactly)"
                % child.t)
        return _shell_prism(node, segments)
    if t == "cyl":
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        r = float(node.d["r"])
        body = se.translate((float(node.d["cu"]), float(node.d["cv"]), lo))(
            se.cylinder(r=r, h=hi - lo, segments=segments_for(r, segments)))
        return _placed(node.d["plane"], body)
    if t == "cone":
        return _frustum(node, segments)
    if t == "sphere":
        r = float(node.d["r"])
        body = se.sphere(r=r, segments=segments_for(r, segments))
        return se.translate((float(node.d["cx"]), float(node.d["cy"]),
                             float(node.d["cz"])))(body)
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
        return se.union()(child, se.mirror(list(normal))(
            lower(node.d["child"], segments)))
    if t == "pattern":
        # Each transform is a RIGID 3x4 matrix, row-major. It used to be
        # (dx, dy, dz, angle_about_Z), which could not express a rotation about any
        # axis but Z -- so CircularPattern.axis was read, validated, and then thrown
        # away. multmatrix() takes the 4x4 straight, so the axis the op names is the
        # axis the instance turns about.
        kids: List[se.ScadNode] = []
        for tr in node.d["transforms"]:
            m = [float(v) for v in tr]
            kids.append(se.multmatrix([
                [m[0], m[1], m[2], m[3]],
                [m[4], m[5], m[6], m[7]],
                [m[8], m[9], m[10], m[11]],
                [0.0, 0.0, 0.0, 1.0],
            ])(lower(node.d["child"], segments)))
        if len(kids) == 1:
            return kids[0]
        return se.union()(*kids)
    if t == "hull":
        # OpenSCAD's native hull() of the child solids -- exact convex hull.
        return se.hull()(*[lower(c, segments) for c in node.d["children"]])
    if t == "minkowski":
        # Minkowski sum with a ball IS OpenSCAD's minkowski(){ solid; sphere(r); }.
        r = float(node.d["radius"])
        return se.minkowski()(lower(node.d["child"], segments),
                              se.sphere(r=r, segments=segments_for(r, segments)))
    if t == "scale":
        # Per-axis scale about the origin -- OpenSCAD's scale([sx,sy,sz]), emitted
        # through the same multmatrix() the pattern lowering uses (a diagonal 4x4),
        # so uniform and non-uniform scales are both exact.
        sx, sy, sz = float(node.d["sx"]), float(node.d["sy"]), float(node.d["sz"])
        return se.multmatrix([
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])(lower(node.d["child"], segments))
    raise OpenScadError("openscad backend: unknown F-rep node kind %r" % t)


def header(segments: int, version: str) -> str:
    """The source header -- and the reason the cache key is honest.

    ``$fn`` and the OpenSCAD version are stamped into the program TEXT, and the
    program text is what the content-addressed cache directory is named after
    (:func:`harnesscad.io.backends.external.program_digest`). So a different
    tessellation, or a different kernel build, is a different artefact -- never a
    stale hit. Blender's and FreeCAD's cache keys both omitted the tool version
    and re-served geometry the previous version had built.
    """
    return ("// Generated by harnesscad OpenScadBackend. Do not edit.\n"
            "// Every curved entity below carries an EXPLICIT $fn (%d), computed by\n"
            "// OpenSCAD's own get_fragments_from_r law -- nothing is left to the\n"
            "// $fa=12 / $fs=2 defaults, so this model's volume is a closed form.\n"
            "// openscad-version: %s" % (int(segments), version))


def render(root: Node, segments: int, version: str = "unknown") -> str:
    """The whole model as OpenSCAD source (deterministic, byte-stable)."""
    return se.scad_render(lower(root, segments), header=header(segments, version))


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
class OpenScadBackend(ExternalToolBackend):
    """A GeometryBackend backed by the OpenSCAD binary (exact CGAL CSG)."""

    TOOL = "openscad"
    #: Ops OpenSCAD cannot honour HONESTLY, refused with a typed diagnostic
    #: rather than approximated. Every reason reduces to the same fact: OpenSCAD
    #: has no 3D erosion (``offset()`` is 2D only; ``minkowski()`` is a sum with
    #: no inverse) and no topological entities at all -- so ``edges`` and
    #: ``faces`` selectors name things its CSG tree does not contain.
    UNSUPPORTED: Dict[str, str] = {
        "fillet": "OpenSCAD is a CSG language with no topological edges, so the "
                  "'edges' selector cannot be honoured; and it has no 3D erosion "
                  "(offset() is 2D-only, minkowski() is a dilation with no "
                  "inverse), so a constant-radius blend that preserves the "
                  "bounding box is not expressible. minkowski() with a sphere "
                  "would GROW a 20x10x5 box to 22x12x7 -- a different part",
        "chamfer": "OpenSCAD is a CSG language with no topological edges, so the "
                   "'edges' selector cannot be honoured, and it has no 3D "
                   "erosion with which to set an edge back",
        "draft": "OpenSCAD has no face entities, so a draft angle cannot be "
                 "applied to a named neutral plane and face set",
        "loft": "OpenSCAD has no loft; hull() of two profiles is a convex hull, "
                "not a ruled/lofted surface through them",
        "sweep": "OpenSCAD has no sweep along a path in the core language",
        "thicken": "OpenSCAD has no 3D offset/erosion (offset() is 2D-only, "
                   "minkowski() is a dilation with no inverse), so growing or "
                   "shrinking a solid by a wall thickness is not expressible",
    }
    #: box/cylinder/cone/sphere are all direct OpenSCAD primitives (cube,
    #: cylinder, cylinder(r1,r2), sphere); torus/wedge have no core primitive.
    PRIMITIVE_SHAPES = ("box", "cylinder", "cone", "sphere")
    FORMATS = ("stl", "stl-ascii", "stl-binary", "glb", "scad")

    #: ``openscad --version``, memoised per executable path. Part of the cache key.
    _VERSIONS: Dict[str, str] = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The op-state model is a composed FRepBackend, and two of frep's refusals
        # are facts about ITS kernel, not about CAD -- so they are not ours to
        # inherit, and frep exposes both as knobs for exactly this reason.
        #
        # 1. frep refuses a shell whose wall is thinner than a few cells of its
        #    SAMPLING GRID. Right for frep (an unresolvable wall would come back as
        #    a different, smaller part); meaningless for OpenSCAD, which is an exact
        #    CSG kernel and samples no grid at all. A 3 mm wall on a 60 mm box is cut
        #    here to the last bit.
        # 2. frep refuses the 'intersection' shell join, because a distance field's
        #    inward offset IS the 'arc' join and it cannot express a sharp corner.
        #    OpenSCAD can: offset(delta=) is precisely the sides-extended-to-their-
        #    intersection join. frep's own diagnostic says so ("use cadquery/freecad/
        #    openscad, whose kernels have both joins").
        #
        # The refusals that ARE ours -- the ones about OpenSCAD's own limits -- are
        # in _check_shell, and they are typed.
        self._frep.SHELL_MIN_WALL_CELLS = 0.0
        self._frep.SHELL_JOINS = ("arc", "intersection")
        # OpenSCAD has a native hull(), so the composed op-state model may build a
        # hull node (which lower() turns into hull()). Minkowski likewise lowers to
        # minkowski(){...} and needs no flag (frep-proper already builds the node).
        self._frep.HULL_SUPPORTED = True

    # -- version (a cache-key input, not a cosmetic banner) -----------------
    def tool_version(self) -> str:
        """The binary's version string. Stamped into the program and the digest.

        OpenSCAD 2021.01 renders CSG through CGAL Nef polyhedra. The Manifold
        backend (``--backend=manifold``) only became selectable in the 2024.09.28
        snapshots, so on a 2021.01 binary there is nothing to select and the
        kernel is CGAL. Two kernels can mesh the same tree differently, which is
        exactly why the version belongs in the key.
        """
        cached = type(self)._VERSIONS.get(self.executable)
        if cached is not None:
            return cached
        try:
            proc = subprocess.run([self.executable, "--version"],
                                  capture_output=True, text=True, timeout=30)
            text = (proc.stdout or "") + (proc.stderr or "")
            version = text.strip().splitlines()[0].strip() if text.strip() else "unknown"
        except Exception:                                   # pragma: no cover
            version = "unknown"
        type(self)._VERSIONS[self.executable] = version
        return version

    def state_digest(self) -> str:
        """Content hash of the model AND of everything that decides its geometry.

        The op stream alone is not enough: the same ops at $fn=16 and $fn=64 are
        different solids, and the same ops on two OpenSCAD builds may be different
        meshes. Both are therefore in the digest. Blender's and FreeCAD's keys both
        omitted the tool version, and silently re-served the geometry that the
        PREVIOUS version of the tool had built.
        """
        blob = "%s|%s|fn=%d|version=%s" % (
            self.TOOL, self._frep.state_digest(), self.segments, self.tool_version())
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- op admission ------------------------------------------------------
    def apply(self, op):
        """Refuse what cannot be honoured, BEFORE anything mutates.

        The base class refuses whole ops (:data:`UNSUPPORTED`). This adds the one
        field-level admission the base cannot do: a Shell whose child is not a
        prism, or which opens a face that is not one of the prism's caps. Both are
        refused rather than silently degraded into a solid block.

        Hole needs nothing here: frep now builds the stepped tool itself and
        validates it, so ``kind`` / ``cbore_*`` / ``csk_*`` / the face datum all
        arrive in the tree and are lowered faithfully.
        """
        if isinstance(op, Shell):
            err = self._check_shell(op)
            if err is not None:
                return err
        return super().apply(op)

    @staticmethod
    def _refuse(code: str, msg: str) -> ApplyResult:
        return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, None)])

    def _check_shell(self, op: Shell) -> Optional[ApplyResult]:
        """Shell is EXACT for a prism, and not expressible for anything else.

        These are the refusals that are genuinely OpenSCAD's. Everything else about
        a Shell -- the selector grammar, the join kind, an unknown face, a
        non-positive thickness -- frep validates for us, so it is not re-checked
        here and cannot drift out of step with it.
        """
        from harnesscad.io.backends.frep import resolve_faces

        bodies = self._frep._bodies
        if not bodies:
            return None                       # frep will refuse it: 'no-solid'
        node = bodies[-1]["node"]
        if node.t != "extrude":
            return self._refuse(
                "unsupported-op",
                "the openscad backend cannot shell a %r: OpenSCAD's offset() is "
                "2D-only and minkowski() has no inverse, so the inward erosion of "
                "a general solid is not expressible. Only a prism (an extruded "
                "profile), whose erosion IS the 2D offset of its profile, can be "
                "shelled exactly" % node.t)
        thickness = float(op.thickness)
        if thickness <= 0.0:
            return None                     # frep will refuse it: 'bad-value'
        try:
            faces = resolve_faces(node_bounds(node), op.faces or ())
        except Exception:
            return None                     # frep will refuse it, with its message

        # The opened faces must be the prism's own CAPS. Opening a side wall is a
        # 2D cut of the profile, which this lowering does not express -- so it is
        # refused, rather than ignored and handed back as a closed box.
        _, _, iw = plane_axes(node.d["plane"])
        for name in faces:
            axis, _sign = SHELL_FACES[name]
            if axis != iw:
                return self._refuse(
                    "unsupported-op",
                    "the openscad backend can only open a prism's CAP faces "
                    "(those normal to the extrusion axis); '%s' is a side wall, "
                    "whose removal is a 2D cut this lowering does not express"
                    % name)
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        if (hi - lo) - (2 - len(set(faces))) * thickness <= 0.0:
            return self._refuse(
                "bad-value",
                "shell thickness %g leaves no cavity in a prism %g tall"
                % (thickness, hi - lo))
        plo_u, plo_v, phi_u, phi_v = node.d["profile"].bounds()
        if min(phi_u - plo_u, phi_v - plo_v) <= 2.0 * thickness:
            return self._refuse(
                "bad-value",
                "shell thickness %g erodes the whole profile away (its smallest "
                "extent is %g)" % (thickness, min(phi_u - plo_u, phi_v - plo_v)))
        return None

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
        source = render(root, self.segments, version=self.tool_version())
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
        openscad_export.check_output_extension(out_path, "binstl")
        with open(plan.scad_path, "w", encoding="utf-8") as fh:
            fh.write(source)
        # A failed run must leave NOTHING behind. The caller's cache is a
        # file-exists check, so a half-written STL from a crashed run would be
        # re-served forever, and reported as a success. Clear before, clear after.
        if os.path.isfile(out_path):
            os.remove(out_path)
        argv = [self.executable, "-o", out_path, "--export-format", "binstl",
                plan.scad_path]
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=self.timeout)
        status, messages = openscad_export.classify_result(proc.returncode, proc.stderr)
        empty = not (os.path.isfile(out_path) and os.path.getsize(out_path) > 0)
        if status != openscad_export.STATUS_OK or empty:
            if os.path.isfile(out_path):
                os.remove(out_path)
            if status != openscad_export.STATUS_OK:
                raise OpenScadError(
                    "openscad %s (exit %d): %s"
                    % (status, proc.returncode,
                       "; ".join(messages) or (proc.stderr or "").strip()))
            raise OpenScadError(
                "openscad exited %d but wrote no geometry to %s"
                % (proc.returncode, out_path))

    # -- export ------------------------------------------------------------
    def export(self, fmt: str):
        if str(fmt).lower() == "scad":
            return self.program()
        return super().export(fmt)
