"""ManifoldBackend — CISP ops driven through the Manifold mesh-boolean kernel.

Why this backend exists (the oracle argument)
----------------------------------------------
The differential oracle (:mod:`harnesscad.eval.selftest.differential`) is only as
strong as the INDEPENDENCE of the engines it cross-checks. Of the earlier six,
``freecad`` and ``cadquery`` are both OCCT, and ``openscad``/``blender``/``frep``
all lower the SAME kernel-neutral F-rep CSG tree -- so a bug that lives in the
shared lowering (or in OCCT) is invisible: every engine "agrees" while every
engine is wrong. That is exactly how 47 op-fields rotted undetected.

Manifold (https://github.com/elalish/manifold) is a genuinely independent voice:
a fast, guaranteed-manifold mesh-boolean kernel with a DIFFERENT algorithm
(exact-predicate mesh booleans over a collider, not Nef polyhedra, not OCCT
B-rep, not a sampled SDF) and therefore a different failure surface. Adding it
makes every future correctness claim strictly stronger -- a third algorithm has
to agree, not just a third wrapper around the same one.

What Manifold is, and is not
----------------------------
Manifold is a MESH kernel. It has:

* guaranteed-2-manifold boolean output (union ``+``, difference ``-``,
  intersection ``^``) -- the whole point of the library;
* ``CrossSection`` for 2D regions, with ``.extrude`` and ``.revolve`` to solids;
* ``CrossSection.offset`` (a real 2D offset with Round / Miter / Square joins);
* and, notably, a true 3D ``minkowski_sum`` / ``minkowski_difference`` and a
  ``level_set`` / ``warp`` for implicit surfaces.

What it does NOT have is any B-rep topology: no persistent edges, no persistent
faces. So -- exactly as for the OpenSCAD backend -- an op that names a
topological entity (``Fillet(edges=(...))`` on a specific edge, ``Draft`` about a
named neutral plane) names something Manifold's data model does not contain, and
the only honest responses are to implement the op faithfully or to REFUSE it with
a typed ``unsupported-op``. Accepting a field and silently dropping it (rounding
every edge when four were asked for) builds a different part while reporting
success -- the precise bug this whole codebase was written to eradicate.

The op -> Manifold mapping
--------------------------
* **Implemented, exactly.** ``extrude`` (``CrossSection.extrude``), ``revolve``
  (``CrossSection.revolve``), ``boolean`` (``+`` / ``-`` / ``^``),
  ``linear_pattern`` / ``circular_pattern`` (rigid ``transform`` of each instance,
  unioned), ``mirror`` (``Manifold.mirror`` across the datum plane, unioned with
  the original), and ``hole`` in full -- ``kind`` / ``cbore_*`` / ``csk_*`` and
  the face datum all arrive as ``cyl`` / ``cone`` nodes in the shared F-rep tree
  and are lowered faithfully (a counterbore is a stacked cylinder; a countersink
  is a ``cylinder(radius_low, radius_high)`` frustum -- a real truncated cone).
  ``shell`` is implemented for a **prism**: the inward erosion of an extruded
  profile IS ``CrossSection.offset`` of that profile, so a 60x40x20 box shelled to
  t=3 has the exact analytic wall volume, not a number near it. Both CadQuery
  join kinds are honoured because Manifold's offset has both: ``JoinType.Round``
  is the "arc" join, ``JoinType.Miter`` is the "intersection" join.
* **Refused, with a typed ``unsupported-op``.** ``fillet``, ``chamfer``,
  ``draft``, ``loft``, ``sweep``, and ``shell`` of a non-prism. The reason is the
  same fact every time: Manifold is a mesh kernel with no B-rep edges/faces (so
  the ``edges``/``faces``/``neutral_plane`` selectors resolve to nothing) and no
  loft/sweep primitive. ``minkowski_difference`` *could* erode a general solid,
  but only by rolling a sphere into every concave corner -- an APPROXIMATE result
  that would round features the op never asked to round and would not preserve the
  bounding box. Shipping that as an exact shell would be the bug, not the feature,
  so it is refused rather than faked.
* **Approximated.** Nothing.

Tessellation is pinned and comparable
-------------------------------------
Every curved entity is faceted through OpenSCAD's own ``get_fragments_from_r``
law (:func:`harnesscad.io.backends.external.segments_for`) at the SAME
``segments`` (``$fn``) the OpenSCAD and Blender backends use. A circle therefore
becomes the identical inscribed polygon in all three mesh backends, so their
volumes are directly comparable -- a curved part lands the same fraction low on
each, and that shared, predictable polygonisation error is not mistaken for a
disagreement. The world placement of every solid reuses the identical basis math
the OpenSCAD/FRep backends use (:func:`_basis`, :func:`_revolve_basis`), so the
two kernels place a sketch on a plane the same way by construction, not by two
conventions that happen to agree.

Both ``segments`` and the Manifold version are folded into
:meth:`ManifoldBackend.state_digest`, so the same op stream at a different ``$fn``
or a different kernel build is a different content hash -- never a stale cache hit.

Absence
-------
``manifold3d`` (the pip wheel exposing ``Manifold`` / ``CrossSection`` / ``Mesh``)
is imported lazily. When it is not installed the constructor raises
:class:`~harnesscad.io.backends.base.BackendUnavailable`, so the CISP server falls
back to the stub with a note and the test suite SKIPs -- it never hangs and never
fails merely because the wheel is absent.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

from harnesscad.core.cisp.ops import Shell
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult, BackendUnavailable
from harnesscad.io.backends.external import (
    ExternalToolBackend, ccw, plane_axes, plane_normal, profile_loops,
    segments_for, slab, to_world,
)
from harnesscad.io.backends.frep import (
    SHELL_FACES, Node, countersink_depth, node_bounds, resolve_faces,
)
from harnesscad.io.formats import stl as stl_fmt

__all__ = ["ManifoldBackend", "ManifoldError", "lower", "render",
           "countersink_depth"]

#: Clearance beyond a face so an opened shell cap is cleanly removed (matches the
#: OpenSCAD backend, so the two kernels open a cap identically).
CUT_PAD = 1.0

#: Miter limit large enough that the 'intersection' shell join runs the offset
#: sides all the way to their true corner rather than beveling it -- the exact
#: analogue of OpenSCAD's ``offset(delta=)`` sharp join.
_MITER_LIMIT = 1.0e6


class ManifoldError(RuntimeError):
    """Manifold ran but produced no usable geometry for this model."""


def _manifold():
    """Lazy import of ``manifold3d`` -> :class:`BackendUnavailable` if absent.

    Raised as :class:`BackendUnavailable` (not a bare ImportError) so a missing
    wheel is never mistaken for a geometry failure: the server falls back to the
    stub and the suite skips, deterministically.
    """
    try:
        import manifold3d  # noqa: WPS433 (deliberately lazy)
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(
            "manifold3d",
            "the manifold backend requires the manifold3d wheel (exposes "
            "Manifold / CrossSection / Mesh): %s (install with: "
            "pip install manifold3d)" % exc,
            ["python: import manifold3d"])
    return manifold3d


# --------------------------------------------------------------------------
# coordinate-frame math -- reimplemented locally (identical to the OpenSCAD /
# FRep basis) so this module is self-contained yet places geometry in the SAME
# world frame the other engines do, which is what makes the oracle comparable.
# --------------------------------------------------------------------------
def _basis(plane: str) -> List[List[float]]:
    """The 4x4 mapping sketch-local (u, v, w) to world for a datum plane.

    Its columns are the world images of the local axes -- identical to frep's
    ``_to_world`` and to the OpenSCAD backend's ``_basis``.
    """
    iu, iv, iw = plane_axes(plane)
    m = [[0.0] * 4 for _ in range(4)]
    m[iu][0] = 1.0
    m[iv][1] = 1.0
    m[iw][2] = 1.0
    m[3][3] = 1.0
    return m


def _revolve_basis(plane: str, axis: Sequence[float]) -> List[List[float]]:
    """The 4x4 taking the revolve's local frame (radius in XY, axis along Z) to
    world -- byte-identical to the OpenSCAD backend's, so a solid of revolution
    lands in the same place on both kernels."""
    au, av, du, dv, nu, nv = (float(x) for x in axis)
    ex = to_world(plane, nu, nv, 0.0)      # in-plane radial direction -> local X
    ey = to_world(plane, 0.0, 0.0, 1.0)    # plane normal            -> local Y
    ez = to_world(plane, du, dv, 0.0)      # revolution axis         -> local Z
    origin = to_world(plane, au, av, 0.0)
    return [
        [ex[0], ey[0], ez[0], origin[0]],
        [ex[1], ey[1], ez[1], origin[1]],
        [ex[2], ey[2], ez[2], origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _m34(basis4: List[List[float]]) -> List[List[float]]:
    """A 4x4 affine as Manifold's ``transform`` 3x4 ([R | t], row-major)."""
    return [list(basis4[0][:4]), list(basis4[1][:4]), list(basis4[2][:4])]


def _placed(plane: str, solid):
    """Put a solid built in sketch-local coordinates onto its sketch plane."""
    if str(plane).upper() == "XY":
        return solid                       # identity basis: no transform at all
    return solid.transform(_m34(_basis(plane)))


# --------------------------------------------------------------------------
# 2D -> CrossSection
# --------------------------------------------------------------------------
def _cross_section(profile, segments: int):
    """A sketch profile as a Manifold ``CrossSection`` (the union of its loops).

    Every entity family (rectangles, circles, polylines) is reduced to one CCW
    polygon by :func:`profile_loops` (circles faceted through OpenSCAD's law, so
    the polygon is the one the mesh backends share). ``FillRule.Positive`` unions
    the CCW loops, which is exactly the OpenSCAD emitter's ``union()`` of the same
    parts -- so an L-shaped profile (two overlapping rectangles) becomes the L,
    not two disjoint rectangles.
    """
    m = _manifold()
    loops = [[(float(x), float(y)) for (x, y) in loop]
             for loop in profile_loops(profile, segments)]
    if not loops:
        raise ManifoldError(
            "manifold backend: empty sketch profile reached the kernel")
    return m.CrossSection(loops, m.FillRule.Positive)


def _revolve_contours(profile, axis: Sequence[float], segments: int):
    """The profile re-expressed in (radius, height) coordinates for revolve.

    ``CrossSection.revolve`` reads the polygon's X as radius and revolves about
    the axis so its Y becomes the axis coordinate -- the same convention as
    OpenSCAD's ``rotate_extrude``. Every loop is transformed into the axis'
    (perp, along) frame, exactly as the OpenSCAD backend does, so the enclosing
    :func:`_revolve_basis` places the swept solid identically.
    """
    m = _manifold()
    au, av, du, dv, nu, nv = (float(x) for x in axis)
    contours = []
    for loop in profile_loops(profile, segments):
        pts = [((u - au) * nu + (v - av) * nv,      # perpendicular = radius (X)
                (u - au) * du + (v - av) * dv)       # along axis     = height (Y)
               for (u, v) in loop]
        contours.append(ccw(pts))
    if not contours:
        raise ManifoldError("manifold backend: empty revolve profile")
    return m.CrossSection(contours, m.FillRule.Positive)


# --------------------------------------------------------------------------
# shell of a prism -- EXACT, and only expressible for a prism
# --------------------------------------------------------------------------
def _shell_prism(shell: Node, segments: int):
    """A shelled prism: the solid minus its inward erosion. EXACT.

    The erosion of an extruded profile by a wall thickness ``t`` is the extrusion
    of the profile's inward 2D offset by ``t``, over a slab inset by ``t`` at each
    closed end. ``CrossSection.offset(-t, ...)`` is exactly that 2D offset, so the
    OUTER surface is untouched -- a shell never grows the bounding box. ``kind`` is
    honoured: ``JoinType.Round`` rolls a radius round a reflex corner (CadQuery's
    "arc"); ``JoinType.Miter`` runs the offset sides to their intersection
    (CadQuery's "intersection"). Those are the two joins Manifold documents and the
    two CadQuery names.
    """
    m = _manifold()
    child = shell.d["child"]
    plane = child.d["plane"]
    t = float(shell.d["thickness"])
    lo, hi = slab(float(child.d["w0"]), float(child.d["w1"]))
    _, _, iw = plane_axes(plane)

    faces = [str(f).strip().lower() for f in (shell.d.get("faces") or ())]
    open_lo = any(SHELL_FACES[f] == (iw, -1) for f in faces)
    open_hi = any(SHELL_FACES[f] == (iw, +1) for f in faces)

    outer_cs = _cross_section(child.d["profile"], segments)
    outer = outer_cs.extrude(hi - lo)
    if lo:
        outer = outer.translate((0.0, 0.0, lo))

    kind = str(shell.d.get("kind", "arc"))
    if kind == "intersection":
        inner_cs = outer_cs.offset(-t, m.JoinType.Miter, _MITER_LIMIT)
    else:
        inner_cs = outer_cs.offset(-t, m.JoinType.Round, 2.0, segments)

    # An opened cap is not inset: the cavity runs out through it (and past it, so
    # the face's wall is actually removed rather than left as a skin).
    ilo = (lo - CUT_PAD) if open_lo else (lo + t)
    ihi = (hi + CUT_PAD) if open_hi else (hi - t)
    inner = inner_cs.extrude(ihi - ilo).translate((0.0, 0.0, ilo))
    return _placed(plane, outer - inner)


# --------------------------------------------------------------------------
# lowering: F-rep tree -> Manifold
# --------------------------------------------------------------------------
def lower(node: Node, segments: int):
    """One F-rep node as a Manifold solid."""
    m = _manifold()
    t = node.t
    if t == "extrude":
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        body = _cross_section(node.d["profile"], segments).extrude(hi - lo)
        if lo:
            body = body.translate((0.0, 0.0, lo))
        return _placed(node.d["plane"], body)
    if t == "shell":
        child = node.d["child"]
        if child.t != "extrude":
            raise ManifoldError(
                "manifold backend: shell of a %r is not expressible (its offset "
                "is 2D-only and minkowski_difference rounds concave corners), so "
                "only a prism can be eroded exactly" % child.t)
        return _shell_prism(node, segments)
    if t == "cyl":
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        r = float(node.d["r"])
        body = m.Manifold.cylinder(
            hi - lo, r, r, segments_for(r, segments)).translate(
            (float(node.d["cu"]), float(node.d["cv"]), lo))
        return _placed(node.d["plane"], body)
    if t == "cone":
        lo, hi = slab(float(node.d["w0"]), float(node.d["w1"]))
        r0, r1 = float(node.d["r0"]), float(node.d["r1"])
        a, b = (r0, r1) if float(node.d["w0"]) <= float(node.d["w1"]) else (r1, r0)
        body = m.Manifold.cylinder(
            hi - lo, a, b, segments_for(max(a, b), segments)).translate(
            (float(node.d["cu"]), float(node.d["cv"]), lo))
        return _placed(node.d["plane"], body)
    if t == "revolve":
        angle = abs(float(node.d.get("angle", 360.0)))
        angle = 360.0 if angle >= 360.0 else angle
        cs = _revolve_contours(node.d["profile"], node.d["axis"], segments)
        body = cs.revolve(circular_segments=segments, revolve_degrees=angle)
        return body.transform(_m34(_revolve_basis(node.d["plane"], node.d["axis"])))
    if t == "bool":
        a = lower(node.d["a"], segments)
        b = lower(node.d["b"], segments)
        op = node.d["op"]
        if op == "union":
            return a + b
        if op == "intersect":
            return a ^ b
        return a - b
    if t == "mirror":
        child = lower(node.d["child"], segments)
        normal = plane_normal(node.d["plane"])
        return child + child.mirror(list(normal))
    if t == "pattern":
        # Each transform is a RIGID 3x4 matrix, row-major -- passed straight to
        # Manifold's transform(), so the axis a CircularPattern names is the axis
        # the instance turns about (not a Z-only rotation).
        child = lower(node.d["child"], segments)
        kids = []
        for tr in node.d["transforms"]:
            v = [float(x) for x in tr]
            kids.append(child.transform([v[0:4], v[4:8], v[8:12]]))
        if len(kids) == 1:
            return kids[0]
        return m.Manifold.batch_boolean(kids, m.OpType.Add)
    raise ManifoldError("manifold backend: unknown F-rep node kind %r" % t)


def manifold_to_stl(solid, tool: bytes = b"harnesscad-manifold") -> bytes:
    """A Manifold's mesh as binary STL. Deterministic (the kernel's own output)."""
    mesh = solid.to_mesh()
    vp = mesh.vert_properties      # (N, >=3) float32
    tv = mesh.tri_verts           # (M, 3) int
    tris: List[stl_fmt.Triangle] = []
    for tri in tv:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        va, vb, vc = vp[a], vp[b], vp[c]
        tris.append(stl_fmt.Triangle(
            (float(va[0]), float(va[1]), float(va[2])),
            (float(vb[0]), float(vb[1]), float(vb[2])),
            (float(vc[0]), float(vc[1]), float(vc[2]))))
    return stl_fmt.write_binary_stl(tris, header=tool)


# --------------------------------------------------------------------------
# program text (for an honest content-addressed cache key + inspection)
# --------------------------------------------------------------------------
def _describe(node: Node) -> str:
    t = node.t
    if t == "bool":
        return "(%s %s %s)" % (node.d["op"], _describe(node.d["a"]),
                               _describe(node.d["b"]))
    if t in ("shell", "mirror"):
        return "%s[%s](%s)" % (t, node.d.get("plane", node.d.get("kind", "")),
                               _describe(node.d["child"]))
    if t == "pattern":
        return "pattern<%d>(%s)" % (len(node.d["transforms"]),
                                    _describe(node.d["child"]))
    keys = sorted(k for k in node.d if k not in ("a", "b", "child"))
    return "%s{%s}" % (t, ",".join("%s=%r" % (k, node.d[k]) for k in keys))


def render(root: Node, segments: int, version: str = "unknown") -> str:
    """The whole model as canonical Manifold-IR text (deterministic)."""
    return "manifold-ir fn=%d version=%s\n%s" % (
        int(segments), version, _describe(root))


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------
class ManifoldBackend(ExternalToolBackend):
    """A GeometryBackend backed by the Manifold mesh-boolean kernel (in-process)."""

    TOOL = "manifold"
    #: Ops Manifold cannot honour HONESTLY -- refused with a typed diagnostic
    #: rather than approximated. Every reason reduces to: Manifold is a mesh
    #: kernel with no B-rep edges/faces, and has no loft/sweep.
    UNSUPPORTED: Dict[str, str] = {
        "fillet": "Manifold is a mesh boolean kernel with no B-rep topology, so "
                  "it has no persistent edges for the 'edges' selector to name; "
                  "and its only offset is 2D (CrossSection.offset), so a "
                  "constant-radius 3D edge blend that preserves the bounding box "
                  "is not expressible (minkowski_difference would round every "
                  "concave corner, a different part)",
        "chamfer": "Manifold has no B-rep edges for the 'edges' selector to name, "
                   "and no 3D erosion with which to set an edge back",
        "draft": "Manifold has no face entities, so a draft angle cannot be "
                 "applied to a named neutral plane and face set",
        "loft": "Manifold has no loft/skinning primitive between profiles (only "
                "extrude, revolve and warp)",
        "sweep": "Manifold has no sweep-along-a-path primitive",
    }
    FORMATS = ("stl", "stl-ascii", "stl-binary", "glb")

    #: Manifold's ``CrossSection.offset`` has BOTH joins (Round = arc, Miter =
    #: intersection), so both shell joins are lowered -- the composed frep model is
    #: widened to match. Declaring a join without lowering it would reintroduce the
    #: dropped-field bug, so this must stay in step with :func:`_shell_prism`.
    SHELL_JOINS = ("arc", "intersection")

    #: ``manifold3d.__version__``, memoised. Part of the cache key.
    _VERSIONS: Dict[str, str] = {}

    @classmethod
    def locate(cls) -> str:
        """Prove ``manifold3d`` is importable; return its version as the sentinel
        'executable' (there is no binary -- the kernel runs in-process)."""
        m = _manifold()
        return "manifold3d-" + str(getattr(m, "__version__", "unknown"))

    def tool_version(self) -> str:
        cached = type(self)._VERSIONS.get(self.executable)
        if cached is not None:
            return cached
        m = _manifold()
        version = str(getattr(m, "__version__", "unknown"))
        type(self)._VERSIONS[self.executable] = version
        return version

    def state_digest(self) -> str:
        """Content hash of the model AND of everything that decides its geometry:
        the op stream, the facet count (``$fn``) and the Manifold version. The same
        ops at fn=16 and fn=64, or on two Manifold builds, are different solids."""
        blob = "%s|%s|fn=%d|version=%s" % (
            self.TOOL, self._frep.state_digest(), self.segments,
            self.tool_version())
        return hashlib.sha256(blob.encode()).hexdigest()

    # -- op admission (shell is EXACT for a prism, refused otherwise) --------
    def apply(self, op):
        if isinstance(op, Shell):
            err = self._check_shell(op)
            if err is not None:
                return err
        return super().apply(op)

    @staticmethod
    def _refuse(code: str, msg: str) -> ApplyResult:
        return ApplyResult(False, [], [Diagnostic(Severity.ERROR, code, msg, None)])

    def _check_shell(self, op: Shell) -> Optional[ApplyResult]:
        """Refuse, BEFORE anything mutates, the shells Manifold cannot build
        exactly: a non-prism child, or opening a side wall (a 2D cut of the
        profile this lowering does not express). Everything else about a Shell --
        the selector grammar, the join kind, thickness sign -- the composed frep
        model validates, so it is not re-checked here and cannot drift."""
        bodies = self._frep._bodies
        if not bodies:
            return None                        # frep will refuse it: 'no-solid'
        node = bodies[-1]["node"]
        if node.t != "extrude":
            return self._refuse(
                "unsupported-op",
                "the manifold backend cannot shell a %r: its offset is 2D-only "
                "and minkowski_difference rounds concave corners, so the inward "
                "erosion of a general solid is not exact. Only a prism (an "
                "extruded profile) can be shelled exactly" % node.t)
        thickness = float(op.thickness)
        if thickness <= 0.0:
            return None                        # frep will refuse it: 'bad-value'
        try:
            faces = resolve_faces(node_bounds(node), op.faces or ())
        except Exception:  # noqa: BLE001
            return None                        # frep will refuse it, with its msg
        _, _, iw = plane_axes(node.d["plane"])
        for name in faces:
            axis, _sign = SHELL_FACES[name]
            if axis != iw:
                return self._refuse(
                    "unsupported-op",
                    "the manifold backend can only open a prism's CAP faces "
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

    # -- geometry (built in-process; no subprocess, no temp files) ----------
    def program(self) -> str:
        root = self.root()
        if root is None:
            raise ManifoldError("manifold backend: no solid to render")
        return render(root, self.segments, version=self.tool_version())

    def stl_bytes(self) -> bytes:
        """Build the model with Manifold and return its mesh as binary STL.

        Cached on the state digest. Manifold runs in-process, so unlike the
        subprocess backends this needs no content-addressed temp directory -- the
        kernel is invoked directly and the result memoised.
        """
        key = self.state_digest()
        if self._stl_cache is not None and self._stl_cache[0] == key:
            return self._stl_cache[1]
        header = b"harnesscad-" + self.TOOL.encode()
        root = self.root()
        if root is None:
            data = stl_fmt.write_binary_stl([], header=header)
            self._stl_cache = (key, data)
            return data
        solid = lower(root, self.segments)
        if solid.is_empty():
            data = stl_fmt.write_binary_stl([], header=header)
        else:
            data = manifold_to_stl(solid, tool=header)
        self._stl_cache = (key, data)
        return data
