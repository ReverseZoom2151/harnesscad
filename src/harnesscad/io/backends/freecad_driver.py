"""The FreeCAD-side driver: lowers an F-rep CSG spec onto real Part B-rep solids.

This file is NEVER imported by the harness. It is read as *text* by
:mod:`harnesscad.io.backends.freecad`, written into a content-addressed work
directory next to its spec, and executed by FreeCAD's own interpreter
(``freecadcmd`` / the bundled ``python``), which is a different Python build
from the host's — so nothing here may import ``harnesscad``.

Contract with the host:

  * reads   ``spec.json``    -- the F-rep root node (``frep.Node.spec()``), the
                               ordered fillet/chamfer blends with their edge
                               selectors, the sketches with their constraints,
                               and the export settings.
  * writes  ``result.json``  -- EXACT B-rep mass properties + topology counts +
                               per-face records (for topological naming) + the
                               Sketcher solver's verdict + the FreeCAD document.
  * writes  ``model.stl`` / ``model.step`` / ``model.brep`` / ``model.iges``.

Results go through a FILE, never stdout: ``freecadcmd`` redirects ``print`` to
FreeCAD's own console (`Headless FreeCAD <https://wiki.freecad.org/Headless_FreeCAD>`_),
so a driver that answered on stdout would return nothing.

Everything is exact: circles are real ``Part.Circle`` arcs (not polygons), so the
volumes this returns are the kernel's analytic volumes, not a tessellation.

The CadQuery **string-selector DSL**
(:mod:`harnesscad.domain.geometry.topology.selector_dsl`) is prepended to this
source verbatim by the host, so ``Entity`` / ``parse`` / ``evaluate`` /
``SelectorError`` are in scope here and edge/face selection uses *exactly* the
same grammar and semantics as the CadQuery backend. That is what makes the
differential oracle an apples-to-apples comparison.
"""

DRIVER_SOURCE = r'''
import json
import math
import os
import traceback

import FreeCAD
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
V = FreeCAD.Vector

# The plane convention is the harness's (frep._PLANES): (u_axis, v_axis, w_axis).
PLANES = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}

#: Angular deflection paired with a linear deflection, by the same law FreeCAD's
#: own ``TopoShape::exportStl`` uses (``defaultAngularDeflection`` in
#: src/Mod/Part/App/TopoShape.cpp): min(0.1, linear * 5 + 0.005) -- but we pass
#: it EXPLICITLY, so the tessellation is declared rather than inherited.
def default_angular_deflection(linear):
    return min(0.5, max(linear * 5.0 + 0.005, 0.005))


def axes(plane):
    return PLANES.get(str(plane).upper(), PLANES["XY"])


def to_world(plane, u, v, w):
    iu, iv, iw = axes(plane)
    p = [0.0, 0.0, 0.0]
    p[iu], p[iv], p[iw] = float(u), float(v), float(w)
    return V(p[0], p[1], p[2])


def normal_of(plane):
    """Unit vector along the plane's w (extrusion) axis."""
    iu, iv, iw = axes(plane)
    n = [0.0, 0.0, 0.0]
    n[iw] = 1.0
    return V(n[0], n[1], n[2])


# -- sub-shape selection (the CadQuery selector DSL, over OCCT sub-shapes) ----
def _geom_type(shape):
    """The CadQuery ``geomType()`` name of an edge/face: LINE, CIRCLE, PLANE...

    ``Edge.Curve`` / ``Face.Surface`` raise ``TypeError: undefined curve type``
    for geometry FreeCAD has no Python wrapper for (some chamfer/fillet output),
    so both accessors are guarded: an unwrappable sub-shape simply has no geom
    type and no axis, which excludes it from direction selection -- exactly what
    CadQuery's ``BaseDirSelector`` does with an edge it cannot take an axis of.
    """
    name = ""
    for attr in ("Curve", "Surface"):
        try:
            name = type(getattr(shape, attr)).__name__
            break
        except Exception:
            continue
    if not name:
        return ""
    name = name.upper()
    if name in ("LINE", "LINESEGMENT"):
        return "LINE"
    if name in ("CIRCLE", "ARCOFCIRCLE"):
        return "CIRCLE"
    if name == "PLANE":
        return "PLANE"
    if name in ("CYLINDER", "CYLINDRICALSURFACE"):
        return "CYLINDER"
    if name in ("SPHERE", "SPHERICALSURFACE"):
        return "SPHERE"
    if name in ("CONE", "CONICALSURFACE"):
        return "CONE"
    if name in ("TORUS", "TOROIDALSURFACE"):
        return "TORUS"
    return name


def edge_axis(edge):
    """The direction a direction-selector tests an EDGE against.

    CadQuery's ``BaseDirSelector`` (cadquery/selectors.py) takes the *tangent* of
    a LINE and the *normal* of a CIRCLE, and excludes every other edge kind from
    direction selection. Mirrored exactly, so ``|Z`` means the same thing on both
    backends.
    """
    gt = _geom_type(edge)
    if gt == "LINE":
        try:
            t = edge.tangentAt(edge.FirstParameter)
            return (t.x, t.y, t.z)
        except Exception:
            return (0.0, 0.0, 0.0)
    if gt == "CIRCLE":
        try:
            a = edge.Curve.Axis
            return (a.x, a.y, a.z)
        except Exception:
            return (0.0, 0.0, 0.0)
    return (0.0, 0.0, 0.0)


def face_axis(face):
    """A PLANE face's normal; other surfaces are excluded from direction tests."""
    if _geom_type(face) != "PLANE":
        return (0.0, 0.0, 0.0)
    try:
        n = face.normalAt(0.0, 0.0)
        return (n.x, n.y, n.z)
    except Exception:
        return (0.0, 0.0, 0.0)


def sub_entities(shape, kind):
    """``(shapes, entities)`` -- OCCT sub-shapes paired with selector Entities."""
    subs = shape.Edges if kind == "edges" else shape.Faces
    axis_of = edge_axis if kind == "edges" else face_axis
    tag = "Edge" if kind == "edges" else "Face"
    ents = []
    for i, s in enumerate(subs):
        c = s.CenterOfMass
        ents.append(Entity(center=(c.x, c.y, c.z), axis=axis_of(s),
                           geom_type=_geom_type(s), name="%s%d" % (tag, i + 1)))
    return subs, ents


def join_selectors(selectors):
    """CISP carries a TUPLE of selectors; the DSL takes ONE string.

    The tuple is the ``or`` (set union) of its members, each parenthesised -- the
    identical rule the CadQuery backend's ``join_selectors`` applies, so a given
    CISP op selects the same sub-shapes on both kernels. Empty tuple -> None,
    which every caller reads as "no filter" (take them all).
    """
    sels = [str(s).strip() for s in (selectors or []) if str(s).strip()]
    if not sels:
        return None
    if len(sels) == 1:
        return sels[0]
    return " or ".join("(%s)" % s for s in sels)


def select_subshapes(shape, kind, selectors, default=None):
    """The sub-shapes a CISP selector tuple picks. Raises on a miss."""
    sel = join_selectors(selectors)
    if sel is None:
        sel = default
    subs, ents = sub_entities(shape, kind)
    if not sel:
        return list(subs)
    picked = evaluate(parse(sel), ents)
    by_name = {e.name: i for i, e in enumerate(ents)}
    out = [subs[by_name[e.name]] for e in picked]
    if not out:
        raise ValueError("selector %r selected no %s" % (sel, kind))
    return out


# -- profiles -> exact faces -------------------------------------------------
def profile_faces(profile, plane, w):
    """Every closed loop of an F-rep profile as an exact Part.Face at height w.

    Circles become genuine Part.Circle arcs (NOT polygons): that is the whole
    point of a B-rep kernel, and it is what makes the volumes analytic.
    """
    faces = []
    normal = normal_of(plane)
    for (x, y, ww, hh) in profile.get("rects", []):
        pts = [(x, y), (x + ww, y), (x + ww, y + hh), (x, y + hh), (x, y)]
        wire = Part.makePolygon([to_world(plane, a, b, w) for (a, b) in pts])
        faces.append(Part.Face(wire))
    for (cx, cy, r) in profile.get("circles", []):
        circ = Part.Circle(to_world(plane, cx, cy, w), normal, float(r))
        faces.append(Part.Face(Part.Wire([circ.toShape()])))
    for verts in profile.get("polys", []):
        if len(verts) < 3:
            continue
        loop = list(verts)
        if loop[0] != loop[-1]:
            loop = loop + [loop[0]]
        wire = Part.makePolygon([to_world(plane, a, b, w) for (a, b) in loop])
        faces.append(Part.Face(wire))
    return faces


def solidify(shape):
    """Unwrap the Compound that OCCT's booleans return around a single solid.

    Part's fuse/cut/common hand back a ``Part.Compound`` even when the result is
    one solid. A Compound carries ``.Volume`` but NOT ``.CenterOfMass`` (and
    reports no shells, so ``isClosed()`` is meaningless on it), so every mass
    property downstream must see the Solid itself.
    """
    if shape is None:
        return None
    if shape.ShapeType == "Compound" and len(shape.Solids) == 1:
        return shape.Solids[0]
    return shape


def center_of_mass(shape):
    """Volume-weighted centroid, valid for a Compound too.

    A pattern or a mirror can legitimately leave several DISJOINT solids, which
    OCCT keeps as a Compound -- and ``Compound.CenterOfMass`` does not exist. The
    volume-weighted mean of the solids' centroids is the exact centroid of the
    whole, and it agrees with ``Solid.CenterOfMass`` in the single-solid case.
    """
    solids = shape.Solids
    if not solids:
        c = shape.BoundBox.Center
        return (c.x, c.y, c.z)
    if len(solids) == 1:
        c = solids[0].CenterOfMass
        return (c.x, c.y, c.z)
    total = sum(s.Volume for s in solids)
    if total <= 0.0:
        c = shape.BoundBox.Center
        return (c.x, c.y, c.z)
    x = sum(s.CenterOfMass.x * s.Volume for s in solids) / total
    y = sum(s.CenterOfMass.y * s.Volume for s in solids) / total
    z = sum(s.CenterOfMass.z * s.Volume for s in solids) / total
    return (x, y, z)


def is_closed(shape):
    """Whether every solid is closed. ``Compound.isClosed()`` is not meaningful."""
    solids = shape.Solids
    if not solids:
        return False
    for s in solids:
        try:
            if not s.isClosed():
                return False
        except Exception:
            return False
    return True


def fuse_all(shapes):
    if not shapes:
        return None
    out = shapes[0]
    for s in shapes[1:]:
        out = out.fuse(s)
    if len(shapes) > 1:
        out = out.removeSplitter()
    return solidify(out)


# -- the Sketcher: REAL geometric constraints, solved by planegcs -------------
#
# https://wiki.freecad.org/Sketcher_scripting -- a Sketcher::SketchObject takes
# Part geometry via addGeometry() and Sketcher.Constraint objects via
# addConstraint(); solve() runs FreeCAD's planegcs solver and .DoF reports the
# remaining degrees of freedom. This is a genuine geometric constraint solver,
# not a DOF counter -- so on this backend a `constrain` op actually MOVES the
# geometry, and the solid is built from the SOLVED profile.
CONSTRAINT_MAP = {
    "horizontal": "Horizontal",
    "vertical": "Vertical",
    "parallel": "Parallel",
    "perpendicular": "Perpendicular",
    "equal": "Equal",
    "coincident": "Coincident",
    "distance": "Distance",
    "radius": "Radius",
}


def _sketch_geometry(sketch):
    """(geometry, entity -> geo indices) for one CISP sketch.

    A CISP rectangle is ONE entity with four sides; the Sketcher has no rectangle
    primitive, so it becomes four LineSegments closed by Coincident constraints
    and squared by Horizontal/Vertical -- exactly what Sketcher's own rectangle
    tool emits. It therefore solves as a rigid rectangle and reads back as one.
    """
    geoms = []
    internal = []
    index = {}
    for ent in sketch.get("entities", []):
        p = ent["params"]
        kind = ent["type"]
        base = len(geoms)
        if kind == "point":
            geoms.append(Part.Point(V(float(p["x"]), float(p["y"]), 0.0)))
            index[ent["id"]] = [base]
        elif kind == "line":
            geoms.append(Part.LineSegment(V(float(p["x1"]), float(p["y1"]), 0.0),
                                          V(float(p["x2"]), float(p["y2"]), 0.0)))
            index[ent["id"]] = [base]
        elif kind == "circle":
            geoms.append(Part.Circle(V(float(p["cx"]), float(p["cy"]), 0.0),
                                     V(0, 0, 1), float(p["r"])))
            index[ent["id"]] = [base]
        elif kind == "rectangle":
            x, y = float(p["x"]), float(p["y"])
            w, h = float(p["w"]), float(p["h"])
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            for i in range(4):
                a = corners[i]
                b = corners[(i + 1) % 4]
                geoms.append(Part.LineSegment(V(a[0], a[1], 0.0), V(b[0], b[1], 0.0)))
            ids = [base + i for i in range(4)]
            index[ent["id"]] = ids
            for i in range(4):
                internal.append(("Coincident", ids[i], 2, ids[(i + 1) % 4], 1))
            internal.append(("Horizontal", ids[0]))
            internal.append(("Vertical", ids[1]))
            internal.append(("Horizontal", ids[2]))
            internal.append(("Vertical", ids[3]))
    return geoms, index, internal


def _constraint_args(con, index):
    """One CISP `constrain` op as Sketcher.Constraint arguments, or None.

    The CISP constraint references *entities*, not vertices, so the vertex-level
    constraints take the idiomatic reading:
      * coincident -> endpoint of A on startpoint of B (the polyline chain);
      * distance with no B -> the LENGTH of line A (Sketcher's 2-arg Distance).
    Sub-element numbering is the wiki's: 1 = start, 2 = end, 3 = centre.
    """
    kind = con["kind"]
    name = CONSTRAINT_MAP.get(kind)
    if name is None:
        return None, "unknown constraint kind '%s'" % kind
    a = index.get(con.get("a"))
    if not a:
        return None, "constraint references unknown entity '%s'" % con.get("a")
    ga = a[0]
    b = index.get(con.get("b")) if con.get("b") else None
    gb = b[0] if b else None
    value = con.get("value")
    if kind in ("horizontal", "vertical"):
        return (name, ga), None
    if kind in ("parallel", "perpendicular", "equal"):
        if gb is None:
            return None, "'%s' needs two entities" % kind
        return (name, ga, gb), None
    if kind == "coincident":
        if gb is None:
            return None, "'coincident' needs two entities"
        return (name, ga, 2, gb, 1), None
    if kind == "radius":
        if value is None:
            return None, "'radius' needs a value"
        return (name, ga, float(value)), None
    if kind == "distance":
        if value is None:
            return None, "'distance' needs a value"
        if gb is None:
            return (name, ga, float(value)), None
        return (name, ga, 1, gb, 1, float(value)), None
    return None, "unhandled constraint kind '%s'" % kind


def _solved_profile(sketch, sk, index):
    """Read the SOLVED sketch back into an F-rep profile dict.

    The entity order and the rect/circle/poly grouping reproduce
    ``frep._profile_of`` exactly, so the solved profile is a drop-in replacement
    for the one the op stream recorded.
    """
    geometry = sk.Geometry
    rects, circles, polys = [], [], []
    lines = []
    for ent in sketch.get("entities", []):
        ids = index.get(ent["id"], [])
        if not ids:
            continue
        kind = ent["type"]
        if kind == "circle":
            g = geometry[ids[0]]
            circles.append([g.Center.x, g.Center.y, g.Radius])
        elif kind == "rectangle":
            xs, ys = [], []
            for gi in ids:
                g = geometry[gi]
                xs.extend([g.StartPoint.x, g.EndPoint.x])
                ys.extend([g.StartPoint.y, g.EndPoint.y])
            rects.append([min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)])
        elif kind == "line":
            g = geometry[ids[0]]
            if not lines:
                lines.append([g.StartPoint.x, g.StartPoint.y])
            lines.append([g.EndPoint.x, g.EndPoint.y])
    if len(lines) >= 4 and lines[0] == lines[-1]:
        lines = lines[:-1]
    if len(lines) >= 3:
        polys.append(lines)
    return {"rects": rects, "circles": circles, "polys": polys}


def solve_sketches(doc, spec):
    """Build + solve every CISP sketch in FreeCAD's Sketcher. Returns
    ``(reports, substitutions)`` where substitutions maps the ORIGINAL profile
    (as the F-rep tree recorded it) to the SOLVED one."""
    import Sketcher

    reports = []
    subs = []
    sketches = spec.get("sketches") or []
    for sketch in sketches:
        rep = {"id": sketch["id"], "plane": sketch.get("plane", "XY"),
               "constraints": len(sketch.get("constraints") or []),
               "solved": False, "dof": None, "fully_constrained": False,
               "conflicting": [], "redundant": [], "malformed": [],
               "status": 0, "errors": []}
        geoms, index, internal = _sketch_geometry(sketch)
        if not geoms:
            reports.append(rep)
            continue
        sk = doc.addObject("Sketcher::SketchObject", "Sketch_" + sketch["id"])
        for g in geoms:
            sk.addGeometry(g, False)
        for args in internal:
            try:
                sk.addConstraint(Sketcher.Constraint(*args))
            except Exception as exc:
                rep["errors"].append("internal %s: %s" % (args[0], exc))
        for con in sketch.get("constraints") or []:
            args, err = _constraint_args(con, index)
            if err is not None:
                rep["errors"].append(err)
                continue
            try:
                sk.addConstraint(Sketcher.Constraint(*args))
            except Exception as exc:
                rep["errors"].append("%s: %s" % (args[0], exc))
        rep["status"] = int(sk.solve())
        doc.recompute()
        rep["solved"] = rep["status"] == 0
        # .DoF is only meaningful AFTER a solve; before one it reads 0 (stale).
        rep["dof"] = int(sk.DoF)
        rep["fully_constrained"] = bool(sk.FullyConstrained)
        rep["conflicting"] = [int(i) for i in sk.ConflictingConstraints]
        rep["redundant"] = [int(i) for i in sk.RedundantConstraints]
        rep["malformed"] = [int(i) for i in sk.MalformedConstraints]
        reports.append(rep)
        # Only a sketch that actually CARRIES a constraint may move the model:
        # an unconstrained sketch is left bit-for-bit as the op stream wrote it,
        # so every existing (unconstrained) op stream is unchanged.
        if rep["solved"] and (sketch.get("constraints") or []):
            subs.append({"original": sketch["original_profile"],
                         "solved": _solved_profile(sketch, sk, index)})
    return reports, subs


def _profile_key(profile):
    return json.dumps(profile, sort_keys=True)


# -- the F-rep tree -> Part shapes -------------------------------------------
def build(node, subs):
    t = node["t"]
    if t == "extrude":
        plane = node["plane"]
        w0, w1 = float(node["w0"]), float(node["w1"])
        faces = profile_faces(subs.get(_profile_key(node["profile"]), node["profile"]),
                              plane, w0)
        if not faces:
            return None
        vec = normal_of(plane).multiply(w1 - w0)
        return fuse_all([f.extrude(vec) for f in faces])
    if t == "cyl":
        plane = node["plane"]
        w0, w1 = float(node["w0"]), float(node["w1"])
        lo, hi = (w0, w1) if w0 <= w1 else (w1, w0)
        base = to_world(plane, node["cu"], node["cv"], lo)
        return Part.makeCylinder(float(node["r"]), hi - lo, base, normal_of(plane))
    if t == "revolve":
        plane = node["plane"]
        au, av, du, dv, nu, nv = node["axis"]
        faces = profile_faces(subs.get(_profile_key(node["profile"]), node["profile"]),
                              plane, 0.0)
        if not faces:
            return None
        base = to_world(plane, au, av, 0.0)
        # the axis direction is in-plane, so it maps through the plane with w=0
        adir = to_world(plane, du, dv, 0.0)
        angle = float(node.get("angle", 360.0))
        return fuse_all([f.revolve(base, adir, angle) for f in faces])
    if t == "bool":
        a = build(node["a"], subs)
        b = build(node["b"], subs)
        if a is None:
            return None
        if b is None:
            return a
        op = node["op"]
        if op == "union":
            return solidify(a.fuse(b).removeSplitter())
        if op == "intersect":
            return solidify(a.common(b).removeSplitter())
        return solidify(a.cut(b).removeSplitter())
    if t == "shell":
        child = build(node["child"], subs)
        if child is None:
            return None
        return shell_solid(child, float(node["thickness"]), node.get("faces") or [])
    if t == "mirror":
        child = build(node["child"], subs)
        if child is None:
            return None
        pl = str(node["plane"]).upper()
        normal = {"XY": V(0, 0, 1), "YZ": V(1, 0, 0)}.get(pl, V(0, 1, 0))
        return child.fuse(child.mirror(V(0, 0, 0), normal)).removeSplitter()
    if t == "pattern":
        child = build(node["child"], subs)
        if child is None:
            return None
        parts = []
        for (dx, dy, dz, ang) in node["transforms"]:
            c = child.copy()
            if ang:
                c.rotate(V(0, 0, 0), V(0, 0, 1), float(ang))
            c.translate(V(float(dx), float(dy), float(dz)))
            parts.append(c)
        return fuse_all(parts)
    raise ValueError("unknown F-rep node kind '%s'" % t)


def shell_solid(child, thickness, faces):
    """Hollow INWARD. The outer surface must NOT move.

    Two cases, and they are different OCCT operations:

    * ``faces`` EMPTY -- a CLOSED hollow: a sealed internal void, no opening.
      ``makeThickness`` cannot express this (it is *defined* in terms of the faces
      it removes: "a hollowed solid is built from an initial solid and a set of
      faces on this solid, which are to be removed" --
      https://wiki.freecad.org/Topological_data_scripting). So the void is built
      directly: ``makeOffsetShape(-t, tol)`` is the solid offset INWARD by t
      (https://wiki.freecad.org/Part_Offset), and cutting it out of the original
      leaves exactly the wall. Measured on a 60x40x20 box at t=3: volume
      22296.0 == 48000 - 54*34*14 exactly, bbox 60x40x20, and the void's corner
      sits at exactly (3, 3, 3) -- a true 3mm wall on every face, including the
      corners. (A bbox check alone would NOT prove this: a wall thinned to
      t/sqrt(3) by an uncorrected corner normal preserves the bbox exactly. The
      analytic volume is what proves it.)

    * ``faces`` NON-EMPTY -- those faces are REMOVED and the rest become walls:
      exactly ``makeThickness(faces, offset, tolerance)``. The SIGN of ``offset``
      is the direction, and it is load-bearing, not luck: POSITIVE grows the part
      OUTWARD (a 60x40x20 box at +3 measures 66x46x23), NEGATIVE hollows INWARD
      and leaves the bbox at 60x40x20. It is the scripting form of PartDesign
      Thickness's "Reversed / make thickness inwards" flag
      (https://wiki.freecad.org/PartDesign_Thickness), whose default is OUTWARD --
      so a backend that omitted the minus sign would silently grow the part.
    """
    t = abs(float(thickness))
    if not faces:
        inner = child.makeOffsetShape(-t, 1e-6)
        if inner is None or not inner.Solids:
            raise ValueError("shell: inward offset of %g collapsed the solid" % t)
        return solidify(child.cut(inner))
    picked = select_subshapes(child, "faces", faces)
    return child.makeThickness(picked, -t, 1e-3)


def apply_blends(shape, blends):
    """Apply each fillet/chamfer op IN ORDER, to the edges IT selected.

    ``Shape.makeFillet(radius, edges)`` / ``makeChamfer(distance, edges)``
    (https://wiki.freecad.org/Part_scripting) take an explicit edge list. The CISP
    op carries a tuple of selector strings; an EMPTY tuple still means every edge,
    so op streams that never selected are unchanged.
    """
    for b in blends or []:
        if shape is None:
            return None
        value = float(b.get("value", 0.0))
        if value <= 0.0:
            continue
        edges = select_subshapes(shape, "edges", b.get("selectors") or [])
        if not edges:
            continue
        if b.get("kind") == "chamfer":
            shape = shape.makeChamfer(value, edges)
        else:
            shape = shape.makeFillet(value, edges)
    return shape


# -- topological naming: per-face records the host fingerprints ---------------
SURFACE_KIND = {"PLANE": "planar", "CYLINDER": "cylindrical",
                "SPHERE": "spherical", "CONE": "conical", "TORUS": "blend"}


def face_records(shape):
    """Each face as (surface kind, normal, centroid, area) -- the attributes
    :mod:`harnesscad.domain.geometry.topology.topological_naming` fingerprints,
    so a stored face reference can be migrated across a rebuild."""
    out = []
    for i, f in enumerate(shape.Faces):
        gt = _geom_type(f)
        n = face_axis(f)
        c = f.CenterOfMass
        out.append({
            "id": "Face%d" % (i + 1),
            "surface": SURFACE_KIND.get(gt, gt.lower() or "other"),
            "normal": None if (n[0] == 0.0 and n[1] == 0.0 and n[2] == 0.0) else list(n),
            "centroid": [c.x, c.y, c.z],
            "area": f.Area,
        })
    return out


def edge_records(shape):
    out = []
    for i, e in enumerate(shape.Edges):
        c = e.CenterOfMass
        out.append({"id": "Edge%d" % (i + 1), "type": _geom_type(e),
                    "center": [c.x, c.y, c.z], "axis": list(edge_axis(e)),
                    "length": e.Length})
    return out


# -- export ------------------------------------------------------------------
def step_schema(path):
    """The FILE_SCHEMA the STEP file actually DECLARES.

    ``Shape.exportStep`` makes no ``Interface_Static`` call (TopoShape.cpp), and
    ``Part.Interface`` / ``ImportGui`` -- the only things that can call
    ``writeStepScheme`` -- are not importable in a console app. So headless FreeCAD
    always writes OCCT's built-in default, AP214 (AUTOMOTIVE_DESIGN, ISO
    10303-214). We do not guess it: we read it back out of the file we wrote.
    """
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if "FILE_SCHEMA" in line:
                    text = line
                    if "10303" not in text:
                        continue
                    if "10303 242" in text or "10303, 242" in text:
                        return "AP242"
                    if "10303 214" in text or "10303, 214" in text:
                        return "AP214"
                    if "10303 203" in text or "10303, 203" in text:
                        return "AP203"
                    return text.strip()
    except Exception:
        pass
    return "unknown"


def step_unit(path):
    """The length unit the STEP file declares. FreeCAD's internal unit is mm
    (https://wiki.freecad.org/Import_Export), and the writer emits it as-is."""
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read(200000)
        if "SI_UNIT(.MILLI.,.METRE.)" in text:
            return "MM"
        if "SI_UNIT($,.METRE.)" in text:
            return "M"
    except Exception:
        pass
    return "unknown"


def write_stl(shape, path, linear, angular):
    """STL with a DECLARED tessellation deflection.

    ``Shape.exportStl(path)`` hard-codes a 0.01 linear deflection
    (``BRepMesh_IncrementalMesh(shape, 0.01, ...)`` in TopoShape.cpp) with no way
    to change it -- an unset deflection is a silent quality bug. ``MeshPart.
    meshFromShape`` takes the deflection explicitly, so the mesh we ship is the
    mesh we asked for. Falls back to exportStl if MeshPart is unavailable.
    """
    try:
        import Mesh
        import MeshPart
        mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=float(linear),
                                      AngularDeflection=float(angular),
                                      Relative=False)
        mesh.write(path)
        return {"linear_deflection": float(linear),
                "angular_deflection": float(angular),
                "facets": int(mesh.CountFacets), "source": "MeshPart.meshFromShape"}
    except ImportError:
        shape.exportStl(path)
        return {"linear_deflection": 0.01, "angular_deflection": 0.055,
                "facets": None, "source": "Shape.exportStl (hard-coded 0.01)"}


# -- the FreeCAD document / feature tree -------------------------------------
def document_tree(doc, shape):
    objects = []
    for obj in doc.Objects:
        entry = {"name": obj.Name, "label": obj.Label, "type": obj.TypeId,
                 "visibility": None}
        pl = getattr(obj, "Placement", None)
        if pl is not None:
            ax = pl.Rotation.Axis
            entry["placement"] = {
                "position": [pl.Base.x, pl.Base.y, pl.Base.z],
                "rotation": [ax.x, ax.y, ax.z, pl.Rotation.Angle],
            }
        shp = getattr(obj, "Shape", None)
        if shp is not None:
            entry["shape"] = {"type": shp.ShapeType,
                              "volume": shp.Volume, "area": shp.Area}
        objects.append(entry)
    return {
        "document": {"name": doc.Name, "filename": doc.FileName or None,
                     "object_count": len(doc.Objects)},
        "objects": objects,
        "view": None,
    }


def main():
    spec = json.load(open(os.path.join(HERE, "spec.json")))
    out = {"ok": False}
    try:
        root = spec.get("root")
        doc = FreeCAD.newDocument("harnesscad")
        # The Sketcher objects live in their OWN document: `query('document')` is
        # the built model's feature tree, and a scratch solver object is not part
        # of it. The solve is identical either way -- planegcs is per-sketch.
        sdoc = FreeCAD.newDocument("harnesscad_sketches")
        sketch_reports, subs_list = solve_sketches(sdoc, spec)
        subs = {}
        for s in subs_list:
            subs[_profile_key(s["original"])] = s["solved"]
        if root is None:
            shape = None
        else:
            shape = solidify(apply_blends(build(root, subs), spec.get("blends") or []))
        if shape is None:
            out = {"ok": True, "solid_present": False,
                   "sketches": sketch_reports,
                   "document": document_tree(doc, None)}
        else:
            # Register the result as a real document object, so the feature tree
            # FreeCAD reports is the one the harness reads back.
            obj = doc.addObject("Part::Feature", "Model")
            obj.Shape = shape
            doc.recompute()
            bb = shape.BoundBox
            com = center_of_mass(shape)
            out = {
                "ok": True,
                "solid_present": True,
                # EXACT B-rep mass properties -- OCCT's analytic integration, not
                # a mesh approximation.
                "volume": shape.Volume,
                "surface_area": shape.Area,
                "bbox": [bb.XLength, bb.YLength, bb.ZLength],
                "bbox_min": [bb.XMin, bb.YMin, bb.ZMin],
                "center_of_mass": [com[0], com[1], com[2]],
                "faces": len(shape.Faces),
                "edges": len(shape.Edges),
                "vertices": len(shape.Vertexes),
                "solids": len(shape.Solids),
                "shells": len(shape.Shells),
                "shape_type": shape.ShapeType,
                "is_valid": bool(shape.isValid()),
                "is_closed": is_closed(shape),
                "sketches": sketch_reports,
                "face_records": face_records(shape),
                "edge_records": edge_records(shape),
                "document": document_tree(doc, shape),
                "export": {},
            }
            linear = float(spec.get("stl_linear_deflection", 0.01))
            angular = float(spec.get("stl_angular_deflection",
                                     default_angular_deflection(linear)))
            for fmt in spec.get("exports", []):
                path = os.path.join(HERE, "model." + fmt)
                if fmt == "stl":
                    out["export"]["stl"] = write_stl(shape, path, linear, angular)
                elif fmt == "step":
                    shape.exportStep(path)
                    out["export"]["step"] = {"schema": step_schema(path),
                                             "unit": step_unit(path)}
                elif fmt == "brep":
                    shape.exportBrep(path)
                    out["export"]["brep"] = {"format": "OpenCASCADE BREP"}
                elif fmt == "iges":
                    shape.exportIges(path)
                    out["export"]["iges"] = {"format": "IGES 5.3"}
    except Exception as exc:
        out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
               "traceback": traceback.format_exc()}
    with open(os.path.join(HERE, "result.json"), "w") as fh:
        json.dump(out, fh, sort_keys=True)


main()
'''
