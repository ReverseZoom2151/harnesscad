"""The FreeCAD-side driver: lowers an F-rep CSG spec onto real Part B-rep solids.

This file is NEVER imported by the harness. It is read as *text* by
:mod:`harnesscad.io.backends.freecad`, written into a content-addressed work
directory next to its spec, and executed by FreeCAD's own interpreter
(``freecadcmd`` / the bundled ``python``), which is a different Python build
from the host's — so nothing here may import ``harnesscad``.

Contract with the host:

  * reads   ``spec.json``    -- the F-rep root node (``frep.Node.spec()``) plus
                               the requested exports and the final blend radius.
  * writes  ``result.json``  -- EXACT B-rep mass properties + topology counts +
                               the FreeCAD document/feature tree.
  * writes  ``model.stl`` / ``model.step`` / ``model.brep`` / ``model.iges``.

Results go through a FILE, never stdout: ``freecadcmd`` redirects ``print`` to
FreeCAD's own console, so a driver that answered on stdout would return nothing.

Everything is exact: circles are real ``Part.Circle`` arcs (not polygons), so the
volumes this returns are the kernel's analytic volumes, not a tessellation.
"""

DRIVER_SOURCE = r'''
import json
import math
import os
import sys
import traceback

import FreeCAD
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
V = FreeCAD.Vector

# The plane convention is the harness's (frep._PLANES): (u_axis, v_axis, w_axis).
PLANES = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}


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


# -- the F-rep tree -> Part shapes -------------------------------------------
def build(node):
    t = node["t"]
    if t == "extrude":
        plane = node["plane"]
        w0, w1 = float(node["w0"]), float(node["w1"])
        faces = profile_faces(node["profile"], plane, w0)
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
        faces = profile_faces(node["profile"], plane, 0.0)
        if not faces:
            return None
        base = to_world(plane, au, av, 0.0)
        # the axis direction is in-plane, so it maps through the plane with w=0
        adir = to_world(plane, du, dv, 0.0)
        angle = float(node.get("angle", 360.0))
        return fuse_all([f.revolve(base, adir, angle) for f in faces])
    if t == "bool":
        a = build(node["a"])
        b = build(node["b"])
        if a is None:
            return None
        if b is None:
            return a
        op = node["op"]
        if op == "union":
            return solidify(a.fuse(b).removeSplitter())
        if op == "intersect":
            return solidify(a.common(b))
        return solidify(a.cut(b))
    if t == "shell":
        child = build(node["child"])
        if child is None:
            return None
        thickness = abs(float(node["thickness"]))
        # Open the face highest along +Z and hollow inward -- the same face the
        # CadQuery backend removes (faces(">Z")), so the two agree.
        best, best_z = None, None
        for f in child.Faces:
            z = f.CenterOfMass.z
            if best_z is None or z > best_z:
                best, best_z = f, z
        return child.makeThickness([best], -thickness, 1e-3)
    if t == "mirror":
        child = build(node["child"])
        if child is None:
            return None
        pl = str(node["plane"]).upper()
        normal = {"XY": V(0, 0, 1), "YZ": V(1, 0, 0)}.get(pl, V(0, 1, 0))
        return child.fuse(child.mirror(V(0, 0, 0), normal)).removeSplitter()
    if t == "pattern":
        child = build(node["child"])
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


def blend(shape, radius, chamfer):
    """A REAL B-rep edge blend on every edge -- a genuine fillet, not a mesh bevel."""
    if shape is None:
        return shape
    edges = shape.Edges
    if not edges:
        return shape
    if radius > 0.0:
        return shape.makeFillet(radius, edges)
    if chamfer > 0.0:
        return shape.makeChamfer(chamfer, edges)
    return shape


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
        if root is None:
            shape = None
        else:
            shape = solidify(blend(build(root), float(spec.get("round", 0.0)),
                                   float(spec.get("chamfer", 0.0))))
        if shape is None:
            out = {"ok": True, "solid_present": False,
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
                "document": document_tree(doc, shape),
            }
            for fmt in spec.get("exports", []):
                path = os.path.join(HERE, "model." + fmt)
                if fmt == "stl":
                    shape.exportStl(path)
                elif fmt == "step":
                    shape.exportStep(path)
                elif fmt == "brep":
                    shape.exportBrep(path)
                elif fmt == "iges":
                    shape.exportIges(path)
    except Exception as exc:
        out = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc),
               "traceback": traceback.format_exc()}
    with open(os.path.join(HERE, "result.json"), "w") as fh:
        json.dump(out, fh, sort_keys=True)


main()
'''
