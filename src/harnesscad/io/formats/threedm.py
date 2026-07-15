"""Rhino ``.3dm`` (openNURBS / File3dm) reader and writer.

Rhino is a widely used commercial CAD/NURBS package, and its native container is
the ``.3dm`` file (openNURBS). ``rhino3dm`` is that library shipped as a
STANDALONE pip wheel -- no Rhino application, no licence: it reads and writes the
``.3dm`` format and carries the openNURBS geometry types (``Mesh``, ``Brep``,
``Extrusion``, curves, surfaces, ``BoundingBox``). The harness could already
speak STL/OBJ/PLY/AMF/3MF/STEP but had no way to hand a part to, or accept one
from, the Rhino ecosystem. This module is that bridge.

WHAT THIS CODEC IS
------------------
A *container* codec, exactly like :mod:`harnesscad.io.formats.ply`: it moves an
indexed triangle mesh in and out of a ``.3dm`` file, carrying the model unit
along with it. It is NOT a geometry kernel -- it does not mesh a Brep, compute a
boolean, or evaluate a NURBS surface. Its job is transport and round-trip.

UNITS ARE EXPLICIT, ON PURPOSE
------------------------------
A ``.3dm`` file has its own document unit system (``File3dm.Settings.
ModelUnitSystem``: Rhino defaults to millimetres but a file can be in metres,
inches, feet, ...). The harness once shipped a glTF exporter that silently
rotated every part; the same class of bug for units would be to write "50" into a
metre-unit file and read it back believing it was 50 mm. So the unit is written
into the document settings on the way out and read back off the settings on the
way in, and mapped through an EXPLICIT table (:data:`_UNIT_TO_RHINO` /
:data:`_RHINO_TO_UNIT`) -- never assumed. A unit that has no exact rhino3dm
equivalent is refused loudly rather than silently coerced.

PUBLIC API
----------
* :func:`write_3dm` -- write ``(vertices, faces)`` + unit to a ``.3dm`` file.
* :func:`serialize_3dm` -- the same, returned as ``bytes``.
* :func:`read_3dm` -- read the first mesh object back as
  ``(vertices, triangles, unit)`` with faces fan-triangulated.
* :func:`measure_3dm` -- read a ``.3dm`` and report its axis-aligned bounding box
  and unit using rhino3dm's own ``GetBoundingBox`` -- an independent measurement
  path for the round-trip test.
* :data:`RHINO3DM_AVAILABLE` -- False when the wheel is not installed, so callers
  (and tests) can skip cleanly instead of crashing.

Determinism note: unlike PLY, a ``.3dm`` file embeds document metadata and is not
guaranteed to be byte-identical across writes. This codec therefore promises
GEOMETRY + UNIT round-trip stability (the mesh you write is the mesh you read,
in the unit you declared), not byte reproducibility -- and the tests assert the
former, never the latter.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]

__all__ = [
    "RHINO3DM_AVAILABLE",
    "ThreeDmError",
    "UNITS",
    "write_3dm",
    "serialize_3dm",
    "read_3dm",
    "measure_3dm",
]


class ThreeDmError(ValueError):
    """Raised when a ``.3dm`` cannot be written or read as an HarnessCAD mesh."""


# rhino3dm is an OPTIONAL dependency. Guard the import so the module (and the
# whole io.formats package) imports cleanly on a machine without the wheel; every
# entry point then raises a clean ThreeDmError instead of an ImportError.
try:  # pragma: no cover - availability depends on the host
    import rhino3dm as _r3  # type: ignore

    RHINO3DM_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "unavailable"
    _r3 = None  # type: ignore
    RHINO3DM_AVAILABLE = False


#: The harness's unit vocabulary (shared with 3MF), mapped to the openNURBS
#: UnitSystem enum. EXPLICIT: a unit not in this table is refused, never guessed.
UNITS: Tuple[str, ...] = (
    "micron", "millimeter", "centimeter", "inch", "foot", "meter")


def _unit_to_rhino(unit: str):
    """Map an HarnessCAD unit name to a rhino3dm ``UnitSystem`` member."""
    table = {
        "micron": _r3.UnitSystem.Microns,
        "millimeter": _r3.UnitSystem.Millimeters,
        "centimeter": _r3.UnitSystem.Centimeters,
        "inch": _r3.UnitSystem.Inches,
        "foot": _r3.UnitSystem.Feet,
        "meter": _r3.UnitSystem.Meters,
    }
    try:
        return table[unit]
    except KeyError:
        raise ThreeDmError(
            "unit %r has no exact .3dm equivalent; use one of %s"
            % (unit, ", ".join(UNITS))) from None


def _rhino_to_unit(system) -> str:
    """Map a rhino3dm ``UnitSystem`` back to an HarnessCAD unit name."""
    table = {
        int(_r3.UnitSystem.Microns): "micron",
        int(_r3.UnitSystem.Millimeters): "millimeter",
        int(_r3.UnitSystem.Centimeters): "centimeter",
        int(_r3.UnitSystem.Inches): "inch",
        int(_r3.UnitSystem.Feet): "foot",
        int(_r3.UnitSystem.Meters): "meter",
    }
    key = int(system)
    if key not in table:
        raise ThreeDmError(
            "the .3dm file is in a unit system (%s) HarnessCAD has no name for"
            % system)
    return table[key]


def _require() -> None:
    if not RHINO3DM_AVAILABLE:
        raise ThreeDmError(
            "rhino3dm is not installed; `pip install rhino3dm` to read/write .3dm")


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------

def _build_mesh(vertices: Sequence[Sequence[float]],
                faces: Sequence[Sequence[int]]):
    """A rhino3dm ``Mesh`` from an indexed ``(vertices, faces)`` soup.

    Triangles and quads are both written natively (openNURBS meshes carry either);
    a face with more than four sides is fan-triangulated. Vertex order and the
    coordinate values are preserved exactly.
    """
    mesh = _r3.Mesh()
    for v in vertices:
        if len(v) != 3:
            raise ThreeDmError("each vertex must have 3 coordinates")
        mesh.Vertices.Add(float(v[0]), float(v[1]), float(v[2]))
    nv = len(vertices)
    for face in faces:
        ids = [int(i) for i in face]
        if len(ids) < 3:
            raise ThreeDmError("a face needs at least 3 vertices, got %d" % len(ids))
        for iv in ids:
            if iv < 0 or iv >= nv:
                raise ThreeDmError(
                    "face index %d out of range 0..%d" % (iv, nv - 1))
        if len(ids) == 3:
            mesh.Faces.AddFace(ids[0], ids[1], ids[2])
        elif len(ids) == 4:
            mesh.Faces.AddFace(ids[0], ids[1], ids[2], ids[3])
        else:
            for k in range(1, len(ids) - 1):
                mesh.Faces.AddFace(ids[0], ids[k], ids[k + 1])
    return mesh


def _new_file(unit: str):
    if unit not in UNITS:
        raise ThreeDmError(
            "unit must be one of %s, got %r" % (", ".join(UNITS), unit))
    f = _r3.File3dm()
    f.ApplicationName = "HarnessCAD"
    # The whole point of the codec: state the unit EXPLICITLY in the document.
    f.Settings.ModelUnitSystem = _unit_to_rhino(unit)
    return f


def write_3dm(path: str,
              vertices: Sequence[Sequence[float]],
              faces: Sequence[Sequence[int]],
              unit: str = "millimeter",
              name: str = "model",
              version: int = 0) -> str:
    """Write an indexed mesh to a ``.3dm`` file. Returns the path written.

    ``version`` is the openNURBS archive version (0 = the wheel's current
    default). The mesh is stored as a single ``File3dmObject`` whose attribute
    name is ``name``; the document unit is set from ``unit``.
    """
    _require()
    f = _new_file(unit)
    mesh = _build_mesh(vertices, faces)
    attr = _r3.ObjectAttributes()
    attr.Name = str(name)
    f.Objects.AddMesh(mesh, attr)
    if not f.Write(str(path), int(version)):
        raise ThreeDmError("rhino3dm failed to write %r" % path)
    return str(path)


def serialize_3dm(vertices: Sequence[Sequence[float]],
                  faces: Sequence[Sequence[int]],
                  unit: str = "millimeter",
                  name: str = "model",
                  version: int = 0) -> bytes:
    """Serialise an indexed mesh to ``.3dm`` bytes (via a temporary file)."""
    _require()
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".3dm")
    os.close(fd)
    try:
        write_3dm(tmp, vertices, faces, unit=unit, name=name, version=version)
        with open(tmp, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------

def _iter_meshes(f):
    """Yield every ``Mesh`` geometry in a File3dm, in object order."""
    for obj in f.Objects:
        geom = obj.Geometry
        if isinstance(geom, _r3.Mesh):
            yield geom


def _mesh_to_indexed(mesh) -> Tuple[List[Vec3], List[Tuple[int, int, int]]]:
    """A rhino3dm mesh -> ``(vertices, triangles)``, faces fan-triangulated."""
    verts: List[Vec3] = []
    for i in range(len(mesh.Vertices)):
        p = mesh.Vertices[i]
        verts.append((float(p.X), float(p.Y), float(p.Z)))
    tris: List[Tuple[int, int, int]] = []
    for i in range(len(mesh.Faces)):
        face = mesh.Faces[i]
        ids = [int(x) for x in face]
        # A rhino triangle face repeats its last index (a==b==c==d is a tri); a
        # genuine quad has four distinct corners. De-duplicate the trailing repeat
        # before fan-triangulating so a triangle does not become a degenerate one.
        if len(ids) == 4 and ids[2] == ids[3]:
            ids = ids[:3]
        for k in range(1, len(ids) - 1):
            tris.append((ids[0], ids[k], ids[k + 1]))
    return verts, tris


def read_3dm(path: str) -> Tuple[List[Vec3], List[Tuple[int, int, int]], str]:
    """Read the mesh objects of a ``.3dm`` file.

    Returns ``(vertices, triangles, unit)``. When the file holds more than one
    mesh object they are concatenated (indices offset) into one soup, matching the
    other multi-object codecs. ``unit`` is read from the document settings.
    """
    _require()
    f = _r3.File3dm.Read(str(path))
    if f is None:
        raise ThreeDmError("rhino3dm could not read %r (not a .3dm file?)" % path)
    unit = _rhino_to_unit(f.Settings.ModelUnitSystem)
    all_verts: List[Vec3] = []
    all_tris: List[Tuple[int, int, int]] = []
    for mesh in _iter_meshes(f):
        verts, tris = _mesh_to_indexed(mesh)
        base = len(all_verts)
        all_verts.extend(verts)
        all_tris.extend((a + base, b + base, c + base) for a, b, c in tris)
    if not all_verts:
        raise ThreeDmError(
            "the .3dm file %r has no mesh objects HarnessCAD can read" % path)
    return all_verts, all_tris, unit


def measure_3dm(path: str) -> dict:
    """Measure a ``.3dm`` with rhino3dm's OWN bounding box -- an independent oracle.

    Returns ``{"unit", "bbox_min", "bbox_max", "bbox", "vertex_count",
    "face_count"}``. The bounding box is taken from the openNURBS geometry's
    ``GetBoundingBox`` (Rhino's kernel), not recomputed here, so the round-trip
    test can compare the harness's measurement against Rhino's.
    """
    _require()
    f = _r3.File3dm.Read(str(path))
    if f is None:
        raise ThreeDmError("rhino3dm could not read %r" % path)
    unit = _rhino_to_unit(f.Settings.ModelUnitSystem)
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    nv = nf = 0
    for mesh in _iter_meshes(f):
        nv += len(mesh.Vertices)
        nf += len(mesh.Faces)
        bb = mesh.GetBoundingBox()
        for i, c in enumerate((bb.Min.X, bb.Min.Y, bb.Min.Z)):
            lo[i] = min(lo[i], float(c))
        for i, c in enumerate((bb.Max.X, bb.Max.Y, bb.Max.Z)):
            hi[i] = max(hi[i], float(c))
    if nv == 0:
        raise ThreeDmError("the .3dm file %r has no mesh objects to measure" % path)
    return {
        "unit": unit,
        "bbox_min": lo,
        "bbox_max": hi,
        "bbox": [hi[i] - lo[i] for i in range(3)],
        "vertex_count": nv,
        "face_count": nf,
    }
