"""AMF (Additive Manufacturing File Format) reader and writer.

AngelCAD reads and writes AMF (``spaceio::amf_io``): ``polyhedron("part.amf")``
loads a mesh, ``amf_io::merge_lumps`` fuses the lumps of a multi-volume file into
one polyhedron, and the xcsg engine emits AMF as one of its output formats.  AMF
is the ISO/ASTM 52915 XML mesh format and is the one thing STL is not: it is
unit-bearing, integer-indexed (so vertices are shared, not repeated per triangle),
supports several *volumes* (lumps) inside one *object* and several objects inside
one file, carries ``<metadata>``, and may be stored either as plain XML or as a
single-entry ZIP archive with the same ``.amf`` extension.

    <amf unit="millimeter" version="1.1">
      <metadata type="name">bracket</metadata>
      <object id="1">
        <mesh>
          <vertices>
            <vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex>
            ...
          </vertices>
          <volume>
            <triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle>
            ...
          </volume>
        </mesh>
      </object>
    </amf>

The harness had STL (``formats.t2cdean_stl_codec``, triangle soup, unitless) and
GLB (``formats.t2cdean_glb_writer``, binary glTF); it had no indexed, unit-bearing
CAD exchange mesh format and no notion of a multi-lump part.  This module adds:

* :func:`dumps` / :func:`loads` -- plain-XML AMF, deterministic byte output;
* :func:`write_amf` / :func:`read_amf` -- files, with ``compress=True`` producing
  the ZIP flavour (fixed timestamps, so the bytes are reproducible) and the
  reader sniffing the ZIP magic so either flavour loads transparently;
* :class:`AmfObject` -- id + list of volumes, sharing one vertex array;
* :func:`merge_lumps` -- AngelCAD's lump fusion: concatenate the meshes with
  index offsets into one :class:`~geometry.angelcad_polyhedron.Polyhedron`;
* :func:`from_polyhedra` -- pack polyhedra (fan-triangulated) into AMF objects.

Units are validated against the AMF-legal set; the unit attribute is the reason
an AMF round trip is lossless where an STL round trip is not.

Pure stdlib (``xml.etree`` + ``zipfile``), deterministic.
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

from geometry.angelcad_polyhedron import Polyhedron

__all__ = [
    "AmfError",
    "AmfObject",
    "UNITS",
    "dumps",
    "loads",
    "write_amf",
    "read_amf",
    "merge_lumps",
    "from_polyhedra",
    "to_polyhedra",
]

UNITS = ("millimeter", "inch", "feet", "meter", "micron")
_VERSION = "1.1"
_ZIP_MAGIC = b"PK\x03\x04"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class AmfError(Exception):
    """Malformed AMF document."""


class AmfObject:
    """One AMF ``<object>``: a shared vertex array plus one or more volumes."""

    __slots__ = ("id", "vertices", "volumes", "metadata")

    def __init__(
        self,
        id: int,
        vertices: Sequence[Sequence[float]],
        volumes: Sequence[Sequence[Sequence[int]]],
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        self.id = int(id)
        self.vertices: List[Tuple[float, float, float]] = [
            (float(p[0]), float(p[1]), float(p[2])) for p in vertices
        ]
        self.volumes: List[List[Tuple[int, int, int]]] = [
            [(int(t[0]), int(t[1]), int(t[2])) for t in vol] for vol in volumes
        ]
        self.metadata: Dict[str, str] = dict(metadata or {})

    def polyhedra(self) -> List[Polyhedron]:
        """One :class:`Polyhedron` per volume, with the vertex array compacted.

        The AMF vertex array is shared by all volumes of the object, but a lump
        is a self-contained solid, so each volume gets only the vertices it uses
        (remapped in order of first use -- deterministic).
        """
        out: List[Polyhedron] = []
        for vol in self.volumes:
            remap: Dict[int, int] = {}
            verts: List[Tuple[float, float, float]] = []
            faces: List[Tuple[int, int, int]] = []
            for tri in vol:
                face = []
                for iv in tri:
                    if iv not in remap:
                        remap[iv] = len(verts)
                        verts.append(self.vertices[iv])
                    face.append(remap[iv])
                faces.append((face[0], face[1], face[2]))
            out.append(Polyhedron(verts, faces))
        return out

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AmfObject)
            and other.id == self.id
            and other.vertices == self.vertices
            and other.volumes == self.volumes
            and other.metadata == self.metadata
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "AmfObject(id=%d, %d vertices, %d volumes)" % (
            self.id,
            len(self.vertices),
            len(self.volumes),
        )


def _num(v: float) -> str:
    return "%.12g" % float(v)


def _indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for i, child in enumerate(elem):
            _indent(child, level + 1)
            last = i == len(elem) - 1
            if not child.tail or not child.tail.strip():
                child.tail = pad if last else pad + "  "
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------


def from_polyhedra(
    polyhedra: Sequence[Polyhedron], one_object: bool = False
) -> List[AmfObject]:
    """Pack polyhedra into AMF objects (faces are fan-triangulated).

    ``one_object=True`` puts every polyhedron in a *single* object as separate
    volumes (lumps) sharing one vertex array -- the multi-lump layout AngelCAD's
    ``merge_lumps`` is designed to undo.
    """
    if not one_object:
        return [
            AmfObject(i + 1, p.vertices, [p.triangles()])
            for i, p in enumerate(polyhedra)
        ]
    verts: List[Tuple[float, float, float]] = []
    volumes: List[List[Tuple[int, int, int]]] = []
    for p in polyhedra:
        off = len(verts)
        verts.extend(p.vertices)
        volumes.append([(a + off, b + off, c + off) for (a, b, c) in p.triangles()])
    return [AmfObject(1, verts, volumes)]


def dumps(
    objects: Sequence[AmfObject],
    unit: str = "millimeter",
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Serialise AMF objects to a plain-XML AMF document."""
    if unit not in UNITS:
        raise AmfError("unit must be one of %s, got %r" % (", ".join(UNITS), unit))
    if not objects:
        raise AmfError("an AMF document needs at least one object")

    root = ET.Element("amf", {"unit": unit, "version": _VERSION})
    for key in sorted(metadata or {}):
        m = ET.SubElement(root, "metadata", {"type": key})
        m.text = str(metadata[key])

    for obj in objects:
        xobj = ET.SubElement(root, "object", {"id": str(obj.id)})
        for key in sorted(obj.metadata):
            m = ET.SubElement(xobj, "metadata", {"type": key})
            m.text = str(obj.metadata[key])
        xmesh = ET.SubElement(xobj, "mesh")
        xverts = ET.SubElement(xmesh, "vertices")
        for p in obj.vertices:
            xv = ET.SubElement(xverts, "vertex")
            xc = ET.SubElement(xv, "coordinates")
            for name, value in zip(("x", "y", "z"), p):
                ET.SubElement(xc, name).text = _num(value)
        for vol in obj.volumes:
            xvol = ET.SubElement(xmesh, "volume")
            for tri in vol:
                xt = ET.SubElement(xvol, "triangle")
                for name, iv in zip(("v1", "v2", "v3"), tri):
                    ET.SubElement(xt, name).text = str(int(iv))

    _indent(root)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )


def write_amf(
    path: str,
    objects: Sequence[AmfObject],
    unit: str = "millimeter",
    metadata: Optional[Dict[str, str]] = None,
    compress: bool = False,
) -> bytes:
    """Write an AMF file; ``compress`` selects the ZIP flavour.  Returns the bytes."""
    text = dumps(objects, unit=unit, metadata=metadata)
    if not compress:
        data = text.encode("utf-8")
    else:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            info = zipfile.ZipInfo(name or "model.amf", date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, text.encode("utf-8"))
        data = buf.getvalue()
    with open(path, "wb") as fh:
        fh.write(data)
    return data


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------


def _text(elem: Optional[ET.Element], what: str) -> str:
    if elem is None or elem.text is None:
        raise AmfError("missing <%s>" % what)
    return elem.text.strip()


def loads(data) -> Tuple[List[AmfObject], str, Dict[str, str]]:
    """Parse an AMF document (``str``, plain ``bytes`` or ZIP ``bytes``).

    Returns ``(objects, unit, metadata)``.
    """
    if isinstance(data, bytes):
        if data.startswith(_ZIP_MAGIC):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    names = zf.namelist()
                    if not names:
                        raise AmfError("empty AMF zip archive")
                    text = zf.read(names[0]).decode("utf-8")
            except zipfile.BadZipFile as exc:
                raise AmfError("invalid AMF zip: %s" % exc)
        else:
            text = data.decode("utf-8")
    else:
        text = data

    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise AmfError("invalid XML: %s" % exc)
    if root.tag != "amf":
        raise AmfError("root element must be <amf>, got <%s>" % root.tag)

    unit = root.get("unit", "millimeter")
    if unit not in UNITS:
        raise AmfError("illegal AMF unit %r" % unit)

    metadata = {
        m.get("type", ""): (m.text or "").strip()
        for m in root.findall("metadata")
    }

    objects: List[AmfObject] = []
    for xobj in root.findall("object"):
        try:
            oid = int(xobj.get("id", "0"))
        except ValueError:
            raise AmfError("object id must be an integer, got %r" % xobj.get("id"))
        xmesh = xobj.find("mesh")
        if xmesh is None:
            raise AmfError("object %d has no <mesh>" % oid)
        xverts = xmesh.find("vertices")
        if xverts is None:
            raise AmfError("object %d has no <vertices>" % oid)

        verts: List[Tuple[float, float, float]] = []
        for xv in xverts.findall("vertex"):
            xc = xv.find("coordinates")
            if xc is None:
                raise AmfError("vertex without <coordinates> in object %d" % oid)
            try:
                verts.append(
                    (
                        float(_text(xc.find("x"), "x")),
                        float(_text(xc.find("y"), "y")),
                        float(_text(xc.find("z"), "z")),
                    )
                )
            except ValueError as exc:
                raise AmfError("bad vertex coordinate in object %d: %s" % (oid, exc))

        volumes: List[List[Tuple[int, int, int]]] = []
        for xvol in xmesh.findall("volume"):
            tris: List[Tuple[int, int, int]] = []
            for xt in xvol.findall("triangle"):
                try:
                    tri = (
                        int(_text(xt.find("v1"), "v1")),
                        int(_text(xt.find("v2"), "v2")),
                        int(_text(xt.find("v3"), "v3")),
                    )
                except ValueError as exc:
                    raise AmfError("bad triangle index in object %d: %s" % (oid, exc))
                for iv in tri:
                    if iv < 0 or iv >= len(verts):
                        raise AmfError(
                            "triangle index %d out of range 0..%d in object %d"
                            % (iv, len(verts) - 1, oid)
                        )
                tris.append(tri)
            volumes.append(tris)
        if not volumes:
            raise AmfError("object %d has no <volume>" % oid)

        obj_meta = {
            m.get("type", ""): (m.text or "").strip() for m in xobj.findall("metadata")
        }
        objects.append(AmfObject(oid, verts, volumes, obj_meta))

    if not objects:
        raise AmfError("AMF document has no <object>")
    return objects, unit, metadata


def read_amf(path: str) -> Tuple[List[AmfObject], str, Dict[str, str]]:
    with open(path, "rb") as fh:
        return loads(fh.read())


# --------------------------------------------------------------------------
# lumps
# --------------------------------------------------------------------------


def to_polyhedra(objects: Sequence[AmfObject]) -> List[Polyhedron]:
    """Every volume of every object as its own polyhedron, in document order."""
    out: List[Polyhedron] = []
    for obj in objects:
        out.extend(obj.polyhedra())
    return out


def merge_lumps(polyhedra: Sequence[Polyhedron]) -> Polyhedron:
    """Fuse several lumps into one polyhedron (AngelCAD ``amf_io::merge_lumps``).

    Vertex arrays are concatenated and face indices offset; no welding is done,
    so the result is a disjoint union -- which is precisely what a multi-lump AMF
    part means.
    """
    if not polyhedra:
        raise AmfError("merge_lumps needs at least one polyhedron")
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, ...]] = []
    for p in polyhedra:
        off = len(verts)
        verts.extend(p.vertices)
        faces.extend(tuple(i + off for i in f) for f in p.faces)
    return Polyhedron(verts, faces)
