"""3MF (3D Manufacturing Format) reader and writer.

3MF is the modern additive-manufacturing exchange format (3MF Consortium, also
ISO/IEC via the OPC container): the thing a 3D printer's slicer actually wants.
Where STL is a unitless triangle soup, a 3MF part is an *indexed* mesh that
carries **units** and **colour**, packaged as an Open Packaging Convention (OPC)
ZIP -- exactly the two facts STL throws away.

A 3MF file is a ZIP archive (the OPC package) with at least three parts::

    [Content_Types].xml     -- MIME types for the .rels and .model extensions
    _rels/.rels             -- the package relationship pointing at the model
    3D/3dmodel.model        -- the mesh, as XML

and the model XML looks like::

    <model unit="millimeter" xml:lang="en-US"
           xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
      <resources>
        <basematerials id="2">
          <base name="colour" displaycolor="#RRGGBBAA"/>
        </basematerials>
        <object id="1" type="model" pid="2" pindex="0">
          <mesh>
            <vertices>
              <vertex x="0" y="0" z="0"/>
              ...
            </vertices>
            <triangles>
              <triangle v1="0" v2="1" v3="2"/>
              ...
            </triangles>
          </mesh>
        </object>
      </resources>
      <build>
        <item objectid="1"/>
      </build>
    </model>

This module adds:

* :func:`dumps_model` / :func:`loads_model` -- the model XML alone, deterministic
  byte output (so the format's units/colour/geometry can be tested without a ZIP);
* :func:`write_3mf` / :func:`read_3mf` -- the whole OPC ZIP, with **fixed
  timestamps** so the bytes are reproducible, and a reader that follows the
  package relationship to find the model part;
* units validated against the 3MF-legal set (note ``foot``/``centimeter``, which
  AMF spells differently), and an optional ``#RRGGBB``/``#RRGGBBAA`` object colour
  that round-trips through a ``<basematerials>`` resource.

Pure stdlib (``xml.etree`` + ``zipfile``), deterministic.
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]

__all__ = [
    "ThreeMFError",
    "UNITS",
    "dumps_model",
    "loads_model",
    "write_3mf",
    "read_3mf",
    "serialize",
]

#: 3MF-legal length units (3MF Core spec, sec. 3.2). Note the spellings differ
#: from AMF's ("foot" vs AMF "feet"; "centimeter" is 3MF-only here).
UNITS = ("micron", "millimeter", "centimeter", "inch", "foot", "meter")

_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_MODEL_PART = "3D/3dmodel.model"
_MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_ZIP_MAGIC = b"PK\x03\x04"
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package'
    '.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package'
    '.3dmanufacturing-3dmodel+xml"/>'
    "</Types>\n"
)

_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rel0" Target="/%s" Type="%s"/>'
    "</Relationships>\n"
) % (_MODEL_PART, _MODEL_REL)


class ThreeMFError(Exception):
    """Malformed 3MF document."""


def _num(v: float) -> str:
    return "%.12g" % float(v)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _validate_color(color: Optional[str]) -> Optional[str]:
    if color is None:
        return None
    c = color.strip()
    if not c.startswith("#") or len(c) not in (7, 9):
        raise ThreeMFError(
            "colour must be #RRGGBB or #RRGGBBAA, got %r" % color)
    try:
        int(c[1:], 16)
    except ValueError:
        raise ThreeMFError("colour has non-hex digits: %r" % color) from None
    return c.upper()


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

def _triangulate(faces: Sequence[Sequence[int]]) -> List[Tuple[int, int, int]]:
    tris: List[Tuple[int, int, int]] = []
    for face in faces:
        ids = [int(i) for i in face]
        if len(ids) < 3:
            raise ThreeMFError("a face needs at least 3 vertices, got %d" % len(ids))
        for k in range(1, len(ids) - 1):
            tris.append((ids[0], ids[k], ids[k + 1]))
    return tris


def dumps_model(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: str = "millimeter",
    color: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Serialise a mesh to a 3MF model-part XML string (deterministic)."""
    if unit not in UNITS:
        raise ThreeMFError(
            "unit must be one of %s, got %r" % (", ".join(UNITS), unit))
    color = _validate_color(color)
    tris = _triangulate(faces)
    for tri in tris:
        for iv in tri:
            if iv < 0 or iv >= len(vertices):
                raise ThreeMFError(
                    "triangle index %d out of range 0..%d" % (iv, len(vertices) - 1))

    lines: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append(
        '<model unit="%s" xml:lang="en-US" xmlns="%s">' % (unit, _CORE_NS))
    if name is not None:
        lines.append(
            '  <metadata name="Title">%s</metadata>' % _xml_escape(name))
    lines.append("  <resources>")
    obj_attrs = 'id="1" type="model"'
    if color is not None:
        lines.append('    <basematerials id="2">')
        lines.append(
            '      <base name="colour" displaycolor="%s"/>' % color)
        lines.append("    </basematerials>")
        obj_attrs += ' pid="2" pindex="0"'
    lines.append('    <object %s>' % obj_attrs)
    lines.append("      <mesh>")
    lines.append("        <vertices>")
    for v in vertices:
        if len(v) != 3:
            raise ThreeMFError("each vertex must have 3 coordinates")
        lines.append('          <vertex x="%s" y="%s" z="%s"/>'
                     % (_num(v[0]), _num(v[1]), _num(v[2])))
    lines.append("        </vertices>")
    lines.append("        <triangles>")
    for tri in tris:
        lines.append('          <triangle v1="%d" v2="%d" v3="%d"/>'
                     % (tri[0], tri[1], tri[2]))
    lines.append("        </triangles>")
    lines.append("      </mesh>")
    lines.append("    </object>")
    lines.append("  </resources>")
    lines.append("  <build>")
    lines.append('    <item objectid="1"/>')
    lines.append("  </build>")
    lines.append("</model>")
    return "\n".join(lines) + "\n"


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def serialize(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: str = "millimeter",
    color: Optional[str] = None,
    name: Optional[str] = None,
) -> bytes:
    """Serialise a mesh to the full 3MF OPC ZIP bytes (reproducible)."""
    model = dumps_model(vertices, faces, unit=unit, color=color, name=name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for part_name, text in (
            ("[Content_Types].xml", _CONTENT_TYPES),
            ("_rels/.rels", _RELS),
            (_MODEL_PART, model),
        ):
            info = zipfile.ZipInfo(part_name, date_time=_ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, text.encode("utf-8"))
    return buf.getvalue()


def write_3mf(
    path: str,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: str = "millimeter",
    color: Optional[str] = None,
    name: Optional[str] = None,
) -> bytes:
    """Write a 3MF file. Returns the ZIP bytes written."""
    data = serialize(vertices, faces, unit=unit, color=color, name=name)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

def loads_model(text) -> Tuple[List[Vec3], List[Tuple[int, int, int]], str, Optional[str]]:
    """Parse a 3MF model-part XML into ``(vertices, triangles, unit, color)``."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ThreeMFError("invalid XML: %s" % exc)
    if _localname(root.tag) != "model":
        raise ThreeMFError("root element must be <model>, got <%s>" % root.tag)

    unit = root.get("unit", "millimeter")
    if unit not in UNITS:
        raise ThreeMFError("illegal 3MF unit %r" % unit)

    resources = _find(root, "resources")
    if resources is None:
        raise ThreeMFError("<model> has no <resources>")

    # Colour: first base of the first basematerials resource, if any.
    color: Optional[str] = None
    for res in resources:
        if _localname(res.tag) == "basematerials":
            for base in res:
                if _localname(base.tag) == "base":
                    color = base.get("displaycolor")
                    break
            if color is not None:
                break

    obj = None
    for res in resources:
        if _localname(res.tag) == "object":
            obj = res
            break
    if obj is None:
        raise ThreeMFError("<resources> has no <object>")
    mesh = _find(obj, "mesh")
    if mesh is None:
        raise ThreeMFError("object has no <mesh>")

    xverts = _find(mesh, "vertices")
    xtris = _find(mesh, "triangles")
    if xverts is None:
        raise ThreeMFError("mesh has no <vertices>")
    if xtris is None:
        raise ThreeMFError("mesh has no <triangles>")

    verts: List[Vec3] = []
    for xv in xverts:
        if _localname(xv.tag) != "vertex":
            continue
        try:
            verts.append((float(xv.get("x")), float(xv.get("y")),
                          float(xv.get("z"))))
        except (TypeError, ValueError) as exc:
            raise ThreeMFError("bad vertex coordinate: %s" % exc)

    tris: List[Tuple[int, int, int]] = []
    for xt in xtris:
        if _localname(xt.tag) != "triangle":
            continue
        try:
            tri = (int(xt.get("v1")), int(xt.get("v2")), int(xt.get("v3")))
        except (TypeError, ValueError) as exc:
            raise ThreeMFError("bad triangle index: %s" % exc)
        for iv in tri:
            if iv < 0 or iv >= len(verts):
                raise ThreeMFError(
                    "triangle index %d out of range 0..%d" % (iv, len(verts) - 1))
        tris.append(tri)
    return verts, tris, unit, color


def _find(elem: ET.Element, local: str) -> Optional[ET.Element]:
    for child in elem:
        if _localname(child.tag) == local:
            return child
    return None


def _model_part_name(data: bytes) -> str:
    """Find the model part by following the package .rels; fall back to default."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            if "_rels/.rels" in names:
                rels = ET.fromstring(zf.read("_rels/.rels"))
                for rel in rels:
                    if rel.get("Type") == _MODEL_REL:
                        target = rel.get("Target", "")
                        return target.lstrip("/")
            if _MODEL_PART in names:
                return _MODEL_PART
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise ThreeMFError("invalid 3MF package: %s" % exc)
    raise ThreeMFError("3MF package has no model part")


def read_3mf(path: str) -> Tuple[List[Vec3], List[Tuple[int, int, int]], str, Optional[str]]:
    """Read a 3MF file into ``(vertices, triangles, unit, color)``."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(_ZIP_MAGIC):
        raise ThreeMFError("not a 3MF package (no ZIP magic)")
    part = _model_part_name(data)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            model = zf.read(part)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ThreeMFError("cannot read model part %r: %s" % (part, exc))
    return loads_model(model)
