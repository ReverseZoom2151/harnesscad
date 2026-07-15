"""3MF extension parts: materials-and-colour, and beam-lattice.

The base :mod:`harnesscad.io.formats.threemf` codec writes a *core* 3MF: an
indexed, unit-bearing mesh in the OPC ZIP, with at most one object-wide
``displaycolor`` via a core ``<basematerials>`` resource. The 3MF Consortium
layers richer capability on top as **extensions**, each a namespaced vocabulary
inside the same model part:

* the **Materials and Properties extension** (namespace ``.../material/2015/02``)
  adds a ``<m:colorgroup>`` of per-vertex/per-triangle colours, so a mesh can be
  painted face by face rather than tinted as a whole;
* the **Beam Lattice extension** (namespace ``.../beamlattice/2017/02``) adds a
  ``<beamlattice>`` inside ``<mesh>``: a graph of capsule ``<beam>`` struts with
  per-end radii, the way lightweight lattices are actually described for additive
  manufacturing (a strut graph, not a tessellated solid).

This module is an *extension*, not a rewrite: it reuses the base codec's OPC
scaffolding (``[Content_Types].xml``, ``_rels/.rels``, the model part name, the
fixed ZIP timestamps and the unit set) and only adds the extension XML. The base
threemf.py is untouched.

AXIS + UNITS. 3MF's build space is **+Z up, right-handed** (slicers treat +Z as
the build/up direction), and the unit is carried EXPLICITLY on ``<model unit=>``
from the 3MF-legal set (``millimeter`` default). Both are asserted on write, the
same way the core codec does. Colours are ``#RRGGBB`` / ``#RRGGBBAA`` and are
validated.

Round trip: the **materials/colour** path writes a normal solid mesh plus a
colour group and reads geometry + unit + per-triangle colours back. The
**beam-lattice** path writes and reads a vertex + beam graph; a lattice is a strut
network, not a closed solid, so it is a specialised builder (see
:func:`write_3mf_beamlattice`) rather than the neutral-mesh write path.

Pure stdlib (``xml.etree`` + ``zipfile``), deterministic.
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Tuple

from harnesscad.io.formats import threemf as _core

Vec3 = Tuple[float, float, float]
Beam = Tuple[int, int, float, float]  # (v1, v2, r1, r2)

__all__ = [
    "ThreeMFExtensionError",
    "UNITS",
    "AXIS",
    "MATERIAL_NS",
    "BEAMLATTICE_NS",
    "dumps_material_model",
    "loads_material_model",
    "write_3mf_materials",
    "read_3mf_materials",
    "dumps_beamlattice_model",
    "loads_beamlattice_model",
    "write_3mf_beamlattice",
    "read_3mf_beamlattice",
]

#: Reuse the core codec's legal unit set and axis, so the extension cannot drift.
UNITS = _core.UNITS
AXIS = "right-handed,+Z-up"

MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
BEAMLATTICE_NS = "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02"

_CORE_NS = _core._CORE_NS
_MODEL_PART = _core._MODEL_PART
_ZIP_MAGIC = b"PK\x03\x04"


class ThreeMFExtensionError(Exception):
    """Malformed or unsupported 3MF extension document."""


def _num(v: float) -> str:
    return "%.12g" % float(v)


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find(elem: "ET.Element", local: str) -> "Optional[ET.Element]":
    for child in elem:
        if _localname(child.tag) == local:
            return child
    return None


def _check_unit(unit: str) -> None:
    if unit not in UNITS:
        raise ThreeMFExtensionError(
            "unit must be one of %s, got %r" % (", ".join(UNITS), unit))


def _opc_package(model_xml: str) -> bytes:
    """Wrap a model-part XML string in the reproducible OPC ZIP (reusing core parts)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for part_name, text in (
            ("[Content_Types].xml", _core._CONTENT_TYPES),
            ("_rels/.rels", _core._RELS),
            (_MODEL_PART, model_xml),
        ):
            info = zipfile.ZipInfo(part_name, date_time=_core._ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, text.encode("utf-8"))
    return buf.getvalue()


def _read_model_part(path: str) -> bytes:
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(_ZIP_MAGIC):
        raise ThreeMFExtensionError("not a 3MF package (no ZIP magic)")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            part = _MODEL_PART
            if part not in names:
                # follow the .rels like the core reader
                if "_rels/.rels" in names:
                    rels = ET.fromstring(zf.read("_rels/.rels"))
                    for rel in rels:
                        if rel.get("Type") == _core._MODEL_REL:
                            part = rel.get("Target", "").lstrip("/")
                            break
            return zf.read(part)
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ThreeMFExtensionError("cannot read model part: %s" % exc) from None


# --------------------------------------------------------------------------
# Materials & colour extension
# --------------------------------------------------------------------------

def dumps_material_model(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    colors: Sequence[str],
    unit: str = "millimeter",
    name: Optional[str] = None,
) -> str:
    """Serialise a solid mesh with **per-triangle** colours to a 3MF model XML.

    ``colors`` is one ``#RRGGBB`` / ``#RRGGBBAA`` per triangle (after fan
    triangulation of ``faces``). Colours are de-duplicated into a single
    ``<m:colorgroup>`` and each triangle references its entry via ``pid``/``p1``.
    """
    _check_unit(unit)
    tris = _core._triangulate(faces)
    if len(colors) != len(tris):
        raise ThreeMFExtensionError(
            "need one colour per triangle: %d colours for %d triangles"
            % (len(colors), len(tris)))
    for tri in tris:
        for iv in tri:
            if iv < 0 or iv >= len(vertices):
                raise ThreeMFExtensionError(
                    "triangle index %d out of range 0..%d"
                    % (iv, len(vertices) - 1))

    # Build the colour palette (stable, first-seen order).
    palette: List[str] = []
    index_of = {}
    for c in colors:
        validated = _core._validate_color(c)
        if validated not in index_of:
            index_of[validated] = len(palette)
            palette.append(validated)

    lines: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append(
        '<model unit="%s" xml:lang="en-US" xmlns="%s" xmlns:m="%s">'
        % (unit, _CORE_NS, MATERIAL_NS))
    if name is not None:
        lines.append('  <metadata name="Title">%s</metadata>'
                     % _core._xml_escape(name))
    lines.append("  <resources>")
    lines.append('    <m:colorgroup id="2">')
    for c in palette:
        lines.append('      <m:color color="%s"/>' % c)
    lines.append("    </m:colorgroup>")
    lines.append('    <object id="1" type="model" pid="2">')
    lines.append("      <mesh>")
    lines.append("        <vertices>")
    for v in vertices:
        if len(v) != 3:
            raise ThreeMFExtensionError("each vertex must have 3 coordinates")
        lines.append('          <vertex x="%s" y="%s" z="%s"/>'
                     % (_num(v[0]), _num(v[1]), _num(v[2])))
    lines.append("        </vertices>")
    lines.append("        <triangles>")
    for tri, col in zip(tris, colors):
        pidx = index_of[_core._validate_color(col)]
        lines.append(
            '          <triangle v1="%d" v2="%d" v3="%d" p1="%d"/>'
            % (tri[0], tri[1], tri[2], pidx))
    lines.append("        </triangles>")
    lines.append("      </mesh>")
    lines.append("    </object>")
    lines.append("  </resources>")
    lines.append("  <build>")
    lines.append('    <item objectid="1"/>')
    lines.append("  </build>")
    lines.append("</model>")
    return "\n".join(lines) + "\n"


def loads_material_model(text) -> Tuple[List[Vec3], List[Tuple[int, int, int]],
                                        str, List[str]]:
    """Parse a materials-extension model XML into ``(verts, tris, unit, colors)``.

    ``colors`` is per-triangle, resolved through the ``<m:colorgroup>`` the
    triangles reference. Triangles without a colour reference get ``""``.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ThreeMFExtensionError("invalid XML: %s" % exc) from None
    if _localname(root.tag) != "model":
        raise ThreeMFExtensionError("root must be <model>")
    unit = root.get("unit", "millimeter")
    _check_unit(unit)

    resources = _find(root, "resources")
    if resources is None:
        raise ThreeMFExtensionError("<model> has no <resources>")

    # colorgroups: id -> [colours]
    groups = {}
    for res in resources:
        if _localname(res.tag) == "colorgroup":
            gid = res.get("id")
            cols = [c.get("color") for c in res if _localname(c.tag) == "color"]
            groups[gid] = cols

    obj = _find(resources, "object")
    if obj is None:
        raise ThreeMFExtensionError("<resources> has no <object>")
    default_pid = obj.get("pid")
    mesh = _find(obj, "mesh")
    if mesh is None:
        raise ThreeMFExtensionError("object has no <mesh>")

    verts: List[Vec3] = []
    xv = _find(mesh, "vertices")
    for node in (xv if xv is not None else []):
        if _localname(node.tag) != "vertex":
            continue
        verts.append((float(node.get("x")), float(node.get("y")),
                      float(node.get("z"))))

    tris: List[Tuple[int, int, int]] = []
    colors: List[str] = []
    xt = _find(mesh, "triangles")
    for node in (xt if xt is not None else []):
        if _localname(node.tag) != "triangle":
            continue
        tri = (int(node.get("v1")), int(node.get("v2")), int(node.get("v3")))
        tris.append(tri)
        pid = node.get("pid", default_pid)
        p1 = node.get("p1")
        if pid in groups and p1 is not None and 0 <= int(p1) < len(groups[pid]):
            colors.append(groups[pid][int(p1)])
        else:
            colors.append("")
    return verts, tris, unit, colors


def write_3mf_materials(
    path: str,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    colors: Sequence[str],
    unit: str = "millimeter",
    name: Optional[str] = None,
) -> bytes:
    """Write a 3MF (materials extension, per-triangle colour). Returns ZIP bytes."""
    model = dumps_material_model(vertices, faces, colors, unit=unit, name=name)
    data = _opc_package(model)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


def read_3mf_materials(path: str) -> Tuple[List[Vec3], List[Tuple[int, int, int]],
                                           str, List[str]]:
    """Read a materials-extension 3MF into ``(verts, tris, unit, colors)``."""
    return loads_material_model(_read_model_part(path))


# --------------------------------------------------------------------------
# Beam-lattice extension
# --------------------------------------------------------------------------

def _norm_beams(beams: Sequence[Sequence[float]], default_radius: float,
                nverts: int) -> List[Beam]:
    out: List[Beam] = []
    for b in beams:
        b = tuple(b)
        if len(b) == 2:
            v1, v2 = int(b[0]), int(b[1])
            r1 = r2 = default_radius
        elif len(b) == 4:
            v1, v2 = int(b[0]), int(b[1])
            r1, r2 = float(b[2]), float(b[3])
        else:
            raise ThreeMFExtensionError(
                "a beam is (v1, v2) or (v1, v2, r1, r2), got %r" % (b,))
        for iv in (v1, v2):
            if iv < 0 or iv >= nverts:
                raise ThreeMFExtensionError(
                    "beam vertex %d out of range 0..%d" % (iv, nverts - 1))
        if v1 == v2:
            raise ThreeMFExtensionError("a beam needs two distinct vertices")
        out.append((v1, v2, r1, r2))
    return out


def dumps_beamlattice_model(
    vertices: Sequence[Sequence[float]],
    beams: Sequence[Sequence[float]],
    radius: float = 1.0,
    min_length: float = 0.0001,
    unit: str = "millimeter",
    name: Optional[str] = None,
) -> str:
    """Serialise a **beam lattice** (vertex graph + struts) to a 3MF model XML.

    ``beams`` are ``(v1, v2)`` or ``(v1, v2, r1, r2)`` into ``vertices``. ``radius``
    is the lattice default strut radius; ``min_length`` is the beamlattice minlength
    attribute. A lattice may carry no triangles (a pure strut network).
    """
    _check_unit(unit)
    if radius <= 0.0:
        raise ThreeMFExtensionError("lattice radius must be positive")
    struts = _norm_beams(beams, radius, len(vertices))

    lines: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append(
        '<model unit="%s" xml:lang="en-US" xmlns="%s" xmlns:b="%s">'
        % (unit, _CORE_NS, BEAMLATTICE_NS))
    if name is not None:
        lines.append('  <metadata name="Title">%s</metadata>'
                     % _core._xml_escape(name))
    lines.append("  <resources>")
    lines.append('    <object id="1" type="model">')
    lines.append("      <mesh>")
    lines.append("        <vertices>")
    for v in vertices:
        if len(v) != 3:
            raise ThreeMFExtensionError("each vertex must have 3 coordinates")
        lines.append('          <vertex x="%s" y="%s" z="%s"/>'
                     % (_num(v[0]), _num(v[1]), _num(v[2])))
    lines.append("        </vertices>")
    lines.append(
        '        <b:beamlattice radius="%s" minlength="%s" clippingmode="none">'
        % (_num(radius), _num(min_length)))
    lines.append("          <b:beams>")
    for (v1, v2, r1, r2) in struts:
        lines.append(
            '            <b:beam v1="%d" v2="%d" r1="%s" r2="%s"/>'
            % (v1, v2, _num(r1), _num(r2)))
    lines.append("          </b:beams>")
    lines.append("        </b:beamlattice>")
    lines.append("      </mesh>")
    lines.append("    </object>")
    lines.append("  </resources>")
    lines.append("  <build>")
    lines.append('    <item objectid="1"/>')
    lines.append("  </build>")
    lines.append("</model>")
    return "\n".join(lines) + "\n"


def loads_beamlattice_model(text) -> Tuple[List[Vec3], List[Beam], float, str]:
    """Parse a beam-lattice model XML into ``(verts, beams, radius, unit)``."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ThreeMFExtensionError("invalid XML: %s" % exc) from None
    if _localname(root.tag) != "model":
        raise ThreeMFExtensionError("root must be <model>")
    unit = root.get("unit", "millimeter")
    _check_unit(unit)

    resources = _find(root, "resources")
    if resources is None:
        raise ThreeMFExtensionError("<model> has no <resources>")
    obj = _find(resources, "object")
    if obj is None:
        raise ThreeMFExtensionError("<resources> has no <object>")
    mesh = _find(obj, "mesh")
    if mesh is None:
        raise ThreeMFExtensionError("object has no <mesh>")

    verts: List[Vec3] = []
    xv = _find(mesh, "vertices")
    for node in (xv if xv is not None else []):
        if _localname(node.tag) == "vertex":
            verts.append((float(node.get("x")), float(node.get("y")),
                          float(node.get("z"))))

    lattice = _find(mesh, "beamlattice")
    if lattice is None:
        raise ThreeMFExtensionError("mesh has no <beamlattice>")
    radius = float(lattice.get("radius", "1"))

    beams: List[Beam] = []
    xbeams = _find(lattice, "beams")
    for node in (xbeams if xbeams is not None else []):
        if _localname(node.tag) != "beam":
            continue
        v1 = int(node.get("v1"))
        v2 = int(node.get("v2"))
        r1 = float(node.get("r1", radius))
        r2 = float(node.get("r2", node.get("r1", radius)))
        beams.append((v1, v2, r1, r2))
    return verts, beams, radius, unit


def write_3mf_beamlattice(
    path: str,
    vertices: Sequence[Sequence[float]],
    beams: Sequence[Sequence[float]],
    radius: float = 1.0,
    min_length: float = 0.0001,
    unit: str = "millimeter",
    name: Optional[str] = None,
) -> bytes:
    """Write a 3MF beam-lattice file. Returns the ZIP bytes written."""
    model = dumps_beamlattice_model(vertices, beams, radius=radius,
                                    min_length=min_length, unit=unit, name=name)
    data = _opc_package(model)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


def read_3mf_beamlattice(path: str) -> Tuple[List[Vec3], List[Beam], float, str]:
    """Read a beam-lattice 3MF into ``(verts, beams, radius, unit)``."""
    return loads_beamlattice_model(_read_model_part(path))
