"""PLY (Polygon File Format / Stanford triangle format) reader and writer.

PLY is the lingua franca of mesh-processing tools (MeshLab, Open3D, CloudCompare,
the Stanford scanning repository): an indexed vertex/face container that every
one of them reads. The harness had STL (triangle soup, unitless), OBJ (indexed
text) and AMF/GLB, but no PLY, so a part could not be handed to the mesh-repair
and inspection tooling that speaks PLY natively.

A PLY file is a short ASCII header followed by a body that is either ASCII or
binary (little- or big-endian). The header declares *elements* (here ``vertex``
and ``face``) and their *properties*::

    ply
    format ascii 1.0
    comment unit millimeter
    element vertex 8
    property float x
    property float y
    property float z
    element face 12
    property list uchar int vertex_indices
    end_header
    0 0 0
    ...
    3 0 1 2
    ...

PLY has no native notion of units. To keep an HarnessCAD round trip lossless we
persist the mesh unit in a ``comment unit <name>`` line -- a legal PLY comment
that other tools ignore but that :func:`parse_ply` reads back, so the unit
survives export -> import where a bare STL would silently drop it.

This module is a *container* codec, not a geometry kernel:

* :func:`parse_ply` auto-detects ascii / binary_little_endian / binary_big_endian
  and returns ``(vertices, faces, unit)`` with faces triangulated (fan) so the
  caller always receives triangles.
* :func:`write_ply` / :func:`dumps_ascii` serialise ``(vertices, faces)`` back
  out; ASCII output is byte-deterministic (fixed float formatting) and binary
  output is deterministic by construction.

Pure stdlib (``struct``), deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import struct
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]

__all__ = [
    "PlyError",
    "parse_ply",
    "write_ply",
    "dumps_ascii",
    "serialize_ply",
]

_ASCII = "ascii"
_LE = "binary_little_endian"
_BE = "binary_big_endian"


class PlyError(ValueError):
    """Raised when a byte string is not a well-formed PLY."""


def _fmt(value: float) -> str:
    """Deterministic float formatting, matching the STL codec's spelling."""
    text = "%.6f" % float(value)
    if text == "-0.000000":
        text = "0.000000"
    return text


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

def _triangulate(faces: Sequence[Sequence[int]]) -> List[Tuple[int, int, int]]:
    tris: List[Tuple[int, int, int]] = []
    for face in faces:
        ids = [int(i) for i in face]
        if len(ids) < 3:
            raise PlyError("a face needs at least 3 vertices, got %d" % len(ids))
        for k in range(1, len(ids) - 1):
            tris.append((ids[0], ids[k], ids[k + 1]))
    return tris


def dumps_ascii(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
) -> str:
    """Serialise a mesh to ASCII PLY text (deterministic)."""
    tris = _triangulate(faces)
    for tri in tris:
        for iv in tri:
            if iv < 0 or iv >= len(vertices):
                raise PlyError(
                    "face index %d out of range 0..%d" % (iv, len(vertices) - 1))
    lines: List[str] = ["ply", "format ascii 1.0"]
    if unit is not None:
        lines.append("comment unit %s" % unit)
    lines.append("element vertex %d" % len(vertices))
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    lines.append("element face %d" % len(tris))
    lines.append("property list uchar int vertex_indices")
    lines.append("end_header")
    for v in vertices:
        if len(v) != 3:
            raise PlyError("each vertex must have 3 coordinates")
        lines.append("%s %s %s" % (_fmt(v[0]), _fmt(v[1]), _fmt(v[2])))
    for tri in tris:
        lines.append("3 %d %d %d" % (tri[0], tri[1], tri[2]))
    return "\n".join(lines) + "\n"


def serialize_ply(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
    binary: bool = False,
) -> bytes:
    """Serialise a mesh to PLY bytes; ``binary`` selects little-endian binary."""
    tris = _triangulate(faces)
    for tri in tris:
        for iv in tri:
            if iv < 0 or iv >= len(vertices):
                raise PlyError(
                    "face index %d out of range 0..%d" % (iv, len(vertices) - 1))
    if not binary:
        return dumps_ascii(vertices, faces, unit=unit).encode("utf-8")
    header: List[str] = ["ply", "format binary_little_endian 1.0"]
    if unit is not None:
        header.append("comment unit %s" % unit)
    header.append("element vertex %d" % len(vertices))
    header.append("property float x")
    header.append("property float y")
    header.append("property float z")
    header.append("element face %d" % len(tris))
    header.append("property list uchar int vertex_indices")
    header.append("end_header")
    out = bytearray(("\n".join(header) + "\n").encode("utf-8"))
    for v in vertices:
        out += struct.pack("<3f", float(v[0]), float(v[1]), float(v[2]))
    for tri in tris:
        out += struct.pack("<B3i", 3, int(tri[0]), int(tri[1]), int(tri[2]))
    return bytes(out)


def write_ply(
    path: str,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
    binary: bool = False,
) -> bytes:
    """Write a PLY file. Returns the bytes written."""
    data = serialize_ply(vertices, faces, unit=unit, binary=binary)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

class _Prop:
    __slots__ = ("name", "typ", "is_list", "count_typ")

    def __init__(self, name: str, typ: str, is_list: bool = False,
                 count_typ: str = "") -> None:
        self.name = name
        self.typ = typ
        self.is_list = is_list
        self.count_typ = count_typ


class _Element:
    __slots__ = ("name", "count", "props")

    def __init__(self, name: str, count: int) -> None:
        self.name = name
        self.count = count
        self.props: List[_Prop] = []


# PLY scalar type -> (struct code, byte size). PLY spells the same width several
# ways; both spellings map to one code.
_SCALAR = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


def _parse_header(data: bytes) -> Tuple[str, List[_Element], Optional[str], int]:
    """Return ``(fmt, elements, unit, body_offset)`` from a PLY header."""
    marker = b"end_header"
    pos = data.find(marker)
    if pos < 0 or not data.startswith(b"ply"):
        raise PlyError("not a PLY file (missing magic or end_header)")
    # The body begins right after the newline that follows end_header.
    nl = data.find(b"\n", pos)
    body_offset = (nl + 1) if nl >= 0 else len(data)
    header_text = data[:pos].decode("ascii", errors="strict")

    fmt: Optional[str] = None
    unit: Optional[str] = None
    elements: List[_Element] = []
    for raw in header_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "ply":
            continue
        if tag == "format":
            fmt = parts[1]
        elif tag == "comment":
            if len(parts) >= 3 and parts[1] == "unit":
                unit = parts[2]
        elif tag == "element":
            elements.append(_Element(parts[1], int(parts[2])))
        elif tag == "property":
            if not elements:
                raise PlyError("property before any element")
            if parts[1] == "list":
                elements[-1].props.append(
                    _Prop(parts[4], parts[3], is_list=True, count_typ=parts[2]))
            else:
                elements[-1].props.append(_Prop(parts[2], parts[1]))
    if fmt is None:
        raise PlyError("PLY header has no format line")
    if fmt not in (_ASCII, _LE, _BE):
        raise PlyError("unsupported PLY format %r" % fmt)
    return fmt, elements, unit, body_offset


def _vertex_axes(elem: _Element) -> Tuple[int, int, int]:
    idx = {p.name: i for i, p in enumerate(elem.props)}
    try:
        return idx["x"], idx["y"], idx["z"]
    except KeyError:
        raise PlyError("vertex element has no x/y/z properties") from None


def _face_prop_index(elem: _Element) -> int:
    for i, p in enumerate(elem.props):
        if p.is_list and p.name in ("vertex_indices", "vertex_index"):
            return i
    raise PlyError("face element has no vertex_indices list property")


def _parse_ascii_body(text: str, elements: List[_Element]
                      ) -> Tuple[List[Vec3], List[Face]]:
    tokens = text.split()
    cur = 0
    verts: List[Vec3] = []
    faces: List[Face] = []
    for elem in elements:
        if elem.name == "vertex":
            ax, ay, az = _vertex_axes(elem)
            width = len(elem.props)
            for _ in range(elem.count):
                row = tokens[cur:cur + width]
                if len(row) < width:
                    raise PlyError("truncated vertex data")
                cur += width
                verts.append((float(row[ax]), float(row[ay]), float(row[az])))
        elif elem.name == "face":
            fp = _face_prop_index(elem)
            for _ in range(elem.count):
                # Only the vertex_indices list is consumed; any leading/trailing
                # scalar properties on a face are rare -- we assume the list is
                # the whole face row, which holds for every mesh we emit and read.
                if cur >= len(tokens):
                    raise PlyError("truncated face data")
                n = int(tokens[cur]); cur += 1
                if cur + n > len(tokens):
                    raise PlyError("truncated face index list")
                idx = tuple(int(tokens[cur + k]) for k in range(n))
                cur += n
                faces.append(idx)
        else:
            # Skip an unknown element's rows (each row is its scalar props;
            # list props on unknown elements are not supported).
            width = len(elem.props)
            cur += elem.count * width
    return verts, faces


def _parse_binary_body(data: bytes, offset: int, elements: List[_Element],
                       little: bool) -> Tuple[List[Vec3], List[Face]]:
    end = "<" if little else ">"
    verts: List[Vec3] = []
    faces: List[Face] = []
    pos = offset
    for elem in elements:
        if elem.name == "vertex":
            ax, ay, az = _vertex_axes(elem)
            codes = [_SCALAR[p.typ][0] for p in elem.props]
            sizes = [_SCALAR[p.typ][1] for p in elem.props]
            fmt = end + "".join(codes)
            row_size = sum(sizes)
            for _ in range(elem.count):
                vals = struct.unpack_from(fmt, data, pos)
                pos += row_size
                verts.append((float(vals[ax]), float(vals[ay]), float(vals[az])))
        elif elem.name == "face":
            fp = _face_prop_index(elem)
            prop = elem.props[fp]
            ccode, csize = _SCALAR[prop.count_typ]
            icode, isize = _SCALAR[prop.typ]
            for _ in range(elem.count):
                (n,) = struct.unpack_from(end + ccode, data, pos)
                pos += csize
                vals = struct.unpack_from(end + icode * n, data, pos)
                pos += isize * n
                faces.append(tuple(int(x) for x in vals))
        else:
            raise PlyError("cannot skip unknown binary element %r" % elem.name)
    return verts, faces


def parse_ply(data: bytes) -> Tuple[List[Vec3], List[Tuple[int, int, int]], Optional[str]]:
    """Parse PLY bytes into ``(vertices, triangles, unit)``.

    Faces are fan-triangulated so triangles are always returned. ``unit`` is the
    value of a ``comment unit <name>`` line if present, else ``None``.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise PlyError("parse_ply expects bytes")
    data = bytes(data)
    fmt, elements, unit, offset = _parse_header(data)
    if fmt == _ASCII:
        verts, faces = _parse_ascii_body(
            data[offset:].decode("ascii", errors="strict"), elements)
    else:
        verts, faces = _parse_binary_body(data, offset, elements, little=(fmt == _LE))
    tris: List[Tuple[int, int, int]] = []
    for face in faces:
        ids = list(face)
        if len(ids) < 3:
            raise PlyError("degenerate face with %d vertices" % len(ids))
        for k in range(1, len(ids) - 1):
            tris.append((ids[0], ids[k], ids[k + 1]))
    return verts, tris, unit
