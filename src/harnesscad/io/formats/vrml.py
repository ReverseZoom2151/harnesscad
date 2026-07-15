"""VRML97 / WRL mesh reader and writer.

VRML (the Virtual Reality Modeling Language, ISO/IEC 14772, ``.wrl``) is the
legacy-but-still-common 3D interchange text format: CAD packages (SolidWorks,
Creo, FreeCAD), 3D printers and web viewers all import it, and it is the ancestor
of X3D. A VRML97 mesh is an ``IndexedFaceSet`` -- a shared ``Coordinate`` point
array plus a ``coordIndex`` list where ``-1`` terminates each face -- which maps
one-to-one onto the harness's indexed ``(vertices, faces)``.

    #VRML V2.0 utf8
    # unit millimeter
    # up-axis Z
    Shape {
      geometry IndexedFaceSet {
        coord Coordinate { point [ 0 0 0, 1 0 0, 0 1 0 ] }
        coordIndex [ 0 1 2 -1 ]
      }
    }

AXIS + UNITS -- stated, not assumed.

    * VRML's scene convention is **+Y up, right-handed** (a browser points its
      default camera down -Z with +Y up). The harness is **+Z up, millimetre**.
      This codec does NOT silently reinterpret one as the other (the glTF Y-up
      trap). By default it writes the coordinates unchanged in the harness +Z-up
      frame and records ``# up-axis Z`` explicitly, so a harness round trip is
      geometry-exact. Passing ``up_axis="Y"`` performs an EXPLICIT, declared
      Z-up -> Y-up rotation ``(x, y, z) -> (x, z, -y)`` for viewers that expect the
      VRML convention; the reader reverses whatever axis the file declares.
    * VRML has no unit; the mesh unit is persisted in a ``# unit`` comment that
      round-trips. ``up_axis`` and ``unit`` are asserted on write.

* :func:`parse_wrl` reads back an ``IndexedFaceSet`` written by this module (or a
  compatible one) into ``(vertices, faces, unit, up_axis)``.
* :func:`dumps_wrl` / :func:`serialize_wrl` / :func:`write_wrl` serialise a mesh;
  ASCII output is byte-deterministic (fixed float formatting).

Pure stdlib, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]

__all__ = [
    "VrmlError",
    "UP_AXES",
    "UNITS",
    "parse_wrl",
    "dumps_wrl",
    "serialize_wrl",
    "write_wrl",
]

#: Up-axis labels this codec understands. "Z" is the harness native frame (no
#: transform); "Y" is the VRML scene convention (an explicit, reversible rotation).
UP_AXES = ("Z", "Y")

UNITS = ("micron", "millimeter", "centimeter", "inch", "foot", "meter")

_HEADER = "#VRML V2.0 utf8"


class VrmlError(ValueError):
    """Raised when text is not a well-formed VRML IndexedFaceSet this codec reads."""


def _fmt(value: float) -> str:
    """Deterministic float formatting, matching the STL/PLY codecs' spelling."""
    text = "%.6f" % float(value)
    if text == "-0.000000":
        text = "0.000000"
    return text


def _to_axis(v: Vec3, up_axis: str) -> Vec3:
    """Map a harness (+Z up) coordinate into the target up-axis frame."""
    if up_axis == "Z":
        return (v[0], v[1], v[2])
    # Z-up -> Y-up: rotate -90 deg about X. (x, y, z) -> (x, z, -y).
    return (v[0], v[2], -v[1])


def _from_axis(v: Vec3, up_axis: str) -> Vec3:
    """Inverse of :func:`_to_axis`: bring a stored coordinate back to +Z up."""
    if up_axis == "Z":
        return (v[0], v[1], v[2])
    # Y-up -> Z-up: (x, y, z) -> (x, -z, y).
    return (v[0], -v[2], v[1])


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

def dumps_wrl(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
    up_axis: str = "Z",
    name: Optional[str] = None,
) -> str:
    """Serialise a mesh to VRML97 text (deterministic)."""
    if unit is not None and unit not in UNITS:
        raise VrmlError(
            "unit must be one of %s (or None), got %r" % (", ".join(UNITS), unit))
    assert up_axis in UP_AXES, (
        "up_axis must be one of %s, got %r" % (", ".join(UP_AXES), up_axis))

    faces_checked: List[Face] = []
    for face in faces:
        ids = tuple(int(i) for i in face)
        if len(ids) < 3:
            raise VrmlError("a face needs at least 3 vertices, got %d" % len(ids))
        for iv in ids:
            if iv < 0 or iv >= len(vertices):
                raise VrmlError(
                    "face index %d out of range 0..%d" % (iv, len(vertices) - 1))
        faces_checked.append(ids)

    lines: List[str] = [_HEADER]
    if unit is not None:
        lines.append("# unit %s" % unit)
    lines.append("# up-axis %s" % up_axis)
    if name is not None:
        lines.append("# name %s" % name.replace("\n", " "))
    lines.append("Shape {")
    lines.append("  geometry IndexedFaceSet {")
    lines.append("    coord Coordinate {")
    lines.append("      point [")
    for v in vertices:
        if len(v) != 3:
            raise VrmlError("each vertex must have 3 coordinates")
        x, y, z = _to_axis((float(v[0]), float(v[1]), float(v[2])), up_axis)
        lines.append("        %s %s %s," % (_fmt(x), _fmt(y), _fmt(z)))
    lines.append("      ]")
    lines.append("    }")
    lines.append("    coordIndex [")
    for face in faces_checked:
        lines.append("      %s, -1," % " ".join(str(i) for i in face))
    lines.append("    ]")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def serialize_wrl(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
    up_axis: str = "Z",
    name: Optional[str] = None,
) -> bytes:
    """Serialise a mesh to VRML bytes (UTF-8 of :func:`dumps_wrl`)."""
    return dumps_wrl(vertices, faces, unit=unit, up_axis=up_axis,
                     name=name).encode("utf-8")


def write_wrl(
    path: str,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
    up_axis: str = "Z",
    name: Optional[str] = None,
) -> bytes:
    """Write a VRML (.wrl) file. Returns the bytes written."""
    data = serialize_wrl(vertices, faces, unit=unit, up_axis=up_axis, name=name)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

def _bracket_block(text: str, start_key: str) -> Optional[str]:
    """The content of the ``[ ... ]`` that follows ``start_key``, or None."""
    pos = text.find(start_key)
    if pos < 0:
        return None
    lb = text.find("[", pos)
    if lb < 0:
        return None
    depth = 0
    for i in range(lb, len(text)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[lb + 1:i]
    raise VrmlError("unterminated '[' after %r" % start_key)


def _strip_comments(text: str) -> Tuple[str, Optional[str], str]:
    """Remove ``#`` line comments; return (code, unit, up_axis)."""
    unit: Optional[str] = None
    up_axis = "Z"
    out: List[str] = []
    for raw in text.splitlines():
        if "#" in raw:
            code, comment = raw.split("#", 1)
            cparts = comment.split()
            if len(cparts) >= 2 and cparts[0] == "unit":
                unit = cparts[1]
            elif len(cparts) >= 2 and cparts[0] in ("up-axis", "up_axis"):
                up_axis = cparts[1].upper()
            out.append(code)
        else:
            out.append(raw)
    return "\n".join(out), unit, up_axis


def parse_wrl(data) -> Tuple[List[Vec3], List[Face], Optional[str], str]:
    """Parse VRML text/bytes into ``(vertices, faces, unit, up_axis)``.

    Reads the first ``IndexedFaceSet``: its ``point`` array and ``coordIndex``
    list (``-1``-terminated faces). Coordinates are brought back to the harness
    +Z-up frame according to the declared ``up-axis`` comment. Handles the mesh
    this module writes and any compatible ``point``/``coordIndex`` block.
    """
    if isinstance(data, (bytes, bytearray)):
        text = bytes(data).decode("utf-8", errors="strict")
    elif isinstance(data, str):
        text = data
    else:
        raise VrmlError("parse_wrl expects bytes or str")

    code, unit, up_axis = _strip_comments(text)
    if up_axis not in UP_AXES:
        raise VrmlError("unknown up-axis %r" % up_axis)

    point_block = _bracket_block(code, "point")
    index_block = _bracket_block(code, "coordIndex")
    if point_block is None:
        raise VrmlError("no 'point [ ... ]' array found")
    if index_block is None:
        raise VrmlError("no 'coordIndex [ ... ]' array found")

    raw_pts = [tok for tok in point_block.replace(",", " ").split()]
    if len(raw_pts) % 3 != 0:
        raise VrmlError("point array length %d is not a multiple of 3"
                        % len(raw_pts))
    verts: List[Vec3] = []
    for i in range(0, len(raw_pts), 3):
        try:
            v = (float(raw_pts[i]), float(raw_pts[i + 1]), float(raw_pts[i + 2]))
        except ValueError as exc:
            raise VrmlError("bad coordinate in point array: %s" % exc) from None
        verts.append(_from_axis(v, up_axis))

    faces: List[Face] = []
    cur: List[int] = []
    for tok in index_block.replace(",", " ").split():
        try:
            n = int(tok)
        except ValueError as exc:
            raise VrmlError("bad index in coordIndex: %s" % exc) from None
        if n == -1:
            if cur:
                faces.append(tuple(cur))
                cur = []
        else:
            if n < 0 or n >= len(verts):
                raise VrmlError(
                    "coordIndex %d out of range 0..%d" % (n, len(verts) - 1))
            cur.append(n)
    if cur:  # a final face without a trailing -1 is legal
        faces.append(tuple(cur))
    return verts, faces, unit, up_axis
