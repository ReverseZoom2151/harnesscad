"""OFF (Object File Format) reader and writer.

OFF is the trivial indexed-mesh interchange format read by nearly every mesh tool
(Geomview, MeshLab, CGAL, libigl, Open3D): a one-word magic line, a counts line,
then the vertices and the polygon faces. Where STL is a unitless triangle soup and
PLY a header-heavy container, OFF is the smallest honest indexed mesh -- shared
vertices, arbitrary-arity faces -- which is exactly why it is the lingua franca of
computational-geometry libraries.

An OFF file is::

    OFF
    <nverts> <nfaces> <nedges>
    x y z            (nverts vertex lines)
    ...
    k i0 i1 ... ik-1 (nfaces face lines: a vertex count then that many indices)
    ...

AXIS + UNITS -- stated, not assumed.

    * OFF has NO axis convention and NO unit of its own: the numbers are raw
      coordinates. This codec writes and reads them in the harness's native frame:
      **right-handed, +Z up, millimetre**. No silent rotation is ever applied (the
      glTF Y-up trap); coordinates go out exactly as given and come back exactly as
      stored, so an OFF round trip is geometry-exact.
    * Because OFF cannot carry a unit, the mesh unit is persisted in a leading
      ``# unit <name>`` comment -- a comment other tools ignore but that
      :func:`parse_off` reads back, so the unit survives export -> import where a
      bare STL would silently drop it. ``AXIS`` and ``UNIT`` are asserted on write.

This is a *container* codec, not a geometry kernel:

* :func:`parse_off` auto-handles the ``OFF`` / ``COFF`` / ``NOFF`` magic variants
  (colour/normal columns are skipped) and returns ``(vertices, faces, unit)`` with
  faces kept as arbitrary-arity index tuples.
* :func:`dumps_off` / :func:`serialize_off` / :func:`write_off` serialise
  ``(vertices, faces)`` back out; ASCII output is byte-deterministic (fixed float
  formatting), matching the STL/PLY codecs' spelling.

Pure stdlib, deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]

__all__ = [
    "OffError",
    "AXIS",
    "UNITS",
    "parse_off",
    "dumps_off",
    "serialize_off",
    "write_off",
]

#: The harness's native coordinate frame. OFF carries no axis, so this is the
#: frame the numbers are written in and read back in -- never silently rotated.
AXIS = "right-handed,+Z-up"

#: Units this codec will persist in the ``# unit`` comment. OFF has no native
#: unit; anything outside this set on write is a caller error.
UNITS = ("micron", "millimeter", "centimeter", "inch", "foot", "meter")

_MAGICS = ("OFF", "COFF", "NOFF", "CNOFF", "4OFF")


class OffError(ValueError):
    """Raised when a byte string / text is not a well-formed OFF."""


def _fmt(value: float) -> str:
    """Deterministic float formatting, matching the STL/PLY codecs' spelling."""
    text = "%.6f" % float(value)
    if text == "-0.000000":
        text = "0.000000"
    return text


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

def _check_faces(vertices: Sequence[Sequence[float]],
                 faces: Sequence[Sequence[int]]) -> List[Face]:
    out: List[Face] = []
    for face in faces:
        ids = tuple(int(i) for i in face)
        if len(ids) < 3:
            raise OffError("a face needs at least 3 vertices, got %d" % len(ids))
        for iv in ids:
            if iv < 0 or iv >= len(vertices):
                raise OffError(
                    "face index %d out of range 0..%d" % (iv, len(vertices) - 1))
        out.append(ids)
    return out


def dumps_off(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
    axis: str = AXIS,
) -> str:
    """Serialise a mesh to ASCII OFF text (deterministic).

    ``unit`` is persisted in a ``# unit`` comment (OFF has no native unit). The
    axis convention is asserted and recorded as a comment; coordinates are written
    unchanged in that frame.
    """
    if unit is not None and unit not in UNITS:
        raise OffError(
            "unit must be one of %s (or None), got %r" % (", ".join(UNITS), unit))
    assert axis == AXIS, (
        "OFF codec only emits the harness native frame %r; refusing a silent "
        "reframe to %r" % (AXIS, axis))
    checked = _check_faces(vertices, faces)

    lines: List[str] = ["OFF"]
    if unit is not None:
        lines.append("# unit %s" % unit)
    lines.append("# axis %s" % axis)
    lines.append("%d %d 0" % (len(vertices), len(checked)))
    for v in vertices:
        if len(v) != 3:
            raise OffError("each vertex must have 3 coordinates")
        lines.append("%s %s %s" % (_fmt(v[0]), _fmt(v[1]), _fmt(v[2])))
    for face in checked:
        lines.append("%d %s" % (len(face), " ".join(str(i) for i in face)))
    return "\n".join(lines) + "\n"


def serialize_off(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
    axis: str = AXIS,
) -> bytes:
    """Serialise a mesh to OFF bytes (UTF-8 of :func:`dumps_off`)."""
    return dumps_off(vertices, faces, unit=unit, axis=axis).encode("utf-8")


def write_off(
    path: str,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: Optional[str] = "millimeter",
    axis: str = AXIS,
) -> bytes:
    """Write an OFF file. Returns the bytes written."""
    data = serialize_off(vertices, faces, unit=unit, axis=axis)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

def _tokens_and_unit(text: str) -> Tuple[List[str], str, Optional[str]]:
    """Strip comments (recording a ``# unit`` line) and return (tokens, magic, unit)."""
    unit: Optional[str] = None
    kept: List[str] = []
    magic: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line[1:].split()
            if len(parts) >= 2 and parts[0] == "unit":
                unit = parts[1]
            continue
        # A trailing comment on a data line is legal too.
        if "#" in line:
            code, comment = line.split("#", 1)
            cparts = comment.split()
            if len(cparts) >= 2 and cparts[0] == "unit":
                unit = cparts[1]
            line = code.strip()
            if not line:
                continue
        if magic is None:
            # The first non-comment token(s): the magic, possibly glued to counts.
            head = line.split()
            token = head[0]
            if token not in _MAGICS:
                raise OffError("not an OFF file (magic was %r)" % token)
            magic = token
            kept.extend(head[1:])
            continue
        kept.extend(line.split())
    if magic is None:
        raise OffError("empty OFF file (no magic line)")
    return kept, magic, unit


def parse_off(data) -> Tuple[List[Vec3], List[Face], Optional[str]]:
    """Parse OFF bytes/text into ``(vertices, faces, unit)``.

    ``COFF``/``NOFF`` colour and normal columns are tolerated and skipped; faces
    keep their native arity (fan-triangulate downstream if triangles are needed).
    ``unit`` is the value of a ``# unit <name>`` comment if present, else ``None``.
    """
    if isinstance(data, (bytes, bytearray)):
        text = bytes(data).decode("utf-8", errors="strict")
    elif isinstance(data, str):
        text = data
    else:
        raise OffError("parse_off expects bytes or str")

    tokens, magic, unit = _tokens_and_unit(text)
    # COFF adds 3-4 colour floats per vertex; NOFF adds a 3-vector normal.
    has_normal = "N" in magic
    has_color = "C" in magic
    dim = 4 if magic.startswith("4") else 3

    it = iter(tokens)

    def take_int(what: str) -> int:
        try:
            return int(next(it))
        except StopIteration:
            raise OffError("truncated OFF: expected %s" % what) from None
        except ValueError as exc:
            raise OffError("bad integer in OFF (%s): %s" % (what, exc)) from None

    def take_float(what: str) -> float:
        try:
            return float(next(it))
        except StopIteration:
            raise OffError("truncated OFF: expected %s" % what) from None
        except ValueError as exc:
            raise OffError("bad float in OFF (%s): %s" % (what, exc)) from None

    nverts = take_int("vertex count")
    nfaces = take_int("face count")
    take_int("edge count")  # nedges: informational, often 0

    verts: List[Vec3] = []
    for _ in range(nverts):
        coords = [take_float("vertex coordinate") for _ in range(dim)]
        if has_normal:
            for _ in range(3):
                take_float("vertex normal")
        if has_color:
            # Colour is 3 or 4 columns; OFF does not say which, so we cannot
            # reliably skip it without ambiguity. Reject rather than guess.
            raise OffError(
                "COFF per-vertex colour columns are ambiguous to skip; this "
                "reader handles OFF/NOFF geometry only")
        verts.append((coords[0], coords[1], coords[2]))

    faces: List[Face] = []
    for _ in range(nfaces):
        k = take_int("face vertex count")
        if k < 3:
            raise OffError("face with %d vertices (need >= 3)" % k)
        idx = tuple(take_int("face index") for _ in range(k))
        for iv in idx:
            if iv < 0 or iv >= nverts:
                raise OffError(
                    "face index %d out of range 0..%d" % (iv, nverts - 1))
        faces.append(idx)
    return verts, faces, unit
