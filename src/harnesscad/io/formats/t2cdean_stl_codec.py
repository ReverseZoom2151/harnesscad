"""t2cdean_stl_codec -- stdlib STL reader/writer (binary + ASCII).

Mined from the PrintX / Text-to-CAD (dean) pipeline, whose text-to-CAD loop ends
in ``OpenSCAD -> .stl -> trimesh -> .glb``.  The repository leans on ``trimesh``
for every mesh hop; the harness has no mesh-file layer at all (``ingest``'s
``import_brep`` only reaches STL through OCCT), so the STL container itself is
reimplemented here in pure stdlib.

The module is a *container* codec, not a geometry kernel:

* ``parse_stl(data)`` auto-detects binary vs ASCII and returns triangles.
* ``write_binary_stl`` / ``write_ascii_stl`` round-trip them back out.

Binary STL layout (little-endian):

    80-byte header | uint32 triangle count | N * 50-byte records
    record := 12 floats (normal, v0, v1, v2) + uint16 attribute byte count

Format detection is the classic trap: an ASCII file starts with the token
``solid``, but so do plenty of binary files whose 80-byte header was filled in
by an exporter that wrote ``solid ...``.  The reliable discriminator is the
*length* check -- a binary file is exactly ``84 + 50 * n`` bytes -- so that is
tried first, and the ASCII path is only taken when the size cannot be explained
by the declared triangle count.

Normals are recomputed (right-hand rule over the vertex winding) when the stored
normal is absent or degenerate (zero-length), which is what most exporters emit;
callers therefore always receive a consistent orientation.  Everything is
deterministic: floats are formatted with a fixed repr, and triangle order is
preserved exactly.
"""

from __future__ import annotations

import math
import struct
from typing import Iterable, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]

# Binary STL: 80-byte header + uint32 count, then 50 bytes per facet.
BINARY_HEADER_SIZE = 80
BINARY_COUNT_SIZE = 4
BINARY_FACET_SIZE = 50

_ZERO: Vec3 = (0.0, 0.0, 0.0)


class StlError(ValueError):
    """Raised when a byte string / text is not a well-formed STL."""


class Triangle:
    """One STL facet: three vertices plus an outward normal."""

    __slots__ = ("v0", "v1", "v2", "normal")

    def __init__(
        self,
        v0: Sequence[float],
        v1: Sequence[float],
        v2: Sequence[float],
        normal: Sequence[float] | None = None,
    ) -> None:
        self.v0 = _vec(v0)
        self.v1 = _vec(v1)
        self.v2 = _vec(v2)
        stored = _vec(normal) if normal is not None else _ZERO
        # A stored normal of (0,0,0) is the exporter's "compute it yourself".
        self.normal = stored if _length(stored) > 0.0 else face_normal(
            self.v0, self.v1, self.v2
        )

    @property
    def vertices(self) -> Tuple[Vec3, Vec3, Vec3]:
        return (self.v0, self.v1, self.v2)

    def area(self) -> float:
        """Triangle area via half the cross-product magnitude."""
        return 0.5 * _length(_cross(_sub(self.v1, self.v0), _sub(self.v2, self.v0)))

    def is_degenerate(self, tol: float = 1e-12) -> bool:
        return self.area() <= tol

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Triangle):
            return NotImplemented
        return (
            self.v0 == other.v0
            and self.v1 == other.v1
            and self.v2 == other.v2
            and self.normal == other.normal
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "Triangle(%r, %r, %r)" % (self.v0, self.v1, self.v2)


def _vec(values: Sequence[float]) -> Vec3:
    if len(values) != 3:
        raise StlError("expected a 3-component vector, got %d" % len(values))
    return (float(values[0]), float(values[1]), float(values[2]))


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def face_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3:
    """Unit normal of the triangle by the right-hand rule; ``(0,0,0)`` if
    degenerate."""
    n = _cross(_sub(v1, v0), _sub(v2, v0))
    ln = _length(n)
    if ln == 0.0:
        return _ZERO
    return (n[0] / ln, n[1] / ln, n[2] / ln)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def is_binary_stl(data: bytes) -> bool:
    """True when ``data`` is a binary STL.

    Length is the discriminator: a binary payload is exactly
    ``84 + 50 * count`` bytes with ``count`` read from the header.  The leading
    ``solid`` token is *not* trusted, because binary exporters routinely write
    it into the 80-byte header.
    """
    if len(data) < BINARY_HEADER_SIZE + BINARY_COUNT_SIZE:
        return False
    (count,) = struct.unpack_from("<I", data, BINARY_HEADER_SIZE)
    expected = BINARY_HEADER_SIZE + BINARY_COUNT_SIZE + BINARY_FACET_SIZE * count
    if len(data) == expected:
        return True
    # A zero-facet binary file is ambiguous with an empty ASCII one; prefer
    # ASCII whenever the text plausibly parses.
    return False


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def parse_binary_stl(data: bytes) -> List[Triangle]:
    if len(data) < BINARY_HEADER_SIZE + BINARY_COUNT_SIZE:
        raise StlError("binary STL truncated: %d bytes" % len(data))
    (count,) = struct.unpack_from("<I", data, BINARY_HEADER_SIZE)
    need = BINARY_HEADER_SIZE + BINARY_COUNT_SIZE + BINARY_FACET_SIZE * count
    if len(data) < need:
        raise StlError(
            "binary STL declares %d facets (%d bytes) but has %d"
            % (count, need, len(data))
        )
    tris: List[Triangle] = []
    offset = BINARY_HEADER_SIZE + BINARY_COUNT_SIZE
    for _ in range(count):
        vals = struct.unpack_from("<12fH", data, offset)
        offset += BINARY_FACET_SIZE
        tris.append(
            Triangle(vals[3:6], vals[6:9], vals[9:12], normal=vals[0:3])
        )
    return tris


def parse_ascii_stl(text: str) -> List[Triangle]:
    tokens = text.split()
    tris: List[Triangle] = []
    i = 0
    n = len(tokens)
    if not tokens or tokens[0] != "solid":
        raise StlError("ASCII STL must begin with 'solid'")
    while i < n:
        tok = tokens[i]
        if tok == "facet":
            normal: Sequence[float] | None = None
            if i + 4 < n and tokens[i + 1] == "normal":
                normal = _floats(tokens[i + 2 : i + 5])
                i += 5
            else:
                i += 1
            verts: List[Vec3] = []
            while i < n and tokens[i] != "endfacet":
                if tokens[i] == "vertex":
                    if i + 3 >= n:
                        raise StlError("truncated vertex in ASCII STL")
                    verts.append(_vec(_floats(tokens[i + 1 : i + 4])))
                    i += 4
                else:
                    i += 1
            if len(verts) != 3:
                raise StlError(
                    "facet must have exactly 3 vertices, got %d" % len(verts)
                )
            tris.append(Triangle(verts[0], verts[1], verts[2], normal=normal))
        else:
            i += 1
    return tris


def _floats(tokens: Sequence[str]) -> List[float]:
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:  # pragma: no cover - message passthrough
        raise StlError("bad float in ASCII STL: %s" % exc) from exc


def parse_stl(data: bytes) -> List[Triangle]:
    """Parse binary or ASCII STL bytes into triangles (order preserved)."""
    if not isinstance(data, (bytes, bytearray)):
        raise StlError("parse_stl expects bytes")
    data = bytes(data)
    if is_binary_stl(data):
        return parse_binary_stl(data)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StlError("not a binary STL and not valid UTF-8 text") from exc
    return parse_ascii_stl(text)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def write_binary_stl(triangles: Iterable[Triangle], header: bytes = b"") -> bytes:
    """Serialise triangles as a binary STL. Header is padded/truncated to 80B."""
    tris = list(triangles)
    out = bytearray()
    out += header[:BINARY_HEADER_SIZE].ljust(BINARY_HEADER_SIZE, b"\x00")
    out += struct.pack("<I", len(tris))
    for t in tris:
        out += struct.pack(
            "<12fH",
            t.normal[0],
            t.normal[1],
            t.normal[2],
            t.v0[0],
            t.v0[1],
            t.v0[2],
            t.v1[0],
            t.v1[1],
            t.v1[2],
            t.v2[0],
            t.v2[1],
            t.v2[2],
            0,
        )
    return bytes(out)


def _fmt(value: float) -> str:
    """Deterministic float formatting: fixed 6-decimal scientific-free form."""
    text = "%.6f" % value
    # Normalise the two spellings of zero so output is byte-stable.
    if text == "-0.000000":
        text = "0.000000"
    return text


def write_ascii_stl(triangles: Iterable[Triangle], name: str = "model") -> str:
    """Serialise triangles as ASCII STL (deterministic float formatting)."""
    lines: List[str] = ["solid %s" % name]
    for t in triangles:
        lines.append(
            "  facet normal %s %s %s"
            % (_fmt(t.normal[0]), _fmt(t.normal[1]), _fmt(t.normal[2]))
        )
        lines.append("    outer loop")
        for v in t.vertices:
            lines.append(
                "      vertex %s %s %s" % (_fmt(v[0]), _fmt(v[1]), _fmt(v[2]))
            )
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid %s" % name)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------


def bounding_box(triangles: Iterable[Triangle]) -> Tuple[Vec3, Vec3]:
    """Axis-aligned bounds ``(min, max)`` over all vertices."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    empty = True
    for t in triangles:
        for v in t.vertices:
            empty = False
            for k in range(3):
                if v[k] < lo[k]:
                    lo[k] = v[k]
                if v[k] > hi[k]:
                    hi[k] = v[k]
    if empty:
        raise StlError("bounding_box of an empty mesh")
    return ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))


def signed_volume(triangles: Iterable[Triangle]) -> float:
    """Signed volume via the divergence theorem (sum of tetra determinants /6).

    Positive for an outward-oriented closed mesh; a printability gate can use
    ``abs()`` and compare against the expected part volume.
    """
    total = 0.0
    for t in triangles:
        a, b, c = t.vertices
        total += (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
    return total / 6.0


def surface_area(triangles: Iterable[Triangle]) -> float:
    return sum(t.area() for t in triangles)
