"""USD / USDZ (Pixar Universal Scene Description) mesh reader and writer.

USD is the modern 3D interchange standard -- the format Apple's AR Quick Look,
NVIDIA Omniverse, Blender, Houdini and Maya all speak, and the one glTF is now
measured against. ``.usda`` is the human-readable ASCII crate; ``.usdz`` is the
zipped, uncompressed AR delivery container (a plain ZIP whose members are stored,
never deflated, and 64-byte aligned so the runtime can mmap them in place).

This module is a *minimal* USD codec: it writes and reads a single ``UsdGeomMesh``
(shared points, ``faceVertexIndices`` + ``faceVertexCounts``) with the two stage
metadata that actually matter for a CAD hand-off:

    #usda 1.0
    (
        metersPerUnit = 0.001
        upAxis = "Z"
    )
    def Mesh "model"
    {
        int[] faceVertexCounts = [3, 3]
        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    }

AXIS + UNITS -- CARRIED EXPLICITLY, and asserted. This is the whole point.

    USD is one of the few mesh formats that states its frame IN THE FILE, so this
    codec refuses to repeat the glTF silent-rotation bug:

    * ``upAxis`` is written explicitly (``"Z"`` for the harness's native +Z-up,
      right-handed frame) and read back; only ``"Y"`` and ``"Z"`` are legal in USD
      and both are asserted. No coordinate is ever silently reinterpreted.
    * ``metersPerUnit`` is written explicitly from the mesh unit (millimetre ->
      0.001) and mapped back to a unit name on read. A stage with no
      ``metersPerUnit`` is UNITLESS by the USD default of 1 metre; this codec makes
      the choice visible rather than guessing.

* :func:`dumps_usda` / :func:`write_usda` -- the ASCII stage;
* :func:`serialize_usdz` / :func:`write_usdz` -- the aligned, stored ZIP container
  (fixed timestamps, so the bytes are reproducible);
* :func:`parse_usda` / :func:`read_usdz` -- read the mesh + upAxis + unit back.

Pure stdlib (``struct`` + ``zipfile``), deterministic.
"""

from __future__ import annotations

import io
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]

__all__ = [
    "UsdError",
    "UP_AXES",
    "UNIT_METERS",
    "parse_usda",
    "dumps_usda",
    "serialize_usda",
    "write_usda",
    "serialize_usdz",
    "write_usdz",
    "read_usdz",
]

#: USD stages are +Y up or +Z up only. The harness is +Z up, right-handed.
UP_AXES = ("Y", "Z")

#: unit name -> metres per unit, the value USD stores in ``metersPerUnit``.
UNIT_METERS: Dict[str, float] = {
    "micron": 1e-6,
    "millimeter": 1e-3,
    "centimeter": 1e-2,
    "inch": 0.0254,
    "foot": 0.3048,
    "meter": 1.0,
}

_USDZ_ALIGN = 64
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_ZIP_MAGIC = b"PK\x03\x04"


class UsdError(ValueError):
    """Raised when text/bytes are not a well-formed USD mesh this codec reads."""


def _fmt(value: float) -> str:
    """Deterministic float formatting, matching the STL/PLY codecs' spelling."""
    text = "%.6f" % float(value)
    if text == "-0.000000":
        text = "0.000000"
    return text


def _meters_for(unit: str) -> float:
    try:
        return UNIT_METERS[unit]
    except KeyError:
        raise UsdError(
            "unit must be one of %s, got %r" % (", ".join(UNIT_METERS), unit)
        ) from None


def _unit_for(meters: float) -> Optional[str]:
    """Nearest known unit name for a ``metersPerUnit`` value, else None."""
    for name, m in UNIT_METERS.items():
        if abs(m - meters) <= 1e-12 + 1e-9 * m:
            return name
    return None


def _sanitize_prim(name: str) -> str:
    """A legal USD prim name: alnum/underscore, not starting with a digit."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch == "_") else "_")
    s = "".join(out) or "model"
    if s[0].isdigit():
        s = "_" + s
    return s


# --------------------------------------------------------------------------
# .usda writer
# --------------------------------------------------------------------------

def dumps_usda(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: str = "millimeter",
    up_axis: str = "Z",
    name: str = "model",
) -> str:
    """Serialise a mesh to a ``.usda`` ASCII stage (deterministic).

    ``upAxis`` and ``metersPerUnit`` are written explicitly and asserted.
    """
    assert up_axis in UP_AXES, (
        "upAxis must be one of %s, got %r" % (", ".join(UP_AXES), up_axis))
    meters = _meters_for(unit)
    assert meters > 0.0, "metersPerUnit must be positive"

    counts: List[int] = []
    indices: List[int] = []
    for face in faces:
        ids = [int(i) for i in face]
        if len(ids) < 3:
            raise UsdError("a face needs at least 3 vertices, got %d" % len(ids))
        for iv in ids:
            if iv < 0 or iv >= len(vertices):
                raise UsdError(
                    "face index %d out of range 0..%d" % (iv, len(vertices) - 1))
        counts.append(len(ids))
        indices.extend(ids)

    prim = _sanitize_prim(name)
    pts: List[str] = []
    for v in vertices:
        if len(v) != 3:
            raise UsdError("each vertex must have 3 coordinates")
        pts.append("(%s, %s, %s)" % (_fmt(v[0]), _fmt(v[1]), _fmt(v[2])))

    lines: List[str] = [
        "#usda 1.0",
        "(",
        "    metersPerUnit = %s" % repr(float(meters)),
        '    upAxis = "%s"' % up_axis,
        ")",
        "",
        'def Mesh "%s"' % prim,
        "{",
        "    int[] faceVertexCounts = [%s]" % ", ".join(str(c) for c in counts),
        "    int[] faceVertexIndices = [%s]" % ", ".join(str(i) for i in indices),
        "    point3f[] points = [%s]" % ", ".join(pts),
        "}",
    ]
    return "\n".join(lines) + "\n"


def serialize_usda(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: str = "millimeter",
    up_axis: str = "Z",
    name: str = "model",
) -> bytes:
    """Serialise a mesh to ``.usda`` bytes (UTF-8 of :func:`dumps_usda`)."""
    return dumps_usda(vertices, faces, unit=unit, up_axis=up_axis,
                      name=name).encode("utf-8")


def write_usda(
    path: str,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: str = "millimeter",
    up_axis: str = "Z",
    name: str = "model",
) -> bytes:
    """Write a ``.usda`` file. Returns the bytes written."""
    data = serialize_usda(vertices, faces, unit=unit, up_axis=up_axis, name=name)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


# --------------------------------------------------------------------------
# .usdz container writer (uncompressed + 64-byte aligned, per the USDZ spec)
# --------------------------------------------------------------------------

def _add_aligned(zf: zipfile.ZipFile, name: str, data: bytes,
                 align: int = _USDZ_ALIGN) -> None:
    """Store ``data`` uncompressed with its file body aligned to ``align`` bytes.

    USDZ mandates ZIP_STORED and that each member's data begins on a 64-byte
    boundary (so the runtime can zero-copy it). We pad the local header's extra
    field to push the data offset onto the boundary.
    """
    info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o600 << 16
    header_offset = zf.fp.tell()
    # Local file header is 30 bytes + filename + extra; align the data start.
    base = header_offset + 30 + len(name.encode("utf-8"))
    pad = (-base) % align
    if pad:
        info.extra = b"\x00" * pad
    zf.writestr(info, data)


def serialize_usdz(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: str = "millimeter",
    up_axis: str = "Z",
    name: str = "model",
) -> bytes:
    """Serialise a mesh to ``.usdz`` container bytes (stored, aligned, reproducible).

    The default layer must be the first member of the archive (USDZ convention).
    """
    stage = serialize_usda(vertices, faces, unit=unit, up_axis=up_axis, name=name)
    prim = _sanitize_prim(name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        _add_aligned(zf, "%s.usda" % prim, stage)
    return buf.getvalue()


def write_usdz(
    path: str,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    unit: str = "millimeter",
    up_axis: str = "Z",
    name: str = "model",
) -> bytes:
    """Write a ``.usdz`` file. Returns the ZIP bytes written."""
    data = serialize_usdz(vertices, faces, unit=unit, up_axis=up_axis, name=name)
    with open(path, "wb") as fh:
        fh.write(data)
    return data


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------

def _scalar_after(text: str, key: str) -> Optional[str]:
    """The token following ``key =`` up to end-of-line, stripped."""
    pos = text.find(key)
    if pos < 0:
        return None
    eq = text.find("=", pos)
    if eq < 0:
        return None
    end = text.find("\n", eq)
    if end < 0:
        end = len(text)
    return text[eq + 1:end].strip()


def _bracket_after(text: str, key: str) -> Optional[str]:
    """The content of the ``[ ... ]`` following ``key``, or None."""
    pos = text.find(key)
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
    raise UsdError("unterminated '[' after %r" % key)


def _ints(block: str, key: str) -> List[int]:
    if block is None:
        raise UsdError("missing %s array" % key)
    out: List[int] = []
    for tok in block.replace(",", " ").split():
        try:
            out.append(int(tok))
        except ValueError as exc:
            raise UsdError("bad integer in %s: %s" % (key, exc)) from None
    return out


def parse_usda(data) -> Tuple[List[Vec3], List[Face], str, str]:
    """Parse a ``.usda`` stage into ``(vertices, faces, unit, up_axis)``.

    Reads the first ``def Mesh``'s ``points`` / ``faceVertexIndices`` /
    ``faceVertexCounts`` plus the stage ``upAxis`` and ``metersPerUnit``. The
    unit is mapped back from ``metersPerUnit`` (default 1 metre when absent).
    """
    if isinstance(data, (bytes, bytearray)):
        text = bytes(data).decode("utf-8", errors="strict")
    elif isinstance(data, str):
        text = data
    else:
        raise UsdError("parse_usda expects bytes or str")

    if "#usda" not in text[:64]:
        raise UsdError("not a .usda stage (missing '#usda' magic)")

    up_raw = _scalar_after(text, "upAxis")
    up_axis = up_raw.strip().strip('"') if up_raw else "Y"  # USD default is Y
    if up_axis not in UP_AXES:
        raise UsdError("illegal upAxis %r" % up_axis)

    mpu_raw = _scalar_after(text, "metersPerUnit")
    try:
        meters = float(mpu_raw) if mpu_raw is not None else 1.0
    except ValueError as exc:
        raise UsdError("bad metersPerUnit %r: %s" % (mpu_raw, exc)) from None
    unit = _unit_for(meters) or "meter"

    counts = _ints(_bracket_after(text, "faceVertexCounts"), "faceVertexCounts")
    indices = _ints(_bracket_after(text, "faceVertexIndices"), "faceVertexIndices")

    pts_block = _bracket_after(text, "points")
    if pts_block is None:
        raise UsdError("missing points array")
    nums: List[float] = []
    for tok in pts_block.replace("(", " ").replace(")", " ").replace(",", " ").split():
        try:
            nums.append(float(tok))
        except ValueError as exc:
            raise UsdError("bad coordinate in points: %s" % exc) from None
    if len(nums) % 3 != 0:
        raise UsdError("points array has %d floats, not a multiple of 3"
                       % len(nums))
    verts: List[Vec3] = [
        (nums[i], nums[i + 1], nums[i + 2]) for i in range(0, len(nums), 3)]

    faces: List[Face] = []
    cursor = 0
    for c in counts:
        if c < 3:
            raise UsdError("face with %d vertices (need >= 3)" % c)
        if cursor + c > len(indices):
            raise UsdError("faceVertexIndices too short for faceVertexCounts")
        face = tuple(indices[cursor:cursor + c])
        for iv in face:
            if iv < 0 or iv >= len(verts):
                raise UsdError(
                    "faceVertexIndex %d out of range 0..%d" % (iv, len(verts) - 1))
        faces.append(face)
        cursor += c
    return verts, faces, unit, up_axis


def read_usdz(path: str) -> Tuple[List[Vec3], List[Face], str, str]:
    """Read a ``.usdz`` container: parse its first ``.usd``/``.usda`` layer."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw.startswith(_ZIP_MAGIC):
        raise UsdError("not a .usdz container (no ZIP magic)")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            member = None
            for nm in zf.namelist():
                if nm.lower().endswith((".usda", ".usd", ".usdc")):
                    member = nm
                    break
            if member is None:
                raise UsdError(".usdz has no .usd/.usda/.usdc layer")
            if member.lower().endswith(".usdc"):
                raise UsdError(
                    "the default layer %r is binary crate (.usdc); this codec "
                    "reads the ASCII .usda flavour only" % member)
            payload = zf.read(member)
    except zipfile.BadZipFile as exc:
        raise UsdError("invalid .usdz package: %s" % exc) from None
    return parse_usda(payload)
