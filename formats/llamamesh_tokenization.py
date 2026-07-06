"""LLaMA-Mesh mesh-as-text tokenization (Wang et al., 2024).

LLaMA-Mesh's deterministic contribution is representing a raw triangle mesh as
*plain OBJ text* that a language model reads and writes directly, with no
learned tokenizer and no vocabulary expansion. The pieces that make the text
short enough for an LLM context window are all deterministic arithmetic /
ordering, and this module reproduces exactly those pieces (the LLM fine-tuning
is external and not implemented here):

* **Coordinate quantization** (Sec. 3.1, Fig. 5). Floating-point vertex
  coordinates split into many BPE tokens. We uniformly scale the mesh into the
  range ``[0, bins]`` (``bins=64`` in the paper) and round each coordinate to
  the nearest integer, so every coordinate is a short integer. Scaling is
  uniform (single factor from the largest axis extent) so aspect ratio is
  preserved; the bounding box is retained so quantization can be inverted.

* **Canonical vertex / face ordering** (Sec. 4.1). Following PivotMesh the
  vertices are sorted by ``z``, then ``y``, then ``x`` (lowest to highest);
  faces are then reindexed and sorted by their lowest vertex index, breaking
  ties on the next-lowest, and so on. Each face is rotated so its smallest
  index comes first (winding preserved). This canonical order shortens and
  regularises the token sequence.

* **OBJ serialization + parsing round-trip** (Sec. 3.1, Fig. 4). ``v x y z``
  and ``f i j k`` lines (1-based indices) are emitted as text and parsed back.
  Quantize -> canonicalize -> serialize -> parse recovers the same integer
  mesh.

* **Token-length / compression metric** (Sec. 3.1). A digit-splitting token
  estimate (LLM tokenizers tend to emit one token per digit) lets us quantify
  how much the quantized integer OBJ compresses the float OBJ.

Pure stdlib, deterministic (no wall clock, no RNG).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

Vertex = Tuple[float, float, float]
Face = Tuple[int, ...]


# --------------------------------------------------------------------------- #
# Coordinate quantization
# --------------------------------------------------------------------------- #
def mesh_bounds(vertices: Sequence[Sequence[float]]) -> Tuple[Vertex, Vertex]:
    """Return axis-aligned ``(min, max)`` corners of ``vertices``."""
    if not vertices:
        raise ValueError("cannot bound an empty vertex list")
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    for vertex in vertices:
        if len(vertex) != 3:
            raise ValueError("each vertex must have 3 coordinates")
        for axis in range(3):
            value = float(vertex[axis])
            if value < lo[axis]:
                lo[axis] = value
            if value > hi[axis]:
                hi[axis] = value
    return tuple(lo), tuple(hi)


def quantize_vertices(
    vertices: Sequence[Sequence[float]], bins: int = 64
) -> Tuple[List[Tuple[int, int, int]], Dict[str, object]]:
    """Uniformly scale ``vertices`` into ``[0, bins]`` and round to integers.

    Returns the quantized integer vertices and a ``bbox`` dict (the bounding-box
    minimum and the uniform scale) sufficient to invert the mapping via
    :func:`dequantize_vertices`. A degenerate mesh (all points coincident) maps
    every vertex to the origin.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    lo, hi = mesh_bounds(vertices)
    extent = max(hi[axis] - lo[axis] for axis in range(3))
    scale = (bins / extent) if extent > 0 else 0.0
    quantized: List[Tuple[int, int, int]] = []
    for vertex in vertices:
        coords = []
        for axis in range(3):
            shifted = (float(vertex[axis]) - lo[axis]) * scale
            level = int(math.floor(shifted + 0.5))       # round-half-up
            if level < 0:
                level = 0
            elif level > bins:
                level = bins
            coords.append(level)
        quantized.append((coords[0], coords[1], coords[2]))
    return quantized, {"min": lo, "scale": scale, "bins": bins}


def dequantize_vertices(
    quantized: Sequence[Sequence[int]], bbox: Dict[str, object]
) -> List[Vertex]:
    """Invert :func:`quantize_vertices` back to (approximate) float coordinates."""
    lo = bbox["min"]  # type: ignore[index]
    scale = float(bbox["scale"])  # type: ignore[arg-type]
    out: List[Vertex] = []
    for vertex in quantized:
        if scale == 0.0:
            out.append((float(lo[0]), float(lo[1]), float(lo[2])))
        else:
            out.append(tuple(float(lo[axis]) + float(vertex[axis]) / scale
                             for axis in range(3)))
    return out


# --------------------------------------------------------------------------- #
# Canonical vertex / face ordering
# --------------------------------------------------------------------------- #
def _rotate_min_first(face: Sequence[int]) -> Face:
    """Cyclically rotate ``face`` so its smallest index leads (winding kept)."""
    if not face:
        raise ValueError("face has no vertices")
    pivot = min(range(len(face)), key=lambda i: face[i])
    return tuple(face[pivot:]) + tuple(face[:pivot])


def canonicalize_mesh(
    vertices: Sequence[Sequence[int]], faces: Sequence[Sequence[int]]
) -> Tuple[List[Tuple[int, ...]], List[Face]]:
    """Sort vertices z-y-x and faces by lowest-index, reindexing faces.

    ``faces`` use 0-based vertex indices. Returns ``(sorted_vertices,
    sorted_faces)`` where vertices are ordered by ``(z, y, x)`` ascending and
    faces are rotated (min index first) then sorted by their full ascending
    index key.
    """
    order = sorted(range(len(vertices)),
                   key=lambda i: (vertices[i][2], vertices[i][1], vertices[i][0]))
    remap = {old: new for new, old in enumerate(order)}
    sorted_vertices = [tuple(vertices[old]) for old in order]

    remapped: List[Face] = []
    for face in faces:
        if len(face) < 3:
            raise ValueError("a face needs at least 3 vertices")
        new_face = [remap[idx] for idx in face]
        remapped.append(_rotate_min_first(new_face))
    remapped.sort(key=lambda f: sorted(f))
    return sorted_vertices, remapped


# --------------------------------------------------------------------------- #
# OBJ serialization + parsing
# --------------------------------------------------------------------------- #
def serialize_obj(
    vertices: Sequence[Sequence[int]], faces: Sequence[Sequence[int]]
) -> str:
    """Serialize an integer mesh to OBJ text (``v``/``f`` lines, 1-based faces)."""
    lines: List[str] = []
    for vertex in vertices:
        lines.append("v " + " ".join(str(int(c)) for c in vertex))
    for face in faces:
        lines.append("f " + " ".join(str(int(idx) + 1) for idx in face))
    return "\n".join(lines) + "\n"


def serialize_obj_float(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    precision: int = 6,
) -> str:
    """Serialize a float mesh to OBJ text (baseline for the compression metric)."""
    lines: List[str] = []
    fmt = "{:." + str(precision) + "f}"
    for vertex in vertices:
        lines.append("v " + " ".join(fmt.format(float(c)) for c in vertex))
    for face in faces:
        lines.append("f " + " ".join(str(int(idx) + 1) for idx in face))
    return "\n".join(lines) + "\n"


def parse_obj(text: str) -> Tuple[List[Tuple[float, ...]], List[Face]]:
    """Parse OBJ ``v``/``f`` lines back into ``(vertices, faces)``.

    Vertex coordinates are returned as floats (integers parse cleanly too);
    faces are returned as 0-based index tuples. Face tokens of the ``i/j/k``
    OBJ form (vertex/texture/normal) keep only the vertex index.
    """
    vertices: List[Tuple[float, ...]] = []
    faces: List[Face] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v":
            if len(parts) < 4:
                raise ValueError("vertex line needs 3 coordinates: " + raw)
            vertices.append(tuple(float(p) for p in parts[1:4]))
        elif tag == "f":
            if len(parts) < 4:
                raise ValueError("face line needs at least 3 vertices: " + raw)
            idx = []
            for token in parts[1:]:
                first = token.split("/")[0]
                idx.append(int(first) - 1)
            faces.append(tuple(idx))
        # other OBJ tags (vn, vt, o, g, ...) are ignored
    return vertices, faces


def roundtrip(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    bins: int = 64,
) -> Tuple[List[Tuple[int, ...]], List[Face], str, Dict[str, object]]:
    """Full quantize -> canonicalize -> serialize -> parse cycle.

    Returns ``(parsed_int_vertices, parsed_faces, obj_text, bbox)``. Parsing the
    emitted text reproduces the canonical integer mesh exactly.
    """
    quantized, bbox = quantize_vertices(vertices, bins=bins)
    cverts, cfaces = canonicalize_mesh(quantized, faces)
    text = serialize_obj(cverts, cfaces)
    pverts, pfaces = parse_obj(text)
    int_verts = [tuple(int(round(c)) for c in v) for v in pverts]
    return int_verts, pfaces, text, bbox


# --------------------------------------------------------------------------- #
# Token-length / compression metric
# --------------------------------------------------------------------------- #
def estimate_token_count(text: str) -> int:
    """Digit-splitting token estimate for OBJ ``text``.

    Approximates an LLM BPE tokenizer where each digit and each sign is its own
    token, ``v``/``f`` markers are one token each, and every newline is a token.
    This is the quantity LLaMA-Mesh shrinks by using integers instead of long
    decimals.
    """
    tokens = 0
    for line in text.splitlines():
        for field in line.split():
            if field in ("v", "f"):
                tokens += 1
            else:
                for ch in field:
                    if ch.isdigit() or ch in "-+.":
                        tokens += 1
                    else:
                        tokens += 1
        tokens += 1  # newline
    return tokens


def compression_ratio(
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    bins: int = 64,
    precision: int = 6,
) -> float:
    """Ratio of float-OBJ tokens to quantized-OBJ tokens (``>1`` means shorter).

    Compares the token estimate of the raw float OBJ against the quantized
    integer OBJ for the same mesh, quantifying LLaMA-Mesh's sequence-length win.
    """
    float_text = serialize_obj_float(vertices, faces, precision=precision)
    quantized, _ = quantize_vertices(vertices, bins=bins)
    cverts, cfaces = canonicalize_mesh(quantized, faces)
    quant_text = serialize_obj(cverts, cfaces)
    quant_tokens = estimate_token_count(quant_text)
    if quant_tokens == 0:
        raise ValueError("quantized mesh produced no tokens")
    return estimate_token_count(float_text) / quant_tokens
