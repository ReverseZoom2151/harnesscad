"""t2cdean_glb_writer -- binary glTF (.glb) writer for triangle meshes.

The PrintX / Text-to-CAD (dean) app ends its pipeline by converting the STL that
OpenSCAD produced into a GLB (``trimesh.load(...).export(file_type='glb')``) and
base64-embedding it into a ``<model-viewer>`` tag, because GLB is the one mesh
container that browsers render natively.  That last hop is the only part of the
app that is not an LLM call or a UI widget -- and it is the part the harness
lacks entirely (no glTF/GLB anywhere).  This module reimplements it in stdlib.

GLB 2.0 container layout (all little-endian):

    header : magic 'glTF' | uint32 version=2 | uint32 total length
    chunk  : uint32 chunkLength | uint32 chunkType | chunkData
             JSON chunk (type 0x4E4F534A) padded with spaces (0x20)
             BIN  chunk (type 0x004E4942) padded with zeros  (0x00)

Both chunks must be 4-byte aligned, and every ``bufferView`` offset must respect
the component alignment -- getting this wrong is the classic reason a viewer
silently shows nothing, so the padding is computed explicitly and asserted.

Beyond the container, the useful deterministic work is **vertex welding**: an
STL is a triangle soup with each vertex repeated per facet, while glTF wants an
indexed mesh.  ``weld_vertices`` deduplicates by quantised position (a stable
first-seen ordering, no dict-iteration nondeterminism) and emits an index buffer,
typically shrinking the vertex count ~6x for a cube.

glTF also *requires* accessor ``min``/``max`` on the POSITION accessor -- viewers
use it to frame the camera, and omitting it is invalid per spec -- so the bounds
are computed from the welded vertices.

The output is byte-for-byte deterministic for a given triangle list, which means
a GLB can be hashed as a build artefact / regression fixture.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Dict, Iterable, List, Sequence, Tuple

from harnesscad.io.formats.t2cdean_stl_codec import Triangle, face_normal

Vec3 = Tuple[float, float, float]

GLB_MAGIC = 0x46546C67  # 'glTF'
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A  # 'JSON'
CHUNK_BIN = 0x004E4942  # 'BIN\0'

# glTF component types.
COMPONENT_UNSIGNED_INT = 5125
COMPONENT_FLOAT = 5126

# glTF buffer view targets.
TARGET_ARRAY_BUFFER = 34962  # vertex attributes
TARGET_ELEMENT_ARRAY_BUFFER = 34963  # indices

MODE_TRIANGLES = 4

# Positions are welded on a quantised grid; 1e-6 is well below any printable
# tolerance yet coarse enough to merge float32 STL round-trip noise.
DEFAULT_WELD_TOLERANCE = 1e-6


class GlbError(ValueError):
    """Raised when a mesh cannot be encoded as GLB."""


def _pad_to_4(length: int) -> int:
    """Bytes needed to round ``length`` up to a 4-byte boundary."""
    return (4 - (length % 4)) % 4


def weld_vertices(
    triangles: Sequence[Triangle], tolerance: float = DEFAULT_WELD_TOLERANCE
) -> Tuple[List[Vec3], List[int]]:
    """Deduplicate the triangle soup into ``(vertices, indices)``.

    Vertices are keyed on their position quantised to ``tolerance``; the first
    occurrence wins and defines the output order, so the result is deterministic
    and independent of hash seeding.
    """
    if tolerance <= 0.0:
        raise GlbError("weld tolerance must be positive")
    inv = 1.0 / tolerance
    seen: Dict[Tuple[int, int, int], int] = {}
    vertices: List[Vec3] = []
    indices: List[int] = []
    for tri in triangles:
        for v in tri.vertices:
            key = (
                int(round(v[0] * inv)),
                int(round(v[1] * inv)),
                int(round(v[2] * inv)),
            )
            idx = seen.get(key)
            if idx is None:
                idx = len(vertices)
                seen[key] = idx
                vertices.append(v)
            indices.append(idx)
    return vertices, indices


def vertex_normals(vertices: Sequence[Vec3], indices: Sequence[int]) -> List[Vec3]:
    """Area-weighted smooth normals for the welded mesh.

    Accumulating the *unnormalised* cross product weights each face by twice its
    area, which is the standard way to keep large faces from being outvoted by
    slivers.  Isolated vertices get ``(0, 0, 1)`` so the buffer never contains
    NaN (a NaN normal makes model-viewer drop the primitive).
    """
    acc = [[0.0, 0.0, 0.0] for _ in vertices]
    for i in range(0, len(indices), 3):
        a, b, c = indices[i], indices[i + 1], indices[i + 2]
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        e1 = (vb[0] - va[0], vb[1] - va[1], vb[2] - va[2])
        e2 = (vc[0] - va[0], vc[1] - va[1], vc[2] - va[2])
        n = (
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        )
        for idx in (a, b, c):
            acc[idx][0] += n[0]
            acc[idx][1] += n[1]
            acc[idx][2] += n[2]
    out: List[Vec3] = []
    for n in acc:
        ln = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) ** 0.5
        if ln == 0.0:
            out.append((0.0, 0.0, 1.0))
        else:
            out.append((n[0] / ln, n[1] / ln, n[2] / ln))
    return out


def _bounds(vertices: Sequence[Vec3]) -> Tuple[List[float], List[float]]:
    lo = [float(vertices[0][k]) for k in range(3)]
    hi = list(lo)
    for v in vertices:
        for k in range(3):
            if v[k] < lo[k]:
                lo[k] = float(v[k])
            if v[k] > hi[k]:
                hi[k] = float(v[k])
    return lo, hi


def build_gltf_json(
    vertices: Sequence[Vec3],
    indices: Sequence[int],
    normals: Sequence[Vec3] | None,
    bin_length: int,
    name: str = "mesh",
) -> dict:
    """Assemble the glTF JSON document describing the BIN chunk layout."""
    pos_bytes = len(vertices) * 12
    nrm_bytes = len(normals) * 12 if normals else 0
    idx_bytes = len(indices) * 4
    lo, hi = _bounds(vertices)

    buffer_views = [
        {
            "buffer": 0,
            "byteOffset": 0,
            "byteLength": pos_bytes,
            "target": TARGET_ARRAY_BUFFER,
        }
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": COMPONENT_FLOAT,
            "count": len(vertices),
            "type": "VEC3",
            # Required by spec for POSITION: viewers frame the camera with it.
            "min": lo,
            "max": hi,
        }
    ]
    attributes = {"POSITION": 0}
    offset = pos_bytes

    if normals:
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": nrm_bytes,
                "target": TARGET_ARRAY_BUFFER,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": COMPONENT_FLOAT,
                "count": len(normals),
                "type": "VEC3",
            }
        )
        attributes["NORMAL"] = len(accessors) - 1
        offset += nrm_bytes

    buffer_views.append(
        {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": idx_bytes,
            "target": TARGET_ELEMENT_ARRAY_BUFFER,
        }
    )
    accessors.append(
        {
            "bufferView": len(buffer_views) - 1,
            "componentType": COMPONENT_UNSIGNED_INT,
            "count": len(indices),
            "type": "SCALAR",
        }
    )
    index_accessor = len(accessors) - 1

    return {
        "asset": {"version": "2.0", "generator": "harnesscad t2cdean_glb_writer"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": index_accessor,
                        "mode": MODE_TRIANGLES,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "default",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": bin_length}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }


def write_glb(
    triangles: Iterable[Triangle],
    name: str = "mesh",
    smooth_normals: bool = True,
    weld_tolerance: float = DEFAULT_WELD_TOLERANCE,
) -> bytes:
    """Encode triangles as a binary glTF (.glb) file.

    Deterministic: the same triangle list always yields the same bytes.
    """
    tris = list(triangles)
    if not tris:
        raise GlbError("cannot write a GLB with no triangles")

    vertices, indices = weld_vertices(tris, weld_tolerance)
    normals = vertex_normals(vertices, indices) if smooth_normals else None

    # --- BIN chunk: positions | normals | indices, each 4-byte aligned. ---
    payload = bytearray()
    for v in vertices:
        payload += struct.pack("<3f", float(v[0]), float(v[1]), float(v[2]))
    if normals:
        for n in normals:
            payload += struct.pack("<3f", float(n[0]), float(n[1]), float(n[2]))
    for i in indices:
        payload += struct.pack("<I", i)
    # Every element is 4 bytes wide, so the sections are already aligned.
    bin_length = len(payload)
    payload += b"\x00" * _pad_to_4(bin_length)

    doc = build_gltf_json(vertices, indices, normals, bin_length, name=name)
    # sort_keys + compact separators => byte-stable JSON chunk.
    json_bytes = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * _pad_to_4(len(json_bytes))  # JSON pads with spaces

    total = 12 + 8 + len(json_bytes) + 8 + len(payload)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, GLB_VERSION, total)
    out += struct.pack("<II", len(json_bytes), CHUNK_JSON)
    out += json_bytes
    out += struct.pack("<II", len(payload), CHUNK_BIN)
    out += payload
    if len(out) != total:  # pragma: no cover - internal invariant
        raise GlbError("GLB length mismatch: %d != %d" % (len(out), total))
    return bytes(out)


def parse_glb(data: bytes) -> Tuple[dict, bytes]:
    """Inverse of :func:`write_glb`: return ``(gltf_json, bin_chunk)``.

    Provided so a produced GLB can be validated (and round-trip tested) without
    a third-party parser.
    """
    if len(data) < 12:
        raise GlbError("GLB truncated")
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GlbError("not a GLB (bad magic)")
    if version != GLB_VERSION:
        raise GlbError("unsupported GLB version %d" % version)
    if total != len(data):
        raise GlbError("GLB header length %d != actual %d" % (total, len(data)))
    offset = 12
    doc: dict | None = None
    binary = b""
    while offset < total:
        if offset + 8 > total:
            raise GlbError("GLB chunk header truncated")
        clen, ctype = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + clen > total:
            raise GlbError("GLB chunk data truncated")
        chunk = data[offset : offset + clen]
        offset += clen
        if ctype == CHUNK_JSON:
            doc = json.loads(chunk.decode("utf-8"))
        elif ctype == CHUNK_BIN:
            binary = chunk
        # Unknown chunk types are skipped, per spec.
    if doc is None:
        raise GlbError("GLB has no JSON chunk")
    return doc, binary


def stl_to_glb(stl_bytes: bytes, name: str = "mesh") -> bytes:
    """The PrintX hop, in stdlib: STL bytes (binary or ASCII) -> GLB bytes."""
    from harnesscad.io.formats.t2cdean_stl_codec import parse_stl

    return write_glb(parse_stl(stl_bytes), name=name)


def glb_data_uri(glb_bytes: bytes) -> str:
    """Base64 ``data:`` URI for a ``<model-viewer src=...>`` embed."""
    encoded = base64.b64encode(glb_bytes).decode("ascii")
    return "data:model/gltf-binary;base64," + encoded


def triangles_from_glb(data: bytes) -> List[Triangle]:
    """Decode a GLB written by :func:`write_glb` back into a triangle soup."""
    doc, binary = parse_glb(data)
    mesh = doc["meshes"][0]["primitives"][0]
    accessors = doc["accessors"]
    views = doc["bufferViews"]

    def read_vec3(accessor_index: int) -> List[Vec3]:
        acc = accessors[accessor_index]
        view = views[acc["bufferView"]]
        start = view["byteOffset"]
        return [
            struct.unpack_from("<3f", binary, start + 12 * i)
            for i in range(acc["count"])
        ]

    positions = read_vec3(mesh["attributes"]["POSITION"])
    acc = accessors[mesh["indices"]]
    view = views[acc["bufferView"]]
    start = view["byteOffset"]
    idx = [
        struct.unpack_from("<I", binary, start + 4 * i)[0] for i in range(acc["count"])
    ]
    tris: List[Triangle] = []
    for i in range(0, len(idx), 3):
        a, b, c = positions[idx[i]], positions[idx[i + 1]], positions[idx[i + 2]]
        tris.append(Triangle(a, b, c, normal=face_normal(a, b, c)))
    return tris
