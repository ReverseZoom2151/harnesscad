"""Format registry -- the one I/O surface the harness exports and imports through.

The repo carries ~20 hand-written codecs under ``harnesscad.io.formats`` (STL,
GLB, OBJ, AMF, STEP, XCSG, SVG, ...). Each is correct and tested, each has its
own idiosyncratic public API (``parse_stl(bytes)``, ``write_amf(path, objects)``,
``get_svg(edges, opts)``, ``serialize(StepFile)``) and *nothing called any of
them*: a HarnessSession could not write a file and the CLI could not read one.

This module is the missing surface:

*   the codecs are **discovered** through the capability registry
    (:func:`harnesscad.registry.find` with ``tag="format"``), never a hardcoded
    list -- a new tagged codec module shows up here as soon as an adapter for it
    exists, and a codec that disappears simply stops being offered;
*   a :class:`FormatSpec` describes each one honestly: extensions, MIME type,
    ``kind`` (mesh / brep / csg / drawing), and whether it can genuinely *read*
    and/or *write*. The capability flags are **verified against the codec's own
    public symbols** (from the static AST index), so this module cannot claim a
    reader a codec does not have -- SVG is write-only and says so, DXF is a bare
    Protocol contract with no concrete serializer and says so;
*   thin **adapters** normalise every codec into the same ``read(path) -> obj`` /
    ``write(obj, path)`` shape. The codec modules themselves are untouched.

Usage::

    from harnesscad.io.formats import registry as fmt

    fmt.write(mesh, "part.stl")          # dispatches on the extension
    mesh = fmt.read("part.stl")
    fmt.supported(kind="mesh", mode="write")
    fmt.capability_matrix()              # the honest report

The neutral in-memory object for the mesh kinds is :class:`Mesh` (a triangle
soup plus a unit and a name); ``brep`` reads/writes a ``step.StepFile``, ``csg``
a ``typed_csg.Node``. :func:`write` coerces what it is given (a ``Mesh``, a list
of ``stl.Triangle``, a ``(vertices, faces)`` pair, a ``Polyhedron``, a
``HarnessSession`` or a raw backend) into whatever the target spec needs.

Everything here is stdlib-only and deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from harnesscad import registry as capabilities
from harnesscad.domain.geometry.mesh.polyhedron import Polyhedron
from harnesscad.io import gate
from harnesscad.domain.programs.ast.typed_csg import Node
from harnesscad.io.formats import amf as amf_codec
from harnesscad.io.formats import dxf as dxf_codec
from harnesscad.io.formats import glb as glb_codec
from harnesscad.io.formats import obj as obj_codec
from harnesscad.io.formats import ply as ply_codec
from harnesscad.io.formats import step as step_codec
from harnesscad.io.formats import stl as stl_codec
from harnesscad.io.formats import threemf as threemf_codec
from harnesscad.io.formats import svg as svg_codec
from harnesscad.io.formats import xcsg as xcsg_codec

__all__ = [
    "FormatError",
    "UnknownFormatError",
    "UnsupportedOperationError",
    "ExportError",
    "Mesh",
    "FormatSpec",
    "FORMAT_TAG",
    "specs",
    "spec_for_extension",
    "spec_for_path",
    "extensions",
    "supported",
    "capability_matrix",
    "format_report",
    "read",
    "write",
    "export_session",
    "to_mesh",
]

FORMAT_TAG = "format"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class FormatError(Exception):
    """Base class for every error raised by the I/O surface."""


class UnknownFormatError(FormatError):
    """The path's extension maps to no registered format."""


class UnsupportedOperationError(FormatError):
    """The format exists but cannot do what was asked (e.g. read a write-only codec)."""


class ExportError(FormatError):
    """A model/session could not be turned into something the target format accepts."""


# ---------------------------------------------------------------------------
# The neutral mesh object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mesh:
    """A triangle soup: the neutral currency of every ``kind="mesh"`` codec."""

    triangles: Tuple[stl_codec.Triangle, ...]
    name: str = "model"
    unit: str = "millimeter"

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    def vertices(self) -> List[Tuple[float, float, float]]:
        """Every triangle corner, in face order (duplicates kept)."""
        out: List[Tuple[float, float, float]] = []
        for t in self.triangles:
            out.extend((t.v0, t.v1, t.v2))
        return out

    def indexed(self) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """Weld exact-duplicate corners into ``(vertices, faces)``. Deterministic."""
        lookup: Dict[Tuple[float, float, float], int] = {}
        verts: List[Tuple[float, float, float]] = []
        faces: List[Tuple[int, int, int]] = []
        for t in self.triangles:
            idx: List[int] = []
            for v in (t.v0, t.v1, t.v2):
                key = (float(v[0]), float(v[1]), float(v[2]))
                if key not in lookup:
                    lookup[key] = len(verts)
                    verts.append(key)
                idx.append(lookup[key])
            faces.append((idx[0], idx[1], idx[2]))
        return verts, faces

    def to_polyhedron(self) -> Polyhedron:
        verts, faces = self.indexed()
        return Polyhedron(verts, faces)

    def edges(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """The unique undirected triangle edges (deterministic order)."""
        seen = set()
        out = []
        for t in self.triangles:
            for a, b in ((t.v0, t.v1), (t.v1, t.v2), (t.v2, t.v0)):
                key = (a, b) if a <= b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                out.append((a, b))
        return out

    @staticmethod
    def from_triangles(triangles: Iterable[stl_codec.Triangle], name: str = "model",
                       unit: str = "millimeter") -> "Mesh":
        return Mesh(tuple(triangles), name=name, unit=unit)

    @staticmethod
    def from_vertices_faces(vertices: Sequence[Sequence[float]],
                            faces: Sequence[Sequence[int]],
                            name: str = "model", unit: str = "millimeter") -> "Mesh":
        """Fan-triangulate index faces into a triangle soup."""
        tris: List[stl_codec.Triangle] = []
        for face in faces:
            ids = [int(i) for i in face]
            for k in range(1, len(ids) - 1):
                tris.append(stl_codec.Triangle(
                    vertices[ids[0]], vertices[ids[k]], vertices[ids[k + 1]]))
        return Mesh(tuple(tris), name=name, unit=unit)


def to_mesh(obj: Any, name: str = "model") -> Mesh:
    """Coerce anything mesh-shaped into a :class:`Mesh`.

    Accepts a ``Mesh``, a sequence of ``stl.Triangle``, a ``(vertices, faces)``
    pair, a ``Polyhedron``, raw STL bytes, or a HarnessSession / GeometryBackend
    (whose ``export("stl")`` output is parsed back into triangles).
    """
    if isinstance(obj, Mesh):
        return obj
    if isinstance(obj, Polyhedron):
        return Mesh.from_vertices_faces(obj.vertices, obj.faces, name=name)
    if isinstance(obj, (bytes, bytearray)):
        return Mesh(tuple(stl_codec.parse_stl(bytes(obj))), name=name)
    backend = _backend_of(obj)
    if backend is not None:
        return _mesh_from_backend(backend, name=name)
    if isinstance(obj, (list, tuple)) and obj and all(
            isinstance(t, stl_codec.Triangle) for t in obj):
        return Mesh(tuple(obj), name=name)
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        verts, faces = obj
        return Mesh.from_vertices_faces(verts, faces, name=name)
    if isinstance(obj, (list, tuple)) and not obj:
        return Mesh((), name=name)
    raise ExportError(f"cannot interpret {type(obj).__name__!r} as a mesh")


def _backend_of(obj: Any) -> Any:
    """The GeometryBackend behind a HarnessSession / backend, or None."""
    backend = getattr(obj, "backend", None)
    if backend is not None and hasattr(backend, "export"):
        return backend
    if hasattr(obj, "export") and hasattr(obj, "state_digest"):
        return obj
    return None


def _mesh_from_backend(backend: Any, name: str = "model") -> Mesh:
    try:
        payload = backend.export("stl")
    except Exception as exc:  # noqa: BLE001 - backend failures become ExportError
        raise ExportError(
            f"backend {type(backend).__name__!r} could not export STL: {exc}") from exc
    data = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    try:
        tris = stl_codec.parse_stl(data)
    except Exception as exc:  # noqa: BLE001
        raise ExportError(
            f"backend {type(backend).__name__!r} returned no usable mesh geometry "
            f"(its export('stl') is not a real STL): {exc}") from exc
    return Mesh(tuple(tris), name=name)


def _step_text(obj: Any) -> str:
    """Coerce anything into STEP part-21 text."""
    if isinstance(obj, step_codec.StepFile):
        return step_codec.serialize(obj)
    if isinstance(obj, str):
        return obj
    backend = _backend_of(obj)
    if backend is not None:
        try:
            payload = backend.export("step")
        except Exception as exc:  # noqa: BLE001
            raise ExportError(
                f"backend {type(backend).__name__!r} could not export STEP: {exc}"
            ) from exc
        return payload if isinstance(payload, str) else payload.decode("utf-8")
    raise ExportError(f"cannot interpret {type(obj).__name__!r} as STEP")


def _csg_node(obj: Any) -> Node:
    if isinstance(obj, Node):
        return obj
    raise ExportError(f"cannot interpret {type(obj).__name__!r} as a CSG tree")


# ---------------------------------------------------------------------------
# Adapters -- one per codec module. The codec modules are NOT modified; each
# adapter only reshapes arguments and return values.
# ---------------------------------------------------------------------------

def _read_stl(path: str, **_: Any) -> Mesh:
    with open(path, "rb") as fh:
        data = fh.read()
    return Mesh(tuple(stl_codec.parse_stl(data)), name=os.path.basename(path))


def _write_stl(obj: Any, path: str, ascii: bool = False, **_: Any) -> None:
    mesh = to_mesh(obj)
    if ascii:
        text = stl_codec.write_ascii_stl(mesh.triangles, name=mesh.name)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return
    data = stl_codec.write_binary_stl(mesh.triangles)
    with open(path, "wb") as fh:
        fh.write(data)


def _read_obj(path: str, **_: Any) -> Mesh:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    vertices, faces = obj_codec.parse_obj(text)
    return Mesh.from_vertices_faces(vertices, faces, name=os.path.basename(path))


def _write_obj(obj: Any, path: str, precision: int = 6, **_: Any) -> None:
    mesh = to_mesh(obj)
    vertices, faces = mesh.indexed()
    text = obj_codec.serialize_obj_float(vertices, faces, precision=precision)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _read_glb(path: str, **_: Any) -> Mesh:
    with open(path, "rb") as fh:
        data = fh.read()
    return Mesh(tuple(glb_codec.triangles_from_glb(data)), name=os.path.basename(path))


def _write_glb(obj: Any, path: str, smooth_normals: bool = True, **_: Any) -> None:
    mesh = to_mesh(obj)
    data = glb_codec.write_glb(mesh.triangles, name=mesh.name,
                               smooth_normals=smooth_normals)
    with open(path, "wb") as fh:
        fh.write(data)


def _read_amf(path: str, **_: Any) -> Mesh:
    objects, unit, _meta = amf_codec.read_amf(path)
    polyhedra = amf_codec.to_polyhedra(objects)
    tris: List[stl_codec.Triangle] = []
    for poly in polyhedra:
        for face in poly.faces:
            ids = [int(i) for i in face]
            for k in range(1, len(ids) - 1):
                tris.append(stl_codec.Triangle(
                    poly.vertices[ids[0]], poly.vertices[ids[k]],
                    poly.vertices[ids[k + 1]]))
    return Mesh(tuple(tris), name=os.path.basename(path), unit=unit)


def _write_amf(obj: Any, path: str, compress: bool = False, **_: Any) -> None:
    mesh = to_mesh(obj)
    objects = amf_codec.from_polyhedra([mesh.to_polyhedron()])
    amf_codec.write_amf(path, objects, unit=mesh.unit, compress=compress)


def _read_ply(path: str, **_: Any) -> Mesh:
    with open(path, "rb") as fh:
        data = fh.read()
    verts, tris, unit = ply_codec.parse_ply(data)
    return Mesh.from_vertices_faces(verts, tris, name=os.path.basename(path),
                                    unit=unit or "millimeter")


def _write_ply(obj: Any, path: str, binary: bool = False, **_: Any) -> None:
    mesh = to_mesh(obj)
    verts, faces = mesh.indexed()
    ply_codec.write_ply(path, verts, faces, unit=mesh.unit, binary=binary)


def _read_3mf(path: str, **_: Any) -> Mesh:
    verts, tris, unit, _color = threemf_codec.read_3mf(path)
    return Mesh.from_vertices_faces(verts, tris, name=os.path.basename(path),
                                    unit=unit)


def _write_3mf(obj: Any, path: str, color: Any = None, **_: Any) -> None:
    mesh = to_mesh(obj)
    verts, faces = mesh.indexed()
    unit = mesh.unit if mesh.unit in threemf_codec.UNITS else "millimeter"
    threemf_codec.write_3mf(path, verts, faces, unit=unit, color=color,
                            name=mesh.name)


def _read_step(path: str, **_: Any) -> step_codec.StepFile:
    with open(path, "r", encoding="utf-8") as fh:
        return step_codec.parse(fh.read())


def _write_step(obj: Any, path: str, **_: Any) -> None:
    text = _step_text(obj)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _read_xcsg(path: str, **_: Any) -> Node:
    return xcsg_codec.read_xcsg(path)


def _write_xcsg(obj: Any, path: str, **kwargs: Any) -> None:
    xcsg_codec.write_xcsg(path, _csg_node(obj), **kwargs)


def _write_png(obj: Any, path: str, **kwargs: Any) -> None:
    """PNG is write-only: a rendered raster of the model, not a model.

    Every other codec here carries geometry; this one carries a *picture* of it.
    :mod:`harnesscad.io.render` scan-converts the mesh with a z-buffer, shades it,
    draws its feature edges over the top and writes the PNG with zlib alone. The
    pixels cannot be turned back into a model, so there is no reader.
    """
    from harnesscad.io import render as render_route

    mesh = to_mesh(obj)
    verts, faces = mesh.indexed()
    if not faces:
        raise ExportError("nothing to render: the model has no faces")
    try:
        render_route.render((verts, faces), path, **kwargs)
    except render_route.RenderError as exc:
        raise ExportError(str(exc)) from exc


def _write_svg(obj: Any, path: str, opts: Optional[dict] = None,
               views: Any = None, **kwargs: Any) -> None:
    """SVG is write-only. Two routes, and the caller picks:

    *   default -- ``svg.get_svg``: one isometric wireframe of every mesh edge;
    *   ``views=...`` -- :func:`harnesscad.io.drawing.orthographic_drawing`: a
        real multi-view engineering drawing (first/third angle, feature edges,
        dashed hidden lines, overall dimensions, one sheet scale).

    ``views=True`` takes the standard three views; a sequence names them
    (``["front", "top"]``). The default route is unchanged.
    """
    mesh = to_mesh(obj)
    if views:
        from harnesscad.io import drawing as drawing_route

        names = ("front", "top", "side") if views is True else tuple(views)
        verts, faces = mesh.indexed()
        if not faces:
            raise ExportError("nothing to export: the model has no faces")
        try:
            text = drawing_route.orthographic_drawing((verts, faces), views=names,
                                                      **kwargs)
        except drawing_route.DrawingError as exc:
            raise ExportError(str(exc)) from exc
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return
    edges = [[a, b] for a, b in mesh.edges()]
    if not edges:
        raise ExportError("nothing to export: the model has no edges")
    text = svg_codec.get_svg(edges, opts=opts)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# ---------------------------------------------------------------------------
# The adapter table: dotted codec path -> how to drive it.
#
# `read_symbols` / `write_symbols` are the codec functions an adapter actually
# calls. A capability is only ever advertised when every symbol it needs is
# genuinely present in that codec's public API (checked against the AST index),
# so this table cannot over-claim.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Adapter:
    extensions: Tuple[str, ...]
    mime: str
    kind: str
    read_symbols: Tuple[str, ...] = ()
    write_symbols: Tuple[str, ...] = ()
    reader: Optional[Callable[..., Any]] = None
    writer: Optional[Callable[..., Any]] = None
    lossless: bool = False
    note: str = ""


_ADAPTERS: Dict[str, _Adapter] = {
    "harnesscad.io.formats.stl": _Adapter(
        extensions=(".stl",), mime="model/stl", kind="mesh",
        read_symbols=("parse_stl",), write_symbols=("write_binary_stl", "write_ascii_stl"),
        reader=_read_stl, writer=_write_stl, lossless=True,
        note="Binary by default (ascii=True for the text flavour). Triangle soup: "
             "no units, no vertex sharing.",
    ),
    "harnesscad.io.formats.obj": _Adapter(
        extensions=(".obj",), mime="model/obj", kind="mesh",
        read_symbols=("parse_obj",), write_symbols=("serialize_obj_float",),
        reader=_read_obj, writer=_write_obj, lossless=True,
        note="Indexed v/f text. Written with precision=6; coordinates round-trip "
             "to that precision.",
    ),
    "harnesscad.io.formats.glb": _Adapter(
        extensions=(".glb",), mime="model/gltf-binary", kind="mesh",
        read_symbols=("triangles_from_glb", "parse_glb"), write_symbols=("write_glb",),
        reader=_read_glb, writer=_write_glb, lossless=True,
        note="Writing welds exact-duplicate vertices and recomputes normals; the "
             "triangle count and corner positions survive the round trip.",
    ),
    "harnesscad.io.formats.amf": _Adapter(
        extensions=(".amf",), mime="application/x-amf", kind="mesh",
        read_symbols=("read_amf", "to_polyhedra"),
        write_symbols=("write_amf", "from_polyhedra"),
        reader=_read_amf, writer=_write_amf, lossless=True,
        note="Unit-bearing, indexed, multi-volume XML (or ZIP with compress=True). "
             "Faces are fan-triangulated on write.",
    ),
    "harnesscad.io.formats.ply": _Adapter(
        extensions=(".ply",), mime="model/ply", kind="mesh",
        read_symbols=("parse_ply",), write_symbols=("write_ply", "serialize_ply"),
        reader=_read_ply, writer=_write_ply, lossless=True,
        note="Stanford PLY (ascii + binary_little_endian). Indexed v/f. PLY has no "
             "native unit, so the mesh unit is persisted in a 'comment unit <name>' "
             "line that round-trips; other tools ignore it.",
    ),
    "harnesscad.io.formats.threemf": _Adapter(
        extensions=(".3mf",), mime="model/3mf", kind="mesh",
        read_symbols=("read_3mf", "loads_model"),
        write_symbols=("write_3mf", "dumps_model"),
        reader=_read_3mf, writer=_write_3mf, lossless=True,
        note="3MF OPC ZIP (Content_Types + .rels + 3D/3dmodel.model). Indexed, "
             "unit-bearing, optional #RRGGBB(AA) object colour. Fixed ZIP "
             "timestamps so the bytes are reproducible.",
    ),
    "harnesscad.io.formats.step": _Adapter(
        extensions=(".step", ".stp"), mime="model/step", kind="brep",
        read_symbols=("parse",), write_symbols=("serialize",),
        reader=_read_step, writer=_write_step, lossless=True,
        note="ISO 10303-21 part-21 text. read() returns a StepFile; write() accepts "
             "a StepFile, raw part-21 text, or a session whose backend exports STEP.",
    ),
    "harnesscad.io.formats.xcsg": _Adapter(
        extensions=(".xcsg",), mime="application/xml", kind="csg",
        read_symbols=("read_xcsg", "loads"), write_symbols=("write_xcsg", "dumps"),
        reader=_read_xcsg, writer=_write_xcsg, lossless=True,
        note="AngelCAD CSG tree (typed_csg.Node) in and out; dumps(loads(x)) == x.",
    ),
    "harnesscad.io.formats.svg": _Adapter(
        extensions=(".svg",), mime="image/svg+xml", kind="drawing",
        read_symbols=(), write_symbols=("get_svg",),
        reader=None, writer=_write_svg, lossless=False,
        note="WRITE-ONLY: the codec projects 3D edges to a 2D wireframe drawing and "
             "has no parser. A drawing cannot be read back into a model. Pass "
             "views=True (or a view list) to take the orthographic engineering-"
             "drawing route instead: first/third angle, hidden lines, dimensions.",
    ),
    "harnesscad.io.render": _Adapter(
        extensions=(".png",), mime="image/png", kind="image",
        read_symbols=(), write_symbols=("render", "write_png"),
        reader=None, writer=_write_png, lossless=False,
        note="WRITE-ONLY: a rendered raster (shaded solid + feature edges), not "
             "geometry. Pixels cannot be decoded back into a model, so there is no "
             "reader. Options pass through to io.render.render(): view=, shading=, "
             "edges=, ssaa=, width=, height=, projection=.",
    ),
    "harnesscad.io.formats.dxf": _Adapter(
        extensions=(".dxf",), mime="image/vnd.dxf", kind="drawing",
        read_symbols=(), write_symbols=(),
        reader=None, writer=None, lossless=False,
        note="CONTRACT ONLY: dxf.py declares DxfDocument plus DxfParser/DxfSerializer "
             "as Protocols. No concrete codec ships, so neither read nor write is "
             "offered.",
    ),
}


# ---------------------------------------------------------------------------
# FormatSpec + discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormatSpec:
    """One usable format: what it is, what it can honestly do, and how to drive it."""

    name: str                      # the codec module's leaf name, e.g. "stl"
    dotted: str                    # harnesscad.io.formats.stl
    extensions: Tuple[str, ...]
    mime: str
    kind: str                      # mesh | brep | csg | drawing | image
    can_read: bool
    can_write: bool
    round_trip: bool
    summary: str
    note: str
    _reader: Optional[Callable[..., Any]] = None
    _writer: Optional[Callable[..., Any]] = None

    def read(self, path: str, **options: Any) -> Any:
        if not self.can_read or self._reader is None:
            raise UnsupportedOperationError(
                f"format {self.name!r} is write-only: {self.note}")
        return self._reader(path, **options)

    def write(self, model: Any, path: str, *, source: Any = None,
              force: bool = False, **options: Any) -> str:
        """Write ``model`` to ``path`` -- THROUGH THE OUTPUT GATE.

        Every codec in this registry is reached through here, so this is the one
        place the gate has to stand. It measures the geometry, compares it with
        the declared intent (when ``source`` is a session/backend carrying an op
        log), and raises :class:`harnesscad.io.gate.InvalidArtifact` *before the
        file is opened* if the two disagree. ``force=True`` writes anyway and
        drops a ``<name>.INVALID.json`` sidecar naming the failures.
        """
        if not self.can_write or self._writer is None:
            raise UnsupportedOperationError(
                f"format {self.name!r} cannot be written: {self.note}")
        return gate.gated_write(self._writer, model, str(path),
                                source=source, force=force, **options)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dotted": self.dotted,
            "extensions": list(self.extensions),
            "mime": self.mime,
            "kind": self.kind,
            "read": self.can_read,
            "write": self.can_write,
            "round_trip": self.round_trip,
            "summary": self.summary,
            "note": self.note,
        }


_SPECS: Optional[Tuple[FormatSpec, ...]] = None


def _codec_symbols(entry: "capabilities.ModuleEntry") -> frozenset:
    return frozenset(entry.symbols)


def _format_entries() -> List["capabilities.ModuleEntry"]:
    """The registry's ``format``-tagged modules, minus this dispatcher itself."""
    return [e for e in capabilities.find(tag=FORMAT_TAG) if e.dotted != __name__]


def _build_specs() -> Tuple[FormatSpec, ...]:
    """Discover codecs through the capability registry and adapt the known ones.

    Only modules the capability registry tags ``format`` are considered, and a
    capability is only advertised when the codec really exposes the symbols the
    adapter calls.
    """
    out: List[FormatSpec] = []
    for entry in _format_entries():
        adapter = _ADAPTERS.get(entry.dotted)
        if adapter is None:
            continue
        symbols = _codec_symbols(entry)
        can_read = bool(adapter.read_symbols) and symbols.issuperset(adapter.read_symbols)
        can_write = bool(adapter.write_symbols) and symbols.issuperset(adapter.write_symbols)
        out.append(FormatSpec(
            name=entry.name,
            dotted=entry.dotted,
            extensions=adapter.extensions,
            mime=adapter.mime,
            kind=adapter.kind,
            can_read=can_read,
            can_write=can_write,
            round_trip=bool(can_read and can_write and adapter.lossless),
            summary=entry.summary,
            note=adapter.note,
            _reader=adapter.reader if can_read else None,
            _writer=adapter.writer if can_write else None,
        ))
    out.sort(key=lambda s: s.name)
    return tuple(out)


def specs(refresh: bool = False) -> Tuple[FormatSpec, ...]:
    """Every adapted format, sorted by name (cached)."""
    global _SPECS
    if refresh or _SPECS is None:
        _SPECS = _build_specs()
    return _SPECS


def unadapted() -> List[str]:
    """Registry modules tagged ``format`` for which no adapter exists (yet)."""
    adapted = {s.dotted for s in specs()}
    return sorted(e.dotted for e in _format_entries() if e.dotted not in adapted)


def extensions() -> Dict[str, FormatSpec]:
    """Lower-cased extension (with the dot) -> spec."""
    table: Dict[str, FormatSpec] = {}
    for spec in specs():
        for ext in spec.extensions:
            table[ext.lower()] = spec
    return table


def spec_for_extension(ext: str) -> FormatSpec:
    if not ext.startswith("."):
        ext = "." + ext
    table = extensions()
    try:
        return table[ext.lower()]
    except KeyError:
        known = ", ".join(sorted(table))
        raise UnknownFormatError(
            f"unknown format extension {ext!r}; known extensions: {known}") from None


def spec_for_path(path: str) -> FormatSpec:
    ext = os.path.splitext(str(path))[1]
    if not ext:
        raise UnknownFormatError(f"cannot infer a format: {path!r} has no extension")
    return spec_for_extension(ext)


def supported(kind: Optional[str] = None, mode: Optional[str] = None) -> List[FormatSpec]:
    """The specs matching a kind (mesh/brep/csg/drawing/image) and/or a mode."""
    if mode not in (None, "read", "write"):
        raise ValueError("mode must be 'read', 'write' or None")
    out = []
    for spec in specs():
        if kind is not None and spec.kind != kind:
            continue
        if mode == "read" and not spec.can_read:
            continue
        if mode == "write" and not spec.can_write:
            continue
        out.append(spec)
    return out


def capability_matrix() -> List[dict]:
    """The honest per-format capability rows."""
    return [spec.to_dict() for spec in specs()]


def format_report() -> dict:
    """Machine-readable report: the matrix, the counts, and what is not adapted."""
    matrix = capability_matrix()
    return {
        "formats": matrix,
        "counts": {
            "total": len(matrix),
            "readable": sum(1 for r in matrix if r["read"]),
            "writable": sum(1 for r in matrix if r["write"]),
            "round_trip": sum(1 for r in matrix if r["round_trip"]),
        },
        "kinds": sorted({r["kind"] for r in matrix}),
        "extensions": sorted(extensions()),
        "tagged_but_unadapted": unadapted(),
    }


# ---------------------------------------------------------------------------
# The unified surface
# ---------------------------------------------------------------------------

def read(path: str, **options: Any) -> Any:
    """Read a file, dispatching on its extension. Raises UnknownFormatError /
    UnsupportedOperationError."""
    return spec_for_path(path).read(str(path), **options)


def write(model_or_mesh: Any, path: str, *, force: bool = False,
          **options: Any) -> str:
    """Write a model/mesh/session to `path`, dispatching on its extension.

    Goes through the output gate: an invalid artifact raises
    :class:`harnesscad.io.gate.InvalidArtifact` and no file is written.
    """
    return spec_for_path(path).write(model_or_mesh, str(path),
                                     source=model_or_mesh, force=force, **options)


def export_session(session: Any, path: str, *, force: bool = False,
                   **options: Any) -> str:
    """Write a HarnessSession's current model to any writable format.

    Mesh/drawing targets go through the backend's STL export; BRep targets go
    through its STEP export. A backend that cannot produce the needed geometry
    raises :class:`ExportError` rather than writing a bogus file.

    The backend is handed to the gate as the ``source``, so even a BRep (STEP)
    export -- whose payload is *text* the gate could not measure -- is judged
    against the real geometry the session built, and against the intent its op
    log declared.
    """
    spec = spec_for_path(path)
    if not spec.can_write:
        raise UnsupportedOperationError(
            f"format {spec.name!r} cannot be written: {spec.note}")
    backend = _backend_of(session)
    if backend is None:
        raise ExportError(
            f"{type(session).__name__!r} is not a session/backend with an export()")
    if spec.kind in ("mesh", "drawing", "image"):
        model: Any = _mesh_from_backend(backend, name=os.path.basename(str(path)))
    elif spec.kind == "brep":
        model = _step_text(backend)
    else:
        raise UnsupportedOperationError(
            f"a session cannot be exported to kind {spec.kind!r} ({spec.name})")
    return spec.write(model, str(path), source=backend, force=force, **options)


def render_matrix(rows: Optional[Sequence[dict]] = None) -> str:
    """The capability matrix as a fixed-width text table (used by the CLI)."""
    rows = list(rows if rows is not None else capability_matrix())
    header = ("FORMAT", "EXT", "KIND", "READ", "WRITE", "ROUNDTRIP", "MIME")
    table = [header]
    for r in rows:
        table.append((
            r["name"],
            ",".join(r["extensions"]),
            r["kind"],
            "yes" if r["read"] else "no",
            "yes" if r["write"] else "no",
            "yes" if r["round_trip"] else "no",
            r["mime"],
        ))
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    lines = []
    for i, row in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)).rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
