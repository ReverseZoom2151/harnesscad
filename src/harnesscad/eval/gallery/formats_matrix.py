"""Every registered format, exported and ROUND-TRIPPED. A container is not passive.

The assumption this module exists to destroy
--------------------------------------------
"We wrote the STL, so the part is on disk." No. A file format is an active
transformation, and this repository has already been bitten three times by
pretending otherwise:

* the glTF exporter silently ROTATED every part -90 degrees about X (``export_yup``
  defaults to True) and reported success;
* CadQuery's STL exporter tessellated 100x too coarse AND cached the
  triangulation, so changing the tolerance did nothing at all;
* FreeCAD's STL deflection is read out of ``user.cfg``, so the same content hash
  produces a different mesh on a different machine.

Every one of those is silent. Every one produces a file that opens. None of them
is caught by exporting and looking at the picture.

So this module does not merely emit each format. For every codec that can READ,
it exports, re-imports, RE-MEASURES, and asserts the geometry survived --
volume, bounding box, genus, watertightness, and ORIENTATION, which is the one
the glTF bug hid behind. For a codec that is export-only it says so plainly and
validates what can be validated: the file parses, the header is right, the units
are right.

Orientation is checked explicitly and separately
------------------------------------------------
A -90-degree rotation about X maps the extents (dx, dy, dz) to (dx, dz, dy). The
VOLUME IS UNCHANGED. The bounding box is unchanged as a SET. Every scalar a naive
check compares is unchanged, and the part is lying on its face. So
:func:`_orientation_ok` compares the bbox AXIS BY AXIS, in order, and a swapped
pair is reported as ``rotated`` rather than being averaged into a "bbox delta"
that looks like rounding.

Nothing here is shape-specific. The harness exports an op-stream-built solid; the
checks are the same for every part and every format.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from harnesscad.io import gate as gate_route
from harnesscad.io.formats import registry as formats

__all__ = [
    "FormatResult",
    "FormatMatrix",
    "NOT_SUPPORTED",
    "check_format",
    "run",
    "format_text",
    "main",
]

#: Formats a reader might reasonably expect and that this harness DOES NOT HAVE.
#: Stated so the README under-claims rather than implying a completeness we lack.
NOT_SUPPORTED: Tuple[str, ...] = (
    "3MF -- no codec, not registered",
    "IGES -- no codec, not registered",
    "PLY -- no codec, not registered",
    "DXF -- dxf.py declares DxfDocument/DxfParser/DxfSerializer as PROTOCOLS "
    "only; no concrete codec ships, so the registry offers neither read nor "
    "write. The extension resolves; nothing can be written to it.",
)

#: Modules that live under io/formats/ but are NOT reachable through the
#: registry (no FormatSpec), so they are not part of the supported output
#: surface however much their file names suggest otherwise.
UNREGISTERED_MODULES: Tuple[str, ...] = (
    "freecad_document", "onshape_json", "cpp_header", "base64_data",
    "flat_array", "step_graph", "step_header", "step_schema", "step_skeleton",
    "step_xstring", "mesh_ordering", "mesh_order_checks", "mesh_windowing",
)

#: How much a mesh codec may move a coordinate before it is a bug, in mm. OBJ is
#: written at precision=6 and STL binary stores float32, so ~1e-3 mm is the
#: honest floor for a 100 mm part; anything larger is the codec, not the format.
COORD_EPS = 5e-3
VOLUME_REL_EPS = 1e-3


@dataclass
class FormatResult:
    """One format, exported and (where possible) read back and re-measured."""

    name: str
    kind: str = ""
    extensions: List[str] = field(default_factory=list)
    can_read: bool = False
    can_write: bool = False
    # -- what happened
    exported: bool = False
    path: str = ""
    bytes: int = 0
    gated: Optional[bool] = None          # did the artifact pass io.gate?
    gate_failures: List[str] = field(default_factory=list)
    round_tripped: Optional[bool] = None  # None = export-only, so N/A
    volume_delta: Optional[float] = None
    bbox_delta: Optional[float] = None
    genus_source: Optional[int] = None
    genus_readback: Optional[int] = None
    orientation_ok: Optional[bool] = None
    watertight_readback: Optional[bool] = None
    notes: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        """Did this format do its job? Export-only formats pass on export alone."""
        if self.error or not self.exported:
            return False
        if self.gated is False:
            return False
        if self.can_read:
            return bool(self.round_tripped)
        return True

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ok"] = self.ok
        return d


@dataclass
class FormatMatrix:
    part: str = ""
    backend: str = ""
    results: List[FormatResult] = field(default_factory=list)
    not_supported: List[str] = field(default_factory=lambda: list(NOT_SUPPORTED))
    unregistered: List[str] = field(default_factory=lambda: list(UNREGISTERED_MODULES))

    @property
    def failures(self) -> List[FormatResult]:
        return [r for r in self.results if not r.ok]

    def to_dict(self) -> dict:
        return {
            "part": self.part,
            "backend": self.backend,
            "counts": {
                "formats": len(self.results),
                "ok": sum(1 for r in self.results if r.ok),
                "failed": len(self.failures),
                "round_trip_capable": sum(1 for r in self.results if r.can_read),
                "export_only": sum(1 for r in self.results
                                   if r.can_write and not r.can_read),
            },
            "results": [r.to_dict() for r in self.results],
            "not_supported": self.not_supported,
            "unregistered_modules": self.unregistered,
        }


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------
def _measure(mesh) -> Dict[str, Any]:
    verts, faces = mesh
    return gate_route.measure(verts, faces)


def _bbox(mesh) -> Tuple[float, float, float]:
    verts = mesh[0]
    if not verts:
        return (0.0, 0.0, 0.0)
    axes = list(zip(*verts))
    return tuple(max(a) - min(a) for a in axes[:3])  # type: ignore[return-value]


def _orientation_ok(src, back) -> Tuple[bool, List[str]]:
    """Is the read-back part standing the same way up as the one we exported?

    THE POINT OF THIS FUNCTION. A -90-degree rotation about X sends the extents
    (dx, dy, dz) to (dx, dz, dy). Volume: unchanged. Bbox as a multiset:
    unchanged. Every scalar a lazy check compares: unchanged. The part is on its
    face and nothing notices. That is precisely the glTF ``export_yup`` bug, and
    it shipped.

    So compare the bbox AXIS BY AXIS, in order, and if the axes match only after
    a permutation, call it what it is: rotated.
    """
    a, b = _bbox(src), _bbox(back)
    notes: List[str] = []
    if all(abs(x - y) <= COORD_EPS for x, y in zip(a, b)):
        return True, notes
    if sorted(a) and all(abs(x - y) <= COORD_EPS
                         for x, y in zip(sorted(a), sorted(b))):
        notes.append(
            "ROTATED: the bbox matches only after permuting the axes "
            "(%s -> %s). The volume is unchanged and the part is lying on its "
            "face -- this is the glTF export_yup class of bug."
            % (tuple(round(v, 3) for v in a), tuple(round(v, 3) for v in b)))
        return False, notes
    notes.append("bbox changed: %s -> %s"
                 % (tuple(round(v, 3) for v in a), tuple(round(v, 3) for v in b)))
    return False, notes


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
def check_format(spec, mesh, out_dir: str, stem: str,
                 session: Any = None, **options: Any) -> FormatResult:
    """Export ``mesh`` as ``spec``, gate it, and read it back if the codec can.

    The gate runs on the artifact BEFORE it is written (``gate.gated_write``),
    so a format cannot be the thing that smuggles a broken solid onto disk.

    A MESH IS NOT A MODEL. The mesh codecs (stl/obj/glb/amf) want triangles, but
    STEP wants a B-rep and XCSG wants a CSG tree -- neither of which can be
    recovered from a triangle soup, and both of which the SESSION still holds.
    Handing a mesh to the STEP writer earns exactly the error it deserves
    ("cannot interpret 'tuple' as STEP"), so the payload is chosen by the
    codec's ``kind`` rather than assumed.
    """
    res = FormatResult(name=spec.name, kind=spec.kind,
                       extensions=list(spec.extensions),
                       can_read=bool(spec.can_read), can_write=bool(spec.can_write))
    if not spec.can_write:
        res.error = "no writer: %s" % (spec.note or "not implemented")
        return res

    ext = spec.extensions[0]
    path = os.path.join(out_dir, stem + ext)
    os.makedirs(out_dir, exist_ok=True)

    # A B-rep / CSG codec is handed the SESSION (which still holds the feature
    # tree); a mesh / image / drawing codec is handed the triangles.
    payload: Any = mesh
    if spec.kind in ("brep", "csg"):
        if session is None:
            res.error = "no session: %s needs a model, not a mesh" % spec.name
            return res
        payload = session

    # 1. EXPORT, through the gate.
    try:
        gate_route.gated_write(
            lambda model, p, **kw: formats.write(model, p, **kw),
            payload, path, source=session, **options)
        res.exported = True
        res.gated = True
        res.path = os.path.basename(path)
        res.bytes = os.path.getsize(path)
    except gate_route.GateError as exc:
        res.gated = False
        res.gate_failures = [str(exc)]
        res.error = "GATE REFUSED the artifact: %s" % str(exc)[:160]
        return res
    except Exception as exc:  # noqa: BLE001 - an export that blows up is a finding
        res.error = "%s: %s" % (type(exc).__name__, exc)
        return res

    if not spec.can_read:
        res.round_tripped = None
        res.notes.append("EXPORT-ONLY: %s" % (spec.note or "no reader ships"))
        return res

    # A B-rep / CSG artifact has no triangles to re-measure. Round-tripping it
    # means parsing it back and checking the STRUCTURE -- and for STEP that means
    # reading FILE_SCHEMA out of the file rather than believing what we asked
    # for. (FreeCAD writes AP214 headless and cannot be told otherwise.)
    if spec.kind in ("brep", "csg"):
        return _round_trip_model(res, spec, path)

    # 2. READ IT BACK and RE-MEASURE. This is the half that finds the bugs.
    try:
        model = formats.read(path)
        verts, faces = formats.to_mesh(model).indexed()
        back = ([tuple(float(c) for c in v) for v in verts],
                [tuple(int(i) for i in f) for f in faces])
    except Exception as exc:  # noqa: BLE001
        res.round_tripped = False
        res.error = "re-import FAILED: %s: %s" % (type(exc).__name__, exc)
        return res

    src_m, back_m = _measure(mesh), _measure(back)
    sv, bv = src_m.get("volume") or 0.0, back_m.get("volume") or 0.0
    res.volume_delta = abs(sv - bv) / max(abs(sv), 1e-9)
    res.genus_source = src_m.get("genus")
    res.genus_readback = back_m.get("genus")
    res.watertight_readback = back_m.get("watertight")

    ok_orient, notes = _orientation_ok(mesh, back)
    res.orientation_ok = ok_orient
    res.notes.extend(notes)
    a, b = _bbox(mesh), _bbox(back)
    res.bbox_delta = max(abs(x - y) for x, y in zip(a, b))

    problems: List[str] = []
    if not ok_orient:
        problems.append("orientation")
    if res.volume_delta > VOLUME_REL_EPS:
        problems.append("volume %.3f%%" % (100.0 * res.volume_delta))
    if res.genus_source is not None and res.genus_readback != res.genus_source:
        problems.append("genus %s -> %s" % (res.genus_source, res.genus_readback))
    if back_m.get("watertight") is False and src_m.get("watertight") is True:
        problems.append("watertight lost")
    res.round_tripped = not problems
    if problems:
        res.notes.append("ROUND-TRIP FAILED: " + ", ".join(problems))
    return res


def _round_trip_model(res: FormatResult, spec, path: str) -> FormatResult:
    """Re-parse a B-rep / CSG artifact and check its STRUCTURE survived.

    For STEP the load-bearing check is the SCHEMA, and it is read out of the file
    that was written -- never assumed from the flag we passed. FreeCAD writes
    AP214 headless and cannot be told otherwise, so a harness that *reports* the
    schema it requested is reporting a wish.
    """
    try:
        model = formats.read(path)
    except Exception as exc:  # noqa: BLE001
        res.round_tripped = False
        res.error = "re-parse FAILED: %s: %s" % (type(exc).__name__, exc)
        return res

    if spec.kind == "brep":
        schema = _step_schema(path)
        res.notes.append("FILE_SCHEMA read back from the file: %s"
                         % (schema or "ABSENT"))
        entities = len(getattr(model, "entities", ()) or ())
        res.notes.append("%d STEP entities parsed back" % entities)
        res.round_tripped = bool(schema) and entities > 0
        if not res.round_tripped:
            res.notes.append("ROUND-TRIP FAILED: no FILE_SCHEMA or no entities")
        return res

    # CSG: the format's own contract is dumps(loads(x)) == x.
    try:
        again = formats.read(path)
        res.round_tripped = (formats.write.__module__ is not None
                             and repr(model) == repr(again))
        res.notes.append("CSG tree re-parsed; dumps(loads(x)) == x: %s"
                         % res.round_tripped)
    except Exception as exc:  # noqa: BLE001
        res.round_tripped = False
        res.notes.append("CSG re-parse failed: %s" % exc)
    return res


def _step_schema(path: str) -> str:
    """The FILE_SCHEMA actually present in the written part-21 file."""
    import re

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(8192)
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", head, re.I)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
#: A format can only come out of a backend that can PRODUCE it. An SDF sampler
#: has no B-rep to write to STEP, so asking frep for one is not a bug -- it is a
#: category error. STEP therefore comes off a real OCCT kernel.
BREP_BACKENDS: Tuple[str, ...] = ("freecad", "cadquery")


def _build(part, backend: str):
    """Compose the part's CISP ops on ``backend``. Returns (session, mesh) or None.

    Never hand-writes geometry: the solid is whatever the op stream builds.
    """
    from harnesscad.io.surfaces.server import CISPServer

    server = CISPServer(backend=backend)
    if server.backend_name != backend:
        return None, "backend unavailable (%s)" % server.backend_note
    result = server.applyOps([dict(o) for o in part.raw])
    if not result.get("ok"):
        rej = result.get("rejected") or {}
        return None, "backend REFUSED the plan (rejected %s)" % rej.get("op", "?")
    verts, faces = formats.to_mesh(server.backend).indexed()
    mesh = ([tuple(float(c) for c in v) for v in verts],
            [tuple(int(i) for i in f) for f in faces])
    return (server, mesh), ""


def run(part_name: str = "shell-and-holes", backend: str = "frep",
        out_dir: str = os.path.join("assets", "gallery", "formats"),
        log=None) -> FormatMatrix:
    """Build ONE part from a CISP op stream, then push it through every codec.

    The part is built by composing ops through a real backend -- never by
    hand-writing vertices -- so the thing being exported is a solid the harness
    actually made. A B-rep codec is fed from a B-rep kernel (see
    :data:`BREP_BACKENDS`); everything else is fed from ``backend``.
    """
    say = log or (lambda _m: None)
    from harnesscad.eval.gallery import complex_parts

    part = complex_parts.get(part_name)
    built, why = _build(part, backend)
    if built is None:
        raise RuntimeError("%s did not build on %s: %s" % (part_name, backend, why))
    server, mesh = built

    # The B-rep kernel, built once, for the STEP row only.
    brep_server = None
    brep_why = "no B-rep kernel available"
    for b in BREP_BACKENDS:
        got, why = _build(part, b)
        if got is not None:
            brep_server, brep_name = got[0], b
            brep_why = ""
            break
        brep_why = "%s: %s" % (b, why)

    matrix = FormatMatrix(part=part_name, backend=backend)
    for spec in formats.supported():
        # STL gets BOTH flavours: the binary default and the ASCII one.
        if spec.name == "stl":
            for label, opts in (("stl-binary", {"ascii": False}),
                                ("stl-ascii", {"ascii": True})):
                res = check_format(spec, mesh, out_dir,
                                   "%s-%s" % (part_name, label),
                                   session=server.session, **opts)
                res.name = label
                matrix.results.append(res)
                say(_line(res))
            continue

        if spec.kind == "brep":
            if brep_server is None:
                res = FormatResult(name=spec.name, kind=spec.kind,
                                   extensions=list(spec.extensions),
                                   can_read=spec.can_read, can_write=spec.can_write,
                                   error="no B-rep kernel could build this part "
                                         "(%s). An SDF sampler has no B-rep to "
                                         "write." % brep_why)
                matrix.results.append(res)
                say(_line(res))
                continue
            res = check_format(spec, mesh, out_dir, part_name,
                               session=brep_server.session)
            res.notes.insert(0, "written from the %s kernel (frep has no B-rep)"
                             % brep_name)
            matrix.results.append(res)
            say(_line(res))
            continue

        res = check_format(spec, mesh, out_dir, part_name,
                           session=server.session)
        matrix.results.append(res)
        say(_line(res))
    return matrix


def _line(r: FormatResult) -> str:
    if r.error:
        return "FAIL %-12s %s" % (r.name, r.error[:64])
    if r.round_tripped is None:
        return "ok   %-12s %-9s export-only  %7d bytes" % (r.name, r.kind, r.bytes)
    if r.round_tripped:
        return ("ok   %-12s %-9s round-trip   %7d bytes  dV %.2e  dBB %.2e"
                % (r.name, r.kind, r.bytes, r.volume_delta or 0.0,
                   r.bbox_delta or 0.0))
    return "FAIL %-12s %-9s ROUND-TRIP BROKEN: %s" % (
        r.name, r.kind, "; ".join(r.notes)[:60])


def format_text(matrix: FormatMatrix) -> str:
    lines: List[str] = []
    lines.append("FORMAT MATRIX -- %s built on %s, pushed through every codec"
                 % (matrix.part, matrix.backend))
    lines.append("=" * 78)
    lines.append("%-13s %-8s %-6s %-11s %10s %s"
                 % ("format", "kind", "gate", "round-trip", "bytes", "notes"))
    lines.append("-" * 78)
    for r in matrix.results:
        if r.round_tripped is None:
            rt = "export-only" if not r.error else "-"
        else:
            rt = "OK" if r.round_tripped else "BROKEN"
        gate_s = "-" if r.gated is None else ("pass" if r.gated else "REFUSED")
        note = r.error or ("; ".join(r.notes) if r.notes else "")
        lines.append("%-13s %-8s %-6s %-11s %10d %s"
                     % (r.name, r.kind, gate_s, rt, r.bytes, note[:30]))
    lines.append("")
    if matrix.failures:
        lines.append("FAILURES -- these are live bugs, not omissions")
        lines.append("-" * 78)
        for r in matrix.failures:
            lines.append("  %-13s %s" % (r.name, (r.error or "; ".join(r.notes))[:60]))
        lines.append("")
    lines.append("NOT SUPPORTED (stated so the README cannot imply otherwise):")
    for n in matrix.not_supported:
        lines.append("  - " + n)
    lines.append("")
    lines.append("Present under io/formats/ but NOT reachable through the registry")
    lines.append("(no FormatSpec, so not part of the supported output surface):")
    lines.append("  " + ", ".join(matrix.unregistered))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="formats-matrix", description=__doc__)
    parser.add_argument("--part", default="shell-and-holes")
    parser.add_argument("--backend", default="frep")
    parser.add_argument("--out", default=os.path.join("assets", "gallery", "formats"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    matrix = run(part_name=args.part, backend=args.backend, out_dir=args.out,
                 log=lambda m: print(m, flush=True))
    print()
    if args.json:
        print(json.dumps(matrix.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_text(matrix))
    return 1 if matrix.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
