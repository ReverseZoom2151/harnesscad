"""Ingest pipeline: tokens / meshes -> CAD command sequence -> editable CISP ops.

The harness can EMIT CAD (planner -> CISP ops -> kernel). This module is the
inverse leg: it takes an existing model -- expressed as one of the CAD token
families the reconstruction fleet already models faithfully, or as a mesh /
point cloud -- and turns it back into a CISP op stream that a
:class:`harnesscad.core.loop.HarnessSession` can apply, verify and *edit*.

Three stages
------------
1. ``decode(TokenSequence, family=...)`` -- a family-specific dequantiser turns
   tokens into a neutral :class:`CommandSequence` (sketch profiles + extrusions).
2. ``to_cisp(CommandSequence)`` -- the neutral sequence becomes ``list[Op]``
   (``NewSketch`` / ``AddLine`` / ``AddCircle`` / ``Extrude`` / ``Boolean``).
3. ``ingest_tokens(...)`` -- the ops are applied to a session, giving a digest,
   a summary and an editable op log (``SetParam`` now works on an ingested model).

THE FAMILIES ARE MUTUALLY INCOMPATIBLE -- THEY ARE NEVER MERGED OR GUESSED
-------------------------------------------------------------------------
Every family quantises with a different rule, and mixing them silently shifts
or rescales all geometry:

=========== ====================================================================
deepcad     256 levels, ``round`` (half-to-even), reconstruction *at the level*
            (:mod:`...tokens.deepcad_quantize`).
skexgen     6-bit (64 levels), *truncating* forward map, reconstruction at the
            level -- biased half a bin low (:mod:`...tokens.skexgen_quantize`).
hnc         8-bit floor quantiser + a **25-frame rotation codebook**: the
            sketch-plane orientation is a categorical index, not an angle
            (:mod:`...tokens.hnc_rotation_codebook`).
vitruvion   floor quantise + **bin-centre** reconstruction over ``[-0.5, 0.5]``
            -- the only unbiased round-trip (:mod:`...tokens.vitruvion_primitives`).
=========== ====================================================================

A :class:`TokenSequence` therefore carries its ``family`` tag, and every entry
point takes an explicit ``family=`` argument. If the two disagree the pipeline
raises :class:`FamilyMismatch` -- decoding DeepCAD tokens with the SkexGen
dequantiser is a real, documented bug and is made impossible here rather than
merely discouraged.

Stdlib only, deterministic, no I/O beyond reading the file you name.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from harnesscad import registry
from harnesscad.core.cisp.ops import (
    AddCircle,
    AddLine,
    Boolean,
    Extrude,
    NewSketch,
    Op,
)
from harnesscad.domain.reconstruction.sequences import sketch_extrude_schema as se_schema
from harnesscad.domain.reconstruction.sketch import deepcad_sketch_plane as dc_plane
from harnesscad.domain.reconstruction.tokens import deepcad_commands as dc_cmd
from harnesscad.domain.reconstruction.tokens import deepcad_quantize as dc_quant
from harnesscad.domain.reconstruction.tokens import deepcad_vector_layout as dc_vec
from harnesscad.domain.reconstruction.tokens import hnc_rotation_codebook as hnc_rot
from harnesscad.domain.reconstruction.tokens import hnc_vector_codec as hnc_codec
from harnesscad.domain.reconstruction.tokens import skexgen_decode as sx_decode
from harnesscad.domain.reconstruction.tokens import skexgen_extrude as sx_extrude
from harnesscad.domain.reconstruction.tokens import skexgen_quantize as sx_quant
from harnesscad.domain.reconstruction.tokens import vitruvion_primitives as vitruvion
from harnesscad.io.ingest.decompile import decompile as _decompile_solid
from harnesscad.io.ingest.point_cloud import canonicalize_cloud as _canonicalize_cloud
from harnesscad.io.ingest.tokenization_audit import audit_tokenization as _audit_tokens
from harnesscad.io.surfaces.server import CISPServer

__all__ = [
    "FAMILIES",
    "IngestError",
    "FamilyMismatch",
    "UnknownFamily",
    "UnsupportedByFamily",
    "TokenSequence",
    "Curve",
    "SketchExtrude",
    "CommandSequence",
    "Decoder",
    "DeepCADDecoder",
    "SkexGenDecoder",
    "HNCDecoder",
    "VitruvionDecoder",
    "discover_decoders",
    "get_decoder",
    "decode",
    "encode",
    "to_cisp",
    "from_cisp",
    "ingest_tokens",
    "ingest_mesh",
    "load_token_file",
    "ingest_file",
]


# --------------------------------------------------------------------------- #
# Families and errors
# --------------------------------------------------------------------------- #
DEEPCAD = "deepcad"
SKEXGEN = "skexgen"
HNC = "hnc"
VITRUVION = "vitruvion"

#: The selectable quantiser families. Never merged, never inferred.
FAMILIES: Tuple[str, ...] = (DEEPCAD, SKEXGEN, HNC, VITRUVION)


class IngestError(ValueError):
    """Base class for every ingest failure (never silently degraded)."""


class UnknownFamily(IngestError):
    """A family name outside :data:`FAMILIES`."""


class FamilyMismatch(IngestError):
    """A token sequence tagged with one family was handed to another's decoder."""


class UnsupportedByFamily(IngestError):
    """The requested operation is not expressible in this family's token format."""


def _check_family(name: str) -> str:
    if name not in FAMILIES:
        raise UnknownFamily(
            "unknown token family %r; the selectable families are %s"
            % (name, ", ".join(FAMILIES)))
    return name


# --------------------------------------------------------------------------- #
# Neutral value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenSequence:
    """A family-tagged token payload. ``tokens`` is family-specific:

    * deepcad   -- a list of 17-int rows (:mod:`...tokens.deepcad_vector_layout`).
    * skexgen   -- ``{"tokens": [...]}``, the merged pixel/extrude stream.
    * hnc       -- ``{"cmds": [...], "params": [[...]], "extrude": [...]}``.
    * vitruvion -- ``{"val": [...], "num_bins": 64}``, the primitive ``val`` stream.
    """

    family: str
    tokens: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_family(self.family)

    def to_dict(self) -> dict:
        return {"family": self.family, "tokens": self.tokens, "meta": dict(self.meta)}

    @staticmethod
    def from_dict(payload: dict) -> "TokenSequence":
        if not isinstance(payload, dict) or "family" not in payload:
            raise IngestError("token document must be a JSON object with a 'family' key")
        if "tokens" not in payload:
            raise IngestError("token document has no 'tokens' key")
        return TokenSequence(str(payload["family"]), payload["tokens"],
                             dict(payload.get("meta") or {}))


@dataclass(frozen=True)
class Curve:
    """A decoded sketch curve in sketch-plane (2D) coordinates.

    ``kind`` is ``line`` (start, end), ``arc`` (start, mid, end) or ``circle``
    (centre + :attr:`radius`).
    """

    kind: str
    points: Tuple[Tuple[float, float], ...] = ()
    radius: Optional[float] = None


@dataclass(frozen=True)
class SketchExtrude:
    """One sketch profile and (optionally) the feature that consumes it.

    ``distance is None`` marks a sketch-only family (Vitruvion has no extrude
    vocabulary at all) -- ingest then produces sketch ops and no solid.
    """

    plane: str = "XY"
    loops: Tuple[Tuple[Curve, ...], ...] = ()
    distance: Optional[float] = None
    operation: str = "new"          # new | join | cut | intersect


@dataclass(frozen=True)
class CommandSequence:
    """The family-neutral CAD command sequence a decoder produces."""

    family: str
    sketches: Tuple[SketchExtrude, ...] = ()

    def __post_init__(self) -> None:
        _check_family(self.family)


# --------------------------------------------------------------------------- #
# Decoder protocol + registry
# --------------------------------------------------------------------------- #
class Decoder(Protocol):
    """A family's adapter: tokens in, neutral command sequence out."""

    family: str
    modules: Tuple[str, ...]

    def decode(self, tokens: Any) -> CommandSequence: ...

    def encode(self, ops: Sequence[Op]) -> TokenSequence: ...


_PKG = "harnesscad.domain.reconstruction"


class _BaseDecoder:
    family: str = ""
    modules: Tuple[str, ...] = ()

    def decode(self, tokens: Any) -> CommandSequence:  # pragma: no cover - abstract
        raise NotImplementedError

    def encode(self, ops: Sequence[Op]) -> TokenSequence:
        raise UnsupportedByFamily(
            "the %r token format cannot represent a CISP op stream directly "
            "(no encoder wired); decode-only family" % self.family)


# --- DeepCAD ---------------------------------------------------------------
_PLANE_NORMALS = {"XY": (0.0, 0.0, 1.0), "XZ": (0.0, 1.0, 0.0), "YZ": (1.0, 0.0, 0.0)}
_PLANE_AXES_2D = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}


def _plane_from_normal(normal: Sequence[float]) -> str:
    """Nearest CISP named plane for a unit normal (deterministic tie-break)."""
    mags = [abs(float(c)) for c in normal]
    axis = max(range(3), key=lambda i: (mags[i], -i))
    return {0: "YZ", 1: "XZ", 2: "XY"}[axis]


def _origin_2d(plane: str, origin3: Sequence[float]) -> Tuple[float, float]:
    i, j = _PLANE_AXES_2D[plane]
    return (float(origin3[i]), float(origin3[j]))


class DeepCADDecoder(_BaseDecoder):
    """DeepCAD family: 256 levels, round-half-even, level-valued.

    Tokens are the 17-int rows of :mod:`...tokens.deepcad_vector_layout`
    (``[cmd, x, y, alpha, f, r, theta, phi, gamma, px, py, pz, s, e1, e2, b, u]``).
    """

    family = DEEPCAD
    modules = (
        _PKG + ".tokens.deepcad_vector_layout",
        _PKG + ".tokens.deepcad_quantize",
        _PKG + ".tokens.deepcad_commands",
        _PKG + ".sketch.deepcad_sketch_plane",
    )

    #: b (boolean op) index -> neutral operation name.
    OPERATIONS = ("new", "join", "cut", "intersect")

    @staticmethod
    def _pixel_scale(bbox_size: float) -> float:
        """The reference's sketch de-normalisation scale (see deepcad_quantize)."""
        return bbox_size / (dc_quant.SKETCH_DIM / 2 * dc_quant.NORM_FACTOR - 1)

    def decode(self, tokens: Any) -> CommandSequence:
        rows = _as_rows(tokens, dc_vec.ROW_LEN, DEEPCAD)
        groups = dc_vec.split_extrudes(dc_vec.trim_eos(rows))
        if not groups:
            raise IngestError("deepcad stream contains no Ext row")

        sketches: List[SketchExtrude] = []
        for group in groups:
            ext = group[-1]
            args = dict(zip(dc_vec.ARG_NAMES, ext[1:]))
            theta = dc_quant.denumericalize_angle(int(args["theta"]))
            phi = dc_quant.denumericalize_angle(int(args["phi"]))
            origin3 = tuple(dc_quant.denumericalize_unit(int(args[k]))
                            for k in ("px", "py", "pz"))
            bbox_size = dc_quant.denumericalize_size(int(args["s"]))
            e1 = dc_quant.denumericalize_unit(int(args["e1"]))
            e2 = dc_quant.denumericalize_unit(int(args["e2"]))
            lo, hi = dc_plane.extrusion_extents(e1, e2, int(args["u"]))
            distance = hi - lo
            op_index = int(args["b"])
            if not 0 <= op_index < len(self.OPERATIONS):
                raise IngestError("deepcad boolean index out of range: %d" % op_index)

            plane = _plane_from_normal(dc_plane.plane_normal(theta, phi))
            ox, oy = _origin_2d(plane, origin3)
            scale = self._pixel_scale(bbox_size)

            loops: List[Tuple[Curve, ...]] = []
            for loop_rows in dc_vec.split_loops(group):
                curves = self._decode_loop(loop_rows[1:], scale, ox, oy)
                if curves:
                    loops.append(curves)
            if not loops:
                raise IngestError("deepcad extrude group has no usable loop")
            sketches.append(SketchExtrude(plane=plane, loops=tuple(loops),
                                          distance=distance,
                                          operation=self.OPERATIONS[op_index]))
        return CommandSequence(DEEPCAD, tuple(sketches))

    def _decode_loop(self, rows: Sequence[Sequence[int]], scale: float,
                     ox: float, oy: float) -> Tuple[Curve, ...]:
        bbox_size = scale * (dc_quant.SKETCH_DIM / 2 * dc_quant.NORM_FACTOR - 1)

        def point(px: int, py: int) -> Tuple[float, float]:
            # The reference de-normalisation: canvas pixel -> sketch-local coords
            # relative to the profile start point, which DeepCAD stores as the
            # sketch-plane origin.
            u, v = dc_quant.denormalize_sketch([(float(px), float(py))], bbox_size)[0]
            return (u + ox, v + oy)

        if len(rows) == 1 and rows[0][0] == dc_vec.CIRCLE_IDX:
            row = rows[0]
            cx, cy = point(int(row[1]), int(row[2]))
            radius = float(row[5]) * scale
            if radius <= 0.0:
                raise IngestError("deepcad circle has non-positive radius")
            return (Curve("circle", ((cx, cy),), radius),)

        pts = [point(int(r[1]), int(r[2])) for r in rows]
        curves: List[Curve] = []
        n = len(pts)
        if n < 2:
            return ()
        for i, row in enumerate(rows):
            start = pts[i - 1] if i > 0 else pts[-1]
            end = pts[i]
            if row[0] == dc_vec.LINE_IDX:
                curves.append(Curve("line", (start, end)))
            elif row[0] == dc_vec.ARC_IDX:
                mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
                curves.append(Curve("arc", (start, mid, end)))
            else:
                raise IngestError("unexpected deepcad row inside a loop: cmd=%d"
                                  % row[0])
        area = se_schema.signed_area([c.points[0] for c in curves])
        if abs(area) <= 0.0:
            return ()
        return tuple(curves)

    # --- encoder (round-trip support) -------------------------------------
    def encode(self, ops: Sequence[Op]) -> TokenSequence:
        groups = _group_ops_by_sketch(ops)
        rows: List[Tuple[int, ...]] = []
        for group in groups:
            plane = group["plane"]
            if plane != "XY":
                raise UnsupportedByFamily(
                    "deepcad encoding currently emits the XY sketch plane only "
                    "(got %r)" % plane)
            points = _group_points(group)
            start = points[0]
            bbox_size = dc_quant.sketch_bbox_size(points, start)
            if bbox_size <= 0.0:
                raise IngestError("degenerate sketch: zero bbox")
            scale = self._pixel_scale(bbox_size)
            half = dc_quant.SKETCH_DIM / 2

            def pixel(p: Sequence[float]) -> Tuple[int, int]:
                return (dc_quant.numericalize_pixel((p[0] - start[0]) / scale + half),
                        dc_quant.numericalize_pixel((p[1] - start[1]) / scale + half))

            for prim in group["prims"]:
                rows.append(dc_vec.sol_row())
                if isinstance(prim, AddCircle):
                    px, py = pixel((prim.cx, prim.cy))
                    rows.append(dc_vec.circle_row(
                        px, py, dc_quant.numericalize_radius(prim.r / scale)))
                else:  # a closed chain of AddLine
                    for line in prim:
                        px, py = pixel((line.x2, line.y2))
                        rows.append(dc_vec.line_row(px, py))
            distance = group["distance"] if group["distance"] is not None else 0.0
            rows.append(dc_vec.ext_row(
                theta=dc_quant.numericalize_angle(0.0),
                phi=dc_quant.numericalize_angle(0.0),
                gamma=dc_quant.numericalize_angle(0.0),
                px=dc_quant.numericalize_unit(start[0]),
                py=dc_quant.numericalize_unit(start[1]),
                pz=dc_quant.numericalize_unit(0.0),
                s=dc_quant.numericalize_size(bbox_size),
                e1=dc_quant.numericalize_unit(distance),
                e2=dc_quant.numericalize_unit(0.0),
                b=self.OPERATIONS.index(group["operation"]),
                u=dc_plane.ONE_SIDED,
            ))
        rows.append(dc_vec.eos_row())
        return TokenSequence(DEEPCAD, [list(r) for r in rows],
                             {"n_commands": len(rows),
                              "vocabulary": list(dc_cmd.COMMAND_TYPES)})


# --- SkexGen ---------------------------------------------------------------
class SkexGenDecoder(_BaseDecoder):
    """SkexGen family: 6-bit truncating quantiser, flat streams."""

    family = SKEXGEN
    modules = (
        _PKG + ".tokens.skexgen_quantize",
        _PKG + ".tokens.skexgen_decode",
        _PKG + ".tokens.skexgen_extrude",
    )

    OPERATIONS = {sx_extrude.OP_ADD: "join", sx_extrude.OP_CUT: "cut",
                  sx_extrude.OP_INTERSECT: "intersect"}

    def decode(self, tokens: Any) -> CommandSequence:
        stream = tokens["tokens"] if isinstance(tokens, dict) else tokens
        bit = int(tokens.get("bit", sx_quant.BIT)) if isinstance(tokens, dict) \
            else sx_quant.BIT
        try:
            flat = [int(t) for t in stream]
        except (TypeError, ValueError) as exc:
            raise IngestError(
                "skexgen tokens must be a flat list of ints (got %r); this is what a "
                "foreign family's payload looks like -- it is refused, not coerced"
                % (type(stream[0]).__name__ if stream else None,)) from exc
        try:
            parts = sx_decode.parse_tokens(flat, bit)
        except (sx_decode.SkexGenParseError, ValueError, IndexError) as exc:
            raise IngestError("skexgen stream did not parse: %s" % exc) from exc

        sketches: List[SketchExtrude] = []
        for i, part in enumerate(parts):
            ext = part["extrude"]
            plane = _plane_from_normal(ext["z_axis"])
            distance = abs(float(ext["value"][0])) + abs(float(ext["value"][1]))
            operation = self.OPERATIONS.get(int(ext["op"]), "join")
            if i == 0:
                operation = "new"
            loops: List[Tuple[Curve, ...]] = []
            for face in part["faces"]:
                for loop in face:
                    curves = tuple(_curve_from_dict(c) for c in loop)
                    if curves:
                        loops.append(curves)
            if not loops:
                raise IngestError("skexgen sketch has no loop")
            sketches.append(SketchExtrude(plane=plane, loops=tuple(loops),
                                          distance=distance, operation=operation))
        return CommandSequence(SKEXGEN, tuple(sketches))


def _curve_from_dict(curve: dict) -> Curve:
    kind = curve["type"]
    if kind == "line":
        return Curve("line", (tuple(curve["start"]), tuple(curve["end"])))
    if kind == "arc":
        return Curve("arc", (tuple(curve["start"]), tuple(curve["mid"]),
                             tuple(curve["end"])))
    if kind == "circle":
        return Curve("circle", (tuple(curve["center"]),), float(curve["radius"]))
    raise IngestError("unknown decoded curve type: %r" % (kind,))


# --- HNC-CAD ---------------------------------------------------------------
class HNCDecoder(_BaseDecoder):
    """HNC-CAD family: floor quantiser + 25-frame rotation codebook.

    Tokens: ``{"cmds": [...], "params": [[8 ints], ...], "extrude": [11 ints]}``
    exactly as :func:`...tokens.hnc_vector_codec.encode_sketch` /
    ``encode_extrude`` emit them.
    """

    family = HNC
    modules = (
        _PKG + ".tokens.hnc_vector_codec",
        _PKG + ".tokens.hnc_rotation_codebook",
    )

    OPERATIONS = {hnc_codec.OP_ADD: "join", hnc_codec.OP_CUT: "cut",
                  hnc_codec.OP_INTERSECT: "intersect"}

    @staticmethod
    def _dequantize(level: int, n_bits: int, lo: float, hi: float) -> float:
        """Inverse of :func:`hnc_vector_codec.quantize` (level-valued, floor-biased)."""
        span = (1 << n_bits) - 1
        return float(level) * (hi - lo) / span + lo

    def decode(self, tokens: Any) -> CommandSequence:
        if not isinstance(tokens, dict):
            raise IngestError("hnc tokens must be a mapping with cmds/params/extrude")
        cmds = [int(c) for c in tokens["cmds"]]
        params = [[int(v) for v in row] for row in tokens["params"]]
        ext = [int(v) for v in tokens["extrude"]]
        n_bits = int(tokens.get("n_bits", 8))
        if len(cmds) != len(params):
            raise IngestError("hnc cmds/params length mismatch")
        if len(ext) != 11:
            raise IngestError("hnc extrude vector must hold 11 slots")

        centre = [self._dequantize(ext[i], n_bits, -1.0, 1.0) for i in range(3)]
        scale = self._dequantize(ext[3], n_bits, 0.0, 1.0)
        ext_values = [self._dequantize(ext[i], n_bits, -1.0, 1.0) for i in (4, 5)]
        origin3 = [self._dequantize(ext[i], n_bits, -hnc_codec.SKETCH_R,
                                    hnc_codec.SKETCH_R) for i in (6, 7, 8)]
        rot_index = ext[9]
        set_op = ext[10]
        _, _, t_z = hnc_rot.frame_axes(rot_index)     # the codebook, not an angle
        plane = _plane_from_normal(t_z)
        ox, oy = _origin_2d(plane, origin3)
        cx, cy = centre[0], centre[1]

        def point(qx: int, qy: int) -> Tuple[float, float]:
            x = self._dequantize(qx, n_bits, -hnc_codec.SKETCH_R, hnc_codec.SKETCH_R)
            y = self._dequantize(qy, n_bits, -hnc_codec.SKETCH_R, hnc_codec.SKETCH_R)
            return (x * scale + cx + ox, y * scale + cy + oy)

        loops: List[Tuple[Curve, ...]] = []
        pending: List[Curve] = []
        starts: List[Tuple[float, float]] = []
        for cmd, row in zip(cmds, params):
            if cmd == hnc_codec.LINE:
                starts.append(point(row[0], row[1]))
                pending.append(Curve("line", (starts[-1], starts[-1])))
            elif cmd == hnc_codec.ARC:
                starts.append(point(row[0], row[1]))
                pending.append(Curve("arc", (starts[-1], point(row[2], row[3]),
                                             starts[-1])))
            elif cmd == hnc_codec.CIRCLE:
                north = point(row[0], row[1])
                south = point(row[2], row[3])
                ccx = (north[0] + south[0]) / 2.0
                ccy = (north[1] + south[1]) / 2.0
                radius = math.dist(north, (ccx, ccy))
                if radius <= 0.0:
                    raise IngestError("hnc circle has non-positive radius")
                pending.append(Curve("circle", ((ccx, ccy),), radius))
                starts.append((ccx, ccy))
            elif cmd in (hnc_codec.LOOP_END, hnc_codec.FACE_END,
                         hnc_codec.SKETCH_END):
                if pending:
                    loops.append(_close_chain(pending, starts))
                pending, starts = [], []
            else:
                raise IngestError("unknown hnc command token: %d" % cmd)
        if not loops:
            raise IngestError("hnc sketch has no loop")

        distance = abs(ext_values[0]) + abs(ext_values[1])
        return CommandSequence(HNC, (SketchExtrude(
            plane=plane, loops=tuple(loops), distance=distance,
            operation=self.OPERATIONS.get(set_op, "join")),))


def _close_chain(curves: Sequence[Curve], starts: Sequence[Tuple[float, float]]
                 ) -> Tuple[Curve, ...]:
    """HNC/SkexGen store only each curve's START; the end is the next curve's start."""
    if len(curves) == 1 and curves[0].kind == "circle":
        return (curves[0],)
    out: List[Curve] = []
    n = len(curves)
    for i, curve in enumerate(curves):
        end = starts[(i + 1) % n]
        if curve.kind == "line":
            out.append(Curve("line", (starts[i], end)))
        elif curve.kind == "arc":
            out.append(Curve("arc", (starts[i], curve.points[1], end)))
        else:
            out.append(curve)
    return tuple(out)


# --- Vitruvion -------------------------------------------------------------
class VitruvionDecoder(_BaseDecoder):
    """Vitruvion family: floor quantise, BIN-CENTRE reconstruction.

    The only unbiased round-trip of the four, and the only sketch-*only* family:
    it has no extrude vocabulary, so the ingested op stream is sketch ops with no
    solid (``distance is None``).
    """

    family = VITRUVION
    modules = (
        _PKG + ".tokens.vitruvion_primitives",
        _PKG + ".tokens.vitruvion_constraints",
    )

    def decode(self, tokens: Any) -> CommandSequence:
        if isinstance(tokens, dict):
            val = [int(t) for t in tokens["val"]]
            n_bins = int(tokens.get("num_bins", vitruvion.DEFAULT_NUM_BINS))
        else:
            val = [int(t) for t in tokens]
            n_bins = vitruvion.DEFAULT_NUM_BINS

        curves: List[Curve] = []
        for bins, _is_construction in vitruvion.param_seq_from_tokens(val, n_bins):
            params = vitruvion.dequantize_params(list(bins), n_bins)
            if len(params) == 4:
                curves.append(Curve("line", ((params[0], params[1]),
                                             (params[2], params[3]))))
            elif len(params) == 3:
                if params[2] <= 0.0:
                    raise IngestError("vitruvion circle has non-positive radius")
                curves.append(Curve("circle", ((params[0], params[1]),), params[2]))
            elif len(params) == 6:
                curves.append(Curve("arc", ((params[0], params[1]),
                                            (params[2], params[3]),
                                            (params[4], params[5]))))
            elif len(params) == 2:
                continue        # a bare point carries no CISP geometry
            else:
                raise IngestError("vitruvion entity with %d params" % len(params))
        if not curves:
            raise IngestError("vitruvion stream decoded to no primitives")
        return CommandSequence(VITRUVION, (SketchExtrude(
            plane="XY", loops=(tuple(curves),), distance=None, operation="new"),))

    def encode(self, ops: Sequence[Op]) -> TokenSequence:
        from harnesscad.domain.geometry.sketch.normalization import entity_from_params

        n_bins = vitruvion.DEFAULT_NUM_BINS
        entities = []
        for op in ops:
            if isinstance(op, AddLine):
                entities.append(entity_from_params([op.x1, op.y1, op.x2, op.y2]))
            elif isinstance(op, AddCircle):
                entities.append(entity_from_params([op.cx, op.cy, op.r]))
            elif isinstance(op, NewSketch):
                continue
            else:
                raise UnsupportedByFamily(
                    "vitruvion tokens carry sketch primitives only; cannot encode %r"
                    % op.OP)
        if not entities:
            raise IngestError("no sketch primitives to encode")
        streams, _gather = vitruvion.tokenize_sketch(entities, n_bins)
        return TokenSequence(VITRUVION,
                             {"val": streams["val"], "num_bins": n_bins},
                             {"coord": streams["coord"], "pos": streams["pos"]})


# --- the decoder registry --------------------------------------------------
_DECODER_CLASSES = (DeepCADDecoder, SkexGenDecoder, HNCDecoder, VitruvionDecoder)


def discover_decoders() -> Dict[str, Decoder]:
    """Discover the family decoders whose backing modules the registry indexes.

    The token/sequence fleet is addressed through :mod:`harnesscad.registry` (the
    static capability index), so a decoder is only offered when every module it
    adapts is actually present in the index -- no silent fallbacks.
    """
    indexed = {entry.dotted for entry in registry.find(package="reconstruction")}
    found: Dict[str, Decoder] = {}
    for cls in _DECODER_CLASSES:
        missing = [m for m in cls.modules if m not in indexed]
        if missing:
            continue
        found[cls.family] = cls()
    return found


def get_decoder(family: str) -> Decoder:
    """The decoder for ``family``. Raises :class:`UnknownFamily` if absent."""
    _check_family(family)
    decoders = discover_decoders()
    if family not in decoders:
        raise UnknownFamily(
            "family %r has no discoverable decoder (its modules are not indexed)"
            % family)
    return decoders[family]


def decoder_modules(family: str) -> Tuple[str, ...]:
    """The reconstruction modules a family's decoder adapts."""
    return tuple(get_decoder(family).modules)


# --------------------------------------------------------------------------- #
# Stage 1: decode / encode  (family is ALWAYS explicit)
# --------------------------------------------------------------------------- #
def decode(sequence: TokenSequence, *, family: str) -> CommandSequence:
    """Decode ``sequence`` with ``family``'s dequantiser.

    Raises :class:`FamilyMismatch` when the sequence is tagged with a different
    family. The four quantisers are mutually incompatible (different level
    counts, different rounding rules, a codebook instead of angles); running one
    family's tokens through another's dequantiser produces geometry that is
    silently wrong, so it is refused rather than approximated.
    """
    _check_family(family)
    if sequence.family != family:
        raise FamilyMismatch(
            "token sequence is tagged family %r but the %r dequantiser was "
            "requested; quantiser families are mutually incompatible and are "
            "never blended (decoding %s tokens with the %s dequantiser silently "
            "rescales every coordinate)"
            % (sequence.family, family, sequence.family, family))
    return get_decoder(family).decode(sequence.tokens)


def encode(ops: Sequence[Op], *, family: str) -> TokenSequence:
    """Encode a CISP op stream back into ``family``'s tokens (where supported)."""
    return get_decoder(family).encode(list(ops))


# --------------------------------------------------------------------------- #
# Stage 2: neutral command sequence -> CISP ops
# --------------------------------------------------------------------------- #
def to_cisp(sequence: CommandSequence, *, arc_policy: str = "chord") -> List[Op]:
    """Convert a decoded command sequence into CISP ops (the editable form).

    Sketch ids follow the backend's deterministic allocation (``sk1``, ``sk2``,
    ...). The CISP op set has no arc primitive: ``arc_policy='chord'`` replaces an
    arc with its chord (lossy but deterministic and flagged in the meta); pass
    ``arc_policy='reject'`` to raise instead.
    """
    if arc_policy not in ("chord", "reject"):
        raise IngestError("arc_policy must be 'chord' or 'reject'")

    ops: List[Op] = []
    solids = 0
    for index, sketch in enumerate(sequence.sketches, start=1):
        sid = "sk%d" % index
        ops.append(NewSketch(plane=sketch.plane))
        n_prims = 0
        for loop in sketch.loops:
            for curve in loop:
                if curve.kind == "circle":
                    ops.append(AddCircle(sketch=sid, cx=curve.points[0][0],
                                         cy=curve.points[0][1],
                                         r=float(curve.radius or 0.0)))
                    n_prims += 1
                elif curve.kind == "line":
                    (x1, y1), (x2, y2) = curve.points[0], curve.points[1]
                    ops.append(AddLine(sketch=sid, x1=x1, y1=y1, x2=x2, y2=y2))
                    n_prims += 1
                elif curve.kind == "arc":
                    if arc_policy == "reject":
                        raise UnsupportedByFamily(
                            "the CISP op set has no arc primitive; "
                            "arc_policy='reject' refused to approximate it")
                    (x1, y1) = curve.points[0]
                    (x2, y2) = curve.points[-1]
                    ops.append(AddLine(sketch=sid, x1=x1, y1=y1, x2=x2, y2=y2))
                    n_prims += 1
                else:
                    raise IngestError("unknown curve kind %r" % (curve.kind,))
        if n_prims == 0:
            raise IngestError("sketch %s decoded to no primitives" % sid)
        if sketch.distance is None:
            continue                        # sketch-only family (Vitruvion)
        if sketch.distance == 0.0:
            raise IngestError("sketch %s has a zero extrude distance" % sid)
        ops.append(Extrude(sketch=sid, distance=float(sketch.distance)))
        solids += 1
        if sketch.operation in ("cut", "intersect") and solids >= 2:
            ops.append(Boolean(kind=sketch.operation, target="solid", tool="last"))
    return ops


def from_cisp(ops: Sequence[Op], *, family: str) -> TokenSequence:
    """CISP ops -> tokens (the inverse of :func:`to_cisp` + :func:`decode`)."""
    return encode(ops, family=family)


def _group_ops_by_sketch(ops: Sequence[Op]) -> List[dict]:
    """Group a CISP op stream into per-sketch {plane, prims, distance, operation}.

    ``prims`` holds either an :class:`AddCircle` or a list of consecutive
    :class:`AddLine` ops forming one closed chain (one loop).
    """
    groups: List[dict] = []
    for op in ops:
        if isinstance(op, NewSketch):
            groups.append({"plane": op.plane, "prims": [], "distance": None,
                           "operation": "new"})
            continue
        if isinstance(op, (AddLine, AddCircle)):
            if not groups:
                raise IngestError("sketch primitive before any new_sketch")
            prims = groups[-1]["prims"]
            if isinstance(op, AddCircle):
                prims.append(op)
            elif prims and isinstance(prims[-1], list):
                prims[-1].append(op)
            else:
                prims.append([op])
            continue
        if isinstance(op, Extrude):
            if not groups:
                raise IngestError("extrude before any new_sketch")
            groups[-1]["distance"] = op.distance
            continue
        if isinstance(op, Boolean):
            if groups:
                groups[-1]["operation"] = op.kind
            continue
        raise UnsupportedByFamily(
            "op %r has no token representation in the sketch-extrude families"
            % op.OP)
    if not groups:
        raise IngestError("op stream contains no sketch")
    return groups


def _group_points(group: dict) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for prim in group["prims"]:
        if isinstance(prim, AddCircle):
            points.extend([(prim.cx - prim.r, prim.cy - prim.r),
                           (prim.cx + prim.r, prim.cy + prim.r)])
        else:
            for line in prim:
                points.append((line.x1, line.y1))
                points.append((line.x2, line.y2))
    if not points:
        raise IngestError("sketch group has no points")
    return points


def _as_rows(tokens: Any, width: int, family: str) -> List[Tuple[int, ...]]:
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise IngestError("%s tokens must be a non-empty list of rows" % family)
    rows: List[Tuple[int, ...]] = []
    for row in tokens:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise IngestError("%s rows must have %d ints (got %r)"
                              % (family, width, row))
        rows.append(tuple(int(v) for v in row))
    return rows


# --------------------------------------------------------------------------- #
# Stage 3: apply to a HarnessSession
# --------------------------------------------------------------------------- #
def _apply(ops: Sequence[Op], backend: str) -> dict:
    server = CISPServer(backend=backend)
    result = server.applyOps([op.to_dict() for op in ops])
    result = dict(result)
    result["summary"] = server.query("summary").get("result", {})
    result["backend"] = server.backend_name
    if server.backend_note:
        result["backend_note"] = server.backend_note
    return result


def ingest_tokens(sequence: TokenSequence, *, family: str, backend: str = "stub",
                  arc_policy: str = "chord") -> dict:
    """tokens -> command sequence -> CISP ops -> a verified HarnessSession.

    Returns ``{ok, applied, digest, diagnostics, summary, ops, family, ...}``.
    The op list IS the editable model: feed it to a session and ``SetParam`` any
    parameter of the ingested part.
    """
    commands = decode(sequence, family=family)
    ops = to_cisp(commands, arc_policy=arc_policy)
    result = _apply(ops, backend)
    result["family"] = family
    result["ops"] = [op.to_dict() for op in ops]
    result["n_sketches"] = len(commands.sketches)
    result["modules"] = list(decoder_modules(family))
    return result


def ingest_mesh(points: Sequence[Sequence[float]], *, backend: str = "stub",
                sample: Optional[int] = None, seed: int = 0) -> dict:
    """Mesh / point-cloud ingestion: vertices -> bbox metrics -> CISP ops.

    Uses :mod:`harnesscad.io.ingest.point_cloud` to canonicalise the cloud and
    :mod:`harnesscad.io.ingest.decompile` to recover a best-effort feature tree
    (a bounding-box prismatic block when no B-rep is available). ``confidence``
    and ``note`` from the decompiler are passed through verbatim -- this path is
    honest about being an approximation.
    """
    cloud, _transform = _canonicalize_cloud(points, count=sample, seed=seed)
    if not cloud:
        raise IngestError("empty point cloud")
    bbox = [max(p[i] for p in cloud) - min(p[i] for p in cloud) for i in range(3)]
    recovered = _decompile_solid(_MetricsSource({"bbox": bbox}))
    if not recovered.ok:
        raise IngestError("mesh ingest recovered no ops: %s" % recovered.note)
    result = _apply(recovered.ops, backend)
    result["family"] = "mesh"
    result["ops"] = [op.to_dict() for op in recovered.ops]
    result["confidence"] = recovered.confidence
    result["note"] = recovered.note
    result["method"] = recovered.method
    return result


class _MetricsSource:
    """Adapter giving :func:`io.ingest.decompile.decompile` a metrics-only source."""

    def __init__(self, metrics: dict) -> None:
        self.metrics = metrics


def audit_round_trip(reference: Sequence[Sequence[float]],
                     recovered: Sequence[Sequence[float]],
                     tolerance: float = 1e-6):
    """Fidelity gate for a tokenise/de-tokenise round trip (io.ingest.tokenization_audit)."""
    return _audit_tokens(
        reference_points=[tuple(p) for p in reference],
        encoded_points=[tuple(p) for p in recovered],
        segment_count=len(reference),
        max_segments=max(1, len(reference)),
        tolerance=tolerance,
    )


# --------------------------------------------------------------------------- #
# File entry points (used by `harnesscad ingest`)
# --------------------------------------------------------------------------- #
_MESH_SUFFIXES = (".obj", ".xyz")


def load_token_file(path: str) -> TokenSequence:
    """Read a ``{"family": ..., "tokens": ...}`` JSON document."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return TokenSequence.from_dict(payload)


def load_mesh_points(path: str) -> List[Tuple[float, float, float]]:
    """Read vertices from a minimal ``.obj`` (``v x y z``) or whitespace ``.xyz``."""
    points: List[Tuple[float, float, float]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                parts = parts[1:4]
            elif len(parts) < 3 or parts[0].startswith(("#", "f", "vn", "vt")):
                continue
            try:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not points:
        raise IngestError("no vertices parsed from %r" % path)
    return points


def ingest_file(path: str, *, family: str, backend: str = "stub",
                arc_policy: str = "chord") -> dict:
    """Ingest a token JSON document or a mesh (``family='mesh'``) from disk."""
    if family == "mesh":
        return ingest_mesh(load_mesh_points(path), backend=backend)
    _check_family(family)
    if os.path.splitext(path)[1].lower() in _MESH_SUFFIXES:
        raise IngestError(
            "%r looks like a mesh; ingest it with --family mesh" % path)
    return ingest_tokens(load_token_file(path), family=family, backend=backend,
                         arc_policy=arc_policy)
