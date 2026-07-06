"""Hierarchical Sketch-Extrusion tree for GeoFusion-CAD (Zhou et al.).

GeoFusion-CAD's central *deterministic* contribution (Sec. 3, Sec. A.4) is a
**hierarchical tree representation** of a parametric Sketch-Extrusion (SE) CAD
program. Unlike the *flat* DeepCAD command list (see
:mod:`reconstruction.deepcad_command_spec`) or the *B-Rep surface-edge adjacency*
of CMT (:mod:`reconstruction.cmt_topology_validity`), GeoFusion encodes the CAD
program as an explicit nested tree ``T = {v_i, e_ij}`` whose nodes are CAD
entities and whose edges are parent->child topological dependencies::

    solid
      +- sketch          (one per SE pair)
      |    +- face
      |    |    +- loop
      |    |    |    +- curve (line / arc / circle)
      +- extrusion       (the operation applied to the sketch)

The paper serializes each tree into a **depth-first command sequence**
``C = [t_1, ..., t_n]`` in which dedicated *end tokens* preserve the hierarchical
closure required for reversible serialization (Sec. A.4). The reserved control
IDs follow Table S1 exactly::

    pad=0  cls=1  esolid=2  esketch=3  eface=4  eloop=5  ec=6  ee=7

Continuous parameters are 8-bit uniformly quantized into ``[11, 266]``.

This module is the fully deterministic, network-agnostic core of that
representation:

* :class:`Curve`, :class:`Loop`, :class:`Face`, :class:`Sketch`,
  :class:`Extrusion`, :class:`SePair`, :class:`Solid` -- the typed tree;
* :func:`serialize` -- depth-first traversal to a token sequence with the
  Table S1 end tokens (top-down, matching "Noise is injected following the
  top-down traversal of the CAD hierarchy");
* :func:`deserialize` -- the exact inverse, rebuilding the tree from the token
  sequence (the paper's "reversible serialization");
* :func:`quantize` / :func:`dequantize` -- the 8-bit ``[11, 266]`` mapping;
* structural helpers (:func:`tree_depth`, :func:`count_nodes`,
  :func:`type_paths`) used by the structure-consistency metric.

Everything is stdlib-only, pure and deterministic. The learned G-Mamba encoder
and diffusion denoiser are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Table S1 reserved control tokens ---------------------------------------
PAD = 0
CLS = 1
ESOLID = 2
ESKETCH = 3
EFACE = 4
ELOOP = 5
EC = 6
EE = 7

# Continuous-parameter quantization range (8-bit into [11, 266]).
QUANT_LO = 11
QUANT_HI = 266
QUANT_LEVELS = QUANT_HI - QUANT_LO  # 255 -> 256 distinct integer codes


def quantize(value: float) -> int:
    """8-bit uniform quantization of ``value in [0, 1]`` into ``[11, 266]``.

    Values outside ``[0, 1]`` are clamped, matching the paper's fixed-range
    tokenization (Sec. A.1.3).
    """
    if value <= 0.0:
        return QUANT_LO
    if value >= 1.0:
        return QUANT_HI
    return QUANT_LO + int(round(value * QUANT_LEVELS))


def dequantize(code: int) -> float:
    """Inverse of :func:`quantize`: map an integer code in ``[11, 266]`` to
    ``[0, 1]`` (the bin centre representative)."""
    if code < QUANT_LO or code > QUANT_HI:
        raise ValueError(f"code {code} outside quantization range [{QUANT_LO}, {QUANT_HI}]")
    return (code - QUANT_LO) / QUANT_LEVELS


# --- typed tree -------------------------------------------------------------

@dataclass(frozen=True)
class Curve:
    """A sketch primitive. ``kind`` in {'line', 'arc', 'circle'}; ``params`` are
    already-quantized integer coordinates in ``[11, 266]`` (their meaning follows
    Table S1: line=(x1,y1,x2,y2), arc=(x1,y1,xm,ym,x2,y2), circle=(cx,cy,r))."""
    kind: str
    params: tuple[int, ...]


@dataclass(frozen=True)
class Loop:
    curves: tuple[Curve, ...]


@dataclass(frozen=True)
class Face:
    loops: tuple[Loop, ...]


@dataclass(frozen=True)
class Sketch:
    faces: tuple[Face, ...]


@dataclass(frozen=True)
class Extrusion:
    """Ten SE extrusion parameters (Sec. 3.2 / A.1.2), pre-quantized:
    ``(theta, phi, gamma, tx, ty, tz, sigma, dplus, dminus, beta)``. ``beta`` is
    the boolean-operation categorical token in ``{7, 8, 9, 10}``."""
    params: tuple[int, ...]


@dataclass(frozen=True)
class SePair:
    sketch: Sketch
    extrusion: Extrusion


@dataclass(frozen=True)
class Solid:
    pairs: tuple[SePair, ...]


# --- serialization ----------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """A serialized command. ``kind`` is one of {'ctl', 'line', 'arc',
    'circle', 'ext'}; for a control token ``payload`` is the reserved integer ID
    (PAD..EE), otherwise it is the tuple of quantized parameters."""
    kind: str
    payload: object


_CURVE_KINDS = ("line", "arc", "circle")


def serialize(solid: Solid) -> tuple[Token, ...]:
    """Depth-first (top-down) serialization of a :class:`Solid` tree into a
    token sequence with Table S1 hierarchical end tokens.

    Order per SE pair: emit the sketch subtree (curves closed by ``ec``, loops by
    ``eloop``, faces by ``eface``, the sketch by ``esketch``), then the extrusion
    params closed by ``ee``. After all pairs, the whole solid is closed by
    ``esolid``. The very first token is ``cls``.
    """
    out: list[Token] = [Token("ctl", CLS)]
    for pair in solid.pairs:
        for face in pair.sketch.faces:
            for loop in face.loops:
                for curve in loop.curves:
                    if curve.kind not in _CURVE_KINDS:
                        raise ValueError(f"unknown curve kind {curve.kind!r}")
                    out.append(Token(curve.kind, tuple(curve.params)))
                    out.append(Token("ctl", EC))
                out.append(Token("ctl", ELOOP))
            out.append(Token("ctl", EFACE))
        out.append(Token("ctl", ESKETCH))
        out.append(Token("ext", tuple(pair.extrusion.params)))
        out.append(Token("ctl", EE))
    out.append(Token("ctl", ESOLID))
    return tuple(out)


def deserialize(tokens: tuple[Token, ...]) -> Solid:
    """Exact inverse of :func:`serialize` -- rebuild the :class:`Solid` tree.

    Raises :class:`ValueError` if the end-token nesting is malformed (i.e. the
    sequence is not a valid hierarchical closure).
    """
    it = iter(tokens)
    seq = list(it)
    i = 0
    n = len(seq)

    def expect_ctl(idx: int, cid: int) -> None:
        if idx >= n or seq[idx].kind != "ctl" or seq[idx].payload != cid:
            raise ValueError(f"expected control token {cid} at position {idx}")

    if n == 0 or seq[0].kind != "ctl" or seq[0].payload != CLS:
        raise ValueError("sequence must start with cls")
    i = 1

    pairs: list[SePair] = []
    while i < n and not (seq[i].kind == "ctl" and seq[i].payload == ESOLID):
        # --- sketch: faces -> loops -> curves ---
        faces: list[Face] = []
        while not (seq[i].kind == "ctl" and seq[i].payload == ESKETCH):
            loops: list[Loop] = []
            while not (seq[i].kind == "ctl" and seq[i].payload == EFACE):
                curves: list[Curve] = []
                while not (seq[i].kind == "ctl" and seq[i].payload == ELOOP):
                    tok = seq[i]
                    if tok.kind not in _CURVE_KINDS:
                        raise ValueError(f"expected a curve at position {i}, got {tok.kind}")
                    i += 1
                    expect_ctl(i, EC)
                    i += 1
                    curves.append(Curve(tok.kind, tuple(tok.payload)))
                i += 1  # consume eloop
                loops.append(Loop(tuple(curves)))
            i += 1  # consume eface
            faces.append(Face(tuple(loops)))
        i += 1  # consume esketch
        # --- extrusion ---
        if i >= n or seq[i].kind != "ext":
            raise ValueError(f"expected extrusion at position {i}")
        ext = Extrusion(tuple(seq[i].payload))
        i += 1
        expect_ctl(i, EE)
        i += 1
        pairs.append(SePair(Sketch(tuple(faces)), ext))
    expect_ctl(i, ESOLID)
    return Solid(tuple(pairs))


# --- structural helpers -----------------------------------------------------

def count_nodes(solid: Solid) -> dict[str, int]:
    """Count nodes of each type in the tree (for structure metrics)."""
    counts = {"solid": 1, "sketch": 0, "face": 0, "loop": 0, "curve": 0,
              "extrusion": 0}
    for pair in solid.pairs:
        counts["sketch"] += 1
        counts["extrusion"] += 1
        for face in pair.sketch.faces:
            counts["face"] += 1
            for loop in face.loops:
                counts["loop"] += 1
                counts["curve"] += len(loop.curves)
    return counts


def tree_depth(solid: Solid) -> int:
    """Maximum root-to-leaf depth (solid=1, sketch=2, face=3, loop=4, curve=5).

    An empty solid has depth 1; a solid whose deepest branch reaches a curve has
    depth 5.
    """
    depth = 1
    for pair in solid.pairs:
        depth = max(depth, 2)  # sketch / extrusion level
        for face in pair.sketch.faces:
            depth = max(depth, 3)
            for loop in face.loops:
                depth = max(depth, 4)
                if loop.curves:
                    depth = max(depth, 5)
    return depth


def type_paths(solid: Solid) -> tuple[tuple[str, ...], ...]:
    """All parent->child type paths from the root (used by structure-F1).

    Each path is a tuple of type names, e.g. ``('solid', 'sketch', 'face',
    'loop', 'line')``. Sibling ordering is preserved by index so structurally
    identical trees yield identical multisets of paths.
    """
    paths: list[tuple[str, ...]] = []
    for pi, pair in enumerate(solid.pairs):
        base = ("solid", f"pair{pi}")
        paths.append(base + ("extrusion",))
        for fi, face in enumerate(pair.sketch.faces):
            fbase = base + ("sketch", f"face{fi}")
            paths.append(fbase)
            for li, loop in enumerate(face.loops):
                lbase = fbase + (f"loop{li}",)
                paths.append(lbase)
                for ci, curve in enumerate(loop.curves):
                    paths.append(lbase + (f"c{ci}:{curve.kind}",))
    return tuple(paths)
