"""Hierarchy-aware CAD -> structured-text serialisation (FlexCAD, Zhang et al. 2024).

FlexCAD ("Unified and Versatile Controllable CAD Generation with Fine-Tuned Large
Language Models") represents a sketch-and-extrude CAD model as a *concise, structured
text* by abstracting each construction hierarchy as a sequence of text tokens
(paper Sec. 3.1, Fig. 3). The construction hierarchy is::

    model  ->  sketch-extrusion (SE)  ->  {sketch, extrusion}
    sketch ->  face  ->  loop  ->  curve  (line | arc | circle)

Faithful text-representation rules (paper Sec. 3.1 + appendix A.1):

* The curve *type* (line / arc / circle) is emitted directly as a textual token.
* Numerical geometry (point coordinates) is expressed as **decimal integers** and
  then as textual tokens -- e.g. the centre ``(31, 31)`` -- rather than the binary
  one-hot ``([0,1,1,1,1,1],[0,1,1,1,1,1])`` used by SkexGen.
* A special ``<H>_end`` token marks the end of every hierarchy,
  ``H in {curve, loop, face, sketch, extrusion}``, instead of one-hot ending flags.
* Tokens of the finer hierarchy are concatenated to form the coarser one; a model is
  the concatenation of its sketch-extrusions.
* An extrusion is a *boolean op* token (add / cut / intersect) followed by its
  numerical attributes (appendix A.1: ``B V V T T T R..R S O O`` = 18 params, the
  op being ``B``; here the op is the token and the remaining 17 are integer attrs).

This module is the **deterministic** core FlexCAD idea: the CAD<->text serialiser and
its round-trip parser, plus a low-level *field masker* (replace one hierarchy field
with a mask token, keep the removed tokens as the infill answer) that the
hierarchy-aware masking scheme and the training-pair constructor build upon. The LLM
fine-tuning itself is out of scope.

Pure stdlib, no wall-clock, fully round-trippable.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- vocabulary ------------------------------------------------------------
LINE = "line"
ARC = "arc"
CIRCLE = "circle"
CURVE_TYPES: tuple[str, ...] = (LINE, ARC, CIRCLE)

# Boolean operations for an extrusion (appendix A.1 "B").
ADD = "add"
CUT = "cut"
INTERSECT = "intersect"
EXTRUSION_OPS: tuple[str, ...] = (ADD, CUT, INTERSECT)

# Hierarchy end-of-field markers.
CURVE_END = "<curve_end>"
LOOP_END = "<loop_end>"
FACE_END = "<face_end>"
SKETCH_END = "<sketch_end>"
EXTRUSION_END = "<extrusion_end>"

# Hierarchy-aware mask tokens (paper Sec. 3.2, Fig. 4). Curve masks are *typed*.
SE_MASK = "[sketch-extrusionmask]"
SKETCH_MASK = "[sketchmask]"
EXTRUSION_MASK = "[extrusionmask]"
FACE_MASK = "[facemask]"
LOOP_MASK = "[loopmask]"
CURVE_MASK: dict[str, str] = {
    LINE: "[linemask]",
    ARC: "[arcmask]",
    CIRCLE: "[circlemask]",
}

ALL_MASK_TOKENS: frozenset[str] = frozenset(
    {SE_MASK, SKETCH_MASK, EXTRUSION_MASK, FACE_MASK, LOOP_MASK, *CURVE_MASK.values()}
)


# --- hierarchical data model ----------------------------------------------
@dataclass(frozen=True)
class Curve:
    """A base-level curve: ``type in {line, arc, circle}`` + integer coordinates.

    Per appendix A.1 a line / arc / circle is defined by 1 / 2 / 4 points; each
    point contributes an (x, y) pair, so ``coords`` holds 2 / 4 / 8 integers. The
    representation is stored explicitly and is not constrained here (the serialiser
    round-trips whatever integers are given).
    """

    type: str
    coords: tuple[int, ...] = ()

    def __post_init__(self):
        if self.type not in CURVE_TYPES:
            raise ValueError(f"unknown curve type: {self.type!r}")


@dataclass(frozen=True)
class Loop:
    """A closed path: one curve (circle) or several (e.g. line-arc-line)."""

    curves: tuple[Curve, ...]


@dataclass(frozen=True)
class Face:
    """A 2D area: an outer loop plus optional inner loops (holes)."""

    loops: tuple[Loop, ...]


@dataclass(frozen=True)
class Sketch:
    """One or more faces sharing a common extrusion command."""

    faces: tuple[Face, ...]


@dataclass(frozen=True)
class Extrusion:
    """A boolean op (add/cut/intersect) plus integer numerical attributes."""

    op: str
    params: tuple[int, ...] = ()

    def __post_init__(self):
        if self.op not in EXTRUSION_OPS:
            raise ValueError(f"unknown extrusion op: {self.op!r}")


@dataclass(frozen=True)
class SketchExtrusion:
    """A single sketch-extrusion 3D body (SE)."""

    sketch: Sketch
    extrusion: Extrusion


@dataclass(frozen=True)
class CADModel:
    """A CAD model M = one or more sketch-extrusion entities."""

    ses: tuple[SketchExtrusion, ...]


# --- convenience constructors (accept plain lists/tuples) ------------------
def curve(ctype: str, *coords: int) -> Curve:
    return Curve(ctype, tuple(int(c) for c in coords))


def loop(*curves: Curve) -> Loop:
    return Loop(tuple(curves))


def face(*loops: Loop) -> Face:
    return Face(tuple(loops))


def sketch(*faces: Face) -> Sketch:
    return Sketch(tuple(faces))


def extrusion(op: str, *params: int) -> Extrusion:
    return Extrusion(op, tuple(int(p) for p in params))


def se(sketch_: Sketch, extrusion_: Extrusion) -> SketchExtrusion:
    return SketchExtrusion(sketch_, extrusion_)


def model(*ses: SketchExtrusion) -> CADModel:
    return CADModel(tuple(ses))


# --- token-level emit helpers ---------------------------------------------
def _emit_curve(out: list[str], c: Curve) -> None:
    out.append(c.type)
    out.extend(str(int(v)) for v in c.coords)
    out.append(CURVE_END)


def _emit_loop(out: list[str], lp: Loop) -> None:
    for c in lp.curves:
        _emit_curve(out, c)
    out.append(LOOP_END)


def _emit_face(out: list[str], f: Face) -> None:
    for lp in f.loops:
        _emit_loop(out, lp)
    out.append(FACE_END)


def _emit_sketch(out: list[str], sk: Sketch) -> None:
    for f in sk.faces:
        _emit_face(out, f)
    out.append(SKETCH_END)


def _emit_extrusion(out: list[str], ex: Extrusion) -> None:
    out.append(ex.op)
    out.extend(str(int(v)) for v in ex.params)
    out.append(EXTRUSION_END)


def _emit_se(out: list[str], s: SketchExtrusion) -> None:
    _emit_sketch(out, s.sketch)
    _emit_extrusion(out, s.extrusion)


def tokenize(m: CADModel) -> list[str]:
    """Serialise a CAD model to its flat list of structured-text tokens."""
    out: list[str] = []
    for s in m.ses:
        _emit_se(out, s)
    return out


def serialize(m: CADModel) -> str:
    """Serialise a CAD model to a single space-separated structured-text string."""
    return " ".join(tokenize(m))


# --- parsing (round-trip inverse) -----------------------------------------
class ParseError(ValueError):
    """Raised when a structured-text token stream is not well-formed."""


def _parse_curve(tokens: list[str], i: int) -> tuple[Curve, int]:
    ctype = tokens[i]
    if ctype not in CURVE_TYPES:
        raise ParseError(f"expected curve type at {i}, got {ctype!r}")
    i += 1
    coords: list[int] = []
    while i < len(tokens) and tokens[i] != CURVE_END:
        try:
            coords.append(int(tokens[i]))
        except ValueError:
            raise ParseError(f"expected integer coord at {i}, got {tokens[i]!r}")
        i += 1
    if i >= len(tokens):
        raise ParseError("unterminated curve (missing <curve_end>)")
    return Curve(ctype, tuple(coords)), i + 1  # skip CURVE_END


def _parse_loop(tokens: list[str], i: int) -> tuple[Loop, int]:
    curves: list[Curve] = []
    while i < len(tokens) and tokens[i] != LOOP_END:
        c, i = _parse_curve(tokens, i)
        curves.append(c)
    if i >= len(tokens):
        raise ParseError("unterminated loop (missing <loop_end>)")
    return Loop(tuple(curves)), i + 1


def _parse_face(tokens: list[str], i: int) -> tuple[Face, int]:
    loops: list[Loop] = []
    while i < len(tokens) and tokens[i] != FACE_END:
        lp, i = _parse_loop(tokens, i)
        loops.append(lp)
    if i >= len(tokens):
        raise ParseError("unterminated face (missing <face_end>)")
    return Face(tuple(loops)), i + 1


def _parse_sketch(tokens: list[str], i: int) -> tuple[Sketch, int]:
    faces: list[Face] = []
    while i < len(tokens) and tokens[i] != SKETCH_END:
        f, i = _parse_face(tokens, i)
        faces.append(f)
    if i >= len(tokens):
        raise ParseError("unterminated sketch (missing <sketch_end>)")
    return Sketch(tuple(faces)), i + 1


def _parse_extrusion(tokens: list[str], i: int) -> tuple[Extrusion, int]:
    if i >= len(tokens):
        raise ParseError("expected extrusion op, reached end")
    op = tokens[i]
    if op not in EXTRUSION_OPS:
        raise ParseError(f"expected extrusion op at {i}, got {op!r}")
    i += 1
    params: list[int] = []
    while i < len(tokens) and tokens[i] != EXTRUSION_END:
        try:
            params.append(int(tokens[i]))
        except ValueError:
            raise ParseError(f"expected integer attr at {i}, got {tokens[i]!r}")
        i += 1
    if i >= len(tokens):
        raise ParseError("unterminated extrusion (missing <extrusion_end>)")
    return Extrusion(op, tuple(params)), i + 1


def parse_tokens(tokens: list[str]) -> CADModel:
    """Inverse of :func:`tokenize`: parse tokens back into a :class:`CADModel`."""
    ses: list[SketchExtrusion] = []
    i = 0
    n = len(tokens)
    while i < n:
        sk, i = _parse_sketch(tokens, i)
        ex, i = _parse_extrusion(tokens, i)
        ses.append(SketchExtrusion(sk, ex))
    if not ses:
        raise ParseError("empty model")
    return CADModel(tuple(ses))


def parse(text: str) -> CADModel:
    """Inverse of :func:`serialize`."""
    tokens = text.split()
    if not tokens:
        raise ParseError("empty text")
    return parse_tokens(tokens)


# --- low-level hierarchy-field masker --------------------------------------
# Levels addressable by the masking scheme (paper Sec. 3.2).
LEVEL_CAD = "cad"
LEVEL_SE = "sketch_extrusion"
LEVEL_SKETCH = "sketch"
LEVEL_EXTRUSION = "extrusion"
LEVEL_FACE = "face"
LEVEL_LOOP = "loop"
LEVEL_CURVE = "curve"

LEVELS: tuple[str, ...] = (
    LEVEL_CAD, LEVEL_SE, LEVEL_SKETCH, LEVEL_EXTRUSION,
    LEVEL_FACE, LEVEL_LOOP, LEVEL_CURVE,
)


@dataclass(frozen=True)
class MaskTarget:
    """Address of the hierarchy field to mask.

    ``se`` / ``face`` / ``loop`` are 0-based indices into the enclosing hierarchy;
    unused indices are ignored per level. ``LEVEL_CAD`` addresses the whole model.
    """

    level: str
    se: int = 0
    face: int = 0
    loop: int = 0


@dataclass(frozen=True)
class MaskResult:
    """Outcome of masking one field.

    ``instruction`` is the token stream with the field replaced by ``mask``
    token(s); ``answer`` is the removed field's tokens. Infilling ``instruction``
    by substituting ``answer`` for the mask run reproduces the original tokens.
    """

    instruction: tuple[str, ...]
    answer: tuple[str, ...]
    mask: tuple[str, ...]
    level: str


def _curve_run_tokens(lp: Loop) -> list[str]:
    out: list[str] = []
    for c in lp.curves:
        _emit_curve(out, c)
    return out


def mask_field(m: CADModel, target: MaskTarget) -> MaskResult:
    """Replace the addressed hierarchy field with its mask token(s).

    Returns a :class:`MaskResult`. The field's normal serialisation becomes the
    ``answer``; the surrounding tokens are preserved verbatim, so
    ``instruction`` with ``mask`` swapped back for ``answer`` == :func:`tokenize`.
    """
    if target.level not in LEVELS:
        raise ValueError(f"unknown level: {target.level!r}")

    tokens = tokenize(m)

    # Locate the field's [start, end) span in the flat token stream and its mask.
    prefix: list[str] = []

    def se_span(idx: int) -> tuple[int, int]:
        start = 0
        for k in range(idx):
            buf: list[str] = []
            _emit_se(buf, m.ses[k])
            start += len(buf)
        buf = []
        _emit_se(buf, m.ses[idx])
        return start, start + len(buf)

    if target.level == LEVEL_CAD:
        span = (0, len(tokens))
        mask = tuple(SE_MASK for _ in m.ses)
    elif target.level == LEVEL_SE:
        start, end = se_span(target.se)
        span = (start, end)
        mask = (SE_MASK,)
    else:
        base, _ = se_span(target.se)
        s = m.ses[target.se]
        # tokens of the SE: [ sketch_tokens..., extrusion_tokens... ]
        sk_buf: list[str] = []
        _emit_sketch(sk_buf, s.sketch)
        sketch_len = len(sk_buf)
        if target.level == LEVEL_SKETCH:
            span = (base, base + sketch_len)
            mask = (SKETCH_MASK,)
        elif target.level == LEVEL_EXTRUSION:
            ex_buf: list[str] = []
            _emit_extrusion(ex_buf, s.extrusion)
            span = (base + sketch_len, base + sketch_len + len(ex_buf))
            mask = (EXTRUSION_MASK,)
        else:
            # face / loop / curve: locate within the sketch token span.
            f_start = base
            for j in range(target.face):
                fb: list[str] = []
                _emit_face(fb, s.sketch.faces[j])
                f_start += len(fb)
            f = s.sketch.faces[target.face]
            if target.level == LEVEL_FACE:
                fb = []
                _emit_face(fb, f)
                span = (f_start, f_start + len(fb))
                mask = (FACE_MASK,)
            else:
                l_start = f_start
                for k in range(target.loop):
                    lb: list[str] = []
                    _emit_loop(lb, f.loops[k])
                    l_start += len(lb)
                lp = f.loops[target.loop]
                if target.level == LEVEL_LOOP:
                    lb = []
                    _emit_loop(lb, lp)
                    span = (l_start, l_start + len(lb))
                    mask = (LOOP_MASK,)
                else:  # LEVEL_CURVE: mask the run of curve tokens (keep <loop_end>)
                    run = _curve_run_tokens(lp)
                    span = (l_start, l_start + len(run))
                    mask = tuple(CURVE_MASK[c.type] for c in lp.curves)

    prefix = tokens[: span[0]]
    answer = tokens[span[0]: span[1]]
    suffix = tokens[span[1]:]
    instruction = tuple(prefix) + mask + tuple(suffix)
    return MaskResult(instruction, tuple(answer), mask, target.level)


def infill(instruction: tuple[str, ...], answer: tuple[str, ...],
           mask: tuple[str, ...]) -> list[str]:
    """Reconstruct the full token stream by swapping ``answer`` for the mask run.

    Locates the contiguous run of ``mask`` tokens in ``instruction`` and replaces
    it with ``answer``. Inverse companion of :func:`mask_field`.
    """
    instr = list(instruction)
    n = len(mask)
    for i in range(len(instr) - n + 1):
        if tuple(instr[i:i + n]) == tuple(mask):
            return instr[:i] + list(answer) + instr[i + n:]
    raise ParseError("mask run not found in instruction")
