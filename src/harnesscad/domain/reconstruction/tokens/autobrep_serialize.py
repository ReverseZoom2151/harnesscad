"""AutoBrep B-rep tokenisation vocabulary and hierarchical serialisation format.

Deterministic re-encoding of the token *format* used by AutoBrep (a multimodal
autoregressive B-rep generator). AutoBrep flattens a solid's boundary
representation into a single integer token stream over a contiguous vocabulary.
The trained FSQ codebooks that supply the *values* of geometry codes are a model
artifact, but the vocabulary layout and the nested container grammar are a fixed,
config-derived specification -- built here in full.

VOCABULARY LAYOUT (contiguous integer ranges)
---------------------------------------------
AutoBrep's ``AutoBrepModel`` (``models/autoregressive.py``) composes the vocabulary
from four blocks in this exact order, parameterised by ``bit`` (position
quantisation bits), ``max_face`` (id capacity) and two codebook sizes::

    flag_pad   = len(MMTokenIndex)            # special/structural flags
    pos_pad    = 2 ** bit                     # quantised position-bit values
    id_pad     = max_face                     # face / edge id slots
    face_z_pad = flag_pad + pos_pad + id_pad  # start of the surface codebook
    edge_off   = face_z_pad + surf_codebook   # start of the edge codebook
    num_tokens = face_z_pad + surf_codebook + edge_codebook

so the ranges are::

    [0,          flag_pad)               -> SPECIAL  (BOS, EOC, BOF, ...)
    [flag_pad,   flag_pad+pos_pad)       -> POSITION (a quantised bbox coordinate)
    [.. ,        face_z_pad)             -> ID       (a face or edge slot index)
    [face_z_pad, face_z_pad+surf)        -> SURFACE  (a surface geometry code)
    [.. ,        num_tokens)             -> EDGE     (an edge geometry code)

The 21 structural flags come from ``data/token_mapping.py`` (``MMTokenIndex``):
begin/end markers for the whole sequence (BOS/EOS), text (BOT/EOT), a CAD B-rep
(BOC/EOC), a level (BOL/EOL), a face (BOF/EOF), a geometry prompt (BOGEOM/EOGEOM),
meta (BOM/EOM), a point cloud (BOPC/EOPC), four complexity tokens and a dummy.

CONTAINER GRAMMAR
-----------------
A B-rep is emitted as a nested, bracketed stream::

    BOC
      BOL                              # a level (a topological shell / group)
        BOF <face-id> <pos>* <surf>    # a face: id, bbox position bits, geom code
            [ <edge-id> <pos>* <edge> ]*   # its edges: id, position bits, geom code
        EOF
        ...
      EOL
      ...
    EOC

The stream is unambiguous because a face's geometry code lives in the SURFACE
range while an edge's lives in the EDGE range: the parser reads position tokens
greedily (POSITION range), a SURFACE token closes a face header, and each edge
group opens with an ID token and closes with an EDGE token.

Position quantisation mirrors ``utils.quantize`` (round-half-to-even mapping of a
``[-1, 1]`` coordinate to ``[0, 2**bit - 1]``) and its inverse ``dequantize``.

Stdlib only, deterministic. Generation of geometry-code *values* is external;
this module owns the layout, the (de)serialisation and structural validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Sequence

__all__ = [
    "MMTokenIndex",
    "AutoBrepVocabulary",
    "TokenKind",
    "Edge",
    "Face",
    "Level",
    "BrepSequence",
    "quantize_coord",
    "dequantize_coord",
    "serialize",
    "parse",
    "validate_tokens",
    "classify_token",
]


class MMTokenIndex(IntEnum):
    """AutoBrep structural flag tokens (``data/token_mapping.py``)."""

    BOS = 0
    EOS = 1
    BOT = 2
    EOT = 3
    BOC = 4
    EOC = 5
    BOL = 6
    EOL = 7
    BOF = 8
    EOF = 9
    BOGEOM = 10
    EOGEOM = 11
    BOM = 12
    EOM = 13
    GEN_EASY = 14
    GEN_MID = 15
    GEN_HARD = 16
    GEN_UNCOND = 17
    BOPC = 18
    EOPC = 19
    DUMMYID = 20


class TokenKind:
    """String tags for the five vocabulary blocks."""

    SPECIAL = "special"
    POSITION = "position"
    ID = "id"
    SURFACE = "surface"
    EDGE = "edge"


@dataclass(frozen=True)
class AutoBrepVocabulary:
    """The contiguous integer vocabulary layout, from AutoBrep's hyperparameters.

    Defaults reproduce AutoBrep's shipped configuration (``bit=10``,
    ``max_face=100``, ``surf_codebook_size = edge_codebook_size = 10000``).
    """

    bit: int = 10
    max_face: int = 100
    surf_codebook_size: int = 10000
    edge_codebook_size: int = 10000

    # --- derived block sizes / offsets (AutoBrepModel.__init__) ---
    @property
    def flag_pad(self) -> int:
        return len(MMTokenIndex.__members__)

    @property
    def pos_pad(self) -> int:
        return 2 ** self.bit

    @property
    def id_pad(self) -> int:
        return self.max_face

    @property
    def pos_offset(self) -> int:
        return self.flag_pad

    @property
    def id_offset(self) -> int:
        return self.flag_pad + self.pos_pad

    @property
    def face_z_pad(self) -> int:
        """Start of the surface codebook (== flag+pos+id block sizes)."""
        return self.flag_pad + self.pos_pad + self.id_pad

    @property
    def surf_offset(self) -> int:
        return self.face_z_pad

    @property
    def edge_offset(self) -> int:
        return self.face_z_pad + self.surf_codebook_size

    @property
    def num_tokens(self) -> int:
        return self.face_z_pad + self.surf_codebook_size + self.edge_codebook_size

    # --- token encoders (value -> global token id) ---
    def flag(self, f: MMTokenIndex) -> int:
        return int(f)

    def pos_token(self, q: int) -> int:
        if not 0 <= q < self.pos_pad:
            raise ValueError(f"position value {q} out of range [0, {self.pos_pad})")
        return self.pos_offset + q

    def id_token(self, i: int) -> int:
        if not 0 <= i < self.id_pad:
            raise ValueError(f"id {i} out of range [0, {self.id_pad})")
        return self.id_offset + i

    def surf_token(self, code: int) -> int:
        if not 0 <= code < self.surf_codebook_size:
            raise ValueError(f"surface code {code} out of range")
        return self.surf_offset + code

    def edge_token(self, code: int) -> int:
        if not 0 <= code < self.edge_codebook_size:
            raise ValueError(f"edge code {code} out of range")
        return self.edge_offset + code

    # --- token classifier (global token id -> (kind, value)) ---
    def classify(self, tok: int) -> tuple[str, int]:
        if not 0 <= tok < self.num_tokens:
            raise ValueError(f"token {tok} out of vocabulary [0, {self.num_tokens})")
        if tok < self.flag_pad:
            return (TokenKind.SPECIAL, tok)
        if tok < self.id_offset:
            return (TokenKind.POSITION, tok - self.pos_offset)
        if tok < self.face_z_pad:
            return (TokenKind.ID, tok - self.id_offset)
        if tok < self.edge_offset:
            return (TokenKind.SURFACE, tok - self.surf_offset)
        return (TokenKind.EDGE, tok - self.edge_offset)

    def ranges(self) -> dict:
        """Human-readable half-open ranges for each vocabulary block."""
        return {
            TokenKind.SPECIAL: (0, self.flag_pad),
            TokenKind.POSITION: (self.pos_offset, self.id_offset),
            TokenKind.ID: (self.id_offset, self.face_z_pad),
            TokenKind.SURFACE: (self.surf_offset, self.edge_offset),
            TokenKind.EDGE: (self.edge_offset, self.num_tokens),
        }


def quantize_coord(x: float, bit: int = 10, lo: float = -1.0, hi: float = 1.0) -> int:
    """Quantise a coordinate in ``[lo, hi]`` to ``[0, 2**bit - 1]`` (AutoBrep utils.quantize)."""
    rng = 2 ** bit - 1
    q = (x - lo) * rng / (hi - lo)
    if q < 0.0:
        q = 0.0
    elif q > rng:
        q = float(rng)
    # round-half-to-even, matching numpy's default .round() used with apply_round.
    return int(round(q))


def dequantize_coord(q: int, bit: int = 10, lo: float = -1.0, hi: float = 1.0) -> float:
    """Inverse of :func:`quantize_coord` (AutoBrep utils.dequantize)."""
    rng = 2 ** bit - 1
    return q * (hi - lo) / rng + lo


@dataclass(frozen=True)
class Edge:
    """An edge inside a face header: an id, quantised bbox position bits, a geom code."""

    edge_id: int
    pos: tuple[int, ...]
    code: int


@dataclass(frozen=True)
class Face:
    """A face: id, quantised bbox position bits, a surface code and its edges."""

    face_id: int
    pos: tuple[int, ...]
    code: int
    edges: tuple[Edge, ...] = ()


@dataclass(frozen=True)
class Level:
    """A level (a topological group of faces) inside a B-rep."""

    faces: tuple[Face, ...]


@dataclass(frozen=True)
class BrepSequence:
    """A whole B-rep as an ordered list of levels."""

    levels: tuple[Level, ...]


def serialize(brep: BrepSequence, vocab: AutoBrepVocabulary | None = None) -> list[int]:
    """Flatten a :class:`BrepSequence` into AutoBrep's bracketed token stream."""
    v = vocab or AutoBrepVocabulary()
    out: list[int] = [v.flag(MMTokenIndex.BOC)]
    for level in brep.levels:
        out.append(v.flag(MMTokenIndex.BOL))
        for face in level.faces:
            out.append(v.flag(MMTokenIndex.BOF))
            out.append(v.id_token(face.face_id))
            for q in face.pos:
                out.append(v.pos_token(q))
            out.append(v.surf_token(face.code))
            for edge in face.edges:
                out.append(v.id_token(edge.edge_id))
                for q in edge.pos:
                    out.append(v.pos_token(q))
                out.append(v.edge_token(edge.code))
            out.append(v.flag(MMTokenIndex.EOF))
        out.append(v.flag(MMTokenIndex.EOL))
    out.append(v.flag(MMTokenIndex.EOC))
    return out


class _Cursor:
    __slots__ = ("toks", "i")

    def __init__(self, toks: Sequence[int]):
        self.toks = toks
        self.i = 0

    def peek(self, vocab: AutoBrepVocabulary):
        if self.i >= len(self.toks):
            return (None, None)
        return vocab.classify(self.toks[self.i])

    def take(self):
        t = self.toks[self.i]
        self.i += 1
        return t


def parse(tokens: Sequence[int], vocab: AutoBrepVocabulary | None = None) -> BrepSequence:
    """Reconstruct a :class:`BrepSequence` from a token stream (inverse of :func:`serialize`)."""
    v = vocab or AutoBrepVocabulary()
    cur = _Cursor(tokens)

    def expect_flag(flag: MMTokenIndex) -> None:
        kind, val = cur.peek(v)
        if kind != TokenKind.SPECIAL or val != int(flag):
            raise ValueError(f"expected {flag.name} at position {cur.i}, got {kind}:{val}")
        cur.take()

    def read_pos() -> tuple[int, ...]:
        pos: list[int] = []
        while True:
            kind, val = cur.peek(v)
            if kind != TokenKind.POSITION:
                break
            pos.append(val)
            cur.take()
        return tuple(pos)

    expect_flag(MMTokenIndex.BOC)
    levels: list[Level] = []
    while True:
        kind, val = cur.peek(v)
        if kind == TokenKind.SPECIAL and val == int(MMTokenIndex.EOC):
            cur.take()
            break
        expect_flag(MMTokenIndex.BOL)
        faces: list[Face] = []
        while True:
            kind, val = cur.peek(v)
            if kind == TokenKind.SPECIAL and val == int(MMTokenIndex.EOL):
                cur.take()
                break
            expect_flag(MMTokenIndex.BOF)
            # face header: id, pos*, surface code
            kind, fid = cur.peek(v)
            if kind != TokenKind.ID:
                raise ValueError(f"expected face id at {cur.i}, got {kind}")
            cur.take()
            face_pos = read_pos()
            kind, scode = cur.peek(v)
            if kind != TokenKind.SURFACE:
                raise ValueError(f"expected surface code at {cur.i}, got {kind}")
            cur.take()
            # edges: (id, pos*, edge code)* until EOF
            edges: list[Edge] = []
            while True:
                kind, eid = cur.peek(v)
                if kind == TokenKind.SPECIAL and eid == int(MMTokenIndex.EOF):
                    cur.take()
                    break
                if kind != TokenKind.ID:
                    raise ValueError(f"expected edge id or EOF at {cur.i}, got {kind}")
                cur.take()
                edge_pos = read_pos()
                kind, ecode = cur.peek(v)
                if kind != TokenKind.EDGE:
                    raise ValueError(f"expected edge code at {cur.i}, got {kind}")
                cur.take()
                edges.append(Edge(edge_id=eid, pos=edge_pos, code=ecode))
            faces.append(Face(face_id=fid, pos=face_pos, code=scode, edges=tuple(edges)))
        levels.append(Level(faces=tuple(faces)))
    return BrepSequence(levels=tuple(levels))


def validate_tokens(tokens: Sequence[int], vocab: AutoBrepVocabulary | None = None) -> bool:
    """True if every token is in-vocabulary and the stream round-trips through parse/serialize."""
    v = vocab or AutoBrepVocabulary()
    for t in tokens:
        v.classify(t)  # raises if out of range
    brep = parse(tokens, v)
    return serialize(brep, v) == list(tokens)


def classify_token(tok: int, vocab: AutoBrepVocabulary | None = None) -> tuple[str, int]:
    """Return ``(kind, value)`` for a single global token id."""
    return (vocab or AutoBrepVocabulary()).classify(tok)
