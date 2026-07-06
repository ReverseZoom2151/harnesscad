"""LEGO(R) brick library and the LEGOGPT plain-text design format.

Distilled from Pun, Deng, Liu et al., *Generating Physically Stable and
Buildable LEGO Designs from Text* (LEGOGPT, CMU 2025), Sections 3-4 and the
appendix training details.

This module captures the parts of that paper that are *LEGO-specific* and that
the sibling generic-brick modules (``brick_*.py`` from the companion paper
"...Buildable Brick Structures from Text") do **not** cover:

* the fixed inventory of eight commonly-available standard bricks
  (1x1, 1x2, 1x4, 1x6, 1x8, 2x2, 2x4, 2x6), all one unit tall;
* the compact custom serialization the paper introduces to *replace* LDraw --
  one line per brick ``"{h}x{w} ({x},{y},{z})"`` -- where the **order of h and
  w encodes the brick's orientation about the vertical axis** (a 2x4 and a 4x2
  are the same physical part, rotated 90 degrees);
* raster-scan ordering of the brick list, bottom-to-top (z, then y, then x);
* the finite-state validity check on the token format used at inference to
  constrain sampling to well-formed bricks (first token a digit, then ``x``,
  and so on), plus the in-library / in-bounds checks.

Everything is stdlib-only and deterministic. Nothing here does collision,
connectivity, force analysis or assembly ordering -- those belong to the
generic ``brick_*`` modules; this module is purely the *representation* layer:
the parts catalog and the text codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Standard brick library (Section 3, "Mesh-to-LEGO")
# --------------------------------------------------------------------------- #
# Each part is an unordered footprint {a, b} in studs; every brick is 1 unit
# tall.  The paper uses eight commonly-available bricks.
STANDARD_FOOTPRINTS: Tuple[Tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 6),
    (1, 8),
    (2, 2),
    (2, 4),
    (2, 6),
)


def _canonical(a: int, b: int) -> Tuple[int, int]:
    """Footprint as (min, max) -- orientation-independent part identity."""
    return (a, b) if a <= b else (b, a)


# The set of canonical (orientation-independent) parts in the library.
LIBRARY: FrozenSet[Tuple[int, int]] = frozenset(
    _canonical(a, b) for a, b in STANDARD_FOOTPRINTS
)

# Allowed *oriented* dimension tokens for the fine-tuning format (appendix
# "Training details"): both orders of every non-square part, one order for
# squares.
ALLOWED_ORIENTED: FrozenSet[Tuple[int, int]] = frozenset(
    {(a, b) for a, b in STANDARD_FOOTPRINTS}
    | {(b, a) for a, b in STANDARD_FOOTPRINTS}
)


def is_library_part(h: int, w: int) -> bool:
    """True if h x w (either orientation) is a standard library brick."""
    return _canonical(h, w) in LIBRARY


def part_area(h: int, w: int) -> int:
    """Number of studs the footprint covers."""
    return h * w


@dataclass(frozen=True)
class Brick:
    """One placed brick.

    ``h`` and ``w`` are the extents along X and Y respectively; their order is
    significant -- it records the brick's orientation about the vertical axis.
    ``(x, y, z)`` is the grid cell of the stud closest to the origin.
    """

    h: int
    w: int
    x: int
    y: int
    z: int

    def footprint(self) -> Tuple[int, int]:
        return _canonical(self.h, self.w)

    def is_valid_part(self) -> bool:
        return is_library_part(self.h, self.w)

    def cells(self) -> List[Tuple[int, int, int]]:
        """Grid cells (studs) occupied by this brick on its layer."""
        return [
            (self.x + dx, self.y + dy, self.z)
            for dx in range(self.h)
            for dy in range(self.w)
        ]

    def in_bounds(self, size_x: int, size_y: int, size_z: int) -> bool:
        return (
            0 <= self.x
            and 0 <= self.y
            and 0 <= self.z < size_z
            and self.x + self.h <= size_x
            and self.y + self.w <= size_y
        )


# --------------------------------------------------------------------------- #
# LEGOGPT text codec  (Section 4.1)
# --------------------------------------------------------------------------- #
def brick_to_line(b: Brick) -> str:
    """Serialize one brick to the paper's format ``"{h}x{w} ({x},{y},{z})"``."""
    return "{h}x{w} ({x},{y},{z})".format(h=b.h, w=b.w, x=b.x, y=b.y, z=b.z)


class BrickFormatError(ValueError):
    """Raised when a line does not match the LEGOGPT brick format."""


def _parse_int(token: str) -> int:
    if not token or not token.isdigit():
        raise BrickFormatError("expected non-negative integer, got %r" % token)
    return int(token)


def line_to_brick(line: str) -> Brick:
    """Parse one ``"{h}x{w} ({x},{y},{z})"`` line into a :class:`Brick`.

    Whitespace around the line is tolerated; the internal grammar is strict.
    """
    s = line.strip()
    if " " not in s:
        raise BrickFormatError("missing space between dims and position: %r" % line)
    dims, _, rest = s.partition(" ")
    rest = rest.strip()
    if "x" not in dims:
        raise BrickFormatError("dimensions must contain 'x': %r" % dims)
    hs, _, ws = dims.partition("x")
    h, w = _parse_int(hs), _parse_int(ws)
    if not (rest.startswith("(") and rest.endswith(")")):
        raise BrickFormatError("position must be parenthesized: %r" % rest)
    inner = rest[1:-1]
    parts = inner.split(",")
    if len(parts) != 3:
        raise BrickFormatError("position needs exactly 3 coords: %r" % rest)
    x, y, z = (_parse_int(p.strip()) for p in parts)
    return Brick(h=h, w=w, x=x, y=y, z=z)


def serialize(bricks: Sequence[Brick]) -> str:
    """Serialize a design to newline-separated LEGOGPT lines."""
    return "\n".join(brick_to_line(b) for b in bricks)


def parse(text: str) -> List[Brick]:
    """Parse a whole design; blank lines are ignored."""
    out: List[Brick] = []
    for raw in text.splitlines():
        if raw.strip():
            out.append(line_to_brick(raw))
    return out


# --------------------------------------------------------------------------- #
# Raster-scan ordering (Section 4.1: bottom-to-top)
# --------------------------------------------------------------------------- #
def raster_scan_sorted(bricks: Sequence[Brick]) -> List[Brick]:
    """Order bricks bottom-to-top in raster-scan order (z, then y, then x)."""
    return sorted(bricks, key=lambda b: (b.z, b.y, b.x, b.h, b.w))


def is_raster_ordered(bricks: Sequence[Brick]) -> bool:
    return list(bricks) == raster_scan_sorted(bricks)


# --------------------------------------------------------------------------- #
# Format finite-state validity check (appendix "Inference Details")
# --------------------------------------------------------------------------- #
# The paper constrains sampling so each output brick is well-formed: first a
# digit, then 'x', then a digit, a space, '(', digits, commas, ')'.  We expose
# the same check as a deterministic scanner over the exact character grammar,
# independent of the semantic library/bounds checks above.
def is_wellformed_line(line: str) -> bool:
    """True iff *line* matches the strict brick token grammar."""
    try:
        b = line_to_brick(line)
    except BrickFormatError:
        return False
    # Reject non-canonical whitespace / re-serialization mismatch so the codec
    # round-trips exactly.
    return brick_to_line(b) == line.strip()


def valid_next_chars(prefix: str) -> FrozenSet[str]:
    """Character-level FSA: the set of characters that may legally follow
    *prefix* in a brick token.  Used to constrain token sampling.

    Returns the empty set when *prefix* is already un-completable.  A prefix
    that is itself a complete, well-formed line still reports no continuation.
    """
    digits = frozenset("0123456789")
    # State machine over: D 'x' D ' ' '(' D ',' D ',' D ')'
    # We walk the prefix; at each point return the allowed next-char class.
    i = 0
    n = len(prefix)

    def _num(idx: int) -> int:
        # consume a run of digits starting at idx; return index after run
        j = idx
        while j < n and prefix[j] in digits:
            j += 1
        return j

    # h
    if i >= n:
        return digits
    if prefix[i] not in digits:
        return frozenset()
    i = _num(i)
    if i >= n:
        return digits | frozenset("x")
    if prefix[i] != "x":
        return frozenset()
    i += 1
    # w
    if i >= n:
        return digits
    if prefix[i] not in digits:
        return frozenset()
    i = _num(i)
    if i >= n:
        return digits | frozenset(" ")
    if prefix[i] != " ":
        return frozenset()
    i += 1
    if i >= n:
        return frozenset("(")
    if prefix[i] != "(":
        return frozenset()
    i += 1
    # three comma-separated coords, then ')'
    for coord in range(3):
        if i >= n:
            return digits
        if prefix[i] not in digits:
            return frozenset()
        i = _num(i)
        sep = "," if coord < 2 else ")"
        if i >= n:
            return digits | frozenset(sep)
        if prefix[i] != sep:
            return frozenset()
        i += 1
    # after ')': complete, nothing more allowed
    return frozenset()
