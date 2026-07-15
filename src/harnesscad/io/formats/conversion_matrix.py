"""Format-conversion MATRIX -- which format converts to which, and by what path.

``action-convert-directory`` is a batch job: point it at a directory and a target
format and it converts every file it can.  The deterministic core of that idea is
not the file I/O, it is the **matrix** -- the table that answers *can format A be
turned into format B, and if so, through which intermediate steps?*  A naive
"any-to-any" table lies: a mesh cannot become a B-rep (the curved faces were
discarded at tessellation), a drawing cannot become a solid, and some formats can
only be written, never read.

This module encodes that table honestly and derives conversions from first
principles rather than hand-listing every pair:

*   each :class:`FormatEntry` declares a format's ``kind`` (what geometry it
    carries) and whether it can be ``read`` and/or ``written``;
*   a **kind lattice** says which kinds can be *lowered* into which -- a ``brep``
    or ``csg`` can be meshed into a ``mesh``, a ``brep`` can be flattened into a
    ``drawing``, but nothing raises a ``mesh`` back to a ``brep``;
*   a direct conversion ``A -> B`` exists iff A can be read, B can be written, and
    ``kind(A)`` can be lowered to ``kind(B)``;
*   :func:`conversion_path` runs a deterministic BFS over that graph to find the
    shortest chain (possibly multi-hop) between two formats, and :func:`convert`
    dispatches an in-memory value through that chain using injected step handlers.

Everything is stdlib-only and deterministic: the same query always yields the same
matrix and the same path.  No file is touched and no codec is imported -- this is
the *plan*, the codecs in :mod:`harnesscad.io.formats` are the *actuators*.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "FormatEntry",
    "FORMATS",
    "KIND_LOWERINGS",
    "kind_can_lower",
    "can_convert",
    "direct_targets",
    "conversion_path",
    "conversion_matrix",
    "ConversionError",
    "convert",
]


class ConversionError(ValueError):
    """Raised when no conversion path exists between two formats."""


# --------------------------------------------------------------------------- #
# format table                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FormatEntry:
    """A registered format: its extension, the geometry ``kind`` it carries, and
    whether the harness can read it, write it, or both."""

    name: str
    extension: str
    kind: str
    read: bool
    write: bool


# The formats the harness actually carries codecs for, described honestly.
FORMATS: Dict[str, FormatEntry] = {
    e.name: e for e in (
        FormatEntry("stl", ".stl", "mesh", read=True, write=True),
        FormatEntry("obj", ".obj", "mesh", read=True, write=True),
        FormatEntry("ply", ".ply", "mesh", read=True, write=True),
        FormatEntry("glb", ".glb", "mesh", read=True, write=True),
        FormatEntry("3mf", ".3mf", "mesh", read=True, write=True),
        FormatEntry("amf", ".amf", "mesh", read=True, write=True),
        FormatEntry("step", ".step", "brep", read=True, write=True),
        FormatEntry("3dm", ".3dm", "brep", read=True, write=True),
        FormatEntry("xcsg", ".xcsg", "csg", read=True, write=True),
        FormatEntry("scad", ".scad", "csg", read=True, write=True),
        FormatEntry("kcl", ".kcl", "program", read=True, write=True),
        FormatEntry("svg", ".svg", "drawing", read=False, write=True),  # write-only
        FormatEntry("dxf", ".dxf", "drawing", read=True, write=True),
        FormatEntry("png", ".png", "image", read=False, write=True),   # write-only
    )
}


# --------------------------------------------------------------------------- #
# kind lattice                                                                #
# --------------------------------------------------------------------------- #
# kind -> the kinds it can be *lowered* to (information is lost going forward,
# never recovered going back).  Every kind can be lowered to itself implicitly.
KIND_LOWERINGS: Dict[str, Tuple[str, ...]] = {
    "program": ("csg", "brep", "mesh"),   # a KCL/program builds a solid
    "csg": ("brep", "mesh"),              # evaluate CSG -> B-rep or tessellate
    "brep": ("mesh", "drawing"),          # tessellate, or project to a drawing
    "mesh": ("mesh", "image"),            # render a mesh to an image
    "drawing": ("drawing", "image"),      # rasterise a drawing
    "image": ("image",),
}


def kind_can_lower(src_kind: str, dst_kind: str) -> bool:
    """True if ``src_kind`` geometry can become ``dst_kind`` geometry."""
    if src_kind == dst_kind:
        return True
    seen = set()
    stack = [src_kind]
    while stack:
        k = stack.pop()
        if k == dst_kind:
            return True
        if k in seen:
            continue
        seen.add(k)
        stack.extend(KIND_LOWERINGS.get(k, ()))
    return False


# --------------------------------------------------------------------------- #
# direct conversions                                                          #
# --------------------------------------------------------------------------- #
def can_convert(src: str, dst: str) -> bool:
    """True if there is a *direct* single-step conversion ``src -> dst``."""
    if src not in FORMATS or dst not in FORMATS:
        raise KeyError(f"unknown format(s): {src!r}, {dst!r}")
    a, b = FORMATS[src], FORMATS[dst]
    if src == dst:
        return a.read or a.write
    return a.read and b.write and kind_can_lower(a.kind, b.kind)


def direct_targets(src: str) -> Tuple[str, ...]:
    """Every format ``src`` can be converted to in one step (sorted, deterministic)."""
    return tuple(sorted(
        dst for dst in FORMATS
        if dst != src and can_convert(src, dst)
    ))


# --------------------------------------------------------------------------- #
# multi-hop paths                                                             #
# --------------------------------------------------------------------------- #
def conversion_path(src: str, dst: str) -> Optional[List[str]]:
    """Shortest chain of formats from ``src`` to ``dst`` (inclusive), or ``None``.

    Deterministic BFS: neighbours are visited in sorted order so the returned path
    is stable.  Returns ``[src]`` when ``src == dst``.
    """
    if src not in FORMATS or dst not in FORMATS:
        raise KeyError(f"unknown format(s): {src!r}, {dst!r}")
    if src == dst:
        return [src]
    queue: deque = deque([src])
    prev: Dict[str, str] = {src: src}
    while queue:
        node = queue.popleft()
        for nxt in direct_targets(node):
            if nxt in prev:
                continue
            prev[nxt] = node
            if nxt == dst:
                # reconstruct
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            queue.append(nxt)
    return None


def conversion_matrix() -> Dict[str, Dict[str, bool]]:
    """The full direct-conversion table: ``matrix[src][dst]`` is True/False."""
    names = sorted(FORMATS)
    return {
        s: {d: (s == d or can_convert(s, d)) for d in names}
        for s in names
    }


# --------------------------------------------------------------------------- #
# dispatch                                                                     #
# --------------------------------------------------------------------------- #
# a step handler converts an in-memory value from one format to the next.
StepHandler = Callable[[object], object]


def convert(
    value: object,
    src: str,
    dst: str,
    handlers: Dict[Tuple[str, str], StepHandler],
) -> object:
    """Convert ``value`` from ``src`` to ``dst`` by dispatching through the planned
    path, applying the injected ``handlers[(a, b)]`` for each hop.

    Raises :class:`ConversionError` if no path exists or a hop has no handler.
    """
    path = conversion_path(src, dst)
    if path is None:
        raise ConversionError(f"no conversion path from {src!r} to {dst!r}")
    for a, b in zip(path, path[1:]):
        handler = handlers.get((a, b))
        if handler is None:
            raise ConversionError(f"no handler for step {a!r} -> {b!r}")
        value = handler(value)
    return value
