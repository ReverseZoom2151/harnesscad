"""System-to-system interoperability matrix for the code-CAD ecosystem.

Grounded in the ``curated-code-cad`` awesome-list (mined into
``adapters/ccc_codecad_ecosystem``) and derived purely from what that list
states, this module answers a question neither existing ccc module answers:

    "I built a model in system A. Which other systems can consume it, and how?"

Two complementary interoperability channels, both stated by the list:

1. FILE-FORMAT handoff. If system A exports a file format that system B imports,
   B can consume A's output through that format. The single-format helpers in
   ``ccc_codecad_ecosystem`` (``exporters_of`` / ``importers_of``) do not join
   the two sides into pairwise edges, paths, or a preferred interchange format;
   this module does. Fidelity of the chosen interchange format follows the
   list's own editorial claim -- exact B-rep exchange (STEP/IGES/BREP) is
   future-proof, mesh formats (STL) lose information, "3mf preserves
   manifoldness" -- so a deterministic fidelity ranking prefers exact formats.

2. Explicit SOURCE-LEVEL bridges the list calls out in prose, which the
   catalogue's format columns do NOT encode:
     * transpilers emit another system's source: SolidPython "outputs OpenSCAD
       code"; scad-clj / scad-hs are "OpenSCAD DSL" that emit ``.scad``.
     * host embedding: FreeCAD ships an OpenSCAD workbench and an external
       CadQuery workbench ("the best in this list at interoperability");
       AngelCAD is "capable of running OpenSCAD script for interoperability".

This module owns NO catalogue data of its own -- every system attribute is read
back from ``ccc_codecad_ecosystem``. It contributes only the interop *edges* and
the graph algorithms over them (pairwise handoff, shared/interchange format,
producers/consumers, shortest handoff path). Deterministic: pure functions over
frozen tables; outputs sorted; ties broken by name.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from harnesscad.io.adapters import ccc_codecad_ecosystem as eco

# ---------------------------------------------------------------------------
# Interop channel kinds
# ---------------------------------------------------------------------------

BRIDGE_FORMAT = "format"  # A exports a file format that B imports
BRIDGE_TRANSPILE = "transpile"  # A emits B's source language (.scad, ...)
BRIDGE_EMBED = "embed"  # A runs/hosts B (workbench, script interpreter)

BRIDGE_KINDS = (BRIDGE_EMBED, BRIDGE_FORMAT, BRIDGE_TRANSPILE)


@dataclass(frozen=True)
class Bridge:
    """A directed interoperability edge: ``src`` output can reach ``dst``."""

    src: str
    dst: str
    kind: str
    via: str  # the file format ("stl") or the mechanism ("openscad-workbench")

    def as_row(self) -> Dict[str, str]:
        return {"src": self.src, "dst": self.dst, "kind": self.kind, "via": self.via}


# ---------------------------------------------------------------------------
# Explicit source-level bridges stated by the curated list
# ---------------------------------------------------------------------------
# Each is quoted/paraphrased from a specific list statement; the catalogue's
# format columns cannot express these, so they are enumerated here by hand and
# validated (every endpoint must be a real catalogue name) at import time.

_EXPLICIT_BRIDGES: Tuple[Bridge, ...] = (
    # Transpilers that emit OpenSCAD source (.scad).
    Bridge("solidpython", "openscad", BRIDGE_TRANSPILE, "scad"),
    Bridge("scad-clj", "openscad", BRIDGE_TRANSPILE, "scad"),
    Bridge("scad-hs", "openscad", BRIDGE_TRANSPILE, "scad"),
    # FreeCAD hosts other systems via workbenches ("best at interoperability").
    Bridge("freecad", "openscad", BRIDGE_EMBED, "openscad-workbench"),
    Bridge("freecad", "cadquery", BRIDGE_EMBED, "cadquery-workbench"),
    # AngelCAD "capable of running OpenSCAD script for interoperability".
    Bridge("angelcad", "openscad", BRIDGE_EMBED, "openscad-script"),
)

# Fidelity ranking of interchange formats, grounded in the list's editorial:
# exact B-rep exchange is future-proof; 3mf "preserves manifoldness"; mesh (stl)
# and 2D (dxf/svg) lose information; source (.scad) needs the target toolchain.
# Lower index == higher fidelity / more preferred as an interchange format.
_FORMAT_FIDELITY: Tuple[str, ...] = (
    "step",
    "iges",
    "brep",
    "3mf",
    "amf",
    "off",
    "gltf",
    "obj",
    "stl",
    "dxf",
    "svg",
    "scad",
)


def _fidelity_key(fmt: str) -> Tuple[int, str]:
    fmt = fmt.lower().lstrip(".")
    try:
        return (_FORMAT_FIDELITY.index(fmt), fmt)
    except ValueError:
        return (len(_FORMAT_FIDELITY), fmt)


# Validate explicit-bridge endpoints against the catalogue at import time so a
# typo can never silently create a phantom system.
for _b in _EXPLICIT_BRIDGES:
    if not eco.has(_b.src):
        raise ValueError("explicit bridge src not in catalogue: %s" % _b.src)
    if not eco.has(_b.dst):
        raise ValueError("explicit bridge dst not in catalogue: %s" % _b.dst)
    if _b.kind not in BRIDGE_KINDS:
        raise ValueError("unknown bridge kind: %s" % _b.kind)


# ---------------------------------------------------------------------------
# Format-handoff edges (derived from the ecosystem catalogue)
# ---------------------------------------------------------------------------

def shared_formats(src: str, dst: str) -> List[str]:
    """File formats ``src`` exports that ``dst`` imports (sorted, by fidelity).

    Ordered most-faithful first per the list's exchange-fidelity editorial.
    """
    a = eco.get(src)
    b = eco.get(dst)
    common = set(a.formats_out) & set(b.formats_in)
    return sorted(common, key=_fidelity_key)


def interchange_format(src: str, dst: str) -> Optional[str]:
    """The single most faithful file format for an A->B handoff, or None."""
    shared = shared_formats(src, dst)
    return shared[0] if shared else None


def format_bridges() -> List[Bridge]:
    """Every file-format handoff edge in the catalogue, deterministically.

    One edge per (src, dst) pair, labelled with the most faithful shared format;
    self-edges are excluded.
    """
    names = eco.system_names()
    bridges: List[Bridge] = []
    for src in names:
        for dst in names:
            if src == dst:
                continue
            fmt = interchange_format(src, dst)
            if fmt is not None:
                bridges.append(Bridge(src, dst, BRIDGE_FORMAT, fmt))
    bridges.sort(key=lambda e: (e.src, e.dst))
    return bridges


def explicit_bridges() -> List[Bridge]:
    """The source-level interop bridges the list states in prose, sorted."""
    return sorted(_EXPLICIT_BRIDGES, key=lambda e: (e.src, e.dst, e.via))


def all_bridges() -> List[Bridge]:
    """Format handoffs plus explicit source-level bridges, sorted."""
    combined = list(format_bridges()) + list(_EXPLICIT_BRIDGES)
    combined.sort(key=lambda e: (e.src, e.dst, e.kind, e.via))
    return combined


# ---------------------------------------------------------------------------
# Adjacency + queries
# ---------------------------------------------------------------------------

def _adjacency(kinds: Optional[Tuple[str, ...]] = None) -> Dict[str, List[Bridge]]:
    """src -> outgoing bridges, restricted to ``kinds`` if given."""
    adj: Dict[str, List[Bridge]] = {n: [] for n in eco.system_names()}
    for edge in all_bridges():
        if kinds is not None and edge.kind not in kinds:
            continue
        adj[edge.src].append(edge)
    for src in adj:
        adj[src].sort(key=lambda e: (e.dst, e.kind, e.via))
    return adj


def can_handoff(src: str, dst: str) -> bool:
    """True if ``src`` output can reach ``dst`` by any single interop bridge."""
    eco.get(src)
    eco.get(dst)
    for edge in all_bridges():
        if edge.src == src and edge.dst == dst:
            return True
    return False


def consumers_of(src: str) -> List[str]:
    """Systems that can directly consume ``src`` output (any bridge), sorted."""
    eco.get(src)
    return sorted({e.dst for e in all_bridges() if e.src == src})


def producers_for(dst: str) -> List[str]:
    """Systems whose output ``dst`` can directly consume (any bridge), sorted."""
    eco.get(dst)
    return sorted({e.src for e in all_bridges() if e.dst == dst})


def transpile_targets(src: str) -> List[str]:
    """Systems ``src`` emits source for (BRIDGE_TRANSPILE), sorted."""
    eco.get(src)
    return sorted({e.dst for e in _EXPLICIT_BRIDGES
                   if e.src == src and e.kind == BRIDGE_TRANSPILE})


def embed_targets(src: str) -> List[str]:
    """Systems ``src`` hosts/runs directly (BRIDGE_EMBED), sorted."""
    eco.get(src)
    return sorted({e.dst for e in _EXPLICIT_BRIDGES
                   if e.src == src and e.kind == BRIDGE_EMBED})


def handoff_path(
    src: str,
    dst: str,
    kinds: Optional[Tuple[str, ...]] = None,
) -> Optional[List[Bridge]]:
    """Shortest chain of bridges taking ``src`` output to ``dst``, or None.

    Breadth-first over the interop graph, so the returned path has the fewest
    hops; neighbours are visited in sorted order making the result
    deterministic. ``kinds`` optionally restricts which bridge kinds may be
    used (e.g. only file formats). Returns ``[]`` when ``src == dst``.
    """
    eco.get(src)
    eco.get(dst)
    if src == dst:
        return []
    adj = _adjacency(kinds)
    prev: Dict[str, Bridge] = {}
    seen = {src}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for edge in adj[node]:
            if edge.dst in seen:
                continue
            seen.add(edge.dst)
            prev[edge.dst] = edge
            if edge.dst == dst:
                # Reconstruct the path from dst back to src.
                path: List[Bridge] = []
                cur = dst
                while cur != src:
                    e = prev[cur]
                    path.append(e)
                    cur = e.src
                path.reverse()
                return path
            queue.append(edge.dst)
    return None


def reachable_from(src: str, kinds: Optional[Tuple[str, ...]] = None) -> List[str]:
    """All systems reachable from ``src`` through any chain of bridges, sorted.

    Excludes ``src`` itself. Deterministic BFS closure.
    """
    eco.get(src)
    adj = _adjacency(kinds)
    seen = {src}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for edge in adj[node]:
            if edge.dst not in seen:
                seen.add(edge.dst)
                queue.append(edge.dst)
    seen.discard(src)
    return sorted(seen)


def interop_hubs(kinds: Optional[Tuple[str, ...]] = None) -> List[Tuple[str, int]]:
    """(system, out-degree) pairs, most connected first then by name.

    Out-degree counts distinct systems each system can hand off to. Surfaces the
    list's claim that FreeCAD is "the best in this list at interoperability" and
    that OpenSCAD is the common transpile target, from the graph rather than by
    assertion.
    """
    adj = _adjacency(kinds)
    scored = [(src, len({e.dst for e in edges})) for src, edges in adj.items()]
    scored.sort(key=lambda p: (-p[1], p[0]))
    return scored


def interop_matrix() -> List[List[int]]:
    """Dense reachability-by-single-bridge matrix over ``eco.system_names()``.

    ``matrix[i][j] == 1`` iff system i can hand off directly to system j. Row/
    column order is the sorted catalogue name list (``eco.system_names()``).
    """
    names = eco.system_names()
    index = {n: i for i, n in enumerate(names)}
    matrix = [[0] * len(names) for _ in names]
    for edge in all_bridges():
        matrix[index[edge.src]][index[edge.dst]] = 1
    return matrix
