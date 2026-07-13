"""Tile-composition: merge per-tile generated CadQuery fragments into one script.

The DST framework generates code for a design specification in one shot, but the
spec-tiling view (see :mod:`spec.spectiling_decompose`) opens a compositional
generation path the paper flags as future work ("adapting to hierarchical
program structures"): generate a code fragment per tile, then **merge** the
fragments into a single executable CadQuery script.

This module performs that merge deterministically and syntactically:

  * Import lines (``import ...`` / ``from ... import ...``) are hoisted to the
    top and de-duplicated, preserving first-seen order.
  * Each fragment's remaining body is emitted in tile (build) order, separated
    by a ``# --- tile N ---`` banner so provenance survives.
  * A fragment's result assignments are renamed to per-tile names (``result``
    -> ``result_0`` ...) when a collision would otherwise shadow an earlier
    tile, and a trailing ``result = result_0 + result_1 + ...`` union line is
    appended when >1 fragment produced a ``result`` symbol -- mirroring how
    CadQuery composes solids with ``+`` (union).

No code is executed; this is a text-level composition so it stays stdlib-only
and deterministic. Callers that need semantic validation should hand the merged
script to their CadQuery compiler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

_IMPORT_RE = re.compile(r"^\s*(?:import\s+\S.*|from\s+\S+\s+import\s+.+)$")
# Matches a top-level assignment to a bare `result` name.
_RESULT_ASSIGN_RE = re.compile(r"^(\s*)result(\s*=\s*)(.+)$")
_RESULT_REF_RE = re.compile(r"\bresult\b")


@dataclass(frozen=True)
class TileFragment:
    """A generated code fragment for one tile."""

    tile_id: int
    code: str


def _split_imports(code: str) -> Tuple[List[str], List[str]]:
    """Return (import_lines, body_lines) preserving order within each."""
    imports: List[str] = []
    body: List[str] = []
    for line in code.splitlines():
        if _IMPORT_RE.match(line):
            imports.append(line.strip())
        else:
            body.append(line.rstrip("\n"))
    # Trim leading/trailing blank body lines.
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return imports, body


def _rename_result(body: Sequence[str], suffix: int) -> Tuple[List[str], bool]:
    """Rename the ``result`` symbol to ``result_<suffix>`` throughout ``body``.

    Returns (new_body, had_result). Only applied when disambiguation is needed.
    """
    had_result = any(_RESULT_ASSIGN_RE.match(ln) for ln in body)
    if not had_result:
        return list(body), False
    new_name = f"result_{suffix}"
    renamed = [_RESULT_REF_RE.sub(new_name, ln) for ln in body]
    return renamed, True


def compose_fragments(
    fragments: Sequence[TileFragment],
    union_results: bool = True,
) -> str:
    """Merge ``fragments`` (already in build order) into one CadQuery script.

    Args:
        fragments: per-tile code fragments, emitted in the given order.
        union_results: when multiple fragments define a top-level ``result``,
            append ``result = result_0 + result_1 + ...`` to union the solids.

    Returns the merged script text. Import de-duplication is order-stable.
    """
    seen_imports = set()
    all_imports: List[str] = []
    rendered_bodies: List[str] = []
    result_names: List[str] = []

    # Decide whether we must disambiguate: >1 fragment defines `result`.
    result_bearing = [
        f for f in fragments
        if any(_RESULT_ASSIGN_RE.match(ln) for ln in _split_imports(f.code)[1])
    ]
    disambiguate = union_results and len(result_bearing) > 1

    for order, frag in enumerate(fragments):
        imports, body = _split_imports(frag.code)
        for imp in imports:
            if imp not in seen_imports:
                seen_imports.add(imp)
                all_imports.append(imp)
        if disambiguate:
            body, had = _rename_result(body, order)
            if had:
                result_names.append(f"result_{order}")
        rendered_bodies.append((frag.tile_id, body))

    out: List[str] = []
    out.extend(all_imports)
    if all_imports:
        out.append("")
    for tile_id, body in rendered_bodies:
        out.append(f"# --- tile {tile_id} ---")
        out.extend(body)
        out.append("")
    if disambiguate and result_names:
        out.append("# --- composed union of tiles ---")
        out.append("result = " + " + ".join(result_names))
    # Drop a single trailing blank line for tidiness.
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)
