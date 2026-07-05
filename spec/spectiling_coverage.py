"""Coverage metrics over design-specification tiles (knowledge-sufficiency).

Given a query specification decomposed into tiles and a set of selected
exemplars, this module reports *how sufficiently* the exemplars cover the
query -- both the paper's global weighted tiling ratio (Eq. 6) and a per-tile
breakdown that pinpoints which sub-specifications remain under-served.

The per-tile coverage view is what the paper's case study (Sec. 4.3, Fig. 5)
argues for qualitatively: similarity top-k can tile some features while
completely missing others ("cylinder with hole" covered, "protruding rod"
missed). Here that is made quantitative -- each tile gets its own tiling ratio
so an under-covered feature is visible before generation.

stdlib-only; deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from context.spectiling_components import (
    ComponentSet,
    DEFAULT_GRANULARITIES,
    tiling_ratio,
    union_components,
    weighted_size,
)
from spec.spectiling_decompose import SpecTile, decompose_spec


@dataclass(frozen=True)
class TileCoverage:
    tile_id: int
    text: str
    ratio: float          # per-tile weighted tiling ratio in [0, 1]
    query_weight: int     # w(C_tile)
    covered_weight: int   # w(C_tile & C(S))


@dataclass(frozen=True)
class CoverageReport:
    global_ratio: float           # whole-query f_suff(S; q)
    tiles: List[TileCoverage]
    fully_covered: List[int]      # tile ids with ratio == 1.0
    uncovered: List[int]          # tile ids with ratio == 0.0

    def min_tile_ratio(self) -> float:
        """Worst-covered tile ratio -- a sufficiency floor across features."""
        return min((t.ratio for t in self.tiles), default=0.0)

    def mean_tile_ratio(self) -> float:
        if not self.tiles:
            return 0.0
        return sum(t.ratio for t in self.tiles) / len(self.tiles)


def coverage_report(
    query_spec: str,
    exemplar_specs: Sequence[str],
    granularities: Sequence[int] = DEFAULT_GRANULARITIES,
) -> CoverageReport:
    """Compute global + per-tile coverage of ``query_spec`` by exemplars.

    ``exemplar_specs`` is the list of *selected* exemplar specifications (their
    text). Each exemplar is turned into a component set and unioned; the query
    and each tile are then tiled against that union.
    """
    query = ComponentSet.from_text(query_spec, granularities)
    exemplar_cs = [
        ComponentSet.from_text(s, granularities) for s in exemplar_specs
    ]
    covered = union_components(exemplar_cs) if exemplar_cs else \
        ComponentSet.empty(granularities)

    global_ratio = tiling_ratio(covered, query)

    tiles: List[SpecTile] = decompose_spec(query_spec, granularities)
    tile_cov: List[TileCoverage] = []
    for t in tiles:
        qw = weighted_size(t.components)
        cw = weighted_size(covered.intersection(t.components))
        ratio = (cw / qw) if qw else 0.0
        tile_cov.append(
            TileCoverage(
                tile_id=t.id,
                text=t.text,
                ratio=ratio,
                query_weight=qw,
                covered_weight=cw,
            )
        )

    return CoverageReport(
        global_ratio=global_ratio,
        tiles=tile_cov,
        fully_covered=[t.tile_id for t in tile_cov if t.ratio >= 1.0],
        uncovered=[t.tile_id for t in tile_cov if t.ratio <= 0.0],
    )
