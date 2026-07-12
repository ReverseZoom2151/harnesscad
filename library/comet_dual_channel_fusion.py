"""Dual-channel (WHAT / WHEN) rank fusion with graph-aware link expansion.

Mined from CoMeT (Cognitive Memory Tree, ``comet/retriever.py``), the memory
substrate behind an autonomous-CAD stack. Its retrieval layer differs from the
single-query hybrid retriever already in this harness (``rag/retriever.py``,
BM25 + hashed-vector RRF) in two respects that are worth transferring:

1. **Dual-channel query decomposition.** Every library entry carries *two*
   independently searchable descriptions: a ``summary`` (WHAT this part /
   feature / recipe is) and a ``trigger`` (WHEN you would reach for it). A
   query is matched against both channels separately and the two rankings are
   fused, with a third optional ``raw`` channel (full text) contributing a
   small weight. A part described as "M6 hex-head cap screw, DIN 933" and
   triggered by "fastening two plates through a clearance hole" is reachable
   from either kind of prompt; a single-channel index has to compromise.

2. **Graph-aware expansion.** After the top-K is fused, entries *linked* from
   the winners (variants, mating parts, prerequisite features) are pulled in at
   a decayed score, two hops deep, with a bonus for entries referenced by
   several winners at once. This recovers the mating washer for a bolt query
   without the washer ever having to match the text.

Both parts are deterministic: fusion is Reciprocal Rank Fusion (scale-free, so
channels with incomparable score scales can be mixed), blended with a
normalised-similarity term, and every sort is broken by ``item_id`` so equal
scores order stably.

Scores are supplied by the caller as *distances* (0 = identical, larger =
worse, as returned by a vector index) together with their per-channel rank, so
this module has no opinion about how the embedding was produced.

Reference: fusion is RRF (Cormack et al., 2009), score(d) = sum_c w_c / (k + rank_c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ChannelHit",
    "FusedHit",
    "ExpandedHit",
    "rank_hits",
    "fuse_channels",
    "expand_by_links",
    "dual_channel_retrieve",
]


@dataclass(frozen=True)
class ChannelHit:
    """One entry returned by one channel of the index.

    ``distance`` is lower-is-better (e.g. cosine distance in [0, 2]).
    ``rank`` is 0-based within its channel.
    """

    item_id: str
    distance: float
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be >= 0")


@dataclass(frozen=True)
class FusedHit:
    item_id: str
    score: float
    rank: int


@dataclass
class ExpandedHit:
    """A fused hit, plus the provenance of how it entered the result set."""

    item_id: str
    score: float
    rank: int
    hop: int = 0  # 0 = matched by text, 1 = direct link, 2 = link-of-link
    via: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "hop": self.hop,
            "via": list(self.via),
        }


def rank_hits(distances: Mapping[str, float]) -> List[ChannelHit]:
    """Turn an ``item_id -> distance`` mapping into ranked ``ChannelHit``s.

    Sorted ascending by distance, ties broken by ``item_id`` so the ranking is
    a deterministic function of the input.
    """
    ordered = sorted(distances.items(), key=lambda kv: (kv[1], kv[0]))
    return [ChannelHit(item_id=i, distance=d, rank=r)
            for r, (i, d) in enumerate(ordered)]


def _similarity(distance: float) -> float:
    """Map a distance to a similarity in [0, 1] (CoMeT's ``max(0, 1 - d)``)."""
    return max(0.0, 1.0 - distance)


def fuse_channels(
    summary_hits: Sequence[ChannelHit],
    trigger_hits: Sequence[ChannelHit],
    raw_hits: Optional[Sequence[ChannelHit]] = None,
    *,
    alpha: float = 0.6,
    raw_weight: float = 0.2,
    rrf_k: int = 5,
    rrf_blend: float = 0.6,
) -> List[FusedHit]:
    """Fuse the summary / trigger (/ raw) channels into one ranking.

    ``alpha`` splits the non-raw weight between summary (WHAT) and trigger
    (WHEN): alpha=1 ignores triggers, alpha=0 ignores summaries. When
    ``raw_hits`` is given it takes ``raw_weight`` of the total and the summary
    and trigger weights are scaled down to fit.

    The final score blends the rank-based RRF term with the best observed
    similarity of the entry across channels:

        score = rrf_blend * RRF + (1 - rrf_blend) * best_similarity

    RRF alone is scale-free but throws away *how* close the match was; the
    similarity term restores that without letting an unbounded raw score
    dominate.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= raw_weight < 1.0:
        raise ValueError("raw_weight must be in [0, 1)")
    if not 0.0 <= rrf_blend <= 1.0:
        raise ValueError("rrf_blend must be in [0, 1]")
    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")

    use_raw = bool(raw_hits)
    if use_raw:
        scale = 1.0 - raw_weight
        weights = (alpha * scale, (1.0 - alpha) * scale, raw_weight)
    else:
        weights = (alpha, 1.0 - alpha, 0.0)

    channels = (summary_hits, trigger_hits, raw_hits or ())

    rrf: Dict[str, float] = {}
    sim: Dict[str, float] = {}
    for weight, hits in zip(weights, channels):
        if weight <= 0.0:
            # Still register similarity so a zero-weighted channel cannot make
            # an entry vanish that another channel found -- but contribute no
            # rank mass.
            pass
        for hit in hits:
            if weight > 0.0:
                rrf[hit.item_id] = rrf.get(hit.item_id, 0.0) + weight / (rrf_k + hit.rank + 1)
            else:
                rrf.setdefault(hit.item_id, 0.0)
            s = _similarity(hit.distance)
            if s > sim.get(hit.item_id, 0.0):
                sim[hit.item_id] = s

    combined = {
        item_id: rrf_blend * score + (1.0 - rrf_blend) * sim.get(item_id, 0.0)
        for item_id, score in rrf.items()
    }
    ordered = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
    return [FusedHit(item_id=i, score=s, rank=r) for r, (i, s) in enumerate(ordered)]


def expand_by_links(
    seeds: Sequence[FusedHit],
    links: Mapping[str, Sequence[str]],
    *,
    hop1_decay: float = 0.5,
    hop2_decay: float = 0.25,
    refcount_weight: float = 0.3,
    known_ids: Optional[Iterable[str]] = None,
) -> List[ExpandedHit]:
    """Pull in entries linked from the seeds, two hops deep, at decayed score.

    A hop-1 entry scores ``seed.score * hop1_decay`` plus
    ``refcount_weight * sum(scores of every seed that links to it)`` -- so an
    entry cited by three winners outranks one cited by a single winner. Hop-2
    entries score ``hop1.score * hop2_decay`` and get no refcount bonus.

    ``links`` need not be symmetric. Links to ids absent from ``known_ids``
    (when supplied) are treated as dangling and skipped. Seeds keep their
    original scores and stay ahead of expanded entries of equal score by the
    ``(hop, -score, item_id)`` ordering.
    """
    if not 0.0 <= hop1_decay <= 1.0 or not 0.0 <= hop2_decay <= 1.0:
        raise ValueError("decays must be in [0, 1]")

    valid = set(known_ids) if known_ids is not None else None

    def _exists(item_id: str) -> bool:
        return valid is None or item_id in valid

    seen = {s.item_id for s in seeds}
    out: List[ExpandedHit] = [
        ExpandedHit(item_id=s.item_id, score=s.score, rank=s.rank, hop=0) for s in seeds
    ]

    # Refcount: total seed mass pointing at each unseen neighbour.
    refcount: Dict[str, float] = {}
    for s in seeds:
        for link_id in links.get(s.item_id, ()):  # type: ignore[arg-type]
            if link_id in seen or not _exists(link_id):
                continue
            refcount[link_id] = refcount.get(link_id, 0.0) + s.score

    hop1: Dict[str, ExpandedHit] = {}
    for s in sorted(seeds, key=lambda h: (h.rank, h.item_id)):
        for link_id in links.get(s.item_id, ()):  # type: ignore[arg-type]
            if link_id in seen or not _exists(link_id) or link_id in hop1:
                continue
            score = s.score * hop1_decay + refcount.get(link_id, 0.0) * refcount_weight
            hop1[link_id] = ExpandedHit(item_id=link_id, score=score, rank=0, hop=1,
                                        via=(s.item_id,))
    seen |= set(hop1)

    hop2: Dict[str, ExpandedHit] = {}
    for parent in sorted(hop1.values(), key=lambda h: (-h.score, h.item_id)):
        for link_id in links.get(parent.item_id, ()):  # type: ignore[arg-type]
            if link_id in seen or not _exists(link_id) or link_id in hop2:
                continue
            hop2[link_id] = ExpandedHit(item_id=link_id, score=parent.score * hop2_decay,
                                        rank=0, hop=2, via=parent.via + (parent.item_id,))
    seen |= set(hop2)

    out.extend(hop1.values())
    out.extend(hop2.values())
    out.sort(key=lambda h: (h.hop, -h.score, h.item_id))
    for rank, hit in enumerate(out):
        hit.rank = rank
    return out


def dual_channel_retrieve(
    summary_distances: Mapping[str, float],
    trigger_distances: Mapping[str, float],
    *,
    raw_distances: Optional[Mapping[str, float]] = None,
    links: Optional[Mapping[str, Sequence[str]]] = None,
    top_k: int = 5,
    known_ids: Optional[Iterable[str]] = None,
    **fusion_kw,
) -> List[ExpandedHit]:
    """End-to-end: rank each channel, fuse, cut to ``top_k``, expand by links."""
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    fused = fuse_channels(
        rank_hits(summary_distances),
        rank_hits(trigger_distances),
        rank_hits(raw_distances) if raw_distances else None,
        **fusion_kw,
    )
    seeds = fused[:top_k]
    if not links:
        return [ExpandedHit(item_id=h.item_id, score=h.score, rank=h.rank, hop=0)
                for h in seeds]
    return expand_by_links(seeds, links, known_ids=known_ids)
