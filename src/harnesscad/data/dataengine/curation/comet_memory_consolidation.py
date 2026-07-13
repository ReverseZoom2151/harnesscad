"""Consolidation pass for a linked knowledge store: dedup, merge, tag-normalise, prune.

Mined from CoMeT (``comet/consolidator.py``). CoMeT periodically sweeps its
memory graph and (a) unions near-duplicate nodes into single-link connected
clusters, (b) merges each cluster into one keeper that absorbs the others'
tags, links and recall counts while every absorbed node's raw content stays
addressable, (c) collapses variant spellings of the same tag, and (d) drops
links pointing at nodes that no longer exist.

The same sweep is exactly what a growing CAD knowledge base needs: repeated
sessions deposit near-identical notes ("M6 clearance hole is 6.6 mm",
"clearance for M6 = 6.6"), tags drift (``fastener`` / ``Fasteners`` /
``fastening``), and deleted parts leave dangling references in assemblies.

Everything here is deterministic and stdlib-only. The clustering is a
union-find (disjoint set) over *pairs* that exceed a similarity threshold --
single-link agglomeration -- so cluster membership does not depend on iteration
order. Similarity is pluggable; the default is a token Jaccard so the module
needs no embeddings, but callers may pass a cosine over their own vectors.

CoMeT's version calls an LLM to validate clusters and rewrite the merged
summary; that step is deliberately excluded (non-deterministic, external). The
keeper's summary is instead chosen by an explicit, reproducible policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "MemoryNode",
    "ConsolidationReport",
    "token_jaccard",
    "find_clusters",
    "merge_cluster",
    "normalize_tags",
    "prune_dangling_links",
    "consolidate",
]

_META_PREFIXES = ("ORIGIN:", "FLAG:", "SESSION:")
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class MemoryNode:
    """One entry of the knowledge store."""

    node_id: str
    summary: str
    trigger: str = ""
    tags: Tuple[str, ...] = ()
    links: Tuple[str, ...] = ()
    importance: int = 1  # higher = keep preferentially
    recall_count: int = 0

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "summary": self.summary,
            "trigger": self.trigger,
            "tags": list(self.tags),
            "links": list(self.links),
            "importance": self.importance,
            "recall_count": self.recall_count,
        }


@dataclass
class ConsolidationReport:
    merged: int = 0
    clusters: List[List[str]] = field(default_factory=list)
    absorbed_into: Dict[str, str] = field(default_factory=dict)
    tags_renamed: Dict[str, str] = field(default_factory=dict)
    nodes_retagged: int = 0
    links_pruned: int = 0

    def to_dict(self) -> dict:
        return {
            "merged": self.merged,
            "clusters": [list(c) for c in self.clusters],
            "absorbed_into": dict(self.absorbed_into),
            "tags_renamed": dict(self.tags_renamed),
            "nodes_retagged": self.nodes_retagged,
            "links_pruned": self.links_pruned,
        }


def _tokens(text: str) -> Set[str]:
    return set(_WORD.findall(text.lower()))


def token_jaccard(a: MemoryNode, b: MemoryNode) -> float:
    """Default similarity: Jaccard over the tokens of summary + trigger."""
    ta = _tokens(a.summary + " " + a.trigger)
    tb = _tokens(b.summary + " " + b.trigger)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


SimilarityFn = Callable[[MemoryNode, MemoryNode], float]


def find_clusters(
    nodes: Sequence[MemoryNode],
    threshold: float = 0.6,
    similarity: Optional[SimilarityFn] = None,
) -> List[List[str]]:
    """Single-link clusters of near-duplicate nodes, via union-find.

    Every pair scoring >= ``threshold`` is unioned. Returns only clusters of
    size >= 2, each sorted by ``node_id``, the list itself sorted by first
    member -- so the output is a pure function of the input set.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    sim = similarity or token_jaccard

    parent: Dict[str, str] = {n.node_id: n.node_id for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Deterministic: the lexicographically smaller root wins.
        if rb < ra:
            ra, rb = rb, ra
        parent[rb] = ra

    ordered = sorted(nodes, key=lambda n: n.node_id)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if sim(a, b) >= threshold:
                union(a.node_id, b.node_id)

    groups: Dict[str, List[str]] = {}
    for n in ordered:
        groups.setdefault(find(n.node_id), []).append(n.node_id)

    clusters = [sorted(members) for members in groups.values() if len(members) > 1]
    clusters.sort(key=lambda c: c[0])
    return clusters


def _keeper_key(node: MemoryNode) -> Tuple[int, int, int, str]:
    """Keeper policy: most important, then most recalled, then richest summary."""
    return (-node.importance, -node.recall_count, -len(node.summary), node.node_id)


def merge_cluster(
    cluster: Sequence[MemoryNode],
) -> Tuple[MemoryNode, List[str]]:
    """Collapse a cluster into one keeper; return (keeper, absorbed_ids).

    The keeper absorbs the union of tags and links (minus self-links and links
    to absorbed nodes) and the sum of recall counts, and takes the maximum
    importance in the cluster. Its summary/trigger are the keeper's own -- no
    text is synthesised, so the result is reproducible.
    """
    if not cluster:
        raise ValueError("cluster must be non-empty")
    ordered = sorted(cluster, key=_keeper_key)
    keeper, absorbed = ordered[0], ordered[1:]
    absorbed_ids = sorted(n.node_id for n in absorbed)
    dead = set(absorbed_ids) | {keeper.node_id}

    tags = set(keeper.tags)
    links = set(keeper.links)
    recall = keeper.recall_count
    importance = keeper.importance
    for n in absorbed:
        tags |= set(n.tags)
        links |= set(n.links)
        recall += n.recall_count
        importance = max(importance, n.importance)

    merged = replace(
        keeper,
        tags=tuple(sorted(tags)),
        links=tuple(sorted(l for l in links if l not in dead)),
        recall_count=recall,
        importance=importance,
    )
    return merged, absorbed_ids


def _canonical_tag_map(tags: Iterable[str]) -> Dict[str, str]:
    """Map each variant tag to its canonical form.

    Two tags collapse when they differ only by case, or when the shorter is a
    case-insensitive substring of the longer (``fastener`` absorbs
    ``fasteners``). The shortest tag wins; among equal-length variants the
    all-lowercase form is preferred (``fastener`` over ``Fastener``), then
    lexicographic order -- so the map never depends on set iteration order.
    Meta-prefixed tags (``ORIGIN:``/``FLAG:``/``SESSION:``) are left alone.
    """
    plain = sorted(
        {t for t in tags if not any(t.startswith(p) for p in _META_PREFIXES)},
        key=lambda t: (len(t), t.lower(), not t.islower(), t),
    )
    mapping: Dict[str, str] = {}
    for i, short in enumerate(plain):
        if short in mapping or len(short) < 2:
            continue
        for long in plain[i + 1:]:
            if long in mapping:
                continue
            if short.lower() == long.lower() or short.lower() in long.lower():
                mapping[long] = short
    return mapping


def normalize_tags(nodes: Sequence[MemoryNode]) -> Tuple[List[MemoryNode], Dict[str, str], int]:
    """Collapse variant tag spellings across the store.

    Returns (nodes, renames, n_nodes_changed).
    """
    all_tags = {t for n in nodes for t in n.tags}
    renames = _canonical_tag_map(all_tags)
    if not renames:
        return list(nodes), {}, 0

    out: List[MemoryNode] = []
    changed = 0
    for n in nodes:
        new = tuple(sorted({renames.get(t, t) for t in n.tags}))
        if new != n.tags:
            changed += 1
            out.append(replace(n, tags=new))
        else:
            out.append(n)
    return out, renames, changed


def prune_dangling_links(nodes: Sequence[MemoryNode]) -> Tuple[List[MemoryNode], int]:
    """Drop links pointing at ids not present in ``nodes`` (and self-links)."""
    alive = {n.node_id for n in nodes}
    out: List[MemoryNode] = []
    pruned = 0
    for n in nodes:
        keep = tuple(l for l in n.links if l in alive and l != n.node_id)
        pruned += len(n.links) - len(keep)
        out.append(replace(n, links=keep) if keep != n.links else n)
    return out, pruned


def consolidate(
    nodes: Sequence[MemoryNode],
    threshold: float = 0.6,
    similarity: Optional[SimilarityFn] = None,
) -> Tuple[List[MemoryNode], ConsolidationReport]:
    """Full sweep: cluster + merge, then tag-normalise, then prune links.

    Links held by *other* nodes to an absorbed node are rewired to the keeper
    (rather than pruned), so the graph stays connected across a merge.
    """
    report = ConsolidationReport()
    by_id = {n.node_id: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate node_id in input")

    clusters = find_clusters(nodes, threshold=threshold, similarity=similarity)
    report.clusters = clusters

    rewire: Dict[str, str] = {}
    survivors: Dict[str, MemoryNode] = dict(by_id)
    for cluster in clusters:
        keeper, absorbed_ids = merge_cluster([by_id[i] for i in cluster])
        survivors[keeper.node_id] = keeper
        for aid in absorbed_ids:
            survivors.pop(aid, None)
            rewire[aid] = keeper.node_id
            report.absorbed_into[aid] = keeper.node_id
        report.merged += len(absorbed_ids)

    if rewire:
        for nid, node in list(survivors.items()):
            new_links = tuple(sorted({
                rewire.get(l, l) for l in node.links
                if rewire.get(l, l) != nid
            }))
            if new_links != node.links:
                survivors[nid] = replace(node, links=new_links)

    out = [survivors[k] for k in sorted(survivors)]
    out, report.tags_renamed, report.nodes_retagged = normalize_tags(out)
    out, report.links_pruned = prune_dangling_links(out)
    return out, report
