"""DFS-based STEP reserialization with locality preservation and CoT annotations.

Paper: *STEP-LLM -- Generating CAD STEP Models from Natural Language with Large
Language Models* (Shi et al., Northwestern, DATE 2026). The paper's core
non-neural contribution is a deterministic **preprocessing** of part-21 files so
that a graph-structured STEP DAG is turned into a locality-preserving linear
sequence an auto-regressive model can consume (Sec. 3.1):

  1. parse the file into a hierarchical tree whose nodes are entities and whose
     children are the directly referenced entities;
  2. serialize the tree by a **depth-first traversal**, so each branch is a
     locally coherent run rather than the scattered, cross-referenced raw order;
  3. **strategic pruning** -- each reference relationship appears exactly once
     (the first DFS visit); later visits become a plain back-reference;
  4. **sequential renumbering** of entity identifiers, eliminating the irregular
     ``#`` gaps of the raw file so reference tracking is simple;
  5. **float precision normalization** -- trim unnecessary digits on real
     literals without altering topology;
  6. **CoT-style structural annotations** -- lightweight per-branch statistics
     (direct child count, subtree size, branch depth) emitted as guidance tokens
     so a downstream reader can reason about global structure while reading the
     local DFS stream.

Everything here is deterministic and stdlib-only. It operates on the parsed
:class:`~harnesscad.io.formats.step.StepFile` produced by :mod:`step` and reuses
:func:`step.entity_refs` for the reference relation and
:mod:`step_graph` root detection. It is distinct from :mod:`step_graph` (which
only *validates* the DAG) and from :mod:`step.serialize` (which round-trips the
*raw* order): this module produces a *reordered, renumbered, annotated* file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harnesscad.io.formats.step import (
    Entity,
    Real,
    Ref,
    StepFile,
    Typed,
    entity_refs,
    serialize_entity,
)
from harnesscad.io.formats.step_graph import ROOT_TYPES

__all__ = [
    "BranchStats",
    "dfs_order",
    "renumber",
    "normalize_reals",
    "reserialize",
    "branch_annotations",
    "annotated_text",
]


# --- root discovery ---------------------------------------------------------

def _root_ids(step: StepFile) -> list:
    """Ids that nothing references (in file order) -- the DFS entry points.

    Prefer entities whose keyword is a known B-rep root type; if none are
    present, fall back to every zero-in-degree instance so the traversal still
    covers the whole file. Order follows ``step.order`` for determinism.
    """
    referenced = set()
    for eid in step.order:
        for tid in entity_refs(step.entities[eid]):
            referenced.add(tid)
    typed_roots = [
        eid for eid in step.order
        if (step.entities[eid].keyword or "") in ROOT_TYPES
    ]
    if typed_roots:
        return typed_roots
    unreferenced = [eid for eid in step.order if eid not in referenced]
    return unreferenced or list(step.order)


# --- DFS traversal ----------------------------------------------------------

def dfs_order(step: StepFile, roots=None) -> list:
    """Ids in depth-first, first-visit order (strategic pruning: each id once).

    Children are visited in :func:`entity_refs` order (deterministic). Ids not
    reachable from the roots are appended afterwards in file order so nothing is
    silently dropped.
    """
    if roots is None:
        roots = _root_ids(step)
    visited = set()
    out: list = []

    def visit(eid: int) -> None:
        if eid in visited or eid not in step.entities:
            return
        visited.add(eid)
        out.append(eid)
        for child in entity_refs(step.entities[eid]):
            visit(child)

    for r in roots:
        visit(r)
    for eid in step.order:  # dead/unreachable instances, still emitted once
        visit(eid)
    return out


# --- value rewriting (renumber refs + normalize reals) ----------------------

def _rewrite_value(value, mapping, digits):
    if isinstance(value, Ref):
        return Ref(mapping.get(value.id, value.id))
    if isinstance(value, Real):
        return Real(_normalize_real_text(value.text, digits))
    if isinstance(value, Typed):
        return Typed(value.keyword, tuple(_rewrite_value(p, mapping, digits)
                                          for p in value.params))
    if isinstance(value, list):
        return [_rewrite_value(v, mapping, digits) for v in value]
    if isinstance(value, tuple):
        return tuple(_rewrite_value(v, mapping, digits) for v in value)
    return value


def _normalize_real_text(text: str, digits) -> str:
    """Trim a real literal to ``digits`` decimals, dropping trailing zeros.

    ``digits=None`` leaves the text untouched. The result always keeps a decimal
    point so it stays a part-21 REAL (e.g. ``2`` -> ``2.``), matching the paper's
    "reduce unnecessary digits while preserving geometric validity".
    """
    if digits is None:
        return text
    try:
        val = float(text)
    except ValueError:
        return text
    rounded = round(val, digits)
    if rounded == int(rounded):
        return f"{int(rounded)}."
    s = f"{rounded:.{digits}f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def normalize_reals(step: StepFile, digits: int = 6) -> StepFile:
    """A copy of ``step`` with every real literal trimmed to ``digits`` decimals."""
    return renumber(step, {eid: eid for eid in step.order}, digits=digits,
                    order=list(step.order))


def renumber(step: StepFile, mapping: dict, digits=None, order=None) -> StepFile:
    """Rebuild ``step`` with ids remapped by ``mapping`` and reals normalized.

    ``order`` (a list of *old* ids) fixes the output order; defaults to the old
    file order sorted by new id. Header is carried over verbatim.
    """
    if order is None:
        order = sorted(step.order, key=lambda eid: mapping.get(eid, eid))
    out = StepFile(header=list(step.header))
    for eid in order:
        ent = step.entities[eid]
        new_params = [_rewrite_value(p, mapping, digits) for p in ent.params]
        out.add(Entity(id=mapping.get(eid, eid), keyword=ent.keyword,
                       params=new_params))
    return out


def reserialize(step: StepFile, digits: int = 6, roots=None) -> StepFile:
    """Full pipeline: DFS order -> sequential renumber -> real normalization.

    Returns a new :class:`StepFile` whose ids are ``1..N`` in DFS order and whose
    reals are trimmed. Topology is preserved exactly (references are remapped, not
    dropped). This is the training-target form the paper feeds to the LLM.
    """
    order = dfs_order(step, roots=roots)
    mapping = {old: new for new, old in enumerate(order, start=1)}
    return renumber(step, mapping, digits=digits, order=order)


# --- CoT-style branch annotations -------------------------------------------

@dataclass
class BranchStats:
    """Per-entity structural statistics used as CoT guidance tokens."""

    ent_id: int
    keyword: str
    children: int          # direct out-references
    subtree_size: int      # distinct ids reachable (incl. self)
    depth: int             # longest downward chain from here (edges)


def branch_annotations(step: StepFile) -> dict:
    """Map each id to its :class:`BranchStats` (subtree size + depth + children).

    Uses memoized DFS over the reference relation; safe on a DAG. On the off
    chance of a cycle, a visited-set guard keeps recursion finite (the cyclic
    edge simply does not extend the subtree).
    """
    size_cache: dict = {}
    depth_cache: dict = {}

    def compute(eid, stack):
        if eid in size_cache:
            return
        if eid not in step.entities:
            size_cache[eid] = 0
            depth_cache[eid] = 0
            return
        reach = {eid}
        depth = 0
        for child in entity_refs(step.entities[eid]):
            if child in stack or child not in step.entities:
                continue
            compute(child, stack | {eid})
            reach |= _reach_cache.get(child, {child})
            depth = max(depth, 1 + depth_cache.get(child, 0))
        _reach_cache[eid] = reach
        size_cache[eid] = len(reach)
        depth_cache[eid] = depth

    _reach_cache: dict = {}
    for eid in step.order:
        compute(eid, frozenset())

    out: dict = {}
    for eid in step.order:
        ent = step.entities[eid]
        out[eid] = BranchStats(
            ent_id=eid,
            keyword=ent.keyword or "COMPLEX",
            children=len(entity_refs(ent)),
            subtree_size=size_cache.get(eid, 1),
            depth=depth_cache.get(eid, 0),
        )
    return out


def annotated_text(step: StepFile, digits: int = 6) -> str:
    """Reserialized DATA lines, each preceded by a CoT-style annotation comment.

    Each entity line is prefixed with ``/* c=<children> n=<subtree> d=<depth> */``
    -- the branch statistics the paper injects as guidance tokens. Deterministic:
    the same file always produces byte-identical output.
    """
    reser = reserialize(step, digits=digits)
    stats = branch_annotations(reser)
    lines: list = []
    for eid in reser.order:
        s = stats[eid]
        ann = f"/* c={s.children} n={s.subtree_size} d={s.depth} */"
        lines.append(f"{ann} {serialize_entity(reser.entities[eid])}")
    return "\n".join(lines)
