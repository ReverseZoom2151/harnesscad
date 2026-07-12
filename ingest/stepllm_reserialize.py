"""DFS-based STEP reserialization with CoT-style structural annotations.

This is STEP-LLM's central preprocessing idea (Shi et al., DATE 2026, Sec. 3.1).
A raw STEP file's cross-reference DAG is *non-sequential*: related entities may be
scattered far apart and long-range identifier dependencies must be recalled
precisely, which is hostile to a left-to-right auto-regressive model. The paper's
remedy is a deterministic, locality-preserving re-serialization:

  1. Parse the file into a hierarchical tree where each node is an entity and its
     children are the entities it directly references.
  2. Serialize via a **depth-first traversal**, so each branch becomes a local,
     coherent sequence and long-range dependency tracking is reduced.
  3. **Strategic pruning**: a shared reference relationship is expanded only once
     (the DAG is a set of instances, each emitted a single time).
  4. **Renumber** entity identifiers sequentially, eliminating the irregular gaps
     of the raw file and simplifying reference tracking.
  5. **Normalize floating-point precision**, reducing unnecessary digits while
     preserving topology (geometric validity).
  6. Emit lightweight **CoT-style branch annotations** (child count, branch depth)
     that summarise branch-level structure and guide global coherence.

Everything here is pure and deterministic. It consumes the parsed model from
:mod:`formats.stepllm_parser` and the root detection from
:mod:`formats.stepllm_graph`.
"""

from __future__ import annotations

from dataclasses import dataclass

from formats.stepllm_parser import (
    Entity, Real, Ref, StepFile, Typed, entity_refs, serialize_entity,
)
from formats.stepllm_graph import build_graph, roots


# --- DFS traversal (step 1-3) ------------------------------------------------

def dfs_order(step: StepFile, start=None) -> list:
    """Pre-order DFS over the reference DAG, visiting each instance once.

    Traversal begins at ``start`` (defaults to detected roots, then any other
    zero-in-degree instances, then remaining ids in file order) so nothing is
    dropped even if the file has several disconnected components.
    """

    graph = build_graph(step)
    if start is None:
        rts = roots(step)
        zero_in = [i for i in step.order
                   if graph.in_degree.get(i, 0) == 0 and i not in rts]
        start = rts + zero_in
    seeds = list(start) + [i for i in step.order if i not in set(start)]

    visited: set = set()
    order: list = []

    def visit(node: int) -> None:
        stack = [node]
        # iterative pre-order to avoid recursion limits on large files
        # (children pushed in reverse so they pop in natural order)
        pending = [node]
        while pending:
            cur = pending.pop()
            if cur in visited or cur not in step.entities:
                continue
            visited.add(cur)
            order.append(cur)
            succ = [s for s in graph.successors(cur) if s in step.entities]
            for s in reversed(succ):
                if s not in visited:
                    pending.append(s)

    for seed in seeds:
        if seed not in visited and seed in step.entities:
            visit(seed)
    return order


# --- reference remap (step 4) ------------------------------------------------

def _remap_value(value, mapping: dict):
    if isinstance(value, Ref):
        return Ref(mapping.get(value.id, value.id))
    if isinstance(value, Typed):
        return Typed(value.keyword,
                     tuple(_remap_value(p, mapping) for p in value.params))
    if isinstance(value, list):
        return [_remap_value(v, mapping) for v in value]
    if isinstance(value, tuple):
        return tuple(_remap_value(v, mapping) for v in value)
    return value


def renumber(step: StepFile, order=None) -> StepFile:
    """Return a new file with ids renumbered ``1..N`` following ``order``."""

    if order is None:
        order = dfs_order(step)
    mapping = {old: new for new, old in enumerate(order, start=1)}
    out = StepFile(header=list(step.header))
    for old in order:
        ent = step.entities[old]
        out.add(Entity(
            id=mapping[old],
            keyword=ent.keyword,
            params=[_remap_value(p, mapping) for p in ent.params],
        ))
    return out


# --- float normalization (step 5) --------------------------------------------

def format_real(value: float, precision: int = 6) -> str:
    """Format a float as a compact part-21 real literal (always keeps a dot)."""

    if value == 0:
        return "0."
    s = f"{value:.{precision}f}"
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            return s
        return s
    return s + "."


def _normalize_value(value, precision: int):
    if isinstance(value, Real):
        return Real(format_real(value.value, precision))
    if isinstance(value, Typed):
        return Typed(value.keyword,
                     tuple(_normalize_value(p, precision) for p in value.params))
    if isinstance(value, list):
        return [_normalize_value(v, precision) for v in value]
    if isinstance(value, tuple):
        return tuple(_normalize_value(v, precision) for v in value)
    return value


def normalize_reals(step: StepFile, precision: int = 6) -> StepFile:
    """Return a new file with every real literal rounded to ``precision``."""

    out = StepFile(header=list(step.header))
    for ent_id in step.order:
        ent = step.entities[ent_id]
        out.add(Entity(
            id=ent.id,
            keyword=ent.keyword,
            params=[_normalize_value(p, precision) for p in ent.params],
        ))
    return out


# --- CoT-style branch annotations (step 6) -----------------------------------

@dataclass(frozen=True)
class BranchStat:
    id: int
    keyword: str | None
    child_count: int   # number of distinct direct references
    depth: int         # longest reference chain from here to a leaf (edges)
    subtree_size: int  # distinct instances reachable, including this one


def branch_stats(step: StepFile) -> dict:
    """Per-instance branch statistics used as CoT guidance tokens."""

    graph = build_graph(step)

    depth_cache: dict = {}
    size_cache: dict = {}

    def compute(node: int, on_path: set):
        if node in depth_cache:
            return depth_cache[node], size_cache[node]
        succ = [s for s in graph.successors(node)
                if s in step.entities and s not in on_path]
        if not succ:
            depth_cache[node] = 0
            size_cache[node] = 1
            return 0, 1
        on_path.add(node)
        max_depth = 0
        reach: set = {node}
        for s in succ:
            d, _ = compute(s, on_path)
            if d + 1 > max_depth:
                max_depth = d + 1
            reach |= _reachable_set(graph, s, step)
        on_path.discard(node)
        depth_cache[node] = max_depth
        size_cache[node] = len(reach)
        return max_depth, len(reach)

    stats: dict = {}
    for ent_id in step.order:
        d, size = compute(ent_id, set())
        ent = step.entities[ent_id]
        stats[ent_id] = BranchStat(
            id=ent_id,
            keyword=ent.keyword,
            child_count=len(entity_refs(ent)),
            depth=d,
            subtree_size=size,
        )
    return stats


def _reachable_set(graph, node, step) -> set:
    seen: set = set()
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in step.entities:
            continue
        seen.add(cur)
        stack.extend(graph.successors(cur))
    return seen


def annotate(step: StepFile) -> str:
    """Render the reserialized file with a leading CoT comment per entity.

    Each entity line is prefixed with a ``/* c=<children> d=<depth>
    n=<subtree_size> */`` structural annotation, matching the paper's
    "lightweight statistical annotations ... branch-level statistics" that act as
    guidance tokens for the auto-regressive model.
    """

    stats = branch_stats(step)
    lines = ["ISO-10303-21;", "HEADER;"]
    from formats.stepllm_parser import serialize_value
    for rec in step.header:
        lines.append(serialize_value(rec) + ";")
    lines.append("ENDSEC;")
    lines.append("DATA;")
    for ent_id in step.order:
        s = stats[ent_id]
        note = f"/* c={s.child_count} d={s.depth} n={s.subtree_size} */"
        lines.append(note + serialize_entity(step.entities[ent_id]))
    lines.append("ENDSEC;")
    lines.append("END-ISO-10303-21;")
    return "\n".join(lines) + "\n"


# --- full pipeline -----------------------------------------------------------

def reserialize(step: StepFile, precision: int = 6) -> StepFile:
    """Apply the full DFS reserialization: DFS order -> renumber -> normalize."""

    ordered = renumber(step, dfs_order(step))
    return normalize_reals(ordered, precision)
