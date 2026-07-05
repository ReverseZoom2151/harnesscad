"""Deterministic population lineage and evolution-budget bookkeeping."""
from __future__ import annotations
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class GeneratorRecord:
    id: str
    name: str
    abstract: str
    detailed: str
    code: str
    parent_ids: tuple[str, ...] = ()

def validate_lineage(records):
    values = tuple(records); ids = [r.id for r in values]; issues = []
    if len(ids) != len(set(ids)): issues.append("duplicate-id")
    known = set(ids)
    for record in values:
        if record.id in record.parent_ids: issues.append(f"{record.id}:self-parent")
        for parent in record.parent_ids:
            if parent not in known: issues.append(f"{record.id}:missing-parent:{parent}")
    graph = {r.id: r.parent_ids for r in values}; visiting=set(); done=set()
    def visit(node):
        if node in visiting: return True
        if node in done: return False
        visiting.add(node)
        cycle = any(parent in graph and visit(parent) for parent in graph.get(node, ()))
        visiting.remove(node); done.add(node)
        return cycle
    if any(visit(node) for node in graph): issues.append("cycle")
    return tuple(dict.fromkeys(issues))

def sample_parents(records, count, *, seed):
    values = sorted(records, key=lambda r: r.id)
    if count <= 0 or count > len(values): raise ValueError("invalid parent count")
    return tuple(random.Random(seed).sample(values, count))

def termination(rounds, *, budget, novelty_window=3, minimum_novelty=0.0):
    values = tuple(rounds)
    if len(values) >= budget: return "budget"
    if len(values) >= novelty_window and all(
        float(row.get("novelty_ratio", 0)) <= minimum_novelty
        for row in values[-novelty_window:]): return "novelty-saturation"
    return None
