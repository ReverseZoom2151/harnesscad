"""Progressive assembly-trace plan structuring.

One-shot object generation becomes a visible *assembly trace*: the object is
built as an ordered sequence of part-addition steps, each
paired with a textual construction rationale, so the structure is inspectable
step by step. The learned part is the rendering; the *trace structure* and its
structure-aware checks (T2S-CompBench: component numeracy, structural topology,
trace stability, rationale alignment) are deterministic.

This module provides the deterministic scaffold:

* :class:`AssemblyStep` / :class:`AssemblyTrace` -- an ordered, monotone plan
  where each step adds one or more named parts and carries a rationale,
* :func:`build_trace` -- structure an unordered part list into a dependency-
  respecting monotone trace (parents before children),
* :func:`component_numeracy` -- T2S-CompBench component-numeracy score: does the
  trace realise the specified part count?,
* :func:`trace_stability` -- is the trace strictly monotone (every step adds new
  parts, nothing is removed or re-added)?,
* :func:`rationale_alignment` -- fraction of steps whose rationale mentions the
  part(s) that step introduces.

No model calls: this constructs and grades plans deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Set, Tuple

__all__ = [
    "AssemblyStep",
    "AssemblyTrace",
    "build_trace",
    "component_numeracy",
    "trace_stability",
    "rationale_alignment",
]


@dataclass(frozen=True)
class AssemblyStep:
    """One construction step: the parts it introduces and its rationale."""

    parts: Tuple[str, ...]
    rationale: str = ""


@dataclass(frozen=True)
class AssemblyTrace:
    """An ordered sequence of assembly steps."""

    steps: Tuple[AssemblyStep, ...]

    def all_parts(self) -> Tuple[str, ...]:
        out: List[str] = []
        for s in self.steps:
            out.extend(s.parts)
        return tuple(out)

    def part_count(self) -> int:
        return len(self.all_parts())


def build_trace(
    parts: Sequence[str],
    dependencies: Mapping[str, Sequence[str]] | None = None,
    *,
    rationales: Mapping[str, str] | None = None,
) -> AssemblyTrace:
    """Order *parts* into a monotone trace that respects *dependencies*.

    ``dependencies[p]`` lists parts that must be placed before ``p`` (its
    supports/parents). The trace is built by a deterministic topological sweep:
    at each step, every part whose dependencies are already placed is added, in
    stable input order. Raises ``ValueError`` on a dependency cycle or an
    unknown dependency.
    """
    deps = {p: list(dependencies.get(p, ())) if dependencies else [] for p in parts}
    part_set = set(parts)
    for p, ds in deps.items():
        for d in ds:
            if d not in part_set:
                raise ValueError(f"part {p!r} depends on unknown part {d!r}")
    rat = dict(rationales or {})
    placed: Set[str] = set()
    remaining = list(parts)
    steps: List[AssemblyStep] = []
    while remaining:
        ready = [p for p in remaining if all(d in placed for d in deps[p])]
        if not ready:
            raise ValueError("dependency cycle among parts")
        for p in ready:
            placed.add(p)
        remaining = [p for p in remaining if p not in placed]
        rationale = "; ".join(rat.get(p, f"add {p}") for p in ready)
        steps.append(AssemblyStep(tuple(ready), rationale))
    return AssemblyTrace(tuple(steps))


def component_numeracy(trace: AssemblyTrace, expected_parts: int) -> float:
    """T2S-CompBench component numeracy: how well the count matches the spec.

    Uses a symmetric ratio ``min(actual, expected) / max(actual, expected)`` in
    ``[0, 1]`` -- 1 when the trace realises exactly the specified number of
    parts, decaying as the count drifts either way.
    """
    if expected_parts <= 0:
        raise ValueError("expected_parts must be positive")
    actual = trace.part_count()
    if actual <= 0:
        return 0.0
    return min(actual, expected_parts) / max(actual, expected_parts)


def trace_stability(trace: AssemblyTrace) -> bool:
    """True iff the trace is strictly monotone: no empty step, no re-added part.

    A stable trace only ever grows -- every step introduces at least one part
    and no part appears in two different steps.
    """
    seen: Set[str] = set()
    for step in trace.steps:
        if not step.parts:
            return False
        for p in step.parts:
            if p in seen:
                return False
            seen.add(p)
    return True


def rationale_alignment(trace: AssemblyTrace) -> float:
    """Fraction of steps whose rationale mentions a part that step introduces.

    A cheap deterministic proxy for T2S-CompBench's rationale alignment: a step
    is aligned when at least one of its introduced part names is a substring of
    its (lower-cased) rationale.
    """
    if not trace.steps:
        return 0.0
    aligned = 0
    for step in trace.steps:
        text = step.rationale.lower()
        if any(p.lower() in text for p in step.parts):
            aligned += 1
    return aligned / len(trace.steps)
