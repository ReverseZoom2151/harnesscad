"""Deterministic CadQuery API-usage profiler.

Mined from the ``cadquery-contrib`` example corpus: a collection of real,
community-written parametric CadQuery programs.  Such a corpus is a ground
truth for *which* parts of the CadQuery fluent API are actually used, with
what arity, in what chain shapes, and with which string selectors.

This module builds that profile deterministically from source text using the
stdlib ``ast`` module (no CadQuery / OCCT runtime needed), and can diff the
profile against a declared method table -- e.g. ``programs.t2cq_ast``'s
``CHAIN_METHODS`` -- to report methods the harness's program AST does not
know about, and methods whose declared positional-arity range is violated by
real-world usage.  Both are correctness findings for the static analysers.

All outputs are sorted / deterministic.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

__all__ = [
    "MethodUsage",
    "ApiProfile",
    "profile_source",
    "profile_sources",
    "unknown_methods",
    "arity_violations",
    "selector_strings",
    "format_profile",
]


@dataclass
class MethodUsage:
    """Aggregated usage facts for one fluent method name."""

    name: str
    count: int = 0
    # positional-argument-count -> number of call sites
    arity_counts: Dict[int, int] = field(default_factory=dict)
    # keyword name -> number of call sites using it
    kwargs: Dict[str, int] = field(default_factory=dict)

    @property
    def min_args(self) -> int:
        return min(self.arity_counts) if self.arity_counts else 0

    @property
    def max_args(self) -> int:
        return max(self.arity_counts) if self.arity_counts else 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "count": self.count,
            "arity_counts": dict(sorted(self.arity_counts.items())),
            "kwargs": dict(sorted(self.kwargs.items())),
            "min_args": self.min_args,
            "max_args": self.max_args,
        }


@dataclass
class ApiProfile:
    """A whole-corpus API usage profile."""

    methods: Dict[str, MethodUsage] = field(default_factory=dict)
    # method-name tuples, longest-first chain shapes -> occurrences
    chains: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    # string literal arguments passed to selector-taking methods
    selectors: Dict[str, int] = field(default_factory=dict)
    sources: int = 0
    parse_errors: List[str] = field(default_factory=list)

    def method_names(self) -> List[str]:
        return sorted(self.methods)

    def top_methods(self, n: int = 10) -> List[Tuple[str, int]]:
        """Most-used methods, ties broken by name for determinism."""
        items = [(m.name, m.count) for m in self.methods.values()]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        return items[:n]

    def top_chains(self, n: int = 10) -> List[Tuple[Tuple[str, ...], int]]:
        items = sorted(self.chains.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[:n]

    def as_dict(self) -> Dict[str, object]:
        return {
            "sources": self.sources,
            "methods": {k: v.as_dict() for k, v in sorted(self.methods.items())},
            "chains": {".".join(k): v for k, v in sorted(self.chains.items())},
            "selectors": dict(sorted(self.selectors.items())),
            "parse_errors": list(self.parse_errors),
        }


# Fluent methods whose first positional argument is a string selector.
SELECTOR_METHODS: frozenset = frozenset({"faces", "edges", "vertices", "solids",
                                         "shells", "wires"})


def _record(profile: ApiProfile, node: ast.Call) -> str | None:
    """Record one ``x.method(...)`` call; return the method name."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    name = func.attr
    usage = profile.methods.setdefault(name, MethodUsage(name))
    usage.count += 1
    n_pos = sum(1 for a in node.args if not isinstance(a, ast.Starred))
    usage.arity_counts[n_pos] = usage.arity_counts.get(n_pos, 0) + 1
    for kw in node.keywords:
        if kw.arg is None:
            continue
        usage.kwargs[kw.arg] = usage.kwargs.get(kw.arg, 0) + 1
    if name in SELECTOR_METHODS and node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            s = first.value
            profile.selectors[s] = profile.selectors.get(s, 0) + 1
    return name


def _chain_of(node: ast.Call) -> List[str]:
    """Walk a fluent call chain outermost->innermost, returning method names."""
    names: List[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
        names.append(cur.func.attr)
        cur = cur.func.value
    names.reverse()
    return names


def profile_source(src: str, profile: ApiProfile | None = None,
                   filename: str = "<src>") -> ApiProfile:
    """Profile a single CadQuery program's source text."""
    prof = profile if profile is not None else ApiProfile()
    prof.sources += 1
    try:
        tree = ast.parse(src, filename=filename)
    except SyntaxError as exc:  # pragma: no cover - defensive
        prof.parse_errors.append(f"{filename}: {exc.msg}")
        return prof

    outermost: List[ast.Call] = []
    inner: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            _record(prof, node)
            recv = node.func.value
            if isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute):
                inner.add(id(recv))
            outermost.append(node)
    for node in outermost:
        if id(node) in inner:
            continue
        chain = tuple(_chain_of(node))
        if len(chain) >= 2:
            prof.chains[chain] = prof.chains.get(chain, 0) + 1
    return prof


def profile_sources(sources: Iterable[str]) -> ApiProfile:
    """Profile many sources into a single corpus-level profile."""
    prof = ApiProfile()
    for i, src in enumerate(sources):
        profile_source(src, prof, filename=f"<src{i}>")
    return prof


def unknown_methods(profile: ApiProfile,
                    known: Mapping[str, Sequence[int]]) -> List[Tuple[str, int]]:
    """Methods observed in the corpus but absent from ``known``.

    ``known`` is a mapping like ``t2cq_ast.CHAIN_METHODS`` (name -> (lo, hi)).
    Result is sorted by descending usage count, then name.
    """
    out = [(m.name, m.count) for m in profile.methods.values() if m.name not in known]
    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def arity_violations(profile: ApiProfile,
                     known: Mapping[str, Sequence[int]]) -> List[Tuple[str, int, int, int]]:
    """Known methods used with a positional arity outside their declared range.

    Returns ``(name, observed_arity, declared_lo, declared_hi)`` tuples, sorted.
    """
    out: List[Tuple[str, int, int, int]] = []
    for name, usage in profile.methods.items():
        if name not in known:
            continue
        lo, hi = known[name][0], known[name][1]
        for arity in sorted(usage.arity_counts):
            if not (lo <= arity <= hi):
                out.append((name, arity, lo, hi))
    out.sort()
    return out


def selector_strings(profile: ApiProfile) -> List[Tuple[str, int]]:
    """Selector string literals, most frequent first."""
    items = sorted(profile.selectors.items(), key=lambda kv: (-kv[1], kv[0]))
    return items


def format_profile(profile: ApiProfile, top: int = 10) -> str:
    """Deterministic plain-text report."""
    lines = [f"sources: {profile.sources}",
             f"distinct methods: {len(profile.methods)}",
             "top methods:"]
    for name, count in profile.top_methods(top):
        u = profile.methods[name]
        lines.append(f"  {name} x{count} args={u.min_args}..{u.max_args}")
    lines.append("top chains:")
    for chain, count in profile.top_chains(top):
        lines.append(f"  {'.'.join(chain)} x{count}")
    lines.append("selectors:")
    for s, count in selector_strings(profile)[:top]:
        lines.append(f"  {s!r} x{count}")
    return "\n".join(lines)
