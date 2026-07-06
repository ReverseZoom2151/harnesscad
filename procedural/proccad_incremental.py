"""Incremental re-evaluation of a procedural program on a parameter edit.

From Séquin, *Interactive Procedural Computer-Aided Design*: the paper's central
demand is that "the key parameters [be] attached to sliders that allowed
real-time interactive fine-tuning of the ... shape", so "a designer [can] quickly
explore many different shapes in just minutes". Real-time slider dragging is only
affordable if editing one parameter re-evaluates *only the part of the procedural
program that depends on it*, not the whole model.

This module implements that as a small demand-driven dependency graph:

* :class:`InputNode` -- a named parameter (a slider);
* :class:`ComputeNode` -- a derived value computed from other nodes by a pure
  function;
* :class:`ProceduralGraph` -- holds the nodes, memoises results, and on
  ``set_input`` marks only the *downstream* cone dirty. ``evaluate`` recomputes
  exactly the dirty nodes and reports how many were recomputed, so callers (and
  tests) can verify the incrementality.

Deterministic: no wall clock, no randomness, functions must be pure.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Set, Tuple


class InputNode:
    """A named parameter node holding a mutable value."""

    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self.value = value
        self.inputs: Tuple[str, ...] = ()

    def compute(self, _deps: Dict[str, object]) -> object:  # pragma: no cover - trivial
        return self.value


class ComputeNode:
    """A derived node: ``fn(dep_values)`` where ``dep_values`` is keyed by input name."""

    def __init__(self, name: str, inputs: Sequence[str], fn: Callable[[Dict[str, object]], object]) -> None:
        self.name = name
        self.inputs = tuple(inputs)
        self.fn = fn

    def compute(self, deps: Dict[str, object]) -> object:
        return self.fn(deps)


class ProceduralGraph:
    """A DAG of input and compute nodes with incremental re-evaluation."""

    def __init__(self) -> None:
        self._nodes: Dict[str, object] = {}
        self._order: List[str] = []  # insertion / topo-consistent order
        self._cache: Dict[str, object] = {}
        self._dirty: Set[str] = set()
        self._dependents: Dict[str, Set[str]] = {}

    # -- construction -------------------------------------------------------

    def add_input(self, name: str, value: object) -> "ProceduralGraph":
        self._register(InputNode(name, value))
        return self

    def add_compute(
        self, name: str, inputs: Sequence[str], fn: Callable[[Dict[str, object]], object]
    ) -> "ProceduralGraph":
        for dep in inputs:
            if dep not in self._nodes:
                raise ValueError(f"unknown dependency '{dep}' for node '{name}'")
        node = ComputeNode(name, inputs, fn)
        self._register(node)
        for dep in inputs:
            self._dependents.setdefault(dep, set()).add(name)
        return self

    def _register(self, node: object) -> None:
        name = node.name  # type: ignore[attr-defined]
        if name in self._nodes:
            raise ValueError(f"duplicate node '{name}'")
        self._nodes[name] = node
        self._order.append(name)
        self._dependents.setdefault(name, set())
        self._dirty.add(name)

    # -- editing ------------------------------------------------------------

    def set_input(self, name: str, value: object) -> Set[str]:
        """Edit a parameter; mark it and its downstream cone dirty.

        Returns the set of node names that became dirty (were invalidated).
        A no-op edit (value unchanged) dirties nothing.
        """
        node = self._nodes.get(name)
        if not isinstance(node, InputNode):
            raise ValueError(f"'{name}' is not an editable input")
        if node.value == value and name not in self._dirty:
            return set()
        node.value = value
        dirtied = self._downstream_cone(name)
        self._dirty |= dirtied
        return dirtied

    def _downstream_cone(self, name: str) -> Set[str]:
        """All nodes transitively depending on ``name`` (including itself)."""
        seen: Set[str] = set()
        stack = [name]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self._dependents.get(cur, ()))
        return seen

    # -- evaluation ---------------------------------------------------------

    def evaluate(self) -> Tuple[Dict[str, object], int]:
        """Recompute only dirty nodes; return (all values, recomputed_count)."""
        recomputed = 0
        for name in self._order:
            if name not in self._dirty:
                continue
            node = self._nodes[name]
            deps = {d: self._cache[d] for d in node.inputs}  # type: ignore[attr-defined]
            self._cache[name] = node.compute(deps)  # type: ignore[attr-defined]
            recomputed += 1
        self._dirty.clear()
        return dict(self._cache), recomputed

    def value(self, name: str) -> object:
        """Current cached value; evaluates first if anything is dirty."""
        if self._dirty:
            self.evaluate()
        return self._cache[name]
