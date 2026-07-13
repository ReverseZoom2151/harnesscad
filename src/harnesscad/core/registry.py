"""The CORE capability surface -- the guards around the loop.

``core/`` grew a set of capabilities that guard the op stream rather than being
part of it, and nothing dispatched to any of them:

*   **op-decoding constraints** -- a generator (an LLM, a sampler, a grammar) may
    only emit an op the CURRENT STATE admits. You cannot extrude before you
    sketch.
*   **routing** -- classify the task, then pick the cheapest model that can do it.
*   **feature tree** -- the parametric DAG: edit a feature, propagate staleness,
    rebuild only what went stale.
*   **constraint hierarchy** -- scoped constraint solving with branch pruning, so
    a local edit stays local.
*   **annotations** -- drafting callouts bound to entity ids that survive a remap.
*   **explicit context** -- typed handles, so 'the face' means one face.

RIVALS ARE SELECTED BY NAME, NEVER BLENDED
------------------------------------------
The two op-decoding constraints work at DIFFERENT GRANULARITIES and are not
interchangeable:

*   ``grammar``     -- a whole-op constraint: a JSON schema / EBNF over the op
    set the state admits, checked against a complete candidate op.
*   ``grammar_fsa`` -- a TOKEN-level finite-state automaton over a
    curve/loop/face/sketch/extrusion token stream: it rejects an illegal token
    the moment it appears, before an op exists at all.

One validates a finished op; the other constrains a decoder mid-stream. Averaging
them is meaningless. :func:`constraints` exposes both, by name.

Adapters only: the core modules are never modified. Deterministic, stdlib-only,
no network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "CoreError",
    "UnknownConstraint",
    "CONSTRAINTS",
    "constraints",
    "allowed_ops",
    "check_op",
    "check_tokens",
    "classify",
    "estimate_cost",
    "feature_tree",
    "rebuild",
    "solve_scoped",
    "remap_annotations",
    "context",
    "RIVAL_FAMILIES",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_CORE = "harnesscad.core."


class CoreError(ValueError):
    """Base class for every core-surface failure."""


class UnknownConstraint(CoreError):
    """An op-decoding constraint name that is not registered."""


# --------------------------------------------------------------------------- #
# Op-decoding constraints (RIVALS -- selected by name)
# --------------------------------------------------------------------------- #
CONSTRAINTS: Tuple[str, ...] = ("grammar", "grammar_fsa")


def constraints() -> Tuple[str, ...]:
    """The selectable op-decoding constraints. RIVALS: pick one, never blend."""
    return tuple(n for n in CONSTRAINTS
                 if _available(_CORE + ("grammar" if n == "grammar" else "grammar_fsa")))


def allowed_ops(has_sketch: bool = False, has_solid: bool = False) -> List[str]:
    """Which op kinds the CURRENT state admits. You cannot extrude before you sketch."""
    from harnesscad.core.grammar import allowed_ops_for_state

    return list(allowed_ops_for_state(bool(has_sketch), bool(has_solid)))


def check_op(candidate: Any, has_sketch: bool = False, has_solid: bool = False):
    """`grammar` -- validate a COMPLETE candidate op against the state's schema.

    Returns the parsed :class:`Op`; raises :class:`GrammarError` if the state
    does not admit it. This is the whole-op rival.
    """
    from harnesscad.core.grammar import constraint_for_state

    return constraint_for_state(bool(has_sketch), bool(has_solid)).check(candidate)


def check_tokens(tokens: Sequence[str]) -> dict:
    """`grammar_fsa` -- run a TOKEN stream through the finite-state automaton.

    Returns the final state and whether the stream is legal. This is the
    token-level rival: it rejects an illegal token before an op exists.
    """
    from harnesscad.core.grammar_fsa import State, allowed, run

    final, diagnostics = run(list(tokens))
    ok = final is not State.DEAD
    return {
        "final_state": final.value,
        "ok": ok,
        "diagnostics": list(diagnostics),
        "next_allowed": sorted(allowed(final)) if ok else [],
    }


# --------------------------------------------------------------------------- #
# Routing: classify the task, then price it
# --------------------------------------------------------------------------- #
def classify(messages: Sequence[Any], hints: Optional[Mapping[str, Any]] = None) -> str:
    """cheap / standard / hard. The HEURISTIC classifier -- no model call."""
    from harnesscad.core.routing import HeuristicClassifier

    task = HeuristicClassifier().classify(list(messages), dict(hints or {}))
    return task.value if hasattr(task, "value") else str(task)


def estimate_cost(model: str, messages: Sequence[Any]) -> float:
    """What this call would cost on this model, from the token count and the price table."""
    from harnesscad.core.routing import CostTable

    return float(CostTable().estimate(model, list(messages)))


# --------------------------------------------------------------------------- #
# Parametric state
# --------------------------------------------------------------------------- #
def feature_tree(nodes: Sequence[Mapping[str, Any]] = ()):
    """A parametric feature DAG. Cycles and missing parents raise, never silently pass."""
    from harnesscad.core.state.feature_tree import FeatureNode, FeatureTree, add_feature

    tree = FeatureTree()
    for n in nodes:
        add_feature(tree, FeatureNode(**dict(n)))
    return tree


def rebuild(tree: Any, builder: Callable[[Any], Any], force: bool = False):
    """Rebuild ONLY what an edit made stale. ``force=True`` rebuilds everything."""
    from harnesscad.core.state.feature_tree import rebuild as _rebuild

    return _rebuild(tree, builder, force=bool(force))


def solve_scoped(root: Any, solver: Callable[[Any], Any]):
    """Solve a constraint hierarchy scope-by-scope, so a local edit stays local."""
    from harnesscad.core.state.constraint_hierarchy import solve_hierarchy

    return solve_hierarchy(root, solver)


# --------------------------------------------------------------------------- #
# Annotations + explicit context
# --------------------------------------------------------------------------- #
def remap_annotations(items: Sequence[Any], entity_map: Mapping[str, str],
                      drop_missing: bool = False):
    """Re-bind drafting callouts after a rebuild renumbered the entities.

    An annotation whose entity vanished raises unless ``drop_missing`` -- a
    dimension pointing at nothing is a bug, not a warning.
    """
    from harnesscad.core.cisp.annotations import remap_annotations as _remap

    return _remap(list(items), dict(entity_map), drop_missing=bool(drop_missing))


def context():
    """A typed-handle :class:`Context`: 'the face' has to mean ONE face, by name."""
    from harnesscad.core.cisp.explicit_context import Context

    return Context()


# --------------------------------------------------------------------------- #
# Rivals
# --------------------------------------------------------------------------- #
RIVAL_FAMILIES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("op-decoding-constraint",
     "A whole-op JSON-schema/EBNF constraint (grammar) vs a TOKEN-level "
     "finite-state automaton (grammar_fsa). One validates a finished op; the "
     "other constrains a decoder mid-stream. Different granularities, never "
     "blended -- select one by name.",
     ("grammar", "grammar_fsa")),
)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e for e in capability_registry.index()
            if e.dotted.startswith(_CORE)}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("decode", "check_op", _CORE + "grammar",
     "whole-op schema/EBNF constraint for the current state (RIVAL)"),
    ("decode", "check_tokens", _CORE + "grammar_fsa",
     "token-level FSA over the curve/loop/face/sketch/extrusion stream (RIVAL)"),
    ("route", "classify", _CORE + "routing",
     "classify the task, then price it against the model cost table"),
    ("state", "feature_tree", _CORE + "state.feature_tree",
     "the parametric feature DAG: edit, propagate staleness, rebuild the stale"),
    ("state", "solve_scoped", _CORE + "state.constraint_hierarchy",
     "scoped constraint solving with branch pruning -- a local edit stays local"),
    ("cisp", "remap_annotations", _CORE + "cisp.annotations",
     "drafting callouts that survive an entity renumber"),
    ("cisp", "context", _CORE + "cisp.explicit_context",
     "typed handles: 'the face' must resolve to exactly one face"),
)


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


UNADAPTED_REASONS: Dict[str, str] = {
    _CORE + "cli": "the CLI itself -- the top of the call graph, by construction",
    _CORE + "harness": "the public facade; callers import it, it imports nothing back",
}


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, UNADAPTED_REASONS.get(d, "no route yet")) for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every core route")
    parser.add_argument("--rivals", action="store_true",
                        help="list the rival families (selected by name, never blended)")
    parser.add_argument("--allowed", action="store_true",
                        help="print the ops the given state admits")
    parser.add_argument("--has-sketch", action="store_true",
                        help="the state already has a sketch")
    parser.add_argument("--has-solid", action="store_true",
                        help="the state already has a solid")
    parser.add_argument("--tokens", default=None, metavar="T,T,T",
                        help="run a token stream through the grammar FSA")
    parser.add_argument("--unadapted", action="store_true",
                        help="list core modules with no route, and why")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "rivals", False):
        for family, doc, members in RIVAL_FAMILIES:
            print("%s: (selected by name, NEVER blended)" % family)
            print("    %s" % doc)
            for m in members:
                print("    - %s" % m)
        return 0

    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "tokens", None):
        result = check_tokens([t.strip() for t in args.tokens.split(",") if t.strip()])
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1

    if getattr(args, "allowed", False):
        for op in allowed_ops(getattr(args, "has_sketch", False),
                              getattr(args, "has_solid", False)):
            print(op)
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-7s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad core",
        description="core surface: op-decoding constraints, routing, feature tree, "
                    "constraint hierarchy, annotations, explicit context")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
