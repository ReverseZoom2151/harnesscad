"""The INTERACTION surface -- what a human (or a frontend) sees and types.

``io/surfaces`` carried a keyboard-first command grammar, a Bayesian
next-command predictor, an adaptive-UX proficiency estimator, confidence
overlays over findings, SVG primitive-ID labels, a debug graph view, an
edit-context/orthographic-view selector, a canonical camera-view set and a
multi-modal intent fuser. All of it was framework-neutral, deterministic, and
reachable from nothing.

This module dispatches into it. Everything here is a PURE PROJECTION of harness
state into something renderable or typeable -- there is no frontend, no window,
no event loop:

    parse("extrude 5")          -> a typed intent (never a shell)
    predict(history)            -> the next command, with a probability
    overlays(diagnostics)       -> confidence bands over the findings
    graph(opdag)                -> the op DAG as JSON / SVG
    labels(anchors)             -> non-overlapping SVG id labels
    views(points)               -> the best orthographic view; the canonical set
    profile(stats)              -> the user's proficiency tier and UX verbosity

WHAT IS NOT HERE: the servers. ``surfaces/mcp``, ``surfaces/acp`` and
``surfaces/a2a_server`` are PROCESS ENTRY POINTS (``python -m ...``) and are
deliberately not routed -- a dispatcher that imported them would start them.

Adapters only: the surface modules are never modified. Deterministic,
stdlib-only, no network.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from harnesscad import registry as capability_registry

__all__ = [
    "SurfaceError",
    "surface",
    "parse",
    "commands",
    "predict",
    "profile",
    "overlays",
    "graph",
    "labels",
    "best_view",
    "canonical_views",
    "fuse",
    "discover",
    "routed_modules",
    "unadapted",
    "add_arguments",
    "run_cli",
    "main",
]

_SRF = "harnesscad.io.surfaces."


class SurfaceError(ValueError):
    """Base class for every interaction-surface failure."""


# --------------------------------------------------------------------------- #
# The command grammar: keyboard-first, SHELL-FREE
# --------------------------------------------------------------------------- #
def surface():
    """A fresh :class:`CommandSurface`. It parses to typed intents, never to a shell."""
    from harnesscad.io.surfaces.commands import CommandSurface

    return CommandSurface()


def parse(text: str, surf: Optional[Any] = None):
    """One typed line -> an Op / Query / Undo / Mode / Help intent.

    A line the grammar does not know raises :class:`CommandParseError`, which
    carries an ``accessible_message()`` -- it never falls through to a shell.
    The surface is STATEFUL (it carries the mode), so pass one in to keep a
    session's mode across lines.
    """
    return (surf if surf is not None else surface()).parse(text)


def commands(surf: Optional[Any] = None) -> List[str]:
    """The commands available RIGHT NOW (mode-dependent, not a static menu)."""
    return list((surf if surf is not None else surface()).available_commands())


# --------------------------------------------------------------------------- #
# Prediction + proficiency
# --------------------------------------------------------------------------- #
def predict(sequences: Sequence[Sequence[str]], history: Sequence[str],
            k: int = 3) -> List[dict]:
    """Next-command prediction from a Bayesian workflow graph over past sessions."""
    from harnesscad.io.surfaces.command_prediction import (
        BayesianCommandPredictor, WorkflowGraph,
    )

    graph = WorkflowGraph()
    graph.add_sequences([list(s) for s in sequences])
    predictor = BayesianCommandPredictor(graph)
    return [{"command": p.command, "probability": p.probability}
            for p in predictor.predict_next(list(history), top_k=int(k))]


def profile(**stats) -> dict:
    """The user's proficiency tier and the interface verbosity it earns."""
    from harnesscad.io.surfaces.adaptive_ux import (
        InteractionStats, ProficiencyEstimator, recommend_ux,
    )

    s = InteractionStats(**stats)
    est = ProficiencyEstimator()
    tier = est.tier(s)
    ux = recommend_ux(tier)
    return {"score": est.score(s), "tier": tier.name if hasattr(tier, "name") else str(tier),
            "verbosity": getattr(ux, "verbosity", None),
            "profile": ux}


# --------------------------------------------------------------------------- #
# Projections of harness state
# --------------------------------------------------------------------------- #
def overlays(findings: Sequence[Mapping[str, Any]]) -> List[dict]:
    """Confidence bands over verifier findings -- frontend-neutral, no rendering."""
    from harnesscad.io.surfaces.confidence import build_overlays

    return [o.to_dict() for o in build_overlays([dict(f) for f in findings])]


def graph(opdag: Any, feature_graph: Optional[Any] = None,
          diagnostics: Optional[Sequence[Any]] = None, fmt: str = "json"):
    """The op DAG (+ feature graph, + diagnostics) as JSON or SVG. Debug view."""
    from harnesscad.io.surfaces.graphview import build_graph_view

    view = build_graph_view(opdag, feature_graph, list(diagnostics or ()))
    if fmt == "svg":
        return view.to_svg()
    if fmt == "json":
        return view.to_json()
    if fmt == "dict":
        return view.to_dict()
    raise SurfaceError("unknown graph format %r; known: json, svg, dict" % fmt)


def labels(anchors: Mapping[str, Sequence[float]], fmt: str = "svg"):
    """Non-overlapping SVG id labels over sketch primitives.

    ``anchors`` maps an entity id ('e1') to its (x, y) anchor. Ids are ESCAPED
    into the SVG, never interpolated -- an entity called ``</text>`` does not get
    to write markup.
    """
    from harnesscad.io.surfaces.id_overlay import overlay_svg, place_labels

    items = {str(k): tuple(float(v) for v in xy) for k, xy in anchors.items()}
    if fmt == "svg":
        svg, _placements = overlay_svg(items)
        return svg
    if fmt == "placements":
        return [dict(zip(type(p).__dataclass_fields__, (
            getattr(p, f) for f in type(p).__dataclass_fields__)))
            for p in place_labels(items)]
    raise SurfaceError("unknown label format %r; known: svg, placements" % fmt)


def best_view(points: Sequence[Sequence[float]]) -> Any:
    """The orthographic view that shows this geometry best (largest projected area)."""
    from harnesscad.io.surfaces.edit_views import best_view as _best

    return _best([tuple(float(v) for v in p) for p in points])


def canonical_views(count: int = 12) -> List[dict]:
    """The versioned, sphere-distributed canonical prompt cameras."""
    from harnesscad.io.surfaces.canonical_views import canonical_views as _views

    return [{"id": v.id, "direction": list(v.direction), "up": list(v.up),
             "version": v.version} for v in _views(int(count))]


def fuse(signals: Sequence[Mapping[str, Any]]):
    """Fuse text / sketch / selection signals into one intent -- or ask for clarification.

    ``needs_clarification()`` is the honest outcome when the modalities disagree;
    the fuser does not pick a winner by confidence alone.
    """
    from harnesscad.io.surfaces.modality_fusion import ModalityFuser, ModalitySignal

    fused = ModalityFuser().fuse([ModalitySignal(**dict(s)) for s in signals])
    return fused


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _index() -> Dict[str, Any]:
    return {e.dotted: e for e in capability_registry.index()
            if e.dotted.startswith(_SRF)}


def _available(dotted: str) -> bool:
    return dotted in _index()


_ROUTES: Tuple[Tuple[str, str, str, str], ...] = (
    ("input", "parse", _SRF + "commands",
     "keyboard-first, SHELL-FREE command grammar -> typed intents"),
    ("input", "predict", _SRF + "command_prediction",
     "Bayesian workflow-graph next-command prediction"),
    ("input", "fuse", _SRF + "modality_fusion",
     "fuse text/sketch/selection signals, or ask for clarification"),
    ("adapt", "profile", _SRF + "adaptive_ux",
     "proficiency tier -> interface verbosity"),
    ("view", "overlays", _SRF + "confidence",
     "confidence bands over verifier findings"),
    ("view", "graph", _SRF + "graphview",
     "the op DAG as JSON / SVG (debug view)"),
    ("view", "labels", _SRF + "id_overlay",
     "non-overlapping SVG primitive-ID labels"),
    ("view", "best_view", _SRF + "edit_views",
     "edit-context selection + the best orthographic view"),
    ("view", "canonical_views", _SRF + "canonical_views",
     "the versioned sphere-distributed canonical prompt cameras"),
)

#: Surface modules deliberately left with NO route, and why.
UNADAPTED_REASONS: Dict[str, str] = {
    _SRF + "mcp.__main__":
        "a PROCESS ENTRY POINT (`python -m harnesscad.io.surfaces.mcp`); a "
        "dispatcher that imported it would start a server",
    _SRF + "acp.__main__":
        "a PROCESS ENTRY POINT (`python -m harnesscad.io.surfaces.acp`)",
    _SRF + "a2a_server.__main__":
        "a PROCESS ENTRY POINT (`python -m harnesscad.io.surfaces.a2a_server`)",
    _SRF + "mcp.gym":
        "an RL ENVIRONMENT (CADGymEnv). Its caller is a training loop, which the "
        "harness does not ship and will not fake",
}


def routed_modules() -> Tuple[str, ...]:
    return tuple(sorted({m for _g, _n, m, _d in _ROUTES if _available(m)}))


def discover() -> List[dict]:
    return [{"group": g, "route": n, "module": m, "doc": d,
             "present": _available(m)}
            for (g, n, m, d) in _ROUTES]


def unadapted() -> List[Tuple[str, str]]:
    routed = set(routed_modules())
    return [(d, UNADAPTED_REASONS.get(d, "no route yet")) for d in sorted(_index())
            if d not in routed and not d.endswith(".registry")]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--list", action="store_true",
                        help="list every interaction route")
    parser.add_argument("--parse", default=None,
                        help="parse one command line into a typed intent")
    parser.add_argument("--commands", action="store_true",
                        help="list the commands available in the initial state")
    parser.add_argument("--views", type=int, default=None, metavar="N",
                        help="print N canonical prompt cameras")
    parser.add_argument("--unadapted", action="store_true",
                        help="list surface modules with no route, and why")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON instead of text")


def run_cli(args: argparse.Namespace) -> int:
    if getattr(args, "unadapted", False):
        for dotted, reason in unadapted():
            print("%s\n    %s" % (dotted, reason))
        return 0

    if getattr(args, "commands", False):
        for c in commands():
            print(c)
        return 0

    if getattr(args, "views", None):
        print(json.dumps(canonical_views(args.views), indent=2, sort_keys=True))
        return 0

    if getattr(args, "parse", None):
        from harnesscad.io.surfaces.commands import CommandParseError

        try:
            intent = parse(args.parse)
        except CommandParseError as exc:
            print("error: %s" % exc.accessible_message())
            return 2
        print("%s: %r" % (type(intent).__name__, intent))
        return 0

    rows = discover()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    width = max(len(r["route"]) for r in rows)
    for r in rows:
        mark = " " if r["present"] else "-"
        print("%s %-6s %-*s  %s" % (mark, r["group"], width, r["route"], r["doc"]))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad ui",
        description="interaction surface: command grammar, prediction, overlays, views")
    add_arguments(parser)
    return run_cli(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
