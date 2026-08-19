"""harnesscad.sdk -- the canonical, programmable SDK facade for HarnessCAD.

THIS MODULE ADDS NO BEHAVIOUR. It is a thin, honest re-export facade: the top
level a researcher or an AI lab imports without spelunking internal module paths.
Every capability here already exists inside the package; the facade only
re-exports the real components and, where noted, composes them lightly (choosing a
sensible default backend, or resolving a backend name to an instance). The
canonical implementations live where they always have:

  - the RL/agent environment  -> ``harnesscad.io.surfaces.mcp.gym.CADGymEnv``
  - the apply/verify session   -> ``harnesscad.core.loop.HarnessSession``
  - the CISP op vocabulary     -> ``harnesscad.core.cisp.ops``
  - the bench scoring dispatch -> ``harnesscad.eval.bench.registry``
  - the MCP server             -> ``harnesscad.io.surfaces.mcp.server.MCPServer``
  - the ReAct trajectory       -> ``harnesscad.agents.agent.tool_trajectory``
  - the sourced task corpus    -> ``harnesscad.eval.corpus.dev`` (the readable split)

Nothing here hides or reinterprets what those modules do; a facade name that
composes rather than re-exports is documented as such below.

The surface (stable, documented, stdlib-only at this layer)::

    Environment(backend="stub", ...)  -> a configured CADGymEnv
    Session(backend="stub", ...)      -> a HarnessSession (non-RL apply+measure)
    Op, parse_op, load_ops            -> the op vocabulary
    metrics(), metric(name), score(name, pred, gold)
    suites(), suite(name), run_suite(name, samples)
    mcp_server(backend="stub", ...)   -> a ready MCPServer
    Trajectory (= ToolTrajectory), TrajectoryStep
    TaskSuite, load_tasks(), Brief    -> the DEV corpus split

Heavy/optional dependencies (a real geometry kernel such as OCCT/cadquery) are
NEVER imported at facade-import time; the underlying components pull them lazily
only when a call actually needs them, exactly as they do today. ``import
harnesscad`` and ``import harnesscad.sdk`` both stay fast and kernel-free.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

# The op vocabulary is pure-stdlib and always wanted, so it is re-exported
# eagerly. Everything heavier is imported lazily inside the functions below.
from harnesscad.core.cisp.ops import Op, parse_op

__all__ = [
    "Environment",
    "Session",
    "mcp_server",
    "Op",
    "parse_op",
    "load_ops",
    "op_vocabulary",
    "metrics",
    "metric",
    "score",
    "suites",
    "suite",
    "run_suite",
    "Trajectory",
    "TrajectoryStep",
    "TaskSuite",
    "load_tasks",
    "Brief",
]


# ---------------------------------------------------------------------------
# Backend resolution (light composition, not new behaviour).
#
# The canonical name->backend construction lives in
# harnesscad.io.surfaces.server._make_backend (falls back to the stub with a note
# when an optional kernel is missing). The facade reuses it so a caller may pass a
# backend NAME ("stub", "frep", "cadquery", ...) or an already-built backend
# instance interchangeably.
# ---------------------------------------------------------------------------

def _resolve_backend(backend):
    """Return a backend *instance* for a name, an instance, or None (-> stub)."""
    if backend is None:
        from harnesscad.io.backends.stub import StubBackend
        return StubBackend()
    if isinstance(backend, str):
        from harnesscad.io.surfaces.server import _make_backend
        return _make_backend(backend)[0]
    return backend  # already a GeometryBackend instance


# ---------------------------------------------------------------------------
# Environment -- the RL/agent Gym interface.
# ---------------------------------------------------------------------------

def Environment(backend="stub", *, verifiers=None, max_steps: Optional[int] = None):
    """A configured :class:`~harnesscad.io.surfaces.mcp.gym.CADGymEnv`.

    ``backend`` is a backend name (resolved via the canonical
    ``server._make_backend``) or a ready backend instance; ``"stub"`` -- the
    dependency-free default -- keeps construction kernel-free. This is a
    constructor alias: the returned object is exactly a ``CADGymEnv`` with its real
    ``reset() -> obs``, ``step(action) -> (obs, reward, done, info)``, ``state()``,
    ``render()``, ``close()`` API and its CISP-op action space. No wrapping.
    """
    from harnesscad.io.surfaces.mcp.gym import CADGymEnv
    return CADGymEnv(backend=_resolve_backend(backend), verifiers=verifiers,
                     max_steps=max_steps)


# ---------------------------------------------------------------------------
# Session -- the non-RL "apply ops and measure" path.
# ---------------------------------------------------------------------------

def Session(backend="stub", **kwargs):
    """A :class:`~harnesscad.core.loop.HarnessSession` for the non-RL path.

    Convenience over ``HarnessSession`` for callers who just want the
    apply -> regen -> verify -> checkpoint spine (``apply_ops``, ``digest``,
    ``summary``, ``export``) without the Gym reward/observation loop. ``backend``
    is a name or an instance (see :func:`Environment`); any further keyword is
    forwarded verbatim to ``HarnessSession`` (``verify_level``, ``tracer``,
    ``record_provenance``, ...).
    """
    from harnesscad.core.loop import HarnessSession
    return HarnessSession(_resolve_backend(backend), **kwargs)


# ---------------------------------------------------------------------------
# The op vocabulary.
# ---------------------------------------------------------------------------

def load_ops(specs: Iterable) -> List[Op]:
    """Parse a sequence of op specs into :class:`Op` instances.

    Each spec is either an ``Op`` (passed through) or a JSON-style dict in
    ``Op.to_dict`` form (``{"op": tag, ...}``), which is reconstructed with
    :func:`parse_op`. This is the batch inverse of ``[op.to_dict() for op in ops]``
    -- a thin composition over ``parse_op``, adding no new parsing rules.
    """
    out: List[Op] = []
    for spec in specs:
        out.append(spec if isinstance(spec, Op) else parse_op(spec))
    return out


def op_vocabulary() -> Tuple[str, ...]:
    """The stable op tags an agent may emit, sorted (the CISP op registry keys)."""
    from harnesscad.core.cisp.ops import _REGISTRY
    return tuple(sorted(_REGISTRY))


# ---------------------------------------------------------------------------
# Bench scoring -- the metric/suite dispatch.
# ---------------------------------------------------------------------------

def metrics(kind: Optional[str] = None, tag: Optional[str] = None):
    """Every discovered bench metric (re-export of ``bench.registry.metrics``)."""
    from harnesscad.eval.bench import registry
    return registry.metrics(kind=kind, tag=tag)


def metric(name: str):
    """One named bench metric (re-export of ``bench.registry.metric``)."""
    from harnesscad.eval.bench import registry
    return registry.metric(name)


def score(name: str, pred: dict, gold: dict):
    """Score a pred/gold pair with one named metric through the bench dispatch.

    Sugar for ``bench.registry.metric(name).score(pred, gold)`` -- returns the
    metric's own value (a float or a dict of named numbers). The metric modules
    and their adapters are untouched; this only looks the metric up and calls it.
    """
    return metric(name).score(pred, gold)


def suites() -> Tuple[str, ...]:
    """The named evaluation suites (re-export of ``bench.registry.suites``)."""
    from harnesscad.eval.bench import registry
    return registry.suites()


def suite(name: str):
    """One named suite (re-export of ``bench.registry.suite``)."""
    from harnesscad.eval.bench import registry
    return registry.suite(name)


def run_suite(name: str, samples: Sequence[dict], extra_metrics: Sequence = ()):
    """Run a named suite over samples (re-export of ``bench.registry.run_suite``)."""
    from harnesscad.eval.bench import registry
    return registry.run_suite(name, samples, extra_metrics=extra_metrics)


# ---------------------------------------------------------------------------
# MCP server.
# ---------------------------------------------------------------------------

def mcp_server(backend: str = "stub", **kwargs):
    """A ready :class:`~harnesscad.io.surfaces.mcp.server.MCPServer`.

    Re-export/constructor alias. ``MCPServer`` already resolves a backend name to
    an instance internally (falling back to the stub with a note), so the string is
    passed through unchanged; any further keyword (``session``, ``catalog``,
    ``approval``) is forwarded verbatim.
    """
    from harnesscad.io.surfaces.mcp.server import MCPServer
    return MCPServer(backend, **kwargs)


# ---------------------------------------------------------------------------
# Trajectory -- the ReAct rollout representation.
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """Lazy re-exports for the class names, so importing this module stays cheap.

    ``Trajectory`` / ``TrajectoryStep`` / ``TaskSuite`` / ``Brief`` resolve their
    backing modules only on first access (PEP 562).
    """
    if name in ("Trajectory", "TrajectoryStep"):
        from harnesscad.agents.agent.tool_trajectory import (
            ToolTrajectory, TrajectoryStep)
        return {"Trajectory": ToolTrajectory, "TrajectoryStep": TrajectoryStep}[name]
    if name == "Brief":
        from harnesscad.eval.corpus.spec import Brief
        return Brief
    if name == "TaskSuite":
        return _TaskSuite
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# TaskSuite -- a thin convenience over the sourced DEV corpus split.
#
# The corpus (harnesscad.eval.corpus.dev) is the genuine task loader: every brief
# there derives its ground truth from arithmetic or a published standard. Only the
# DEV split is exposed -- the held-out split is import-restricted by design and is
# never touched here. This class only wraps the existing BRIEFS tuple with
# lookup/iteration helpers; it invents no tasks.
# ---------------------------------------------------------------------------

def load_tasks() -> Tuple:
    """The DEV corpus briefs (re-export of ``eval.corpus.dev.BRIEFS``).

    A tuple of :class:`Brief`; each carries a prompt (``text``), a reference op
    stream and independently-sourced ground truth (volume/bbox/genus).
    """
    from harnesscad.eval.corpus import dev
    return dev.BRIEFS


class _TaskSuite:
    """A lookup/iteration convenience over a tuple of corpus :class:`Brief`.

    Construct via :meth:`dev` (the readable split). Wraps, never rebuilds:
    ``ids()`` / ``by_id()`` delegate to the same briefs ``eval.corpus.dev``
    exposes.
    """

    def __init__(self, briefs: Sequence) -> None:
        self.briefs: Tuple = tuple(briefs)

    @classmethod
    def dev(cls) -> "_TaskSuite":
        """The DEV split -- the corpus you are allowed to look at."""
        return cls(load_tasks())

    def ids(self) -> List[str]:
        return [b.id for b in self.briefs]

    def by_id(self, bid: str):
        for b in self.briefs:
            if b.id == bid:
                return b
        raise KeyError(f"unknown task {bid!r}; known: {', '.join(self.ids())}")

    def __iter__(self):
        return iter(self.briefs)

    def __len__(self) -> int:
        return len(self.briefs)
