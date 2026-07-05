"""CLI entry point: ``python -m surfaces.a2a_server --backend stub --port 9100``.

Wires a default ``harness_factory`` (fresh backend + session + planner per run)
into the HTTP transport and serves forever. The backend is chosen with the same
``_make_backend`` helper the CISP stdio server uses.

The default planner (``_PlatePlanner``) is a dependency-free, offline stand-in
that emits a simple constrained plate so the server produces a valid STEP without
a network/LLM dependency. Swap in ``agent.planner.Planner`` (backed by a real
``LLM``) for genuine text-to-CAD once a model client is configured.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, List, Optional

from cisp.ops import AddRectangle, Constrain, Extrude, NewSketch
from harness import AgentHarness
from llm.structured import ParsedOps
from loop import HarnessSession
from surfaces.a2a_server.app import serve
from surfaces.server import _make_backend


class _PlatePlanner:
    """Offline deterministic planner: one constrained rectangular plate, extruded.

    Emits the plan once; once a solid exists it emits an empty plan so the
    harness converges. Ignores the brief's specifics — a placeholder for a real
    LLM planner, present so the server is runnable with zero external deps.
    """

    def plan_parsed(
        self,
        brief: str,
        state_summary: Optional[dict] = None,
        diagnostics: Optional[List[Any]] = None,
    ) -> ParsedOps:
        if state_summary and state_summary.get("solid_present"):
            return ParsedOps([])
        ops = (
            [NewSketch(), AddRectangle(sketch="sk1")]
            + [Constrain(kind="distance", a="e1", value=20.0) for _ in range(4)]
            + [Extrude(sketch="sk1", distance=5.0)]
        )
        return ParsedOps(list(ops))


def default_harness_factory(backend_name: str = "stub") -> Callable[[], AgentHarness]:
    """Return a zero-arg factory that mints a fresh AgentHarness per run."""

    def factory() -> AgentHarness:
        backend, _name, _note = _make_backend(backend_name)
        session = HarnessSession(backend)
        return AgentHarness(session, _PlatePlanner())

    return factory


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m surfaces.a2a_server",
        description="HarnessCAD A2A (Agent-to-Agent) protocol server.",
    )
    parser.add_argument("--backend", default="stub", choices=["stub", "cadquery"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    args = parser.parse_args(argv)

    factory = default_harness_factory(args.backend)
    print(
        f"HarnessCAD A2A server on http://{args.host}:{args.port} "
        f"(backend={args.backend}); card at "
        f"http://{args.host}:{args.port}/.well-known/agent-card.json"
    )
    serve(factory, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
