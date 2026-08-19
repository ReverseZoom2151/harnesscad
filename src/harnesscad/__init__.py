"""HarnessCAD: a verifier-first agentic harness for text-to-CAD.

Layers:
    core/        the CISP op spine, harness loop, pipeline, CLI
    domain/      geometry, numerics, reconstruction, drawings, CAD domain
    io/          formats, ingestion, kernel backends, adapters, surfaces
    eval/        benchmarks, quality analysis, verifiers, reliability
    agents/      agent loop, LLM layer, generation, RAG, memory, protocols
    data/        dataset engine and generators
    governance/  security, research provenance, audit closure
"""

__version__ = "0.1.0"

# --- programmable SDK surface (lazy) --------------------------------------
# The headline names live in `harnesscad.sdk` (a thin re-export facade). They are
# exposed at top level too, but ONLY through a lazy module `__getattr__` (PEP 562)
# so `import harnesscad` stays cheap and kernel-free: nothing is imported until a
# name is actually accessed. This preserves the package's lazy-import contract --
# many internal modules are import-safe only because the package never eagerly
# pulls a geometry kernel (OCCT/cadquery) at import time.

#: The names re-exported from `harnesscad.sdk` and resolvable at top level.
_SDK_EXPORTS = frozenset({
    "Environment", "Session", "mcp_server",
    "Op", "parse_op", "load_ops", "op_vocabulary",
    "metrics", "metric", "score", "suites", "suite", "run_suite",
    "Trajectory", "TrajectoryStep",
    "TaskSuite", "load_tasks", "Brief",
})


def __getattr__(name):
    """Resolve an SDK headline name lazily from `harnesscad.sdk` on first access."""
    if name in _SDK_EXPORTS:
        from harnesscad import sdk
        return getattr(sdk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_SDK_EXPORTS))
