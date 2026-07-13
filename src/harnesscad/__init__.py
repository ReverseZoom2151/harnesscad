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
