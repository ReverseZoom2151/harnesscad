"""memory — the grounding layer (blueprint sec.8).

Four memory types (MemoryStore: working / episodic / semantic / procedural) plus
a Voyager-style, execution-verified SkillLibrary of parametric CAD skills that
grows monotonically. Dependency-free (stdlib only); retrieval uses an embedding-
free similarity with a pluggable interface for a real embedder later.
"""

from __future__ import annotations

from harnesscad.agents.memory.store import (
    Episode,
    MemoryStore,
    Similarity,
    TokenOverlapSimilarity,
)
from harnesscad.agents.memory.similarity import (
    BM25Similarity,
    EmbeddingSimilarity,
    default_similarity,
    make_similarity,
)
from harnesscad.agents.memory.skills import (
    Skill,
    SkillLibrary,
    build_default_library,
    default_expanders,
    plate_skill,
    bracket_skill,
    plate_ops,
    bracket_ops,
)
from harnesscad.agents.memory.harness_memory import (
    HarnessMemory,
    OracleVerdict,
    Recalled,
    gate_oracle,
)

__all__ = [
    "Episode",
    "HarnessMemory",
    "OracleVerdict",
    "Recalled",
    "gate_oracle",
    "MemoryStore",
    "Similarity",
    "TokenOverlapSimilarity",
    "BM25Similarity",
    "EmbeddingSimilarity",
    "default_similarity",
    "make_similarity",
    "Skill",
    "SkillLibrary",
    "build_default_library",
    "default_expanders",
    "plate_skill",
    "bracket_skill",
    "plate_ops",
    "bracket_ops",
]
