"""memory — the grounding layer (blueprint sec.8).

Four memory types (MemoryStore: working / episodic / semantic / procedural) plus
an execution-verified SkillLibrary of parametric CAD skills that
grows monotonically. Dependency-free (stdlib only); retrieval uses an embedding-
free similarity with a pluggable interface for a real embedder later.

On top of those sits the DUAL-TRACK agent memory with utility-learned retrieval
(``case_library`` + ``skill_utility`` + ``learned_retrieval``): a case track
``C = (I, T, O)`` written back only after verification, a skill track
``K = (Script, Doc, params, applicability, U, stats)`` auto-internalised from
successful trajectories, and a retrieval rule that anneals from similarity
toward MEASURED utility so recall stops preferring cases that merely look right.
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
from harnesscad.agents.memory.case_library import (
    Case,
    CaseIntent,
    CaseLibrary,
    CaseOutcome,
    ToolCall,
)
from harnesscad.agents.memory.skill_utility import (
    SkillDoc,
    UtilitySkill,
    UtilitySkillLibrary,
    internalise_trajectory,
)
from harnesscad.agents.memory.learned_retrieval import (
    CaseValueModel,
    DualTrackAgentMemory,
    EpisodeSelection,
    terminal_reward,
)
from harnesscad.agents.memory.harness_memory import (
    HarnessMemory,
    OracleVerdict,
    Recalled,
    gate_oracle,
)

__all__ = [
    "Case",
    "CaseIntent",
    "CaseLibrary",
    "CaseOutcome",
    "CaseValueModel",
    "DualTrackAgentMemory",
    "EpisodeSelection",
    "SkillDoc",
    "ToolCall",
    "UtilitySkill",
    "UtilitySkillLibrary",
    "internalise_trajectory",
    "terminal_reward",
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
