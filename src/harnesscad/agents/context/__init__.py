"""Context management layer — the finite, precious window, budgeted explicitly.

This package implements docs/blueprint.md sec.3 (`ContextManager`) and sec.7
(context management + the Context Staging Area). Two public surfaces:

- `context.manager` — the `ContextManager` that owns the token window: an
  explicit budget `C >= S + M + T + H + R`, a pluggable `TokenCounter`
  (stdlib `HeuristicCounter` default; an exact tokenizer is a drop-in), a
  pre-flight guard that reports/raises BEFORE a call (never silently truncates),
  a prefix-cache-friendly `assemble` that PINS the system prompt + first user
  message and evicts from the middle (trailing tool results first), and a
  compact `feature_tree_summary` (never a full B-rep dump).
- `context.staging` — the transparent Context Staging Area: a per-task
  `task-context/` model (`01_BRIEF.md`, `02_MODEL/`, `03_DOCS/`) driven by a
  `context.toml` manifest. Explicit, auditable control over what the model sees
  each turn — no black-box RAG.
"""

from __future__ import annotations

from harnesscad.agents.context.manager import (
    BudgetReport,
    ContextManager,
    ContextOverflowError,
    HeuristicCounter,
    TokenCounter,
    AssembledContext,
)
from harnesscad.agents.context.staging import StagingArea

__all__ = [
    "BudgetReport",
    "ContextManager",
    "ContextOverflowError",
    "HeuristicCounter",
    "TokenCounter",
    "AssembledContext",
    "StagingArea",
]
