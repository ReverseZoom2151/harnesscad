"""The local model fleet, and the one place an `LLM` is constructed.

Ten open-weight models served by a local ollama. They are reached through the
harness's existing provider seam -- `agents.llm.litellm_backend.LiteLLMClient`,
which speaks ollama through litellm -- so nothing model-specific leaks into the
loop: the planner sees an `LLM`, exactly as it would with a frontier API.

Determinism: temperature 0 and a fixed `seed` are sent on every request, and the
seed is recorded with every result, so a part in the showcase can be rebuilt by
replaying (model, seed, prompt).
"""

from __future__ import annotations

import os
from typing import Any, List, Tuple

__all__ = ["MODELS", "SEED", "TEMPERATURE", "OLLAMA_API_BASE", "make_llm", "model_slug"]


#: Every model on the box, smallest first (so a sweep fails fast and cheap).
MODELS: Tuple[str, ...] = (
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:14b",
    "deepseek-coder-v2:16b",
    "codellama:7b",
    "llama3.1:8b",
    "mistral:7b",
    "starcoder2:7b",
    "granite-code:8b",
)

SEED = 7
TEMPERATURE = 0.0
OLLAMA_API_BASE = os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434")

#: Ollama's default context window (2048) truncates a long op stream mid-JSON.
#: The system prompt alone is ~1.5k tokens, so the fleet gets a real window.
NUM_CTX = 8192
MAX_TOKENS = 1200
REQUEST_TIMEOUT = 240


def model_slug(model: str) -> str:
    """File-name-safe form of a model id ('qwen2.5-coder:7b' -> 'qwen2.5-coder-7b')."""
    return model.replace(":", "-").replace("/", "-")


def make_llm(model: str, seed: int = SEED, **overrides: Any):
    """A seeded, temperature-0 `LLM` for a local ollama model.

    `model` is the bare ollama tag ('qwen2.5-coder:7b'); the 'ollama/' provider
    prefix litellm needs is added here so callers never carry it around.
    """
    from harnesscad.agents.llm.litellm_backend import LiteLLMClient

    opts = {
        "api_base": OLLAMA_API_BASE,
        "seed": seed,
        "num_ctx": NUM_CTX,
        "max_tokens": MAX_TOKENS,
        "request_timeout": REQUEST_TIMEOUT,
    }
    opts.update(overrides)
    tag = model if model.startswith("ollama/") else f"ollama/{model}"
    return LiteLLMClient(model=tag, temperature=TEMPERATURE, **opts)


def resolve_models(names: List[str] | None) -> List[str]:
    """Validate a --model selection against the fleet (or return the whole fleet)."""
    if not names:
        return list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise KeyError(
            "unknown model(s): %s; known: %s" % (", ".join(unknown), ", ".join(MODELS)))
    return list(names)
