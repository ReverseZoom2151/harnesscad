"""models — local Ollama model discovery and LLM construction.

The agent is graded on geometry, so the model is swappable and the point of this
module is to make swapping it a one-liner and to report WHICH model built WHICH
part. Everything goes through the provider-neutral seam
(:class:`harnesscad.agents.llm.litellm_backend.LiteLLMClient`); nothing here
imports Ollama directly except the tags probe, which is a plain HTTP GET so this
module imports fine on a machine with no Ollama at all.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

DEFAULT_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

#: Models worth pointing at a GUI, largest first. A CAD plan is a strict-format
#: structured-output task, so the larger checkpoint of each family leads. This is
#: a PREFERENCE order, not a requirement — :func:`discover_models` returns whatever
#: is actually installed, and :func:`select_models` takes ``preferred=`` so the
#: lineup is a PARAMETER, never a hard-coded tag baked into the loop.
#:
#: The eventual run uses exactly these two families; there is no other model.
#: (Left as data, and NOT run here — the campaign is invoked separately with
#: ``--live``.)
PREFERRED = (
    "qwen3.6:35b", "qwen3.6:27b",
    "ornith:35b", "ornith:9b",
)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_gb: float


def discover_models(base: str = DEFAULT_BASE, timeout: float = 5.0) -> List[ModelInfo]:
    """Installed Ollama models, largest first. Empty (never raises) if unreachable."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/api/tags", timeout=timeout) as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 - no Ollama here is a fact, not a crash
        return []
    out = [ModelInfo(m["name"], round(m.get("size", 0) / 1e9, 1))
           for m in data.get("models", [])]
    out.sort(key=lambda m: m.size_gb, reverse=True)
    return out


def largest_available(base: str = DEFAULT_BASE) -> Optional[str]:
    models = discover_models(base)
    return models[0].name if models else None


def select_models(base: str = DEFAULT_BASE, limit: int = 3,
                  preferred: Sequence[str] = PREFERRED) -> List[str]:
    """The models to run the campaign against: the largest installed of the
    ``preferred`` set, deduplicated, capped at ``limit``.

    ``preferred`` is a PARAMETER (default :data:`PREFERRED`) so the lineup is
    swappable at the call site and no dead tag is baked into the loop. A vision
    model is NOT required and never selected here: the design does not need vision
    for tier 0/1, and tier-2 picks are COMPUTED, not seen.
    """
    installed = {m.name for m in discover_models(base)}
    chosen: List[str] = []
    for name in preferred:
        if name in installed and name not in chosen:
            chosen.append(name)
        if len(chosen) >= limit:
            break
    # Fall back to whatever is installed if none of the preferred set is.
    if not chosen:
        chosen = [m.name for m in discover_models(base)[:limit]]
    return chosen


#: The one instruction that turns a single-object JSON reply into the ARRAY the
#: op parser needs. Ollama's ``format='json'`` guarantees a valid JSON value with
#: no prose or code fences (which ``ops_from_json`` requires — it json.loads the
#: whole reply), but left alone the model emits ONE op object; measured, both
#: qwen2.5-coder:7b and :14b returned only ``new_sketch``. Wrapping the ops in an
#: object the parser already unwraps (``_coerce_to_list`` reads the ``ops`` key)
#: is what makes the model emit the full sequence.
_ARRAY_NUDGE = (
    'Return ONLY a JSON object of the exact form {"ops": [ ... ]}. The "ops" array '
    "MUST contain the COMPLETE ordered sequence of operations for the whole part "
    "-- for a simple block that is new_sketch, then add_rectangle (with x, y, w, h), "
    "then extrude (with distance). Include every numeric field. Emit no prose."
)


class _ArrayNudgeLLM:
    """Wraps an :class:`LLM` to (a) force Ollama's JSON mode and (b) append the
    array nudge as a final user turn, so a local model returns the whole op
    sequence rather than a single op object. A pure decorator: it adds one message
    and forwards everything else."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.model = getattr(inner, "model", "ollama")

    def complete(self, messages, tools=None, response_schema=None, **opts):
        from harnesscad.agents.llm.base import user
        return self._inner.complete(list(messages) + [user(_ARRAY_NUDGE)],
                                    tools=tools, response_schema=response_schema,
                                    **opts)

    def stream(self, messages, tools=None, response_schema=None, **opts):
        from harnesscad.agents.llm.base import user
        return self._inner.stream(list(messages) + [user(_ARRAY_NUDGE)],
                                  tools=tools, response_schema=response_schema, **opts)


def make_llm(model: str, base: str = DEFAULT_BASE, temperature: float = 0.0,
             timeout: float = 600.0, nudge: bool = True):
    """A local-Ollama :class:`LLM`, JSON-mode, wrapped in the array nudge.

    Temperature 0 for determinism. A generous timeout: a 32B model with VRAM
    offload on 16 GB is not fast. Set ``nudge=False`` for the bare client.
    """
    from harnesscad.agents.llm.litellm_backend import LiteLLMClient

    client = LiteLLMClient(
        model="ollama/" + model,
        temperature=temperature,
        api_base=base,
        request_timeout=timeout,
        format="json",
    )
    return _ArrayNudgeLLM(client) if nudge else client
