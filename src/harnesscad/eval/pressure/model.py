"""The model seam for the pressure test.

Three pieces:

``OllamaClient``  a thin, seeded ollama caller. It goes through litellm (the
                  repo's existing optional LLM dep, already wired in
                  ``agents/llm/litellm_backend.py``) and it always passes
                  ``seed`` and ``temperature`` so a run is reproducible at the
                  model level, not just at the cache level.
``CachedClient``  wraps any client in the disk cache. This is what the runner
                  actually uses, so a re-run costs nothing and produces the same
                  bytes.
``ScriptedClient`` a deterministic, offline stand-in used by the test suite: it
                  replays a canned list of responses. The tests exercise the real
                  loops, the real grader and the real cache with this in place,
                  so the suite never needs ollama running.

``extract_ops`` is the funnel from a raw completion string to CISP ops. Small
models fence their JSON in ```json blocks and prepend prose no matter how loudly
the system prompt forbids it, so we strip fences and take the first balanced JSON
array before handing off to the repo's own ``agents.llm.structured`` validator.
Note that this leniency is applied IDENTICALLY in both arms -- it changes the
absolute solve rate, never the A/B difference.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from harnesscad.agents.llm.structured import ParsedOps, validate_raw
from harnesscad.eval.pressure.cache import CompletionCache, cache_key

DEFAULT_API_BASE = "http://localhost:11434"


# --------------------------------------------------------------------------- #
# Provider routing
#
# litellm is ONE call shape across ~100 providers, so the only thing that ever
# pinned this seam to local models was a hardcoded ``ollama/`` prefix. A model
# string that already names a provider is passed through untouched, which is what
# lets the same runner, cache, solver and graders drive a frontier model.
# --------------------------------------------------------------------------- #
def is_provider_qualified(model: str) -> bool:
    """True when ``model`` already names a litellm provider.

    The rule is a ``/`` BEFORE any ``:``, which is litellm's own convention and
    needs no hardcoded provider list (so a provider added upstream tomorrow works
    today). Ollama tags carry a colon and no slash -- ``qwen3.6:27b``, ``ornith:9b``
    -- while provider-qualified names carry a slash -- ``anthropic/claude-...``,
    ``openai/gpt-...``. The one ambiguous case is a namespaced *local* tag such as
    ``library/llama3``; name it ``ollama/library/llama3`` to force local routing.
    """
    return "/" in model.split(":", 1)[0]


def qualified_model(model: str) -> str:
    """The name to hand litellm: bare tags are ollama, everything else passes through."""
    return model if is_provider_qualified(model) else "ollama/%s" % model


def is_local(model: str) -> bool:
    """Whether this model runs against the local ollama daemon."""
    return qualified_model(model).startswith("ollama/")


#: Providers known to honour a ``seed``. Everything else may ignore it, so a run
#: is reproducible only through the completion CACHE, not at the model level.
SEEDED_PROVIDERS = ("ollama/", "openai/", "azure/")


def seed_is_honoured(model: str) -> bool:
    """Whether ``seed`` actually pins this model's sampling.

    This repo claims determinism, so the claim has to be qualified rather than
    quietly broken: ollama and OpenAI honour ``seed``; most hosted providers
    ignore or reject it. When this returns False a re-run may produce different
    text, and reproducibility comes only from the on-disk completion cache. The
    runners record this alongside the results so a published row cannot imply a
    reproducibility it does not have.
    """
    return qualified_model(model).startswith(SEEDED_PROVIDERS)


class Client(Protocol):
    """Everything the loop needs from a model.

    ``seed`` and ``temperature`` are per-call OVERRIDES, and they exist for
    exactly one reason: Best-of-N cannot be run at temperature 0. Greedy decoding
    is a function of the prompt, so N samples of one prompt are N copies of one
    sample (verified against ollama: qwen2.5-coder:3b at T=0 returns byte-
    identical text for three different seeds). The iterative arms pass neither
    override and are therefore unchanged from v1, byte for byte.
    """

    name: str

    def complete(self, messages: List[Dict[str, str]], attempt: int,
                 seed: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        """Return the raw assistant text for this message list."""
        ...


# --------------------------------------------------------------------------- #
# ollama
# --------------------------------------------------------------------------- #
class ModelClient:
    """A seeded chat client for ANY litellm provider, local or hosted.

    Named for what it does now. ``OllamaClient`` remains as an alias below: six
    call sites construct it, and a bare tag still routes to ollama, so every one
    of them gained frontier providers without changing a line.

    ``api_base`` is applied ONLY for local models. Sending ollama's localhost URL
    to a hosted provider would either 404 or, worse, silently retarget the call.
    """

    def __init__(self, model: str, seed: int = 0, temperature: float = 0.0,
                 api_base: str = DEFAULT_API_BASE, max_tokens: int = 1024,
                 timeout: float = 300.0) -> None:
        self.name = model
        self.model = model
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.api_base = api_base
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)

    @property
    def is_local(self) -> bool:
        return is_local(self.model)

    @property
    def seed_is_honoured(self) -> bool:
        return seed_is_honoured(self.model)

    def complete(self, messages: List[Dict[str, str]], attempt: int,
                 seed: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        import litellm  # lazy: the package imports fine without it

        litellm.suppress_debug_info = True
        kwargs: Dict[str, Any] = {
            "model": qualified_model(self.model),
            "messages": messages,
            "temperature": (self.temperature if temperature is None
                            else float(temperature)),
            "seed": (self.seed if seed is None else int(seed)),
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.is_local:
            kwargs["api_base"] = self.api_base
        else:
            # Hosted providers reject params they do not implement (`seed` on
            # Anthropic, for one). Dropping them is what makes one call shape
            # work everywhere; `seed_is_honoured` is how the loss of model-level
            # reproducibility is REPORTED rather than hidden.
            litellm.drop_params = True
        resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content or ""


#: Back-compat alias. A bare tag still means ollama, so existing call sites are
#: unchanged; the name is kept because six modules construct it.
OllamaClient = ModelClient


class CachedClient:
    """Memoises any Client on disk, keyed by (model, seed, temperature, attempt,
    messages). The cached bytes are the *raw completion text*, so replaying a run
    exercises the parser, the loops and the grader for real -- only the network
    call is skipped."""

    def __init__(self, inner: Client, cache: CompletionCache, seed: int,
                 temperature: float) -> None:
        self.inner = inner
        self.name = inner.name
        self.cache = cache
        self.seed = int(seed)
        self.temperature = float(temperature)

    def complete(self, messages: List[Dict[str, str]], attempt: int,
                 seed: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        s = self.seed if seed is None else int(seed)
        t = self.temperature if temperature is None else float(temperature)
        # The key carries the seed and the temperature, so a T=0.8 draw at seed
        # 20260714 can never be served from the T=0.0 cell's cache entry.
        key = cache_key(self.name, s, t, attempt, messages)
        hit = self.cache.get(key)
        if hit is not None:
            return hit["text"]
        # Only forward the overrides when they are actually set, so a Client that
        # implements the plain two-argument protocol still works untouched -- and
        # so the iterative arms reach the model through the exact call v1 made.
        if seed is None and temperature is None:
            text = self.inner.complete(messages, attempt)
        else:
            text = self.inner.complete(messages, attempt, seed=seed,
                                       temperature=temperature)
        self.cache.put(key, {
            "model": self.name,
            "seed": s,
            "temperature": t,
            "attempt": attempt,
            "messages": messages,
            "text": text,
        })
        return text


class ScriptedClient:
    """An offline Client that replays canned responses in order.

    Used by the test suite so the loops, the metrics and the cache can be proven
    without ollama. ``responses`` may be strings (returned verbatim) or lists of
    op dicts (serialised to JSON for you).
    """

    def __init__(self, responses: Sequence[Any], name: str = "scripted") -> None:
        self.name = name
        self._responses = list(responses)
        self.calls: List[Tuple[int, List[Dict[str, str]]]] = []
        self.draws: List[Tuple[Optional[int], Optional[float]]] = []

    def complete(self, messages: List[Dict[str, str]], attempt: int,
                 seed: Optional[int] = None,
                 temperature: Optional[float] = None) -> str:
        self.calls.append((attempt, [dict(m) for m in messages]))
        self.draws.append((seed, temperature))
        if not self._responses:
            return "[]"
        r = self._responses.pop(0)
        if isinstance(r, str):
            return r
        return json.dumps(r)


# --------------------------------------------------------------------------- #
# raw text -> ops
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _first_json_array(text: str) -> Optional[str]:
    """Return the first balanced top-level [...] in `text`, honouring strings."""
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_ops(raw: str) -> ParsedOps:
    """Raw completion text -> ParsedOps (ops, or an error string to feed back).

    Tries, in order: a fenced block, the first balanced JSON array, the whole
    string. Whichever first yields ops wins. On total failure the error from the
    most promising candidate is returned, because that is the message the model
    will be shown next.
    """
    candidates: List[str] = []
    for m in _FENCE.finditer(raw or ""):
        candidates.append(m.group(1).strip())
    arr = _first_json_array(raw or "")
    if arr:
        candidates.append(arr)
    if raw and raw.strip():
        candidates.append(raw.strip())

    if not candidates:
        return ParsedOps([], error="empty response; expected a JSON array of ops")

    first_error: Optional[str] = None
    for c in candidates:
        parsed = validate_raw(c)
        if parsed.ok:
            return parsed
        if first_error is None:
            first_error = parsed.error
    return ParsedOps([], error=first_error or "could not parse a JSON array of ops")


def ops_to_dicts(parsed: ParsedOps) -> List[dict]:
    return [op.to_dict() for op in parsed.ops]
