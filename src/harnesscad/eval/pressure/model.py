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


class Client(Protocol):
    """Everything the loop needs from a model."""

    name: str

    def complete(self, messages: List[Dict[str, str]], attempt: int) -> str:
        """Return the raw assistant text for this message list."""
        ...


# --------------------------------------------------------------------------- #
# ollama
# --------------------------------------------------------------------------- #
class OllamaClient:
    """A seeded ollama chat client, via litellm."""

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

    def complete(self, messages: List[Dict[str, str]], attempt: int) -> str:
        import litellm  # lazy: the package imports fine without it

        litellm.suppress_debug_info = True
        resp = litellm.completion(
            model=f"ollama/{self.model}",
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
            api_base=self.api_base,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content or ""


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

    def complete(self, messages: List[Dict[str, str]], attempt: int) -> str:
        key = cache_key(self.name, self.seed, self.temperature, attempt, messages)
        hit = self.cache.get(key)
        if hit is not None:
            return hit["text"]
        text = self.inner.complete(messages, attempt)
        self.cache.put(key, {
            "model": self.name,
            "seed": self.seed,
            "temperature": self.temperature,
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

    def complete(self, messages: List[Dict[str, str]], attempt: int) -> str:
        self.calls.append((attempt, [dict(m) for m in messages]))
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
