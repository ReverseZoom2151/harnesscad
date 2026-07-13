"""`LiteLLMClient` — an `LLM` backed by litellm.

litellm gives us one call shape (`litellm.completion`) across ~100 providers, so
this is the concrete backend behind the provider-agnostic seam. We import litellm
*lazily* inside the methods so this module (and the whole `llm` package) imports
fine even when litellm isn't installed — only actually calling the model needs it.

Temperature defaults to 0 (deterministic-as-possible planning). Tool/function
calls and JSON / structured output are mapped from our neutral types onto
litellm's OpenAI-flavoured arguments.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from harnesscad.agents.llm.base import (
    CompletionResult,
    LLM,
    Message,
    ToolCall,
    ToolSpec,
    messages_to_dicts,
)


class LiteLLMClient(LLM):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, **opts: Any) -> None:
        self.model = model
        self.temperature = temperature
        self.opts = opts  # forwarded to litellm.completion (api_key, base_url, ...)

    # --- request construction -------------------------------------------
    def _build_kwargs(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]],
        response_schema: Optional[Dict[str, Any]],
        stream: bool,
        opts: Dict[str, Any],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(self.opts)
        kwargs.update(opts)
        kwargs["model"] = kwargs.pop("model", self.model)
        kwargs["messages"] = messages_to_dicts(messages)
        kwargs.setdefault("temperature", self.temperature)
        if stream:
            kwargs["stream"] = True
        if tools:
            kwargs["tools"] = [t.to_openai_tool() for t in tools]
            kwargs.setdefault("tool_choice", "auto")
        if response_schema is not None:
            # If given a bare JSON Schema, wrap it as an OpenAI json_schema
            # response_format; if given a ready response_format dict, pass through.
            if response_schema.get("type") == "json_object" or "json_schema" in response_schema:
                kwargs["response_format"] = response_schema
            else:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_schema.get("title", "response"),
                        "schema": response_schema,
                    },
                }
        return kwargs

    # --- LLM protocol ----------------------------------------------------
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **opts: Any,
    ) -> CompletionResult:
        import litellm  # lazy: module imports without litellm installed

        kwargs = self._build_kwargs(messages, tools, response_schema, stream=False, opts=opts)
        resp = litellm.completion(**kwargs)
        return _parse_response(resp)

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        **opts: Any,
    ) -> Iterator[str]:
        import litellm  # lazy

        kwargs = self._build_kwargs(messages, tools, response_schema, stream=True, opts=opts)
        for chunk in litellm.completion(**kwargs):
            text = _chunk_text(chunk)
            if text:
                yield text


# --- response adapters (module-level so they're unit-testable) -------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from either an attr (litellm model objects) or a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_response(resp: Any) -> CompletionResult:
    choices = _get(resp, "choices") or []
    if not choices:
        return CompletionResult(text="", raw=resp)
    choice = choices[0]
    message = _get(choice, "message") or {}
    text = _get(message, "content") or ""
    finish = _get(choice, "finish_reason")

    tool_calls: List[ToolCall] = []
    for tc in _get(message, "tool_calls") or []:
        fn = _get(tc, "function") or {}
        tool_calls.append(
            ToolCall(
                name=_get(fn, "name") or "",
                arguments=_get(fn, "arguments") or "",
                id=_get(tc, "id"),
            )
        )
    return CompletionResult(
        text=text or "",
        tool_calls=tool_calls,
        finish_reason=finish,
        raw=resp,
    )


def _chunk_text(chunk: Any) -> str:
    choices = _get(chunk, "choices") or []
    if not choices:
        return ""
    delta = _get(choices[0], "delta") or {}
    return _get(delta, "content") or ""
