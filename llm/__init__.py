"""Provider-agnostic LLM layer for the harness.

The point of this package is provider independence: everything above talks to
the `LLM` Protocol (llm.base), never to a concrete SDK. `LiteLLMClient` is one
implementation (any model litellm speaks); a `MockLLM` in the tests is another.
"""

from llm.base import (
    LLM,
    Message,
    ToolSpec,
    ToolCall,
    CompletionResult,
    system,
    user,
    assistant,
)

__all__ = [
    "LLM",
    "Message",
    "ToolSpec",
    "ToolCall",
    "CompletionResult",
    "system",
    "user",
    "assistant",
]
