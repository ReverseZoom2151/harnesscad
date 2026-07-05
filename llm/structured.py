"""Turn a raw model response into validated CISP ops.

The model is asked to emit a JSON array of ops (or to call a tool whose argument
is that array). This module is the funnel that converts either shape into a
`list[cisp.ops.Op]` via `cisp.ops.parse_op`, and — crucially for the correction
loop — never lets a malformed response blow up the harness: `validate_ops`
returns an error string the caller can feed back to the model to re-prompt.

If `instructor` is installed it is used to coax structured output out of a live
client; otherwise (and always in tests) we fall back to json + parse_op with our
own lightweight validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from cisp.ops import Op, parse_op, _REGISTRY
from llm.base import CompletionResult

try:  # optional, never required
    import instructor  # noqa: F401

    INSTRUCTOR_AVAILABLE = True
except Exception:  # pragma: no cover - depends on env
    INSTRUCTOR_AVAILABLE = False


class OpParseError(ValueError):
    """Raised (or returned as a message) when a response can't become valid ops."""


@dataclass
class ParsedOps:
    """Outcome of a parse attempt. Exactly one of `ops`/`error` is meaningful."""

    ops: List[Op]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --- extraction ------------------------------------------------------------
def _coerce_to_list(data: Any) -> List[dict]:
    """Accept an array of ops, or an object wrapping them under 'ops'/'operations'."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("ops", "operations", "plan"):
            if isinstance(data.get(key), list):
                return data[key]
        # A single op object.
        if "op" in data:
            return [data]
    raise OpParseError(
        "expected a JSON array of ops (or an object with an 'ops' array), "
        f"got {type(data).__name__}"
    )


def ops_from_json(raw: str) -> List[Op]:
    """Parse a JSON string into a list of validated Ops (raises OpParseError)."""
    if not isinstance(raw, str) or not raw.strip():
        raise OpParseError("empty response; expected a JSON array of ops")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise OpParseError(f"response was not valid JSON: {e}") from e
    return ops_from_obj(data)


def ops_from_obj(data: Any) -> List[Op]:
    """Parse an already-decoded object/list into validated Ops."""
    items = _coerce_to_list(data)
    ops: List[Op] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise OpParseError(f"op[{i}] must be a JSON object, got {type(item).__name__}")
        if "op" not in item:
            raise OpParseError(f"op[{i}] is missing the required 'op' tag")
        tag = item["op"]
        if tag not in _REGISTRY:
            valid = ", ".join(sorted(_REGISTRY))
            raise OpParseError(f"op[{i}] has unknown op '{tag}'; valid ops: {valid}")
        try:
            ops.append(parse_op(item))
        except TypeError as e:
            # parse_op passes dict keys as kwargs; bad/extra params land here.
            raise OpParseError(f"op[{i}] ('{tag}') has invalid parameters: {e}") from e
    return ops


def ops_from_completion(result: CompletionResult) -> List[Op]:
    """Extract ops from a CompletionResult: tool-call arguments first, else text.

    Raises OpParseError on failure.
    """
    if result.has_tool_calls:
        # Concatenate ops across any tool calls the model emitted.
        collected: List[Op] = []
        for tc in result.tool_calls:
            collected.extend(ops_from_json(tc.arguments))
        return collected
    return ops_from_json(result.text)


# --- validate-and-retry ----------------------------------------------------
def validate_ops(result: CompletionResult) -> ParsedOps:
    """Non-raising wrapper: return ParsedOps carrying either ops or an error.

    The caller (planner/runner) can hand the error string straight back to the
    model as a correction prompt — the whole point of a re-promptable seam.
    """
    try:
        ops = ops_from_completion(result)
    except OpParseError as e:
        return ParsedOps([], error=str(e))
    if not ops:
        return ParsedOps([], error="no ops were produced; emit at least one op")
    return ParsedOps(ops)


def validate_raw(raw: str) -> ParsedOps:
    """Same as `validate_ops` but from a raw JSON string."""
    try:
        return ParsedOps(ops_from_json(raw))
    except OpParseError as e:
        return ParsedOps([], error=str(e))
