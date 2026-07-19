"""Deterministic single-prompt compaction under a character budget.

Harness gap filled: HarnessCAD had no way to shrink ONE oversized prompt
string. This module complements harnesscad.agents.context.manager
(ContextManager), which enforces a token budget across a WHOLE message window
by evicting entire messages from the middle of the history. That is
message-level eviction; this is character-level compaction INSIDE a single
prompt string. The two compose: the manager decides which messages survive,
this module makes any single surviving prompt fit a provider character cap.

Strategy ladder:
  (a) find embedded single-line JSON objects that look like a large structured
      spec (the "looks like spec" predicate is pluggable; the default checks
      for components / placements / dimensions style keys) and re-serialize
      them through descending detail profiles -- standard / tight / minimal
      budgets for list lengths and text truncation, generic over any dict --
      with a final title + dimensions + note fallback;
  (b) trim low-value lines past the budget by a pluggable marker tuple;
  (c) emergency head/tail fit with an explicit inserted notice preserving
      roughly 68 percent of the head.
Raises ValueError if the prompt still exceeds max_chars after all three.

Deterministic: no wall clock, no randomness, no environment reads. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_PROMPT_MAX_CHARS = 32000
DEFAULT_PROMPT_TARGET_CHARS = 28000

# Line markers considered low-value once the budget is exceeded. Pluggable via
# the `low_value_markers` argument of compact_if_needed().
DEFAULT_LOW_VALUE_MARKERS: Tuple[str, ...] = (
    "failure modes",
    "fabrication notes",
    "cad sources",
    "hardware kit",
    "do not add labels",
    "background color",
    "render style",
    "lighting note",
)

# Detail profiles: descending budgets for list lengths and text truncation.
# Generic over any dict -- `list_limit` caps every list, `text_limit` truncates
# every scalar rendered as text, `nested_list_limit` caps lists inside nested
# dicts one level down.
_PROFILES: Tuple[Tuple[str, Dict[str, int]], ...] = (
    ("standard", {"list_limit": 20, "nested_list_limit": 6, "text_limit": 150}),
    ("tight", {"list_limit": 12, "nested_list_limit": 4, "text_limit": 110}),
    ("minimal", {"list_limit": 8, "nested_list_limit": 2, "text_limit": 80}),
)

_EMPTY = (None, "", [], {})


@dataclass(frozen=True)
class PromptCompactionResult:
    """Outcome of a compaction attempt."""

    prompt: str
    original_length: int
    final_length: int
    was_compacted: bool
    strategy: str = "none"


def _truncate(value: Any, limit: int) -> str:
    text = str(value if value is not None else "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def default_spec_predicate(value: Dict[str, Any]) -> bool:
    """Default "looks like a large structured spec" test.

    True when the dict carries CAD-spec style structure: a components list plus
    placements, or components plus dimensions, or an assembly model.
    """
    has_components = bool(value.get("components"))
    has_placements = bool(
        value.get("component_placements") or value.get("placements")
    )
    has_dimensions = bool(
        value.get("dimensions")
        or value.get("dimensions_mm")
        or value.get("external_dimensions_mm")
    )
    return bool(
        value.get("design_assembly_model")
        or (has_components and has_placements)
        or (has_components and has_dimensions)
    )


def _compact_value(value: Any, *, list_limit: int, text_limit: int) -> Any:
    """Compact any JSON value: cap lists, truncate scalars, recurse one level."""
    if isinstance(value, dict):
        compacted = {}
        for key, item in value.items():
            if isinstance(item, list):
                compacted[key] = [
                    _compact_value(entry, list_limit=list_limit, text_limit=text_limit)
                    for entry in item[:list_limit]
                ]
            elif isinstance(item, dict):
                inner = {
                    inner_key: _compact_value(
                        inner_value, list_limit=list_limit, text_limit=text_limit
                    )
                    for inner_key, inner_value in item.items()
                    if inner_value not in _EMPTY
                }
                compacted[key] = inner
            elif isinstance(item, (int, float, bool)) or item is None:
                compacted[key] = item
            else:
                compacted[key] = _truncate(item, text_limit)
        return {k: v for k, v in compacted.items() if v not in _EMPTY}
    if isinstance(value, list):
        return [
            _compact_value(entry, list_limit=list_limit, text_limit=text_limit)
            for entry in value[:list_limit]
        ]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate(value, text_limit)


def _compact_spec(spec: Dict[str, Any], profile: Dict[str, int]) -> Dict[str, Any]:
    """Compact a spec dict under one detail profile, generic over any keys.

    Top-level lists are capped at `list_limit`; lists nested inside dicts are
    capped at `nested_list_limit`; every free-text scalar is truncated to
    `text_limit`; empty values are dropped.
    """
    list_limit = profile["list_limit"]
    nested_limit = profile["nested_list_limit"]
    text_limit = profile["text_limit"]
    compact: Dict[str, Any] = {}
    for key, value in spec.items():
        if isinstance(value, list):
            compact[key] = [
                _compact_value(entry, list_limit=nested_limit, text_limit=text_limit)
                for entry in value[:list_limit]
            ]
        else:
            compact[key] = _compact_value(
                value, list_limit=nested_limit, text_limit=text_limit
            )
    return {k: v for k, v in compact.items() if v not in _EMPTY}


def _fallback_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Last-resort summary: title + dimensions + note."""
    dimensions = {}
    for key in (
        "dimensions",
        "dimensions_mm",
        "external_dimensions_mm",
        "internal_usable_dimensions_mm",
    ):
        if spec.get(key) not in _EMPTY:
            dimensions[key] = spec[key]
    fallback: Dict[str, Any] = {
        "title": _truncate(spec.get("title") or spec.get("name"), 80),
        "note": "spec compacted to prompt character budget",
    }
    if dimensions:
        fallback["dimensions"] = dimensions
    return {k: v for k, v in fallback.items() if v not in _EMPTY}


def _spec_summary_json(spec: Dict[str, Any], budget: int) -> str:
    for _name, profile in _PROFILES:
        text = json.dumps(
            _compact_spec(spec, profile), separators=(",", ":"), ensure_ascii=True
        )
        if len(text) <= budget:
            return text
    return json.dumps(_fallback_spec(spec), separators=(",", ":"), ensure_ascii=True)


def _replace_spec_json(
    prompt: str,
    *,
    target_chars: int,
    spec_predicate: Callable[[Dict[str, Any]], bool],
) -> str:
    lines = prompt.splitlines()
    changed = False
    non_json_chars = sum(len(line) + 1 for line in lines)

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or not spec_predicate(parsed):
            continue
        spec_budget = max(2000, target_chars - (non_json_chars - len(line)) - 500)
        replacement = _spec_summary_json(parsed, spec_budget)
        if replacement != stripped:
            lines[index] = replacement
            changed = True

    return "\n".join(lines) if changed else prompt


def _trim_low_value_lines(
    prompt: str, target_chars: int, markers: Sequence[str]
) -> str:
    lines = prompt.splitlines()
    lowered_markers = tuple(marker.lower() for marker in markers)
    kept: List[str] = []
    kept_chars = 0
    for line in lines:
        lowered = line.lower()
        if kept_chars > target_chars and any(
            marker in lowered for marker in lowered_markers
        ):
            continue
        kept.append(line)
        kept_chars += len(line) + 1
    text = "\n".join(kept)
    if len(text) <= target_chars:
        return text

    shortened = []
    for line in kept:
        if len(line) > 3000:
            line = line[:2999].rstrip() + "..."
        shortened.append(line)
    return "\n".join(shortened)


def _emergency_fit(prompt: str, max_chars: int) -> str:
    notice = (
        "\n[Prompt compacted to provider limit: middle detail removed, preserve "
        "stated dimensions, components, mounting planes, and stage rules.]\n"
    )
    budget = max_chars - len(notice) - 1
    if budget <= 0:
        return prompt[:max_chars]
    head_chars = int(budget * 0.68)
    tail_chars = budget - head_chars
    return prompt[:head_chars].rstrip() + notice + prompt[-tail_chars:].lstrip()


def compact_if_needed(
    prompt: str,
    *,
    max_chars: int = DEFAULT_PROMPT_MAX_CHARS,
    target_chars: int = DEFAULT_PROMPT_TARGET_CHARS,
    label: str = "prompt",
    spec_predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    low_value_markers: Sequence[str] = DEFAULT_LOW_VALUE_MARKERS,
) -> PromptCompactionResult:
    """Compact `prompt` under `target_chars` if it exceeds the budget.

    Applies the strategy ladder documented in the module docstring. Raises
    ValueError if the result still exceeds `max_chars`.
    """
    prompt = prompt or ""
    predicate = spec_predicate or default_spec_predicate
    max_chars = max(1000, int(max_chars or DEFAULT_PROMPT_MAX_CHARS))
    target_chars = max(
        1000, min(int(target_chars or DEFAULT_PROMPT_TARGET_CHARS), max_chars - 1)
    )
    original_length = len(prompt)
    if original_length <= target_chars:
        return PromptCompactionResult(prompt, original_length, original_length, False)

    compacted = _replace_spec_json(
        prompt, target_chars=target_chars, spec_predicate=predicate
    )
    strategy = "spec_summary" if compacted != prompt else "line_budget"
    if len(compacted) > target_chars:
        compacted = _trim_low_value_lines(compacted, target_chars, low_value_markers)
        strategy = f"{strategy}+line_trim"
    if len(compacted) > max_chars:
        compacted = _emergency_fit(compacted, max_chars)
        strategy = f"{strategy}+emergency_fit"

    final_length = len(compacted)
    if final_length > max_chars:
        raise ValueError(
            f"{label} compaction failed: {final_length} chars still exceeds "
            f"{max_chars}."
        )

    was_compacted = compacted != prompt
    return PromptCompactionResult(
        compacted, original_length, final_length, was_compacted, strategy
    )


# --- selfcheck ---------------------------------------------------------------
def _selfcheck() -> int:
    # 1. Under budget: untouched.
    small = compact_if_needed("short prompt", max_chars=2000, target_chars=1500)
    assert not small.was_compacted
    assert small.strategy == "none"
    assert small.prompt == "short prompt"

    # 2. Embedded spec JSON gets summarized through the profile ladder.
    spec = {
        "title": "Test bracket assembly",
        "description": "d" * 900,
        "external_dimensions_mm": [120, 80, 40],
        "components": [
            {"ref_des": f"C{i}", "name": "component " + "x" * 300, "category": "cat"}
            for i in range(60)
        ],
        "component_placements": [
            {"ref_des": f"C{i}", "position_mm": [i, i, 0]} for i in range(60)
        ],
        "notes": ["n" * 400 for _ in range(30)],
    }
    spec_line = json.dumps(spec, separators=(",", ":"))
    prompt = "Render the following assembly.\n" + spec_line + "\nKeep it clean."
    assert default_spec_predicate(spec)
    result = compact_if_needed(prompt, max_chars=8000, target_chars=6000)
    assert result.was_compacted
    assert result.final_length <= 8000
    assert result.final_length < result.original_length
    assert "spec_summary" in result.strategy
    assert "Render the following assembly." in result.prompt
    assert "Keep it clean." in result.prompt

    # 3. Pluggable predicate: a predicate that rejects everything skips (a).
    result_no_spec = compact_if_needed(
        prompt,
        max_chars=8000,
        target_chars=6000,
        spec_predicate=lambda value: False,
    )
    assert result_no_spec.strategy.startswith("line_budget")

    # 4. Low-value line trim by pluggable marker tuple.
    filler = "\n".join("core requirement line " + str(i) for i in range(200))
    junk = "\n".join("fabrication notes: irrelevant detail " + "z" * 50 for _ in range(80))
    trim_prompt = filler + "\n" + junk
    trimmed = compact_if_needed(
        trim_prompt, max_chars=len(trim_prompt) + 100, target_chars=len(filler) + 200
    )
    assert trimmed.was_compacted
    assert "line_trim" in trimmed.strategy
    assert trimmed.prompt.count("fabrication notes") < 80
    assert "core requirement line 0" in trimmed.prompt

    # 5. Emergency head/tail fit inserts the notice and preserves head + tail.
    # Every line is below the per-line shortening threshold and carries no
    # low-value marker, so only the emergency stage can shrink it.
    big = "HEAD-START\n" + "\n".join("m" * 80 for _ in range(200)) + "\nEND-TAIL"
    emergency = compact_if_needed(big, max_chars=5000, target_chars=4000)
    assert emergency.final_length <= 5000
    assert "emergency_fit" in emergency.strategy
    assert "Prompt compacted to provider limit" in emergency.prompt
    assert emergency.prompt.startswith("HEAD-START")
    assert emergency.prompt.endswith("END-TAIL")
    notice_pos = emergency.prompt.index("[Prompt compacted")
    assert notice_pos / emergency.final_length > 0.5  # head-weighted split

    # 6. Determinism: same input, same output.
    again = compact_if_needed(prompt, max_chars=8000, target_chars=6000)
    assert again == result

    print(
        "PASS prompt_compaction selfcheck: 6 scenarios "
        f"(spec {result.original_length}->{result.final_length} chars via "
        f"{result.strategy})"
    )
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic single-prompt compaction."
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="exercise the strategy ladder on synthetic prompts and print PASS.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print(f"SELFCHECK FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
