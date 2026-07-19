"""Tolerant structured-JSON recovery for malformed or truncated model output.

The escalation ladder strips markdown, extracts a JSON document, repairs
truncated structure, and prunes a truncated tail before shape validation. The
repairer is stdlib-only: a character-by-character walk tracks the open
container stack and in-string state, then closes only structure left open.

Gap filled: harnesscad.agents.llm.structured assumes the provider text is at
least syntactically valid JSON -- json.loads either succeeds or the whole
response is rejected and re-prompted. Real providers routinely wrap JSON in
markdown fences, prepend prose, or get cut off mid-array by a token budget.
This module is the lower-level tolerant-repair layer that structured.py lacks:
it recovers a decodable Python object from such text BEFORE any op-stream
shape validation. It complements (and never duplicates) structured.py -- the
output of loads_with_salvage can be handed to structured.ops_from_obj, which
remains the sole authority on whether the ops themselves are valid.

Deterministic and stdlib-only: no wall clock, no randomness, no third-party
repair library. Salvage only closes structure and drops trailing garbage; it
never invents field content, so a semantically wrong record still fails the
downstream validation that gates the result.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional, Sequence, Tuple

__all__ = [
    "strip_json_markdown",
    "extract_json_document",
    "salvage_truncated_json",
    "prune_truncated_tail",
    "prune_last_list_item",
    "loads_with_salvage",
]

# Note text emitted when salvage had to append closers; loads_with_salvage
# keys the prune step off this marker.
_FORCE_CLOSED = "force-closed"


# --- step 1: peel markdown fences -------------------------------------------
def strip_json_markdown(text: str) -> str:
    """Strip a surrounding markdown code fence (``` or ```json) if present.

    Mirrors Forma-OSS _strip_json_markdown: drop the first line when it opens
    a fence and the last line when it closes one, otherwise return the text
    stripped of outer whitespace.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


# --- step 2: extract an embedded complete document ---------------------------
def extract_json_document(text: str) -> Optional[str]:
    """Return the first complete top-level JSON object/array substring, or None.

    Mirrors Forma-OSS _extract_json_document: scan for a '{' or '[' and let the
    stdlib decoder attempt a raw_decode from there (which itself honours
    strings and escapes), returning the exact matched substring on success.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in {"{", "["}:
            continue
        try:
            _, end_index = decoder.raw_decode(text[index:])
            return text[index : index + end_index]
        except json.JSONDecodeError:
            continue
    return None


# --- step 3: repair a truncated document -------------------------------------
def _scan(text: str) -> Tuple[List[str], bool, Optional[int], List[Tuple[int, List[str]]]]:
    """Walk `text` (which starts at a '{' or '[') tracking JSON structure.

    Returns (open_stack, in_string_at_end, complete_end_index, safe_points):
      - open_stack: the openers ('{'/'[') still unclosed at end of text;
      - in_string_at_end: True if the walk ended inside a string literal;
      - complete_end_index: index one past a balanced top-level close, if the
        document actually completed (then no salvage is needed past it);
      - safe_points: (index, stack_snapshot) for each structural comma, i.e.
        positions where the text can be cut and the remainder discarded while
        keeping every previously completed element.
    """
    stack: List[str] = []
    in_string = False
    escape = False
    safe_points: List[Tuple[int, List[str]]] = []

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()
                if not stack:
                    return stack, False, i + 1, safe_points
            else:
                # Mismatched closer: treat everything from here on as garbage.
                return stack, False, None, safe_points
        elif ch == ",":
            safe_points.append((i, list(stack)))
    return stack, in_string, None, safe_points


def _closers_for(stack: Sequence[str]) -> str:
    return "".join("}" if opener == "{" else "]" for opener in reversed(stack))


def salvage_truncated_json(text: str) -> Tuple[Optional[Any], List[str]]:
    """Repair a truncated/malformed JSON document from the first '{' or '[' onward.

    Stdlib reimplementation of the json_repair role in Forma-OSS
    _salvage_json_text: from the first opener, walk char-by-char tracking the
    open-container stack and in-string state; close any unterminated string,
    drop a trailing comma or half-written token, and append the missing
    closers, then json.loads the result. Falls back to cutting at the last
    structural comma (dropping the partial trailing element) when the direct
    repair still does not decode.

    Returns (object_or_None, notes). Never raises on bad input.
    """
    notes: List[str] = []
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
    if not starts:
        return None, ["no JSON object or array found to salvage"]
    start = min(starts)
    body = text[start:]

    stack, in_string, complete_end, safe_points = _scan(body)

    if complete_end is not None:
        # The document actually completed; decode the balanced prefix.
        try:
            obj = json.loads(body[:complete_end])
        except json.JSONDecodeError:
            return None, ["balanced JSON prefix failed to decode"]
        if complete_end < len(body.rstrip()):
            notes.append("dropped trailing garbage after complete document")
        return obj, notes

    # Direct repair: close string, tidy the tail, append closers.
    candidate = body
    if in_string:
        candidate += '"'
        notes.append("closed unterminated string")
    trimmed = candidate.rstrip()
    if trimmed.endswith(":"):
        trimmed += " null"
        notes.append("completed dangling key with null")
    elif trimmed.endswith(","):
        trimmed = trimmed[:-1]
        notes.append("dropped trailing comma")
    if stack:
        notes.append(
            "%s %d open container(s)" % (_FORCE_CLOSED, len(stack))
        )
    try:
        return json.loads(trimmed + _closers_for(stack)), notes
    except json.JSONDecodeError:
        pass

    # Fallback: cut at the last structural comma (drop the partial trailing
    # token/element entirely) and close whatever was open at that point.
    for cut_index, snapshot in reversed(safe_points):
        cut_notes = list(notes) + [
            "dropped partial trailing element after truncation"
        ]
        if not any(note.startswith(_FORCE_CLOSED) for note in cut_notes) and snapshot:
            cut_notes.append("%s %d open container(s)" % (_FORCE_CLOSED, len(snapshot)))
        try:
            obj = json.loads(body[:cut_index] + _closers_for(snapshot))
        except json.JSONDecodeError:
            continue
        return obj, cut_notes

    return None, notes + ["salvage failed: no decodable repair found"]


# --- step 4: prune the half-written tail --------------------------------------
def prune_last_list_item(obj: Any, path: Sequence[Any]) -> bool:
    """Drop the last element of the list found at `path` inside `obj`.

    `path` is a sequence of dict keys / list indices leading to a list. This is
    the low-level primitive; loads_with_salvage decides (via the salvage notes)
    whether pruning is warranted. Returns True if an element was removed.
    """
    node = obj
    for step in path:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return False
    if isinstance(node, list) and node:
        node.pop()
        return True
    return False


def prune_truncated_tail(obj: Any) -> Tuple[Any, List[str]]:
    """Drop the half-written trailing list item that truncation leaves behind.

    Mirrors Forma-OSS _prune_truncated_tail: after a force-closed salvage, the
    LAST element of the list that was being written is usually incomplete (a
    dict missing keys its siblings all have). Detect exactly that -- last item
    of a list of >= 2 dicts whose key set is a proper subset of the first
    item's -- and drop it. Only the tail of each list is considered, so
    legitimately sparse items elsewhere are kept.
    """
    notes: List[str] = []

    def _prune(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _prune(value, "%s.%s" % (path, key) if path else str(key))
        elif isinstance(node, list) and len(node) >= 2:
            first, last = node[0], node[-1]
            if isinstance(first, dict) and isinstance(last, dict) and set(last) < set(first):
                node.pop()
                notes.append("dropped incomplete trailing item in %s" % (path or "<root>"))
            for i, value in enumerate(node):
                _prune(value, "%s[%d]" % (path, i))

    _prune(obj, "")
    return obj, notes


# --- umbrella -----------------------------------------------------------------
def loads_with_salvage(text: str) -> Tuple[Optional[Any], List[str]]:
    """Decode provider text into a Python object, escalating through repairs.

    Tries in order (mirroring the Forma-OSS _validate_structured_json ladder,
    minus the pydantic step which belongs to the caller):
      1. direct json.loads;
      2. json.loads after stripping markdown fences;
      3. json.loads of the first embedded complete JSON document;
      4. truncation salvage, followed by tail pruning ONLY when a salvage note
         says containers were force-closed (so intact documents are never
         mutated).

    Returns (object_or_None, notes). An empty notes list means the text was
    already clean JSON. Shape validation of the result is the caller's job,
    e.g. harnesscad.agents.llm.structured.ops_from_obj.
    """
    if not isinstance(text, str) or not text.strip():
        return None, ["empty or non-string input"]

    try:
        return json.loads(text), []
    except json.JSONDecodeError:
        pass

    stripped = strip_json_markdown(text)
    if stripped != text.strip():
        try:
            return json.loads(stripped), ["stripped markdown fences"]
        except json.JSONDecodeError:
            pass

    # Accept an extracted document only when it starts at the FIRST opener in
    # the text (i.e. it really is the top-level document with prose around it).
    # An extraction starting later means the top-level document is broken and
    # the "complete" match is just an inner fragment; Forma-OSS relies on
    # downstream pydantic validation to reject such fragments, here we must
    # prefer salvage of the real document instead.
    extracted = extract_json_document(stripped)
    starts = [idx for idx in (stripped.find("{"), stripped.find("[")) if idx != -1]
    first_opener = min(starts) if starts else -1
    extracted_is_top_level = (
        extracted is not None
        and first_opener != -1
        and stripped[first_opener : first_opener + len(extracted)] == extracted
    )
    if extracted is not None and extracted_is_top_level:
        try:
            return json.loads(extracted), ["extracted embedded JSON document"]
        except json.JSONDecodeError:
            pass

    obj, notes = salvage_truncated_json(stripped)
    if obj is not None and any(note.startswith(_FORCE_CLOSED) for note in notes):
        obj, prune_notes = prune_truncated_tail(obj)
        notes = notes + prune_notes
    if obj is not None:
        return obj, notes

    # Last resort: an inner complete fragment is better than nothing.
    if extracted is not None:
        try:
            return json.loads(extracted), ["extracted embedded JSON fragment (top-level document unrecoverable)"]
        except json.JSONDecodeError:
            pass
    return None, notes


# --- CLI ------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. ``--selfcheck`` exercises every rung of the salvage
    ladder on synthetic malformed payloads and asserts the recovered objects."""
    parser = argparse.ArgumentParser(
        prog="python -m harnesscad.agents.llm.json_salvage",
        description="Tolerant structured-JSON recovery (fence stripping, "
        "document extraction, truncation salvage, tail pruning).",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run deterministic salvage checks on synthetic malformed JSON and exit 0 on success.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.selfcheck:
        parser.print_help()
        return 0

    failures: List[str] = []

    def check(label: str, condition: bool) -> None:
        if not condition:
            failures.append(label)

    # 1. Clean JSON passes through untouched.
    obj, notes = loads_with_salvage('{"a": 1}')
    check("clean passthrough", obj == {"a": 1} and notes == [])

    # 2. Markdown fences are peeled.
    obj, notes = loads_with_salvage('```json\n{"a": [1, 2]}\n```')
    check("fence strip", obj == {"a": [1, 2]} and "stripped markdown fences" in notes)

    # 3. Embedded document amid prose is extracted.
    obj, notes = loads_with_salvage('Sure, here you go: {"ops": [{"op": "x"}]} hope that helps')
    check(
        "document extraction",
        obj == {"ops": [{"op": "x"}]} and "extracted embedded JSON document" in notes,
    )

    # 4. Truncated JSON: cut mid-string inside the second element of a list.
    truncated = '{"parts": [{"id": "p1", "len": 4}, {"id": "p2", "le'
    obj, notes = loads_with_salvage(truncated)
    check("truncated salvage decodes", isinstance(obj, dict))
    check(
        "truncated salvage force-closed",
        any(note.startswith("force-closed") for note in notes),
    )
    # The half-written trailing dict must be pruned (subset of first item's keys).
    check(
        "truncated tail pruned",
        isinstance(obj, dict) and obj.get("parts") == [{"id": "p1", "len": 4}],
    )

    # 5. Truncated mid-value with a trailing comma.
    obj, notes = loads_with_salvage('[1, 2, 3,')
    check("trailing comma salvage", obj == [1, 2, 3])

    # 6. Dangling key gets a null, structure closes.
    obj, notes = loads_with_salvage('{"a": 1, "b":')
    check("dangling key salvage", obj == {"a": 1, "b": None})

    # 7. Partial bare token is dropped via the safe-point fallback.
    obj, notes = loads_with_salvage('{"vals": [10, 20, tru')
    check("partial token salvage", obj == {"vals": [10, 20]})

    # 8. Hopeless input returns (None, notes) without raising.
    obj, notes = loads_with_salvage("no json here at all")
    check("hopeless input", obj is None and len(notes) > 0)

    # 9. prune_last_list_item primitive.
    doc = {"a": {"b": [1, 2, 3]}}
    check(
        "prune_last_list_item",
        prune_last_list_item(doc, ["a", "b"]) and doc == {"a": {"b": [1, 2]}},
    )
    check("prune_last_list_item bad path", prune_last_list_item(doc, ["a", "zz"]) is False)

    # 10. prune_truncated_tail leaves intact sibling lists alone.
    doc = {"items": [{"k": 1, "v": 2}, {"k": 3, "v": 4}]}
    doc, prune_notes = prune_truncated_tail(doc)
    check("prune leaves complete tails", doc["items"] == [{"k": 1, "v": 2}, {"k": 3, "v": 4}] and prune_notes == [])

    if failures:
        print("SELFCHECK FAILED: %s" % ", ".join(failures), file=sys.stderr)
        return 1
    print("PASS: json_salvage selfcheck (10 checks, salvage ladder exercised end to end)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
