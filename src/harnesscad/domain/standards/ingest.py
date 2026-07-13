"""Document-to-ruleset ingestion — turn clause text into typed Rule records.

Two paths, same output (:class:`standards.registry.Rule` records, cited to the
exact clause):

  * **With an injected LLM** (:mod:`llm.base` shape) — structured extraction
    against :func:`rule_schema`. Any provider error or unparseable answer falls
    straight back to the heuristic, so ingestion never hard-depends on a network.
  * **Without** — a deterministic regex heuristic that recognises the phrasings
    real standards use ("minimum wall thickness shall be 2 mm", "hole diameter
    must be >= 3 mm", "fillet radius not less than 0.5 mm") and pulls out
    ``(parameter, comparator, limit)`` plus the clause id and a citation string.

Stdlib only; deterministic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from harnesscad.domain.standards.registry import Rule, RulePack


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def rule_schema() -> Dict[str, Any]:
    """JSON schema for one extracted rule (used for LLM structured extraction).

    No network — a plain dict the caller can hand to an LLM backend as a
    ``response_schema`` / tool parameter set.
    """
    return {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "clause": {
                            "type": "string",
                            "description": "The exact clause id, e.g. '3.2'.",
                        },
                        "parameter": {
                            "type": "string",
                            "description": "The constrained quantity, e.g. "
                                           "'wall thickness'.",
                        },
                        "comparator": {
                            "type": "string",
                            "enum": ["<=", ">=", "==", "in", "near"],
                        },
                        "limit": {
                            "type": ["number", "null"],
                            "description": "Numeric bound for <=,>=,==,near.",
                        },
                        "values": {
                            "type": ["array", "null"],
                            "description": "Allowed set for the 'in' comparator.",
                        },
                        "unit": {"type": ["string", "null"]},
                    },
                    "required": ["parameter", "comparator"],
                },
            }
        },
        "required": ["rules"],
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def ingest_standard(text: str, standard: str, version: str,
                    llm: Optional[Any] = None,
                    source: str = "") -> RulePack:
    """Translate clause ``text`` into a :class:`RulePack` of typed rules.

    ``llm`` is optional: when given (an :class:`llm.base.LLM`), structured
    extraction is attempted first and the heuristic is the fallback; when
    ``None`` the heuristic runs directly. Every rule carries its clause id and a
    citation string. Deterministic.
    """
    rules: List[Rule] = []
    if llm is not None:
        try:
            rules = _ingest_with_llm(text, standard, version, llm)
        except Exception:  # noqa: BLE001 - any LLM failure degrades to heuristic
            rules = []
    if not rules:
        rules = ingest_heuristic(text, standard, version)
    return RulePack(
        name=standard,
        version=version,
        rules=rules,
        source=source or f"ingested:{standard}@{version}",
    )


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #
def _ingest_with_llm(text: str, standard: str, version: str,
                     llm: Any) -> List[Rule]:
    """Structured extraction via an injected LLM; returns [] to trigger fallback."""
    from harnesscad.agents.llm.base import Message  # local import keeps registry import-light

    schema = rule_schema()
    system = Message(
        "system",
        "You extract machine-readable engineering rules from standards text. "
        "Return ONLY JSON matching the provided schema. Each rule needs a "
        "parameter, a comparator (<=,>=,==,in,near), a numeric limit (or a "
        "values list for 'in'), and the exact clause id.")
    user = Message("user", text)
    result = llm.complete([system, user], response_schema=schema)

    payload = _extract_json(result)
    if not payload:
        return []
    raw_rules = payload.get("rules") if isinstance(payload, dict) else None
    if not raw_rules:
        return []

    rules: List[Rule] = []
    for i, rr in enumerate(raw_rules):
        try:
            clause = str(rr.get("clause") or f"r{i + 1}")
            comparator = str(rr["comparator"])
            parameter = _norm_param(str(rr["parameter"]))
            limit = rr.get("limit")
            values = rr.get("values")
            rules.append(Rule(
                id=_rule_id(standard, version, clause, i),
                standard=standard, version=version,
                parameter=parameter, comparator=comparator,
                limit=None if limit is None else float(limit),
                values=list(values) if values else None,
                clause=clause,
                citation=_citation(standard, version, clause),
                scope={},
                unit=rr.get("unit"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return rules


def _extract_json(result: Any) -> Optional[dict]:
    """Pull a JSON object out of a CompletionResult (text or first tool call)."""
    # Prefer a structured tool call if present.
    tool_calls = getattr(result, "tool_calls", None) or []
    for tc in tool_calls:
        obj = _try_json(getattr(tc, "arguments", "") or "")
        if obj is not None:
            return obj
    text = getattr(result, "text", None)
    if isinstance(result, str):
        text = result
    if not text:
        return None
    obj = _try_json(text)
    if obj is not None:
        return obj
    # Salvage the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return _try_json(text[start:end + 1])
    return None


def _try_json(s: str) -> Optional[dict]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Heuristic path
# --------------------------------------------------------------------------- #
# Comparator phrases in priority order (longer / negated forms first so
# "not less than" wins over "less than").
_COMPARATOR_PHRASES = [
    (">=", ">="), ("<=", "<="), ("==", "=="),
    ("greater than or equal to", ">="), ("less than or equal to", "<="),
    ("not less than", ">="), ("no less than", ">="),
    ("not greater than", "<="), ("not more than", "<="),
    ("no more than", "<="), ("shall not exceed", "<="),
    ("must not exceed", "<="), ("not exceed", "<="),
    ("at least", ">="), ("at most", "<="),
    ("no smaller than", ">="), ("no larger than", "<="),
    ("greater than", ">="), ("less than", "<="),
    ("minimum of", ">="), ("maximum of", "<="),
    ("minimum", ">="), ("maximum", "<="),
    ("up to", "<="),
    ("exactly", "=="), ("equal to", "=="),
    ("approximately", "near"), ("about", "near"),
    ("nominally", "near"), ("nominal", "near"), ("circa", "near"),
]

# Words that mark where the parameter phrase ends and the value clause begins.
_CUT_MARKERS = [
    "shall", "must", "should", "will",
    "not less than", "no less than", "not greater than", "not more than",
    "no more than", "not exceed", "at least", "at most", "up to",
    "greater than", "less than", "equal to", "exactly",
    "approximately", "about", "nominally",
    ">=", "<=", "==", "=", " is ", " are ", " be ", " of ",
]

# Leading qualifier words to strip from a parameter phrase.
_LEADING_QUALIFIERS = [
    "the ", "a ", "an ", "each ", "every ", "all ", "any ",
    "minimum ", "maximum ", "min ", "max ", "nominal ", "overall ",
]

_CLAUSE_ID_RE = re.compile(
    r"^\s*(?:clause\s+|section\s+|§\s*)?(\d+(?:\.\d+)*)[\.\)]?\s+",
    re.IGNORECASE)

# A number with an optional unit (mm, cm, m, deg, degrees, °, %).
_NUMBER_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(mm|cm|m|deg(?:rees)?|°|%)?", re.IGNORECASE)


def ingest_heuristic(text: str, standard: str, version: str) -> List[Rule]:
    """Deterministic regex extraction of rules from clause text (no network)."""
    rules: List[Rule] = []
    for i, line in enumerate(_clause_lines(text)):
        rule = _parse_clause(line, standard, version, i)
        if rule is not None:
            rules.append(rule)
    return rules


def _clause_lines(text: str) -> List[str]:
    """Split raw text into candidate clause strings (one per line, trimmed)."""
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            out.append(line)
    return out


def _parse_clause(line: str, standard: str, version: str,
                  index: int) -> Optional[Rule]:
    clause_id: Optional[str] = None
    body = line
    m = _CLAUSE_ID_RE.match(line)
    if m:
        clause_id = m.group(1)
        body = line[m.end():]

    num_match = _NUMBER_RE.search(body)
    if not num_match:
        return None
    limit = float(num_match.group(1))
    unit = num_match.group(2)
    if unit:
        unit = unit.lower()
        if unit == "degrees":
            unit = "deg"

    pre = body[:num_match.start()]
    comparator = _detect_comparator(body.lower())
    if comparator is None:
        # A bare "shall be N" with no min/max qualifier is an equality.
        if re.search(r"\b(shall|must|should)\b", pre.lower()):
            comparator = "=="
        else:
            return None

    parameter = _extract_parameter(pre)
    if not parameter:
        return None

    clause = clause_id or f"r{index + 1}"
    return Rule(
        id=_rule_id(standard, version, clause, index),
        standard=standard,
        version=version,
        parameter=parameter,
        comparator=comparator,
        limit=limit,
        clause=clause,
        citation=_citation(standard, version, clause),
        scope={},
        unit=unit,
    )


def _detect_comparator(lowered: str) -> Optional[str]:
    for phrase, comp in _COMPARATOR_PHRASES:
        if phrase in lowered:
            return comp
    return None


def _extract_parameter(pre: str) -> str:
    """The parameter phrase: text before the first cut marker, de-qualified."""
    lowered = pre.lower()
    cut = len(pre)
    for marker in _CUT_MARKERS:
        idx = lowered.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    param = pre[:cut]
    return _norm_param(param)


def _norm_param(param: str) -> str:
    param = " ".join(param.strip().split()).lower()
    changed = True
    while changed:
        changed = False
        for q in _LEADING_QUALIFIERS:
            if param.startswith(q):
                param = param[len(q):]
                changed = True
    # Trim trailing filler.
    param = param.strip(" ,.;:")
    return param


def _rule_id(standard: str, version: str, clause: str, index: int) -> str:
    base = clause if clause else f"r{index + 1}"
    return f"{standard}:{version}:{base}"


def _citation(standard: str, version: str, clause: str) -> str:
    return f"{standard} {version}, clause {clause}"
