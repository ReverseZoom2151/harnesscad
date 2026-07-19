"""Skill trigger routing: list-valued frontmatter + deterministic dispatch.

Source: ``resources/cad_repos/AgentSCAD-main`` (the 13 ``skills/*/SKILL.md``
files and the CLAUDE.md "Skill routing" table). AgentSCAD's skill dialect
differs from the one :mod:`harnesscad.agents.memory.skillpack` already
ingests (earthtojake/text-to-cad) in two ways this module covers:

1. **List-valued frontmatter.** AgentSCAD declares explicit trigger phrases
   as a YAML list::

       triggers:
         - generate cad
         - new cad artifact

   ``skillpack._parse_frontmatter`` folds continuation lines into one string,
   so the list structure is lost. :func:`parse_frontmatter_lists` parses the
   ``key:`` + indented ``- item`` block subset properly (single-line values
   still work), without editing the existing module.

2. **Deterministic routing.** AgentSCAD routes a request to a skill by
   trigger-phrase matching, and its CLAUDE.md states the routing doctrine:
   "When in doubt, invoke the skill. A false positive is cheaper than a false
   negative." :class:`TriggerRouter` implements the matching half as pure
   scoring -- exact-phrase containment beats token overlap, ties break on
   registration order then name -- so skill selection is reproducible and
   auditable rather than a hidden model choice.

VERIFICATION-FIRST INVARIANT: a routed skill name is *retrieval*, not
knowledge. Router entries carry provenance and an ``unverified=True`` flag by
default; the router never surfaces skill body text, and promotion to the
model channel remains the job of the existing Voyager gate
(:meth:`harnesscad.agents.memory.skills.SkillLibrary.add_verified`) via
:mod:`harnesscad.agents.memory.skillpack`.

Stdlib only, deterministic, absolute imports. ``--selfcheck`` covers the
frontmatter list parser and the routing order.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "parse_frontmatter_lists",
    "TriggerEntry",
    "RouteMatch",
    "TriggerRouter",
    "main",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

FrontmatterValue = Union[str, List[str]]


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def parse_frontmatter_lists(text: str) -> Tuple[Dict[str, FrontmatterValue], str]:
    """Split ``---`` frontmatter, preserving ``- item`` list values.

    Understands the subset AgentSCAD's SKILL.md files use: ``key: value``
    single-line entries, ``key:`` followed by indented ``- item`` lines
    (a list), and indented plain continuations (folded into the previous
    single-line value). Returns ``(meta, body)``; a file without frontmatter
    yields ``({}, text)``.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: Dict[str, FrontmatterValue] = {}
    current_key: Optional[str] = None
    body_start = len(lines)
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            body_start = i + 1
            break
        stripped = line.strip()
        indented = line[:1] in (" ", "\t")
        if indented and current_key is not None:
            if stripped.startswith("- "):
                value = meta.get(current_key)
                if isinstance(value, list):
                    value.append(stripped[2:].strip().strip("\"'"))
                elif value in ("", None):
                    meta[current_key] = [stripped[2:].strip().strip("\"'")]
                else:  # scalar followed by a dash line: fold as text
                    meta[current_key] = f"{value} {stripped}".strip()
            else:
                value = meta.get(current_key)
                if isinstance(value, list) and value:
                    value[-1] = f"{value[-1]} {stripped}".strip()
                else:
                    prior = value if isinstance(value, str) else ""
                    meta[current_key] = f"{prior} {stripped}".strip()
            continue
        if ":" in line and not indented:
            key, _, value = line.partition(":")
            current_key = key.strip()
            meta[current_key] = value.strip().strip("\"'")
    return meta, "\n".join(lines[body_start:])


@dataclass(frozen=True)
class TriggerEntry:
    """One routable skill: its name, trigger phrases, and provenance."""
    name: str
    triggers: Tuple[str, ...]
    description: str = ""
    provenance: str = ""
    unverified: bool = True


@dataclass(frozen=True)
class RouteMatch:
    """One scored route: which trigger matched and how."""
    name: str
    score: float
    matched_trigger: str
    kind: str            # "phrase" | "tokens"
    unverified: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "score": self.score,
                "matched_trigger": self.matched_trigger,
                "kind": self.kind, "unverified": self.unverified}


class TriggerRouter:
    """Deterministic request -> skill routing over trigger phrases.

    Scoring: an exact phrase containment scores ``1.0 + phrase-token-count /
    100`` (longer, more specific phrases win); otherwise the best trigger's
    token-overlap Jaccard scores in ``(0, 1)``. Entries below ``min_score``
    are dropped. Ties break on registration order, then name -- never on
    hash order.
    """

    def __init__(self, min_score: float = 0.34) -> None:
        self._entries: List[TriggerEntry] = []
        self.min_score = float(min_score)

    def register(self, entry: TriggerEntry) -> TriggerEntry:
        if not entry.name:
            raise ValueError("a trigger entry needs a name")
        self._entries.append(entry)
        return entry

    def register_skill_md(self, text: str, provenance: str = "") -> TriggerEntry:
        """Register directly from AgentSCAD-dialect SKILL.md text."""
        meta, _body = parse_frontmatter_lists(text)
        name = str(meta.get("name", "")).strip()
        description = str(meta.get("description", "")).strip()
        raw = meta.get("triggers", [])
        triggers: List[str] = list(raw) if isinstance(raw, list) else (
            [raw] if raw else [])
        if not triggers and description:
            triggers = [description]
        return self.register(TriggerEntry(
            name=name, triggers=tuple(t.lower() for t in triggers if t),
            description=description, provenance=provenance))

    @property
    def names(self) -> List[str]:
        return [e.name for e in self._entries]

    def _score_entry(self, entry: TriggerEntry,
                     request_lower: str,
                     request_tokens: frozenset) -> Optional[RouteMatch]:
        best: Optional[RouteMatch] = None
        for trigger in entry.triggers:
            if trigger and trigger in request_lower:
                phrase_tokens = len(_tokens(trigger))
                score = 1.0 + phrase_tokens / 100.0
                candidate = RouteMatch(entry.name, score, trigger, "phrase",
                                       entry.unverified)
            else:
                t_tokens = frozenset(_tokens(trigger))
                if not t_tokens or not request_tokens:
                    continue
                overlap = len(t_tokens & request_tokens)
                if overlap == 0:
                    continue
                score = overlap / len(t_tokens | request_tokens)
                candidate = RouteMatch(entry.name, score, trigger, "tokens",
                                       entry.unverified)
            if best is None or candidate.score > best.score:
                best = candidate
        return best

    def route(self, request: str, limit: int = 3) -> List[RouteMatch]:
        """Ranked matches for a request, best first, deterministic."""
        request_lower = request.lower()
        request_tokens = frozenset(_tokens(request))
        scored: List[Tuple[float, int, str, RouteMatch]] = []
        for order, entry in enumerate(self._entries):
            match = self._score_entry(entry, request_lower, request_tokens)
            if match is not None and match.score >= self.min_score:
                scored.append((-match.score, order, entry.name, match))
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in scored[:max(0, limit)]]

    def best(self, request: str) -> Optional[RouteMatch]:
        matches = self.route(request, limit=1)
        return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

_GENERATION_MD = """\
---
name: scad-generation
description: Generate new CAD artifacts from natural-language requests.
triggers:
  - generate cad
  - new cad artifact
  - create openscad
---

# body
"""

_REPAIR_MD = """\
---
name: scad-repair
description: Repair OpenSCAD after validation failures.
triggers:
  - repair scad
  - render failed
  - validation failed
  - fix broken openscad
---

# body
"""

_CHAT_MD = """\
---
name: scad-chat
description: Answer questions about an existing CAD artifact.
---

# body
"""


def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    meta, body = parse_frontmatter_lists(_GENERATION_MD)
    check(meta.get("name") == "scad-generation", "scalar value parsed")
    check(meta.get("triggers") == ["generate cad", "new cad artifact",
                                   "create openscad"], "list value parsed")
    check(body.strip() == "# body", "body preserved")
    check(parse_frontmatter_lists("no frontmatter")[0] == {}, "plain text ok")

    router = TriggerRouter()
    router.register_skill_md(_GENERATION_MD, provenance="AgentSCAD-main")
    router.register_skill_md(_REPAIR_MD, provenance="AgentSCAD-main")
    chat = router.register_skill_md(_CHAT_MD, provenance="AgentSCAD-main")
    check(chat.triggers == ("answer questions about an existing cad artifact.",),
          "description doubles as trigger when triggers absent")

    best = router.best("the render failed with a syntax error, please fix it")
    check(best is not None and best.name == "scad-repair", "phrase routing")
    check(best is not None and best.kind == "phrase", "phrase kind")

    best2 = router.best("please generate cad for a mounting plate")
    check(best2 is not None and best2.name == "scad-generation",
          "generation routed")

    check(router.best("what is the weather like") is None,
          "unrelated request routes nowhere")

    ranked = router.route("validation failed while trying to generate cad")
    check(len(ranked) >= 2 and ranked[0].score >= ranked[1].score,
          "ranked output ordered")
    check(all(m.unverified for m in ranked), "matches marked unverified")

    # Determinism: same input, same output.
    check([m.to_dict() for m in ranked]
          == [m.to_dict() for m in router.route(
              "validation failed while trying to generate cad")],
          "routing is deterministic")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("skill_triggers selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Skill trigger routing")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
