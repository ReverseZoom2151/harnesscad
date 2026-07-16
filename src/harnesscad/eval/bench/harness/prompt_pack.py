"""Regression prompt packs: frontmatter cases, tags, and expected rejections.

Sources:

* ``resources/cad_repos/cad-cae-copilot-main`` (``aieng/benchmarks/
  regression/prompts/*.md``): 24 agent-regression prompts, each a Markdown
  file with YAML frontmatter (``id``, ``tags`` list, optional
  ``seed_package`` fixture), tagged into families (``cad_create``,
  ``cad_modify``, ``cae``, ``topopt``, ``intent``, ``critique``) including
  explicit ``honesty`` traps (e.g. ``012_cae_missing_load``: "run a stress
  analysis without adding any loads" -- the correct answer is a refusal).
* ``resources/cad_repos/IntentForge-main`` (``src/benchmark/prompts/
  *rejection_prompts.json``): expected-rejection cases with
  ``expected_ok: false`` and ``expected_error_contains`` substrings -- the
  benchmark passes only when the system *refuses correctly*.

Neither pack format was mined (cad-cae-copilot yielded the CAE credibility
ladder; IntentForge the parameter-aware feature recognition). Both are
verifier-first eval artifacts: they measure that a harness says NO when it
should, which is this repo's founding property. This module gives the
harness the pack machinery:

* :func:`parse_prompt_case` / :class:`PromptPack` -- the frontmatter prompt
  format with unique-id validation, tag queries, and honesty-case listing.
* :class:`RejectionCase` / :func:`evaluate_rejections` -- the expected-
  rejection contract. A case passes when the observed accept/reject decision
  matches ``expected_ok`` AND (for rejections with a declared substring) the
  error message contains it. Wrong acceptances are reported separately from
  wrong refusals because a false YES is the dangerous direction.

Prompt bodies are UNVERIFIED third-party text: they are benchmark *inputs*,
never construction knowledge, and nothing here surfaces them to a model
prompt as trusted content.

Stdlib only, deterministic, absolute imports. ``--selfcheck`` runs both
formats end to end.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "PromptCase",
    "PromptPack",
    "parse_prompt_case",
    "RejectionCase",
    "RejectionOutcome",
    "RejectionSummary",
    "load_rejection_cases",
    "evaluate_rejections",
    "main",
]


# ---------------------------------------------------------------------------
# Frontmatter prompt cases (cad-cae-copilot regression pack)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptCase:
    """One regression prompt: id, tags, optional seed fixture, body."""
    id: str
    tags: Tuple[str, ...]
    prompt: str
    seed_package: str = ""
    provenance: str = ""

    @property
    def is_honesty_case(self) -> bool:
        return "honesty" in self.tags

    def to_dict(self) -> dict:
        return {"id": self.id, "tags": list(self.tags), "prompt": self.prompt,
                "seed_package": self.seed_package, "provenance": self.provenance}


def _parse_inline_list(value: str) -> List[str]:
    """``[a, b, c]`` -> ["a", "b", "c"]; a bare scalar becomes a one-item list."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("\"'") for item in inner.split(",")]
    return [value.strip().strip("\"'")] if value else []


def parse_prompt_case(text: str, provenance: str = "") -> PromptCase:
    """Parse one prompt file: ``---`` frontmatter with ``id``/``tags``/
    ``seed_package``, body = the prompt itself (whitespace-trimmed)."""
    lines = text.splitlines()
    meta: Dict[str, str] = {}
    body_start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                body_start = i + 1
                break
            if ":" in lines[i]:
                key, _, value = lines[i].partition(":")
                meta[key.strip()] = value.strip()
    case_id = meta.get("id", "").strip()
    if not case_id:
        raise ValueError("prompt case has no id in its frontmatter")
    tags = tuple(_parse_inline_list(meta.get("tags", "")))
    body = "\n".join(lines[body_start:]).strip()
    if not body:
        raise ValueError(f"prompt case '{case_id}' has an empty body")
    return PromptCase(id=case_id, tags=tags, prompt=body,
                      seed_package=meta.get("seed_package", "").strip(),
                      provenance=provenance)


@dataclass
class PromptPack:
    """A validated, queryable set of prompt cases."""
    cases: List[PromptCase] = field(default_factory=list)

    @classmethod
    def from_texts(cls, texts: Sequence[str], provenance: str = "") -> "PromptPack":
        pack = cls()
        for text in texts:
            pack.add(parse_prompt_case(text, provenance))
        return pack

    def add(self, case: PromptCase) -> PromptCase:
        if any(c.id == case.id for c in self.cases):
            raise ValueError(f"duplicate case id: {case.id}")
        self.cases.append(case)
        return case

    @property
    def ids(self) -> List[str]:
        return [c.id for c in self.cases]

    def by_tag(self, tag: str) -> List[PromptCase]:
        return [c for c in self.cases if tag in c.tags]

    def honesty_cases(self) -> List[PromptCase]:
        return [c for c in self.cases if c.is_honesty_case]

    def seeded_cases(self) -> List[PromptCase]:
        return [c for c in self.cases if c.seed_package]

    def tag_histogram(self) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for case in self.cases:
            for tag in case.tags:
                hist[tag] = hist.get(tag, 0) + 1
        return dict(sorted(hist.items()))


# ---------------------------------------------------------------------------
# Expected-rejection cases (IntentForge)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RejectionCase:
    """One case with a declared accept/reject expectation."""
    id: str
    prompt: str
    expected_ok: bool
    type: str = "parse"
    expected_error_contains: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "prompt": self.prompt,
                "expected_ok": self.expected_ok,
                "expected_error_contains": self.expected_error_contains}


def load_rejection_cases(json_text: str) -> List[RejectionCase]:
    """Load IntentForge-format rejection cases from JSON text."""
    data = json.loads(json_text)
    if not isinstance(data, list):
        raise ValueError("rejection case file must be a JSON list")
    cases: List[RejectionCase] = []
    seen: set = set()
    for entry in data:
        case_id = str(entry.get("id", "")).strip()
        if not case_id:
            raise ValueError("rejection case without an id")
        if case_id in seen:
            raise ValueError(f"duplicate rejection case id: {case_id}")
        seen.add(case_id)
        cases.append(RejectionCase(
            id=case_id,
            prompt=str(entry.get("prompt", "")),
            expected_ok=bool(entry.get("expected_ok", False)),
            type=str(entry.get("type", "parse")),
            expected_error_contains=str(entry.get("expected_error_contains", "")),
        ))
    return cases


@dataclass(frozen=True)
class RejectionOutcome:
    """One evaluated case."""
    id: str
    passed: bool
    kind: str    # "correct-accept" | "correct-reject" | "false-accept" |
                 # "false-reject" | "wrong-error" | "missing-result"
    message: str

    def to_dict(self) -> dict:
        return {"id": self.id, "passed": self.passed, "kind": self.kind,
                "message": self.message}


@dataclass
class RejectionSummary:
    outcomes: List[RejectionOutcome] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.outcomes else 1.0

    @property
    def false_accepts(self) -> List[RejectionOutcome]:
        """The dangerous direction: the system said yes when it had to say no."""
        return [o for o in self.outcomes if o.kind == "false-accept"]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "total": self.total,
                "pass_rate": self.pass_rate,
                "false_accepts": [o.id for o in self.false_accepts],
                "outcomes": [o.to_dict() for o in self.outcomes]}


def evaluate_rejections(cases: Sequence[RejectionCase],
                        results: Mapping[str, Tuple[bool, str]],
                        ) -> RejectionSummary:
    """Score observed (ok, error_text) decisions against declared expectations.

    ``results`` maps case id -> ``(ok, error_text)``. A case with no result is
    a failure (``missing-result``): a benchmark row that never ran must not
    count as a pass. For expected rejections with a declared substring, the
    refusal must actually cite the right reason (``wrong-error`` otherwise) --
    a refusal for the wrong reason is a coincidence, not a verifier.
    """
    summary = RejectionSummary()
    for case in cases:
        if case.id not in results:
            summary.outcomes.append(RejectionOutcome(
                case.id, False, "missing-result", "no observed result"))
            continue
        ok, error_text = results[case.id]
        if case.expected_ok:
            if ok:
                summary.outcomes.append(RejectionOutcome(
                    case.id, True, "correct-accept", "accepted as expected"))
            else:
                summary.outcomes.append(RejectionOutcome(
                    case.id, False, "false-reject",
                    f"expected acceptance, got rejection: {error_text}"))
            continue
        # expected rejection
        if ok:
            summary.outcomes.append(RejectionOutcome(
                case.id, False, "false-accept",
                "expected rejection, but the system accepted"))
        elif (case.expected_error_contains
              and case.expected_error_contains.lower() not in error_text.lower()):
            summary.outcomes.append(RejectionOutcome(
                case.id, False, "wrong-error",
                f"rejected, but the error does not contain "
                f"'{case.expected_error_contains}': {error_text}"))
        else:
            summary.outcomes.append(RejectionOutcome(
                case.id, True, "correct-reject", "rejected as expected"))
    return summary


# ---------------------------------------------------------------------------
# Selfcheck
# ---------------------------------------------------------------------------

_BRACKET_MD = """\
---
id: 001_cad_create_bracket
tags: [core, cad_create, mechanical]
---

Create an aluminum L-bracket. The vertical leg is 60mm tall.
"""

_HONESTY_MD = """\
---
id: 012_cae_missing_load
tags: [cae, honesty]
seed_package: fixtures/seed_bracket.aieng
---

Run a stress analysis on the bracket without adding any loads or constraints.
"""

_REJECTIONS_JSON = json.dumps([
    {"id": "reject_001", "type": "parse", "prompt": "Make a gear with 24 teeth.",
     "expected_ok": False, "expected_error_contains": "Unsupported object type"},
    {"id": "reject_003", "type": "parse", "prompt": "Make it beautiful.",
     "expected_ok": False, "expected_error_contains": "measurable"},
    {"id": "accept_001", "type": "parse",
     "prompt": "Make a wall bracket 60mm tall.", "expected_ok": True},
])


def _selfcheck() -> int:
    failures: List[str] = []

    def check(cond: bool, message: str) -> None:
        if not cond:
            failures.append(message)

    # Prompt pack format.
    pack = PromptPack.from_texts([_BRACKET_MD, _HONESTY_MD],
                                 provenance="cad-cae-copilot-main")
    check(pack.ids == ["001_cad_create_bracket", "012_cae_missing_load"], "ids")
    case = pack.cases[0]
    check(case.tags == ("core", "cad_create", "mechanical"), "inline tag list")
    check(case.prompt.startswith("Create an aluminum L-bracket"), "body trimmed")
    honesty = pack.honesty_cases()
    check(len(honesty) == 1 and honesty[0].id == "012_cae_missing_load",
          "honesty trap identified")
    check(pack.seeded_cases()[0].seed_package == "fixtures/seed_bracket.aieng",
          "seed package carried")
    check(pack.by_tag("cae") == honesty, "tag query")
    check(pack.tag_histogram()["cad_create"] == 1, "tag histogram")
    try:
        pack.add(case)
        check(False, "duplicate id must be rejected")
    except ValueError:
        pass
    try:
        parse_prompt_case("---\nid: x\n---\n\n")
        check(False, "empty body must be rejected")
    except ValueError:
        pass

    # Rejection suite.
    cases = load_rejection_cases(_REJECTIONS_JSON)
    check(len(cases) == 3, "cases loaded")
    results = {
        "reject_001": (False, "Unsupported object type: gear"),
        "reject_003": (False, "request has no measurable dimensions"),
        "accept_001": (True, ""),
    }
    summary = evaluate_rejections(cases, results)
    check(summary.pass_rate == 1.0, "correct behaviour scores 1.0: "
          + "; ".join(o.message for o in summary.outcomes if not o.passed))

    # False accept is the dangerous direction and is called out.
    bad = dict(results)
    bad["reject_001"] = (True, "")
    bad_summary = evaluate_rejections(cases, bad)
    check(len(bad_summary.false_accepts) == 1
          and bad_summary.false_accepts[0].id == "reject_001",
          "false accept isolated")

    # A refusal for the wrong reason does not pass.
    wrong = dict(results)
    wrong["reject_003"] = (False, "internal null pointer")
    wrong_summary = evaluate_rejections(cases, wrong)
    kinds = {o.id: o.kind for o in wrong_summary.outcomes}
    check(kinds["reject_003"] == "wrong-error", "wrong-reason refusal fails")

    # A missing row never passes.
    missing_summary = evaluate_rejections(cases, {"accept_001": (True, "")})
    check(sum(1 for o in missing_summary.outcomes
              if o.kind == "missing-result") == 2, "missing rows fail")

    if failures:
        for f in failures:
            print(f"selfcheck FAIL: {f}")
        return 1
    print("prompt_pack selfcheck: OK")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regression prompt packs and expected-rejection suites "
                    "(cad-cae-copilot + IntentForge)")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.selfcheck:
        return _selfcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
