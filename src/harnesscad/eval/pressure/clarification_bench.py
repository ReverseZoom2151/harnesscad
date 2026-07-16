"""Clarification-quality bench: misleading-brief mutators + question grader.

Ported from Pro-CAD (Pro-CAD-main ``config/``): the harness has a
needs_clarification pathway but NO eval of whether the questions it asks are
any good. Pro-CAD's pipeline fixes that with three deterministic pieces:

1. **Misleading-brief mutators** (source:
   ``config/ambiguity_under_specified.py`` and
   ``config/direct_conflict_same_feature_two_values.py`` few-shot corpora):
   take a clean, fully-specified brief and inject exactly one ambiguity,
   recording WHAT_I_CHANGED and the GROUND-TRUTH clarifying question(s) an
   ideal assistant would ask (plus the ground-truth answers):
     * ``under_specified``  -- remove one numeric dimension so multiple
       geometries satisfy the text;
     * ``direct_conflict``  -- state the SAME feature with TWO different
       values so the text is self-contradictory.

2. **Matched-vs-Hallucinated grader** (source:
   ``config/clarification.py`` JUDGE_QUESTION_QUALITY prompts): every
   generated question is categorized as *Matched* (asks for the same missing
   variable as a ground-truth question) or *Hallucinated* (asks for
   something irrelevant); unmatched ground truths are *Missed*. The source
   uses a judge LLM; this port substitutes a token/number-overlap similarity
   -- clearly labeled HEURISTIC -- so the bench runs deterministically with
   no model. Phrasing differences are tolerated (shared content tokens);
   intent differences are not (disjoint tokens score ~0).

Brief shapes follow ``eval/pressure/briefs.py``: mutators operate on the
``Brief.text`` string (the only thing a model is shown), so any brief in the
pressure corpus can be turned into a clarification probe.

Attribution: Pro-CAD (config/clarification.py,
config/ambiguity_under_specified.py,
config/direct_conflict_same_feature_two_values.py). Pure stdlib,
deterministic; no kernel, no model.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: A numeric dimension token: integer or decimal, optionally signed.
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\.?\d)(?!\w)")

#: Words carrying no feature intent, dropped before overlap scoring.
_STOPWORDS = frozenset("""
a an and are as at be by for from in is it its of on or that the this to
was what which with should would please could you your correct value
""".split())

UNDER_SPECIFIED = "ambiguity_under_specified"
DIRECT_CONFLICT = "direct_conflict_same_feature_two_values"


# --------------------------------------------------------------------------- #
# misleading-brief mutators
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MisleadingBrief:
    """A mutated brief plus the ground truth the grader scores against."""

    mutation_type: str
    original_text: str
    misleading_text: str
    what_changed: str
    ground_truth_questions: Tuple[str, ...]
    ground_truth_answers: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "mutation_type": self.mutation_type,
            "original_text": self.original_text,
            "misleading_text": self.misleading_text,
            "what_changed": self.what_changed,
            "ground_truth_questions": list(self.ground_truth_questions),
            "ground_truth_answers": list(self.ground_truth_answers),
        }


def _numeric_occurrences(text: str) -> List[re.Match]:
    return list(_NUMBER_RE.finditer(text))


def _context_window(text: str, start: int, end: int, words: int = 4) -> str:
    """Up to ``words`` words on each side of [start, end), for question text."""
    before = text[:start].split()[-words:]
    after = text[end:].split()[:words]
    return " ".join(before + ["<value>"] + after)


def mutate_under_specified(text: str, occurrence: int = 0) -> MisleadingBrief:
    """Remove one numeric dimension so multiple geometries fit the text.

    Deterministic: the ``occurrence``-th number (0-based, in text order) is
    replaced by the token ``an unspecified value``. The ground-truth question
    asks for exactly that dimension, quoting its surrounding context; the
    ground-truth answer restores the removed value -- mirroring the source's
    QUESTIONS_TO_ASK / ANSWER_TO_QUESTIONS few-shot structure.
    """
    matches = _numeric_occurrences(text)
    if not matches:
        raise ValueError("brief text contains no numeric dimension to remove")
    if not 0 <= occurrence < len(matches):
        raise ValueError(f"occurrence {occurrence} out of range "
                         f"(text has {len(matches)} numbers)")
    m = matches[occurrence]
    value = m.group(0)
    context = _context_window(text, m.start(), m.end())
    mutated = text[:m.start()] + "an unspecified value" + text[m.end():]
    question = (f"What is the missing value in "
                f"\"{context.replace('<value>', '...')}\"?")
    answer = f"The missing value is {value}."
    return MisleadingBrief(
        mutation_type=UNDER_SPECIFIED,
        original_text=text,
        misleading_text=mutated,
        what_changed=(f"{UNDER_SPECIFIED} -- removed the value {value!r} "
                      f"(context: \"{context}\"), so multiple geometries "
                      f"satisfy the description"),
        ground_truth_questions=(question,),
        ground_truth_answers=(answer,),
    )


def _conflicting_value(value: str) -> str:
    """A deterministic same-feature different-value: shift the leading digit.

    Keeps magnitude and format plausible (200 -> 300, 16 -> 26, 7.5 -> 8.5)
    while guaranteeing the two values differ.
    """
    if value.startswith("-"):
        return "-" + _conflicting_value(value[1:])
    lead = int(value[0])
    return str((lead % 9) + 1) + value[1:]


def mutate_direct_conflict(text: str, occurrence: int = 0) -> MisleadingBrief:
    """State the same feature with two different values (self-contradiction).

    The ``occurrence``-th number keeps its original mention, and a second,
    CONFLICTING restatement of the same feature is appended, exactly the
    source's same-feature-two-values pattern. The ground-truth question asks
    which value is correct; the answer picks the original.
    """
    matches = _numeric_occurrences(text)
    if not matches:
        raise ValueError("brief text contains no numeric dimension to conflict")
    if not 0 <= occurrence < len(matches):
        raise ValueError(f"occurrence {occurrence} out of range "
                         f"(text has {len(matches)} numbers)")
    m = matches[occurrence]
    value = m.group(0)
    wrong = _conflicting_value(value)
    context = _context_window(text, m.start(), m.end())
    restatement = (f" Note: the dimension given as "
                   f"\"{context.replace('<value>', value)}\" is {wrong}.")
    mutated = text.rstrip() + restatement
    question = (f"For \"{context.replace('<value>', '...')}\", which value "
                f"is correct: {value} or {wrong}?")
    answer = f"The correct value is {value}."
    return MisleadingBrief(
        mutation_type=DIRECT_CONFLICT,
        original_text=text,
        misleading_text=mutated,
        what_changed=(f"{DIRECT_CONFLICT} -- \"{value}\" vs \"{wrong}\" for "
                      f"the same feature (context: \"{context}\"); the text "
                      f"now specifies it twice with different values"),
        ground_truth_questions=(question,),
        ground_truth_answers=(answer,),
    )


# --------------------------------------------------------------------------- #
# Matched-vs-Hallucinated grader (HEURISTIC)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class QuestionMatch:
    generated_question: str
    matched_ground_truth: str
    score: float

    def to_dict(self) -> dict:
        return {"generated_question": self.generated_question,
                "matched_ground_truth": self.matched_ground_truth,
                "score": round(self.score, 4)}


@dataclass
class ClarificationGrade:
    """The source judge's output shape: matched vs hallucinated (+ missed)."""

    matched: List[QuestionMatch] = field(default_factory=list)
    hallucinated: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        total = len(self.matched) + len(self.hallucinated)
        return len(self.matched) / total if total else 0.0

    @property
    def recall(self) -> float:
        total = len(self.matched) + len(self.missed)
        return len(self.matched) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "matched_questions": [m.to_dict() for m in self.matched],
            "hallucinated_questions": list(self.hallucinated),
            "missed_ground_truth": list(self.missed),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
        }


def _tokens(question: str) -> frozenset:
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", question.lower())
    return frozenset(w for w in words if w not in _STOPWORDS)


def question_similarity(a: str, b: str) -> float:
    """HEURISTIC similarity between two clarification questions.

    Token-overlap coefficient (|intersection| / min size) over lowercased
    content words and numbers. This is a deterministic, model-free stand-in
    for the source's judge-LLM intent matching: same-variable questions share
    the feature nouns and values; unrelated questions do not. It is a
    HEURISTIC -- semantic paraphrases with zero token overlap will be missed.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def grade_clarification(
    generated_questions: Sequence[str],
    ground_truth_questions: Sequence[str],
    threshold: float = 0.5,
) -> ClarificationGrade:
    """Categorize every generated question as Matched or Hallucinated.

    Mirrors the source judge's contract: a generated question is *Matched*
    when it asks for the same missing variable as some ground-truth question
    (similarity >= ``threshold``; each ground truth matches at most once,
    greedy best-score-first for determinism), else *Hallucinated*.
    Ground truths no generated question covered are *Missed*.
    """
    pairs: List[Tuple[float, int, int]] = []
    for gi, gen in enumerate(generated_questions):
        for ti, gt in enumerate(ground_truth_questions):
            score = question_similarity(gen, gt)
            if score >= threshold:
                pairs.append((score, gi, ti))
    # Greedy: best score first; ties break on (generated, ground-truth) index.
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    grade = ClarificationGrade()
    used_gen: set = set()
    used_gt: set = set()
    for score, gi, ti in pairs:
        if gi in used_gen or ti in used_gt:
            continue
        used_gen.add(gi)
        used_gt.add(ti)
        grade.matched.append(QuestionMatch(
            generated_question=generated_questions[gi],
            matched_ground_truth=ground_truth_questions[ti],
            score=score))
    grade.hallucinated = [q for i, q in enumerate(generated_questions)
                          if i not in used_gen]
    grade.missed = [q for i, q in enumerate(ground_truth_questions)
                    if i not in used_gt]
    return grade


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #

_BRIEF = ("This is a rectangular prismatic block with a 200 by 73 "
          "rectangular footprint and a thickness of 22. Sketch a closed "
          "rectangle spanning X=0 to 200 and Y=0 to 73, then extrude it 22 "
          "in the positive normal direction. No other cuts or features.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Misleading-brief mutators + Matched-vs-Hallucinated "
                    "clarification grader, heuristic and model-free "
                    "(Pro-CAD config/ port).",
    )
    parser.add_argument("--selfcheck", action="store_true",
                        help="mutate a synthetic brief both ways and grade "
                             "good, paraphrased, and off-topic questions.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0

    # 1. Under-specified: remove the thickness (occurrence 2 = "22").
    mb = mutate_under_specified(_BRIEF, occurrence=2)
    assert mb.mutation_type == UNDER_SPECIFIED
    assert "thickness of an unspecified value" in mb.misleading_text
    assert "200 by 73" in mb.misleading_text  # only one value removed
    assert mb.ground_truth_questions and "missing value" in \
        mb.ground_truth_questions[0]
    assert "22" in mb.ground_truth_answers[0]
    print(f"[selfcheck] under_specified: {mb.what_changed[:70]}...")

    # 2. Direct conflict: the footprint length 200 restated as 300.
    mb2 = mutate_direct_conflict(_BRIEF, occurrence=0)
    assert mb2.mutation_type == DIRECT_CONFLICT
    assert "200" in mb2.misleading_text and "300" in mb2.misleading_text
    assert "200 or 300" in mb2.ground_truth_questions[0]
    assert "200" in mb2.ground_truth_answers[0]
    print(f"[selfcheck] direct_conflict: {mb2.what_changed[:70]}...")

    # 3. Grader: an on-target paraphrase matches, an off-topic question is
    #    hallucinated, an unasked ground truth is missed.
    gts = list(mb.ground_truth_questions) + [
        "What is the fillet radius on the top edges?"]
    generated = [
        "What thickness should the block be extruded to?",   # paraphrase-ish
        "What color should the part be?",                    # hallucinated
    ]
    # The paraphrase shares tokens with the GT via the quoted context.
    g = grade_clarification(generated, gts, threshold=0.3)
    assert len(g.matched) == 1, g.to_dict()
    assert g.matched[0].generated_question == generated[0]
    assert g.hallucinated == [generated[1]], g.to_dict()
    assert g.missed == [gts[1]], g.to_dict()
    assert 0.0 < g.precision < 1.0 and 0.0 < g.recall < 1.0
    print(f"[selfcheck] grader: matched=1 hallucinated=1 missed=1 "
          f"(precision={g.precision:.2f}, recall={g.recall:.2f}) [HEURISTIC]")

    # 4. Exact-question echo scores 1.0; disjoint questions 0.0.
    assert question_similarity(gts[0], gts[0]) == 1.0
    assert question_similarity("What color?", gts[0]) == 0.0
    print("[selfcheck] similarity extremes: echo=1.0, disjoint=0.0")

    # 5. Grading is deterministic and one-to-one.
    two_gen = [gts[0], gts[0]]  # duplicate generated question
    g2 = grade_clarification(two_gen, [gts[0]])
    assert len(g2.matched) == 1 and len(g2.hallucinated) == 1
    assert grade_clarification(generated, gts, 0.3).to_dict() == g.to_dict()
    print("[selfcheck] deterministic, one ground truth matches at most once")

    # 6. Mutators reject briefs with no numbers.
    try:
        mutate_under_specified("a cube")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("[selfcheck] numberless brief rejected")
    print("[selfcheck] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
