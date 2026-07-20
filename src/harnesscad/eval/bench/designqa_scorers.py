"""Six automatic QA graders for FSAE-rules multimodal QA (DesignQA family).

DesignQA (Doris et al., ASME JCISE 2025; MIT DECODE Lab) ships a six-scorer
*automatic* grading protocol -- one deterministic scorer per QA subset, no LLM
judge. Every one of the six is a PUBLIC, textbook NLP metric (SQuAD v1.1
token-F1, set F1, bag-of-characters F1, yes/no accuracy, BLEU-2, ROUGE-L); this
module reimplements each from its standard published definition. Nothing is
copied from DesignQA's ``eval/metrics/metrics.py`` and no DesignQA corpus is
vendored -- the harness carries the ALGORITHMS, cited to their origin papers.

The harness already scores CAD-QA for *other* benchmarks (QueryCAD
Correct/Partial/Wrong in ``judges.qa_grade_scale``, Query2CAD VQAScore in
``judges.vqa_score``, Text2CAD-Bench VLM rubric in
``protocols.vlm_rubric_scorecard``, CVCAD dimension error in
``verifiers.dimension_qa``, textbook 2-of-3 in ``protocols.qa_scoring``). None
of them implements the SQuAD/ScienceQA token-overlap family below -- that is the
delta this module fills.

The six scorers, each a pure function returning a float in ``[0, 1]``:

  1. :func:`token_f1`            -- SQuAD-style bag-of-WORDS F1 after the
     standard SQuAD normalisation (lowercase, drop punctuation, drop the
     articles a/an/the, collapse whitespace). Used for verbatim rule RETRIEVAL.
  2. :func:`rule_number_f1`      -- set/bag F1 over a list of rule-number
     identifiers (rule COMPILATION: "which rules mention X").
  3. :func:`character_f1`        -- bag-of-CHARACTERS F1 with a synonym-max
     (score against each ``;``-separated synonym, take the best). Component
     DEFINITION naming, where surface spelling matters more than word overlap.
  4. :func:`yesno_accuracy`      -- yes/no accuracy via first-yes/no extraction,
     falling back to the prediction-side sentinel :data:`NOANSWER` (rule
     PRESENCE, and the yes/no half of dimensioning / functional-performance).
  5. :func:`bleu2`               -- BLEU-2 (bigram, clipped modified precision,
     standard brevity penalty) of a generated EXPLANATION against a reference.
  6. :func:`rouge_l`             -- ROUGE-L: longest-common-subsequence F1 of a
     generated explanation against a reference.

Plus :func:`score`, a small dispatcher over :data:`SCORERS`.

IMPORTANT -- what this module does NOT provide: a REFUSAL / cannot-determine
ground-truth class. DesignQA has none; every ground truth is determinate. The
:data:`NOANSWER` sentinel is a PREDICTION-SIDE token (the model emitted no
yes/no), never a gradable ground-truth label, and it always scores 0 against a
real yes/no answer. Do not read it as refusal supervision.

Argument order is uniformly ``(prediction, reference)`` -- the model output
first, the ground truth second -- even where DesignQA's own helpers took the
reference first. Pure stdlib, deterministic, ASCII.
"""

from __future__ import annotations

import argparse
import math
import re
import string
import sys
from collections import Counter
from typing import Callable, Dict, Iterable, List, Sequence, Union

__all__ = [
    "NOANSWER",
    "normalize_answer",
    "token_f1",
    "rule_number_f1",
    "character_f1",
    "extract_yes_no",
    "yesno_accuracy",
    "bleu2",
    "rouge_l",
    "SCORERS",
    "score",
    "main",
]

#: Prediction-side sentinel emitted when no yes/no can be extracted from a
#: prediction. It is NOT a ground-truth label (DesignQA has no refusal GT); it
#: exists only so a non-committal prediction scores 0 against a real yes/no.
NOANSWER = "noanswer"

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


# --------------------------------------------------------------------------- #
# Normalisation (SQuAD v1.1)
# --------------------------------------------------------------------------- #
def normalize_answer(s: str) -> str:
    """The standard SQuAD v1.1 answer normalisation.

    Lowercase, delete punctuation, remove the articles ``a``/``an``/``the``
    (as whole words), and collapse all runs of whitespace to single spaces.
    Punctuation is DELETED (not spaced), matching SQuAD, so ``"F.1.1"`` becomes
    ``"f11"`` -- callers that need the dotted identifier kept whole must not
    route it through here (see :func:`rule_number_f1`).
    """
    text = str(s).lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _f1(pred_tokens: Sequence, ref_tokens: Sequence) -> float:
    """SQuAD bag F1 over two token multisets.

    Both empty -> 1.0 (they agree on emptiness). Exactly one empty, or no
    shared token -> 0.0. Otherwise the harmonic mean of precision
    (shared/len(pred)) and recall (shared/len(ref)).
    """
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2.0 * precision * recall) / (precision + recall)


# --------------------------------------------------------------------------- #
# 1. SQuAD token-F1 (retrieval)
# --------------------------------------------------------------------------- #
def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style bag-of-words F1 after :func:`normalize_answer`.

    The retrieval metric: how well a predicted verbatim span overlaps the
    ground-truth span, order-independent, at word granularity. Range ``[0, 1]``;
    identical (post-normalisation) strings score 1.0, disjoint vocabularies 0.0.
    """
    return _f1(normalize_answer(prediction).split(),
               normalize_answer(reference).split())


# --------------------------------------------------------------------------- #
# 2. Rule-number list F1 (compilation)
# --------------------------------------------------------------------------- #
def _as_rule_list(value: Union[str, Iterable[str]]) -> List[str]:
    """Normalise a rule-number field into a clean list of identifiers.

    Accepts a real list/tuple (ground truth) or a delimited string (a model
    prediction such as ``"T.1.1, T.2.2"``). Identifiers are split on commas,
    stripped, lowercased and de-whitespaced; empties are dropped. The dots in
    ``T.1.1`` are PRESERVED -- rule numbers are compared whole, not tokenised.
    """
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    else:
        parts = str(value).split(",")
    out = []
    for p in parts:
        tok = "".join(str(p).split()).strip().lower()
        if tok:
            out.append(tok)
    return out


def rule_number_f1(prediction: Union[str, Iterable[str]],
                   reference: Union[str, Iterable[str]]) -> float:
    """Bag/set F1 over a list of rule-number identifiers.

    Compilation questions ("list every rule that mentions the accumulator")
    have a LIST ground truth; the prediction is a delimited string. Both sides
    are parsed by :func:`_as_rule_list` and scored with the same bag F1 as
    :func:`token_f1`, but over whole identifiers rather than words. Range
    ``[0, 1]``; identical sets -> 1.0, disjoint -> 0.0.
    """
    return _f1(_as_rule_list(prediction), _as_rule_list(reference))


# --------------------------------------------------------------------------- #
# 3. Bag-of-characters F1 with synonym-max (definition)
# --------------------------------------------------------------------------- #
def _char_bag(s: str) -> List[str]:
    """Normalised, space-free character list of a string."""
    return list(normalize_answer(s).replace(" ", ""))


def character_f1(prediction: str, reference: str) -> float:
    """Bag-of-characters F1, taking the MAX over ``;``-separated synonyms.

    Component-definition answers ("what is this part called?") are short names
    where character overlap is a better signal than word overlap and where the
    ground truth may list several acceptable spellings separated by ``;``
    (e.g. ``"tractive system;ts"``). Score against each synonym and keep the
    best. Range ``[0, 1]``; an exact spelling of any synonym scores 1.0.
    """
    pred = _char_bag(prediction)
    synonyms = [syn for syn in str(reference).split(";")]
    if not synonyms:
        synonyms = [str(reference)]
    return max(_f1(pred, _char_bag(syn)) for syn in synonyms)


# --------------------------------------------------------------------------- #
# 4. Yes/No accuracy with the noanswer sentinel (presence)
# --------------------------------------------------------------------------- #
def extract_yes_no(prediction: str) -> str:
    """Return the FIRST ``"yes"``/``"no"`` word in a prediction, else NOANSWER.

    Matches DesignQA's presence extraction: normalise, scan tokens, take the
    first polar word. A prediction with neither yields :data:`NOANSWER` -- a
    prediction-side "did not answer" marker, never a ground-truth class.
    """
    for tok in normalize_answer(prediction).split():
        if tok == "yes":
            return "yes"
        if tok == "no":
            return "no"
    return NOANSWER


def yesno_accuracy(prediction: str, reference: str) -> float:
    """1.0 if the prediction's first yes/no matches the ground truth, else 0.0.

    The ground truth is itself normalised to its first yes/no. A prediction
    from which no yes/no can be read scores 0.0 (its extracted token is
    :data:`NOANSWER`, which never equals a real ``yes``/``no``). This is the
    exact-match accuracy DesignQA reports for the presence subset and for the
    yes/no half of the dimensioning / functional-performance subsets.
    """
    ref = extract_yes_no(reference)
    if ref == NOANSWER:
        # Ground truth must be a real yes/no; if the caller passed something
        # non-polar, fall back to normalised string equality so the contract
        # (identical -> 1.0) still holds.
        ref_norm = normalize_answer(reference)
        return 1.0 if normalize_answer(prediction) == ref_norm and ref_norm \
            else 0.0
    return 1.0 if extract_yes_no(prediction) == ref else 0.0


# --------------------------------------------------------------------------- #
# 5. BLEU-2 (bigram)
# --------------------------------------------------------------------------- #
def _bleu_tokens(text: str) -> List[str]:
    """ScienceQA-style tokenisation: split on whitespace and periods."""
    return [t for t in re.split(r"\s|\.", str(text)) if t]


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(hyp: Sequence[str], ref: Sequence[str], n: int) -> float:
    hyp_ngrams = _ngram_counts(hyp, n)
    total = sum(hyp_ngrams.values())
    if total == 0:
        return 0.0
    ref_ngrams = _ngram_counts(ref, n)
    clipped = sum(min(c, ref_ngrams.get(g, 0)) for g, c in hyp_ngrams.items())
    return clipped / total


def bleu2(prediction: str, reference: str) -> float:
    """BLEU-2 of a prediction against a single reference (weights 1/2, 1/2).

    Geometric mean of the clipped modified unigram and bigram precisions, times
    the standard brevity penalty ``BP = 1 if c > r else exp(1 - r/c)`` (c = #
    prediction tokens, r = # reference tokens). No smoothing: if either
    precision is 0 (e.g. the prediction has no matching bigram, or fewer than 2
    tokens) BLEU-2 is 0.0, matching the unsmoothed sentence-BLEU DesignQA uses.
    Range ``[0, 1]``; identical token streams score 1.0.
    """
    hyp = _bleu_tokens(prediction)
    ref = _bleu_tokens(reference)
    c, r = len(hyp), len(ref)
    if c == 0 or r == 0:
        return 0.0
    p1 = _modified_precision(hyp, ref, 1)
    p2 = _modified_precision(hyp, ref, 2)
    if p1 == 0.0 or p2 == 0.0:
        return 0.0
    geo_mean = math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2))
    bp = 1.0 if c > r else math.exp(1.0 - r / c)
    return bp * geo_mean


# --------------------------------------------------------------------------- #
# 6. ROUGE-L (LCS F1)
# --------------------------------------------------------------------------- #
def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of two token sequences."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L: balanced (beta=1) longest-common-subsequence F1.

    ``P = LCS / len(prediction)``, ``R = LCS / len(reference)`` over
    whitespace/period tokens; the score is their harmonic mean
    ``2PR/(P+R)``. Range ``[0, 1]``; identical token streams score 1.0, no
    shared subsequence 0.0. (The balanced F1 is the most standard "LCS F1";
    DesignQA leans on a recall-tilted beta, but a symmetric F1 is the neutral
    published definition and is what this module documents and tests.)
    """
    hyp = _bleu_tokens(prediction)
    ref = _bleu_tokens(reference)
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return (2.0 * precision * recall) / (precision + recall)


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
#: The six scorers by DesignQA subset name. Every value is
#: ``Callable[[prediction, reference], float]`` returning a value in ``[0, 1]``.
SCORERS: Dict[str, Callable[..., float]] = {
    "retrieval": token_f1,          # SQuAD token-F1
    "compilation": rule_number_f1,  # rule-number list F1
    "definition": character_f1,     # bag-of-characters F1, synonym-max
    "presence": yesno_accuracy,     # yes/no accuracy (noanswer sentinel)
    "bleu2": bleu2,                 # BLEU-2 explanation score
    "rouge_l": rouge_l,             # ROUGE-L explanation score
}


def score(metric: str, prediction, reference) -> float:
    """Dispatch to one of the six scorers by name.

    ``metric`` is a key of :data:`SCORERS` (a DesignQA subset or explanation
    metric). Raises ``KeyError`` with the valid names on an unknown metric.
    """
    try:
        fn = SCORERS[metric]
    except KeyError:
        raise KeyError("unknown metric %r; valid: %s"
                       % (metric, ", ".join(sorted(SCORERS)))) from None
    return fn(prediction, reference)


# --------------------------------------------------------------------------- #
# Self-check: every value below is hand-derived in the comment beside it.
# --------------------------------------------------------------------------- #
def _isclose(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _selfcheck() -> int:
    # -- identity: every scorer returns 1.0 on identical input -------------- #
    assert token_f1("the quick brown fox", "the quick brown fox") == 1.0
    assert rule_number_f1("T.1.1, T.2.2", ["T.1.1", "T.2.2"]) == 1.0
    assert character_f1("widget", "widget") == 1.0
    assert yesno_accuracy("yes", "yes") == 1.0
    assert bleu2("the quick brown fox", "the quick brown fox") == 1.0
    assert rouge_l("the quick brown fox", "the quick brown fox") == 1.0

    # -- disjoint: every scorer returns 0.0 on non-overlapping input -------- #
    assert token_f1("red green blue", "one two three") == 0.0
    assert rule_number_f1("A.1", ["Z.9"]) == 0.0
    assert character_f1("xyz", "qw") == 0.0            # {x,y,z} vs {q,w}
    assert yesno_accuracy("no", "yes") == 0.0
    assert bleu2("alpha beta gamma", "one two three") == 0.0
    assert rouge_l("alpha beta", "one two") == 0.0

    # -- token_f1 partial: "the red box" vs "the blue box" ------------------ #
    # normalise drops the article "the" -> [red, box] vs [blue, box].
    # shared = {box}; precision 1/2, recall 1/2; F1 = 2*.25/1 = 0.5.
    assert _isclose(token_f1("the red box", "the blue box"), 0.5)

    # -- rule_number_f1 partial: pred has an extra spurious rule ------------ #
    # pred [t.1.1, t.2.2, v.3] vs gt [t.1.1, t.2.2]: shared 2,
    # precision 2/3, recall 2/2=1; F1 = 2*(2/3)*1 / (2/3 + 1) = 4/5 = 0.8.
    assert _isclose(rule_number_f1("T.1.1, T.2.2, V.3", ["T.1.1", "T.2.2"]),
                    0.8)

    # -- character_f1 synonym-max: exact match on the 2nd synonym ----------- #
    # pred "ts" vs "tractive system;ts": best synonym is "ts" -> 1.0.
    assert character_f1("ts", "tractive system;ts") == 1.0
    # character_f1 partial: "cat" vs "car": chars {c,a,t} vs {c,a,r},
    # shared 2, precision 2/3, recall 2/3; F1 = 2/3.
    assert _isclose(character_f1("cat", "car"), 2.0 / 3.0)

    # -- yes/no sentinel: a non-committal prediction scores 0 --------------- #
    # "I cannot tell" has no yes/no -> extract == NOANSWER -> 0 vs a real GT,
    # against BOTH an answerable "yes" and "no". NOANSWER is prediction-side
    # only: it is proven as a sentinel, NOT as refusal ground truth.
    assert extract_yes_no("I cannot tell from the image") == NOANSWER
    assert yesno_accuracy("I cannot tell from the image", "yes") == 0.0
    assert yesno_accuracy("I cannot tell from the image", "no") == 0.0
    # first-polar-word extraction: "No, wait, yes" -> first is "no".
    assert extract_yes_no("No, wait, yes") == "no"
    assert yesno_accuracy("Yes, the bracket is present.", "yes") == 1.0

    # -- BLEU-2 partial: "a b c d" (pred) vs "a b c e" (ref) ----------------- #
    # unigrams: a,b,c match (d not in ref) -> p1 = 3/4.
    # bigrams (a,b),(b,c),(c,d) vs (a,b),(b,c),(c,e): match 2 -> p2 = 2/3.
    # c=4, r=4 -> BP=1. BLEU-2 = sqrt(3/4 * 2/3) = sqrt(1/2) = 0.70710678...
    assert _isclose(bleu2("a b c d", "a b c e"), math.sqrt(0.5))
    # BLEU-2 brevity penalty: pred shorter than ref.
    # pred "a b" vs ref "a b c d": p1=2/2=1, p2 (a,b)=1/1=1; c=2,r=4,
    # BP = exp(1 - 4/2) = exp(-1). BLEU-2 = exp(-1) * 1 = 0.36787944...
    assert _isclose(bleu2("a b", "a b c d"), math.exp(-1.0))
    # BLEU-2 is 0 when the prediction is a single token (no bigram exists).
    assert bleu2("hello", "hello world foo") == 0.0

    # -- ROUGE-L partial: LCS length 3 ------------------------------------- #
    # pred "a b c e" vs ref "a b c d": LCS = "a b c" = 3.
    # P = 3/4, R = 3/4; F1 = 2*(3/4)*(3/4)/(3/4+3/4) = 0.75.
    assert _isclose(rouge_l("a b c e", "a b c d"), 0.75)
    # ROUGE-L with differing lengths: pred "a b c" vs ref "a b c d e".
    # LCS 3; P = 3/3 = 1, R = 3/5 = 0.6; F1 = 2*1*0.6/(1+0.6) = 0.75.
    assert _isclose(rouge_l("a b c", "a b c d e"), 0.75)
    # LCS is subsequence, not substring: "a x b y c" vs "a b c" -> LCS 3.
    # P = 3/5, R = 3/3 = 1; F1 = 2*0.6*1/(0.6+1) = 0.75.
    assert _isclose(rouge_l("a x b y c", "a b c"), 0.75)

    # -- dispatcher routes to the right scorer ------------------------------ #
    assert _isclose(score("retrieval", "the red box", "the blue box"), 0.5)
    assert _isclose(score("compilation", "T.1.1, T.2.2, V.3",
                          ["T.1.1", "T.2.2"]), 0.8)
    assert _isclose(score("definition", "cat", "car"), 2.0 / 3.0)
    assert score("presence", "yes", "yes") == 1.0
    assert _isclose(score("bleu2", "a b c d", "a b c e"), math.sqrt(0.5))
    assert _isclose(score("rouge_l", "a b c e", "a b c d"), 0.75)
    try:
        score("nonesuch", "a", "b")
    except KeyError:
        pass
    else:
        raise AssertionError("dispatcher accepted an unknown metric")

    # -- every scorer's range is [0, 1] on the cases above ------------------ #
    for name, fn in SCORERS.items():
        v = fn("a b c d", "a b c e") if name not in ("compilation",) \
            else fn("a, b, c, d", ["a", "b", "c", "e"])
        assert 0.0 <= v <= 1.0, "%s out of range: %r" % (name, v)

    print("SELFCHECK OK: 6 scorers verified against hand-computed values "
          "(identity=1.0, disjoint=0.0, token_f1=0.5, rule_f1=0.8, "
          "char_f1=2/3, BLEU-2=sqrt(1/2), ROUGE-L=0.75); noanswer proven as a "
          "prediction-side sentinel (NOT refusal ground truth); dispatcher OK.")
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="DesignQA-family automatic QA graders: six public NLP "
                    "metrics (SQuAD token-F1, rule-number F1, "
                    "bag-of-characters F1, yes/no accuracy, BLEU-2, ROUGE-L).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="prove each scorer against hand-computed values; "
                             "exit 0 on success.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
