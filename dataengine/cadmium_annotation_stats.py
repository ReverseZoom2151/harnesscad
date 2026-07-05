"""Corpus-level annotation quality statistics from CADmium.

CADmium (Govindarajan et al., TMLR 01/2026) motivates its GPT-4.1 re-annotation
of DeepCAD by *measuring* corpus properties rather than trusting a model. It shows
that good, human-like CAD descriptions are (Figure 2):

  * concise           -- most annotations fall between 100 and 200 words;
  * lexically diverse -- a higher ratio of unique words per annotation and a
                         faster-rising vocabulary-growth curve (Heaps-style unique
                         words as a function of cumulative token count);
  * naturally precise -- numbers carry a *natural* number of decimal places rather
                         than the excessively long decimal expansions of prior
                         template-driven annotations.

This module computes those descriptive statistics deterministically from raw text
(pure stdlib, no models). It complements ``dataengine.annotation_scorecard`` (which
gates individual candidates by cross-view agreement); here we characterise a whole
*corpus* and can compare two corpora head-to-head on the CADmium axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median
import re
from typing import Iterable, Sequence

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
# A number with an optional fractional part; used for decimal-precision analysis.
_NUMBER = re.compile(r"\d+\.\d+|\d+")


# --------------------------------------------------------------------------- #
# tokenisation primitives
# --------------------------------------------------------------------------- #
def words(text: str) -> tuple:
    """Lower-cased alphabetic word tokens (digits/punctuation excluded)."""
    return tuple(m.group(0).lower() for m in _WORD.finditer(text))


def word_count(text: str) -> int:
    return len(words(text))


def unique_word_count(text: str) -> int:
    return len(set(words(text)))


def unique_word_ratio(text: str) -> float:
    """Type-token ratio: unique words / total words (0.0 for empty text)."""
    toks = words(text)
    return len(set(toks)) / len(toks) if toks else 0.0


def decimal_places(text: str) -> tuple:
    """Number of fractional digits for each numeric literal in ``text``.

    Integers contribute ``0``. ``"3.14 and 2"`` -> ``(2, 0)``.
    """
    out = []
    for m in _NUMBER.finditer(text):
        token = m.group(0)
        out.append(len(token.split(".")[1]) if "." in token else 0)
    return tuple(out)


# --------------------------------------------------------------------------- #
# vocabulary growth curve (Heaps-style)
# --------------------------------------------------------------------------- #
def vocabulary_growth(annotations: Sequence[str]) -> tuple:
    """Cumulative ``(tokens_seen, unique_words_seen)`` after each annotation.

    Annotations are consumed in the given order, so the curve is deterministic.
    Mirrors CADmium Figure 2a (vocabulary as a function of token count).
    """
    seen: set = set()
    tokens_seen = 0
    curve = []
    for text in annotations:
        toks = words(text)
        tokens_seen += len(toks)
        seen.update(toks)
        curve.append((tokens_seen, len(seen)))
    return tuple(curve)


# --------------------------------------------------------------------------- #
# corpus summary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CorpusStats:
    annotations: int
    vocabulary_size: int
    total_words: int
    mean_words: float
    median_words: float
    fraction_in_concise_band: float   # share with 100 <= words <= 200 (Fig. 2c)
    mean_unique_word_ratio: float
    numbers: int
    mean_decimal_places: float
    max_decimal_places: int


def corpus_stats(
    annotations: Sequence[str],
    *,
    concise_band: tuple = (100, 200),
) -> CorpusStats:
    """Descriptive statistics over a corpus of annotation strings."""
    texts = list(annotations)
    if not texts:
        raise ValueError("corpus must contain at least one annotation")
    low, high = concise_band
    counts = [word_count(t) for t in texts]
    ratios = [unique_word_ratio(t) for t in texts]
    vocab: set = set()
    for t in texts:
        vocab.update(words(t))
    decimals = [d for t in texts for d in decimal_places(t)]
    in_band = sum(1 for c in counts if low <= c <= high)
    return CorpusStats(
        annotations=len(texts),
        vocabulary_size=len(vocab),
        total_words=sum(counts),
        mean_words=fmean(counts),
        median_words=float(median(counts)),
        fraction_in_concise_band=in_band / len(texts),
        mean_unique_word_ratio=fmean(ratios),
        numbers=len(decimals),
        mean_decimal_places=fmean(decimals) if decimals else 0.0,
        max_decimal_places=max(decimals) if decimals else 0,
    )


# --------------------------------------------------------------------------- #
# head-to-head corpus comparison on the CADmium axes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CorpusComparison:
    """Which corpus is more concise / diverse / naturally-precise.

    ``more_concise`` prefers the smaller mean word count; ``more_diverse`` the
    larger vocabulary and unique-word ratio; ``more_natural_precision`` the
    smaller mean decimal places (CADmium's argument against excessive precision).
    Each label is ``"a"``, ``"b"`` or ``"tie"``.
    """

    a: CorpusStats
    b: CorpusStats
    more_concise: str
    more_diverse: str
    more_natural_precision: str


def _pick(x: float, y: float, *, prefer_smaller: bool, tol: float = 1e-9) -> str:
    if abs(x - y) <= tol:
        return "tie"
    if prefer_smaller:
        return "a" if x < y else "b"
    return "a" if x > y else "b"


def compare_corpora(
    corpus_a: Iterable[str],
    corpus_b: Iterable[str],
    *,
    concise_band: tuple = (100, 200),
) -> CorpusComparison:
    a = corpus_stats(list(corpus_a), concise_band=concise_band)
    b = corpus_stats(list(corpus_b), concise_band=concise_band)
    diverse_a = a.vocabulary_size + a.mean_unique_word_ratio
    diverse_b = b.vocabulary_size + b.mean_unique_word_ratio
    return CorpusComparison(
        a=a,
        b=b,
        more_concise=_pick(a.mean_words, b.mean_words, prefer_smaller=True),
        more_diverse=_pick(diverse_a, diverse_b, prefer_smaller=False),
        more_natural_precision=_pick(
            a.mean_decimal_places, b.mean_decimal_places, prefer_smaller=True),
    )
