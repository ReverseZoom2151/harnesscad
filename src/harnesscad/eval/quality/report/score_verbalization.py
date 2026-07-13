"""Graded numeric-to-text verbalisation (ChatCAD+ "prob2text", generalised).

ChatCAD+ (Zhao et al.) observes that raw classifier probabilities are a poor
input to an LLM: a bare ``"cardiomegaly: 0.87"`` gives the model no calibrated
sense of how much to trust or emphasise it. Their fix is a *rule-based*
``prob2text`` layer that maps each numeric score onto a graded natural-language
phrase before it ever reaches the LLM. The paper defines three schemes
(Table I): ``P1`` direct (echo the number), ``P2`` simplistic (two bands), and
``P3`` illustrative (four severity bands) -- and finds the illustrative,
human-styled wording produces the best downstream reports.

That mechanism is entirely domain-agnostic and transfers cleanly to a mechanical
text-to-CAD harness: any scalar in ``[0, 1]`` that will be fed to an LLM as
context -- a verifier's pass-confidence, a tolerance-violation risk, a geometric
fit score, a manufacturability likelihood -- reads better and steers generation
more reliably when it is *verbalised into calibrated language* rather than
handed over as a naked float. This module provides that band-based verbaliser
with the paper's three named schemes plus arbitrary custom band sets, and a
helper to turn a whole ``{label: score}`` mapping into a deterministic,
LLM-ready description block.

Pure stdlib, deterministic, no I/O. No CAD/medical assumptions baked in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "Band",
    "BandScheme",
    "SCHEME_DIRECT",
    "SCHEME_SIMPLISTIC",
    "SCHEME_ILLUSTRATIVE",
    "SCHEMES",
    "verbalise",
    "verbalise_scores",
]


@dataclass(frozen=True)
class Band:
    """One half-open score band ``[lo, hi)`` and its phrase template.

    ``template`` may reference ``{label}`` and ``{score}`` via ``str.format``.
    ``{score}`` is passed as a float so callers can use format specs such as
    ``{score:.2f}``. The final band in a scheme is treated as closed on the
    right so a score of exactly ``hi_max`` (typically ``1.0``) is covered.
    """

    lo: float
    hi: float
    template: str

    def render(self, label: str, score: float) -> str:
        return self.template.format(label=label, score=score)


class BandScheme:
    """An ordered, contiguous set of bands covering ``[lo_min, hi_max]``.

    Bands are validated to be sorted, non-overlapping and gap-free so that every
    score in range maps to exactly one phrase. Lookup is deterministic.
    """

    def __init__(self, name: str, bands: Sequence[Band]) -> None:
        if not bands:
            raise ValueError("a band scheme needs at least one band")
        ordered = list(bands)
        for a, b in zip(ordered, ordered[1:]):
            if not (a.hi <= b.lo + 1e-12 and a.hi >= b.lo - 1e-12):
                raise ValueError(
                    "bands must be contiguous and non-overlapping: "
                    f"{a.hi} != {b.lo}"
                )
            if a.lo > a.hi:
                raise ValueError(f"band has lo > hi: {a}")
        self.name = name
        self.bands: Tuple[Band, ...] = tuple(ordered)
        self.lo_min = ordered[0].lo
        self.hi_max = ordered[-1].hi

    def band_for(self, score: float) -> Band:
        """Return the band containing ``score`` (last band closed on the right)."""
        if score < self.lo_min - 1e-12 or score > self.hi_max + 1e-12:
            raise ValueError(
                f"score {score} outside scheme range "
                f"[{self.lo_min}, {self.hi_max}]"
            )
        for band in self.bands:
            if score < band.hi:
                return band
        return self.bands[-1]  # score == hi_max

    def render(self, label: str, score: float) -> str:
        return self.band_for(score).render(label, score)


# --- The three named schemes from ChatCAD+ Table I, kept domain-neutral. ------

# P1: echo the raw number. Useful as a baseline / when the LLM should reason
# over the exact value rather than a bucket.
SCHEME_DIRECT = BandScheme(
    "direct",
    [Band(0.0, 1.0, "{label} score: {score:.2f}")],
)

# P2: coarse two-band split at the 0.5 decision boundary.
SCHEME_SIMPLISTIC = BandScheme(
    "simplistic",
    [
        Band(0.0, 0.5, "No {label}"),
        Band(0.5, 1.0, "The prediction is {label}"),
    ],
)

# P3: four calibrated severity bands -- the paper's recommended setting.
SCHEME_ILLUSTRATIVE = BandScheme(
    "illustrative",
    [
        Band(0.0, 0.2, "No sign of {label}"),
        Band(0.2, 0.5, "Small possibility of {label}"),
        Band(0.5, 0.9, "Likely to have {label}"),
        Band(0.9, 1.0, "Definitely has {label}"),
    ],
)

SCHEMES: Dict[str, BandScheme] = {
    s.name: s for s in (SCHEME_DIRECT, SCHEME_SIMPLISTIC, SCHEME_ILLUSTRATIVE)
}


def _resolve(scheme) -> BandScheme:
    if isinstance(scheme, BandScheme):
        return scheme
    if isinstance(scheme, str):
        try:
            return SCHEMES[scheme]
        except KeyError:
            raise ValueError(
                f"unknown scheme {scheme!r}; known: {sorted(SCHEMES)}"
            )
    raise TypeError("scheme must be a BandScheme or a scheme name")


def verbalise(label: str, score: float, scheme="illustrative") -> str:
    """Verbalise a single ``score`` for ``label`` using the given scheme.

    ``scheme`` is a scheme name (``"direct"``/``"simplistic"``/``"illustrative"``)
    or a :class:`BandScheme` for custom bands.
    """
    return _resolve(scheme).render(label, float(score))


def verbalise_scores(
    scores: Mapping[str, float],
    scheme="illustrative",
    *,
    sort: str = "score",
    bullet: str = "- ",
) -> str:
    """Verbalise a whole ``{label: score}`` mapping into an LLM-ready block.

    ``sort`` controls ordering of the lines, which matters because LLMs weight
    earlier context more:

    * ``"score"`` (default): descending score, then label ascending as a stable
      tie-break -- most salient findings first.
    * ``"label"``: alphabetical by label.
    * ``"none"``: preserve the mapping's own iteration order.

    The output is deterministic for a given input and ``sort`` choice.
    """
    resolved = _resolve(scheme)
    items: List[Tuple[str, float]] = [(k, float(v)) for k, v in scores.items()]
    if sort == "score":
        items.sort(key=lambda kv: (-kv[1], kv[0]))
    elif sort == "label":
        items.sort(key=lambda kv: kv[0])
    elif sort == "none":
        pass
    else:
        raise ValueError(f"unknown sort {sort!r}; use score|label|none")
    return "\n".join(bullet + resolved.render(k, v) for k, v in items)
