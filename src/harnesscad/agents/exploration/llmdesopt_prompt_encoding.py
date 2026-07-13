"""Text-prompt design-variable encodings for text-to-3D design optimisation.

Paper: T. Rios, S. Menzel, B. Sendhoff, "Large Language and Text-to-3D Models
for Engineering Design Optimization" (Honda Research Institute Europe).

Section III-A defines two ways to turn a continuous/integer design vector (the
thing the evolution strategy optimises) into a *text prompt* that the external
text-to-3D model consumes:

  (a) Bag-of-words (BoW).  A prompt template
          "A <adjective> car in the shape of <noun>"
      is completed from word *sets*.  Each candidate word is pre-encoded by its
      Wu & Palmer similarity to a reference word ("fast" for the adjective,
      "wing" for the noun).  The optimiser works in that 1-D similarity space
      per slot; to reconstruct a prompt from a real design value, the word whose
      encoded similarity is *closest* to the value is recovered.  Because the
      sets only contain real English words, the generated prompts stay
      human-readable.

  (b) Tokenisation.  The template
          "A car in the shape of <string>"
      is completed by a sequence of integer token ids (the paper uses GPT-4's
      byte-pair-encoding vocabulary, ids in [0, 32768)).  The optimiser's real
      values are rounded to the nearest integer and clamped to the token range;
      arbitrary character strings -- often illegible -- can result.

Both the real BPE vocabulary and the text-to-3D model are external.  This module
implements the *deterministic encoding machinery*: template rendering, the
nearest-value word recovery for BoW, and an injectable token-id -> text codec
for tokenisation (with a trivial reversible default so tests are self-contained).
No randomness of its own, no wall clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Bag-of-words encoding
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WordSlot:
    """A prompt slot backed by words pre-encoded to a scalar value.

    ``encoding`` maps each word -> its scalar design value (e.g. WUP similarity
    to a reference word).  ``decode`` recovers the word whose value is nearest
    to a queried design value.
    """

    encoding: Dict[str, float]

    def __post_init__(self) -> None:
        if not self.encoding:
            raise ValueError("WordSlot needs at least one word")

    def words(self) -> List[str]:
        return sorted(self.encoding)

    def value_of(self, word: str) -> float:
        return self.encoding[word]

    def decode(self, value: float) -> str:
        """Recover the word whose encoded value is closest to ``value``.

        Ties broken alphabetically for determinism.
        """
        best_word = None
        best_key: Optional[Tuple[float, str]] = None
        for word in sorted(self.encoding):
            key = (abs(self.encoding[word] - value), word)
            if best_key is None or key < best_key:
                best_key = key
                best_word = word
        assert best_word is not None
        return best_word


class BagOfWordsPrompt:
    """Render a template from per-slot nearest-value word recovery.

    ``template`` uses ``{name}`` placeholders; ``slots`` maps each name to a
    :class:`WordSlot`.  A design vector supplies one scalar per slot, in the
    order given by ``slot_order``.
    """

    def __init__(self, template: str, slots: Dict[str, WordSlot],
                 slot_order: Sequence[str]) -> None:
        if set(slot_order) != set(slots):
            raise ValueError("slot_order must list exactly the slot names")
        self.template = template
        self.slots = slots
        self.slot_order = list(slot_order)

    def decode(self, design: Sequence[float]) -> str:
        if len(design) != len(self.slot_order):
            raise ValueError("design length must equal number of slots")
        chosen = {
            name: self.slots[name].decode(value)
            for name, value in zip(self.slot_order, design)
        }
        return self.template.format(**chosen)

    def encode(self, words: Dict[str, str]) -> List[float]:
        """Inverse: map a chosen word per slot back to its design values."""
        return [self.slots[name].value_of(words[name]) for name in self.slot_order]


# ---------------------------------------------------------------------------
# Tokenisation encoding
# ---------------------------------------------------------------------------

TokenCodec = Callable[[Sequence[int]], str]

DEFAULT_VOCAB_SIZE = 32768


def clamp_tokens(values: Sequence[float],
                 vocab_size: int = DEFAULT_VOCAB_SIZE) -> List[int]:
    """Round real design values to nearest integer token ids in [0, vocab_size).

    Reproduces Sec. III-B: "generate the token values by approximating the
    generated parameter values to corresponding nearest integers, and limit the
    values to available range of tokens ([0, 32768))."
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    out: List[int] = []
    for v in values:
        t = int(round(v))
        if t < 0:
            t = 0
        elif t > vocab_size - 1:
            t = vocab_size - 1
        out.append(t)
    return out


def _default_codec(tokens: Sequence[int]) -> str:
    """Trivial reversible codec: tokens -> space-joined ids.

    Stands in for a real BPE detokeniser (external).  Reversible so tests can
    round-trip without depending on a vocabulary file.
    """
    return " ".join(str(t) for t in tokens)


class TokenisationPrompt:
    """Render 'A car in the shape of <string>' from integer tokens."""

    def __init__(self, template: str = "A car in the shape of {string}",
                 codec: Optional[TokenCodec] = None,
                 vocab_size: int = DEFAULT_VOCAB_SIZE) -> None:
        if "{string}" not in template:
            raise ValueError("template must contain a {string} placeholder")
        self.template = template
        self.codec = codec or _default_codec
        self.vocab_size = vocab_size

    def decode(self, design: Sequence[float]) -> str:
        tokens = clamp_tokens(design, self.vocab_size)
        return self.template.format(string=self.codec(tokens))

    def tokens(self, design: Sequence[float]) -> List[int]:
        return clamp_tokens(design, self.vocab_size)
