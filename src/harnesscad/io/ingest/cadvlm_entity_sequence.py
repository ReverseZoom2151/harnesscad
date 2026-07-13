"""Entity-level embedding sequence construction (CadVLM inductive bias).

Section 5.3 ("Entity Level Modelling", Table 4) of CadVLM adds an *inductive bias*
to the flat primitive token sequence: each sketch is also processed entity-by-entity.
Concretely the paper (i) prepends a special ``<ENTITY>`` token to every entity's
input sequence, feeds the per-entity sequences through the encoder in parallel,
gathers each entity's embedding, and then (ii) concatenates those per-entity segments
with the *original* full token sequence, using another special ``<TOKEN>`` token as
the delimiter between the two parts.

The embeddings are learned, but the *sequence layout* -- the ``<ENTITY>`` prefixing,
the parallel per-entity segmentation, and the ``<TOKEN>``-delimited concatenation with
the flat sequence -- is a pure, reversible token-manipulation. This module builds and
parses that layout deterministically. Nothing else in the repository implements this
entity-level packing (``ingest.cadvlm_codec`` only produces flat per-entity tuples).
"""

from __future__ import annotations

from dataclasses import dataclass


ENTITY = "<ENTITY>"    # per-entity segment marker
TOKEN = "<TOKEN>"      # delimiter before the flat original sequence
SPECIAL = frozenset({ENTITY, TOKEN})


def _check(entities):
    values = tuple(tuple(e) for e in entities)
    for ent in values:
        if not ent:
            raise ValueError("entity token sequence must be non-empty")
        for tok in ent:
            if tok in SPECIAL:
                raise ValueError(f"entity tokens must not contain the reserved "
                                 f"token {tok!r}")
    return values


def entity_segments(entities) -> tuple:
    """Per-entity segments, each ``(<ENTITY>, *tokens)`` (the parallel encoder input)."""
    return tuple((ENTITY,) + ent for ent in _check(entities))


def flat_sequence(entities) -> tuple:
    """The original, undelimited concatenation of all entity tokens."""
    values = _check(entities)
    return tuple(tok for ent in values for tok in ent)


def build_sequence(entities) -> tuple:
    """Full entity-level sequence: ``[<ENTITY> e_i ...] + [<TOKEN>] + flat``.

    The first part is every entity's ``<ENTITY>``-prefixed segment concatenated in
    order; ``<TOKEN>`` then delimits the flat original token sequence (the "second
    part of the sequence" in the paper).
    """
    values = _check(entities)
    head = tuple(tok for seg in entity_segments(values) for tok in seg)
    return head + (TOKEN,) + flat_sequence(values)


@dataclass(frozen=True)
class ParsedSequence:
    """Recovered structure of a :func:`build_sequence` output."""

    entities: tuple        # tuple of per-entity token tuples (no <ENTITY> prefix)
    flat: tuple            # the flat tail after <TOKEN>

    @property
    def entity_count(self) -> int:
        return len(self.entities)


def parse_sequence(sequence) -> ParsedSequence:
    """Inverse of :func:`build_sequence`; round-trips the entity/flat split.

    Raises ``ValueError`` if the layout is malformed (no ``<TOKEN>`` delimiter, a
    head not starting with ``<ENTITY>``, or a tail that disagrees with the head).
    """
    seq = tuple(sequence)
    if TOKEN not in seq:
        raise ValueError("sequence missing <TOKEN> delimiter")
    split = seq.index(TOKEN)
    head, flat = seq[:split], seq[split + 1:]
    if TOKEN in flat:
        raise ValueError("multiple <TOKEN> delimiters")
    entities = []
    current = None
    for tok in head:
        if tok == ENTITY:
            if current is not None:
                entities.append(tuple(current))
            current = []
        else:
            if current is None:
                raise ValueError("head must start with <ENTITY>")
            current.append(tok)
    if current is not None:
        entities.append(tuple(current))
    entities = tuple(entities)
    if tuple(tok for ent in entities for tok in ent) != tuple(flat):
        raise ValueError("head entities disagree with flat tail")
    return ParsedSequence(entities=entities, flat=tuple(flat))
