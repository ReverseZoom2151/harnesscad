"""Deterministic token-sequence packing for fixed-context training data.

Ported from mesh-transformer-jax's ``create_finetune_tfrecords.py`` -- the
data-format / sequence-packing pipeline that turns variable-length tokenised
documents into the equal-length token windows a fixed-context transformer
consumes.  Only the deterministic *data* transformation is ported here; nothing
about the JAX/TPU model, TensorFlow ``tfrecords`` serialisation, or the GPT-2
tokeniser is reproduced (those are the trained/infra parts).

The packing algorithm (mirrors the reference exactly)
-----------------------------------------------------
1.  **EOT splitting** (:func:`eot_split`) -- each document is split internally on
    the end-of-text token, so a file that embeds several logical documents
    becomes several token arrays (empty pieces dropped).  Mirror of
    ``eot_splitting_generator``.
2.  **Concatenate + chunk** (:func:`arrays_to_sequences`) -- all token arrays are
    concatenated into one stream and cut into ``sequence_length``-token windows.
    Whenever the accumulator exceeds one window it is split, every full window
    is yielded, and the remainder is carried forward.  A final short window (the
    *trailing data*) is yielded last.  Mirror of ``arrays_to_sequences``.
3.  **Min-unique filtering** (:func:`enforce_min_unique`) -- windows made of
    fewer than ``min_unique_tokens`` distinct tokens are dropped (repetitive
    windows produce large gradients).  Mirror of ``enforce_min_unique``.
4.  **Repack epochs** (:func:`pack_sequences`) -- the data can be re-packed
    ``n_repack_epochs`` times.  With ``preserve_data_order`` the trailing data of
    the previous epoch is prepended before re-chunking (so no tokens are wasted
    and every epoch is a *different* shift); otherwise documents are reshuffled
    with a seeded RNG.  Mirror of ``create_tfrecords`` / ``chunk_and_finalize``.

Determinism
-----------
``preserve_data_order=True`` (the default here, unlike the CLI) makes the whole
pipeline a pure function of the input.  When ``preserve_data_order=False`` all
shuffling is driven by a single :class:`random.Random` seeded with ``seed``, so
runs are still reproducible.  No global ``random`` state is touched.

Pure stdlib.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "split_list",
    "eot_split",
    "append_separator",
    "arrays_to_sequences",
    "enforce_min_unique",
    "chunk_and_finalize",
    "pack_sequences",
]

Tokens = Sequence[int]


def split_list(seq: Sequence[int], n: int) -> List[List[int]]:
    """Split ``seq`` into consecutive chunks of size ``n`` (last may be short)."""
    if n <= 0:
        raise ValueError("chunk size must be positive")
    return [list(seq[i:i + n]) for i in range(0, len(seq), n)]


def eot_split(docs: Iterable[Tokens], eot_token: int) -> Iterator[List[int]]:
    """Split each document on ``eot_token``, yielding the non-empty pieces.

    Mirror of ``eot_splitting_generator`` but on token ids rather than strings.
    """
    for doc in docs:
        piece: List[int] = []
        for tok in doc:
            if tok == eot_token:
                if piece:
                    yield piece
                piece = []
            else:
                piece.append(tok)
        if piece:
            yield piece


def append_separator(docs: Iterable[Tokens], eos_token: int) -> Iterator[List[int]]:
    """Append ``eos_token`` to every document (the model's document separator).

    Mirror of ``tokens = encoder.encode(doc) + [encoder.eos_token_id]``.
    """
    for doc in docs:
        yield list(doc) + [eos_token]


def arrays_to_sequences(
    token_lists: Iterable[Tokens],
    sequence_length: int = 2049,
) -> Iterator[List[int]]:
    """Concat variable-length arrays and cut into ``sequence_length`` windows.

    Yields every full window, then a final (possibly shorter) trailing window if
    any tokens remain.  Exact mirror of ``arrays_to_sequences``.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    accum: List[int] = []
    for arr in token_lists:
        accum.extend(arr)
        if len(accum) > sequence_length:
            chunks = split_list(accum, sequence_length)
            for c in chunks[:-1]:
                yield c
            accum = chunks[-1]
    if len(accum) > 0:
        yield accum


def enforce_min_unique(
    seqs: Iterable[Tokens],
    min_unique_tokens: int,
) -> Iterator[List[int]]:
    """Drop windows with fewer than ``min_unique_tokens`` distinct tokens."""
    for seq in seqs:
        if len(set(seq)) >= min_unique_tokens:
            yield list(seq)


def chunk_and_finalize(
    arrays: Iterable[Tokens],
    sequence_length: int,
    *,
    min_unique_tokens: int = 0,
    preserve_data_order: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """Chunk arrays into full windows + a trailing remainder.

    Returns ``(full_windows, trailing_tokens)``.  Mirror of
    ``chunk_and_finalize``: split into sequences, peel off the last (short)
    sequence as trailing data, optionally min-unique-filter, optionally shuffle.
    """
    sequences = list(arrays_to_sequences(arrays, sequence_length))
    if not sequences:
        return [], []
    # Faithful to the reference: the last sequence is always peeled off as
    # trailing data, even when it happens to be full-length (the model's data
    # loader ignores sub-batch trailing data, so the reference drops it here).
    full_seqs, trailing = sequences[:-1], sequences[-1]
    if min_unique_tokens > 0:
        full_seqs = list(enforce_min_unique(full_seqs, min_unique_tokens))
    if not preserve_data_order and rng is not None:
        rng.shuffle(full_seqs)
    return full_seqs, list(trailing)


def pack_sequences(
    docs: Sequence[Tokens],
    sequence_length: int = 2049,
    *,
    eos_token: Optional[int] = None,
    eot_token: Optional[int] = None,
    min_unique_tokens: int = 0,
    n_repack_epochs: int = 1,
    preserve_data_order: bool = True,
    seed: int = 10,
) -> Tuple[List[List[int]], int]:
    """Full packing pipeline over tokenised ``docs``.

    Parameters mirror the reference CLI's flags.  ``docs`` is a sequence of
    already-tokenised documents (lists of ints).

    * ``eot_token`` -- if given, documents are first EOT-split on this id.
    * ``eos_token`` -- if given, this separator id is appended to every document
      (done *after* EOT splitting, matching the reference order).
    * ``n_repack_epochs`` -- repeat the packing this many times.  With
      ``preserve_data_order`` each epoch is prefixed with the previous epoch's
      trailing data and re-chunked (a different shift each time); otherwise the
      documents are reshuffled with the seeded RNG per epoch.

    Returns ``(sequences, dropped_trailing_token_count)`` where ``sequences`` is
    the concatenation of full windows across all epochs and the second value is
    the number of trailing tokens ultimately discarded.
    """
    if n_repack_epochs < 1:
        raise ValueError("n_repack_epochs must be >= 1")
    rng = random.Random(seed)

    prepared: List[List[int]] = [list(d) for d in docs]
    if eot_token is not None:
        prepared = list(eot_split(prepared, eot_token))
    if eos_token is not None:
        prepared = list(append_separator(prepared, eos_token))

    if not preserve_data_order:
        rng.shuffle(prepared)

    all_sequences: List[List[int]] = []
    full_seqs, trailing = chunk_and_finalize(
        prepared,
        sequence_length,
        min_unique_tokens=min_unique_tokens,
        preserve_data_order=preserve_data_order,
        rng=rng,
    )
    all_sequences.extend(full_seqs)

    for _ in range(1, n_repack_epochs):
        if not preserve_data_order:
            rng.shuffle(prepared)
            full_seqs, trailing = chunk_and_finalize(
                prepared,
                sequence_length,
                min_unique_tokens=min_unique_tokens,
                preserve_data_order=preserve_data_order,
                rng=rng,
            )
        else:
            seqs_with_prefix = [trailing] + full_seqs
            full_seqs, trailing = chunk_and_finalize(
                seqs_with_prefix,
                sequence_length,
                min_unique_tokens=min_unique_tokens,
                preserve_data_order=True,
                rng=rng,
            )
        all_sequences.extend(full_seqs)

    return all_sequences, len(trailing)
