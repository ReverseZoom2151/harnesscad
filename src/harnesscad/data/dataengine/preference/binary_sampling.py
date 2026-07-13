"""Seeded stratified selection of already-labelled binary preferences."""

from __future__ import annotations

import random


def sample_binary(records, count, *, positive_fraction=.5, seed=0,
                  stratum=lambda record: record.provenance.get("family", "")):
    if count < 0 or not 0 <= positive_fraction <= 1:
        raise ValueError("invalid sampling request")
    positives = [record for record in records if record.desirable]
    negatives = [record for record in records if not record.desirable]
    rng = random.Random(seed)
    positives.sort(key=lambda item: (stratum(item), item.candidate_digest))
    negatives.sort(key=lambda item: (stratum(item), item.candidate_digest))
    rng.shuffle(positives); rng.shuffle(negatives)
    want_positive = round(count * positive_fraction)
    chosen = positives[:want_positive] + negatives[:count-want_positive]
    chosen.sort(key=lambda item: (item.prompt, item.candidate_digest))
    return {
        "records": tuple(chosen), "requested": count,
        "selected": len(chosen),
        "positive_fraction": (sum(item.desirable for item in chosen) / len(chosen)
                              if chosen else None),
        "shortage": count - len(chosen),
    }
