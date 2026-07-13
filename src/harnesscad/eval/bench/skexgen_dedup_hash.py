"""SkexGen branch-wise dataset de-duplication (``utils/deduplicate.py``).

SkexGen trains its sketch branch and its extrude branch on *separately
de-duplicated* corpora: two CAD models with an identical sketch but different
extrusions are duplicates for the sketch branch and not for the extrude branch.
De-duplication is done by hashing the token stream of the branch in question:

* ``"s"``  -> hash of the sketch (pixel) tokens only,
* ``"e"``  -> hash of the extrude tokens only,
* ``"se"`` -> hash of both, concatenated.

Each per-SE stream is shifted by ``EXTRA_PAD`` and the whole branch stream is
terminated with a ``0``, exactly as at training time, so the hash is taken over
the tokens the model actually sees.  Records with no extrusion hash to nothing
and are dropped.

Complements ``bench/diffusioncad_generation_metrics`` (which measures unique /
novel / invalid percentages of *generated* sets by comparing command tuples):
this module is about *training-set* de-duplication and about doing it per
disentangled branch.

Deterministic, stdlib only.
"""
from __future__ import annotations

import struct
from hashlib import sha256
from typing import Dict, Iterable, List, Sequence

EXTRA_PAD = 1
BRANCHES = ("s", "e", "se")


def _shift(stream: Sequence[int]) -> List[int]:
    return [int(t) + EXTRA_PAD for t in stream]


def branch_tokens(record: Dict, branch: str) -> List[int]:
    """Flatten a record's per-SE streams into the token list hashed for ``branch``.

    ``record`` holds ``se_pix`` and ``se_ext``: lists (one entry per
    sketch-extrude pair) of raw token streams.
    """
    if branch not in BRANCHES:
        raise ValueError("branch must be one of %r" % (BRANCHES,))
    pix: List[int] = []
    for stream in record.get("se_pix", ()):
        pix.extend(_shift(stream))
    pix.append(0)
    ext: List[int] = []
    for stream in record.get("se_ext", ()):
        ext.extend(_shift(stream))
    ext.append(0)
    if branch == "s":
        return pix
    if branch == "e":
        return ext
    return pix + ext


def token_hash(tokens: Sequence[int]) -> str:
    """SHA-256 over a canonical little-endian int64 encoding of the tokens."""
    buf = b"".join(struct.pack("<q", int(t)) for t in tokens)
    return sha256(buf).hexdigest()


def record_hash(record: Dict, branch: str = "se") -> str:
    """Hash one record's branch stream; ``""`` if the record has no extrusion."""
    if not record.get("se_ext"):
        return ""
    return token_hash(branch_tokens(record, branch))


def duplicate_groups(records: Iterable[Dict], branch: str = "se") -> Dict[str, List[int]]:
    """Map branch hash -> list of record indices (insertion ordered)."""
    groups: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        h = record_hash(record, branch)
        if not h:
            continue
        groups.setdefault(h, []).append(record.get("uid", idx))
    return groups


def unique_percent(records: Sequence[Dict], branch: str = "se") -> float:
    """Percentage of records whose branch stream occurs exactly once.

    Matches SkexGen's report: groups of size 1 over the *total* record count.
    """
    if not records:
        return 0.0
    groups = duplicate_groups(records, branch)
    singletons = sum(1 for g in groups.values() if len(g) == 1)
    return 100.0 * singletons / len(records)


def duplicate_percent(records: Sequence[Dict], branch: str = "se") -> float:
    """Percentage of records dropped by de-duplication."""
    if not records:
        return 0.0
    kept = len(duplicate_groups(records, branch))
    return 100.0 * (len(records) - kept) / len(records)


def deduplicate(records: Sequence[Dict], branch: str = "se") -> List[Dict]:
    """Keep the first record of every hash group (records with no extrude drop)."""
    seen = set()
    out: List[Dict] = []
    for record in records:
        h = record_hash(record, branch)
        if not h or h in seen:
            continue
        seen.add(h)
        out.append(record)
    return out


def dedup_report(records: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """Unique / duplicate / kept counts for every branch."""
    report: Dict[str, Dict[str, float]] = {}
    for branch in BRANCHES:
        kept = deduplicate(records, branch)
        report[branch] = {
            "total": float(len(records)),
            "kept": float(len(kept)),
            "unique_percent": unique_percent(records, branch),
            "duplicate_percent": duplicate_percent(records, branch),
        }
    return report
