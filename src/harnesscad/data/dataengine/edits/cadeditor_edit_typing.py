"""Edit-type classification and grouped variant pairing for CAD editing data.

From CAD-Editor (Yuan et al., ICML 2025, ``data/pair.py``). To synthesise a
text-based CAD-editing corpus, design variations of one seed model are grouped,
then every ordered pair of variants within a group becomes a training triplet
(original -> edited). Each pair carries a *semantic edit type* -- ``add``,
``delete`` or ``modify`` -- inferred purely from which member is the untouched
original input, marked by a sentinel substring (``origInput``) in its name:

* the original is the untouched seed and the edited adds geometry -> ``add``;
* the edited is the untouched seed (the original had extra geometry) -> ``delete``;
* neither member is the untouched seed -> ``modify``.

Two long-tail controls from the paper are preserved: variants are **bucketed by a
name-prefix key** (default first 8 chars = the seed id) so only variations of the
*same* seed are paired, and each bucket is **capped** so prolific seeds do not
dominate the dataset.

The existing ``datagen.edit_triplets`` enumerates base/variant pairs with lineage
and forward/reverse/cross *directions*; it does not assign the semantic
add/delete/modify type nor apply prefix-bucketing with a per-bucket cap. This
module adds exactly those pieces.

Deterministic (input order preserved; ``itertools.combinations``); stdlib only;
absolute imports.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Mapping, Sequence

DEFAULT_SENTINEL = "origInput"
DEFAULT_KEY_LEN = 8
DEFAULT_CAP = 56


def classify_edit_type(original_name: str, edited_name: str,
                       sentinel: str = DEFAULT_SENTINEL) -> str:
    """Classify an edit's semantic type from which name is the untouched seed.

    * ``sentinel in original_name`` -> the original is the seed; the edit added
      geometry -> ``"add"``.
    * else ``sentinel in edited_name`` -> the edited side is the seed; the
      original had extra geometry that was removed -> ``"delete"``.
    * otherwise both are derived variants -> ``"modify"``.
    """
    if sentinel in original_name:
        return "add"
    if sentinel in edited_name:
        return "delete"
    return "modify"


def group_by_prefix(items: Sequence[Mapping[str, Any]],
                    key_len: int = DEFAULT_KEY_LEN,
                    name_field: str = "name") -> "OrderedDict[str, List[Mapping[str, Any]]]":
    """Bucket items by the first ``key_len`` chars of their name field.

    Insertion order of both buckets and members is preserved so downstream
    pairing is deterministic.
    """
    groups: "OrderedDict[str, List[Mapping[str, Any]]]" = OrderedDict()
    for item in items:
        key = str(item[name_field])[:key_len]
        groups.setdefault(key, []).append(item)
    return groups


@dataclass(frozen=True)
class EditPairing:
    """One directed edit pair with its inferred semantic type."""
    original_name: str
    edited_name: str
    original_sequence: Any
    edited_sequence: Any
    edit_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_pic_name": self.original_name,
            "edited_pic_name": self.edited_name,
            "original_sequence": self.original_sequence,
            "edited_sequence": self.edited_sequence,
            "type": self.edit_type,
        }


def pair_group(items: Sequence[Mapping[str, Any]],
               cap: int = DEFAULT_CAP,
               sentinel: str = DEFAULT_SENTINEL,
               name_field: str = "name",
               seq_field: str = "original_sequence") -> List[EditPairing]:
    """Enumerate typed forward+reverse edit pairs within one bucket, capped.

    For every unordered pair ``(a, b)`` (via ``itertools.combinations``) both the
    forward (a->b) and reverse (b->a) directed pairs are emitted with independently
    inferred edit types. Emission stops once ``cap`` pairs are produced, matching
    the paper's per-bucket long-tail control. Buckets with fewer than two members
    yield nothing.
    """
    out: List[EditPairing] = []
    if len(items) < 2:
        return out
    for a, b in combinations(items, 2):
        na, nb = str(a[name_field]), str(b[name_field])
        out.append(EditPairing(na, nb, a[seq_field], b[seq_field],
                               classify_edit_type(na, nb, sentinel)))
        out.append(EditPairing(nb, na, b[seq_field], a[seq_field],
                               classify_edit_type(nb, na, sentinel)))
        if len(out) >= cap:
            return out[:cap]
    return out


def pair_variants(items: Sequence[Mapping[str, Any]],
                  key_len: int = DEFAULT_KEY_LEN,
                  cap: int = DEFAULT_CAP,
                  sentinel: str = DEFAULT_SENTINEL,
                  name_field: str = "name",
                  seq_field: str = "original_sequence") -> List[EditPairing]:
    """Full pipeline: bucket variants by seed prefix, then typed-pair each bucket.

    Returns a flat, deterministic list of :class:`EditPairing` across all buckets.
    """
    result: List[EditPairing] = []
    for _key, group in group_by_prefix(items, key_len, name_field).items():
        result.extend(pair_group(group, cap, sentinel, name_field, seq_field))
    return result
