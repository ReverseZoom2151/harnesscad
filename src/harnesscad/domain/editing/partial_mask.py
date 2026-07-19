"""Fine-grained partial-token masking for locate-then-infill CAD editing.

The
"Locate" stage needs a *ground-truth masked sequence*: given an original CAD
token sequence and its edited counterpart, produce the original with every span
that must change replaced by a ``<mask>`` placeholder, so the "Infill" stage only
has to regenerate the masked spans while the surrounding context is frozen.

The harness already has a *whole-token* locate mask (``editing.locate_infill``):
a token either survives verbatim or is masked. The refinement captured
here is **finer granularity** -- when two tokens align but differ only in some of
their comma-separated components (e.g. ``line,14,14`` vs ``line,13,13``), the
shared command word and the matching components are *preserved* and only the
differing components become ``<mask>`` (``line,<mask>,<mask>``). That keeps far
more immutable context for the infill model than masking the whole token.

Alignment uses :class:`difflib.SequenceMatcher` opcodes:

* ``equal``   -> copy the tokens unchanged.
* ``replace`` -> zip the two equal-run slices pairwise and partial-mask each
  aligned pair with :func:`compare_tokens`; any unpaired remainder (unequal run
  lengths) collapses to a single ``<mask>``.
* ``delete`` / ``insert`` -> the changed span collapses to a single ``<mask>``.

Finally :func:`merge_consecutive_masks` coalesces adjacent whole-token masks so
the infill model sees one placeholder per contiguous edit region.

Deterministic (difflib is order-deterministic); stdlib only; absolute imports.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Sequence

MASK = "<mask>"


def parse_components(token: str, sep: str = ",") -> List[str]:
    """Split a token into its components (command word + parameters).

    Such tokens look like ``line,14,14`` / ``circle,7,3,7,9`` -- a leading
    command word followed by comma-separated integer parameters.
    """
    return token.split(sep)


def compare_tokens(token1: str, token2: str, sep: str = ",") -> str:
    """Partial-mask ``token1`` against ``token2`` at component granularity.

    * If the two tokens are identical, ``token1`` is returned verbatim.
    * If their command words (first component) differ, the whole token is masked
      -- a different primitive type cannot be a partial edit of the other.
    * Otherwise the command word and every matching component are preserved, and
      only the differing components become ``<mask>``. Any length mismatch in the
      trailing components masks the extra positions on the longer side.

    Example: ``compare_tokens("line,14,14", "line,13,13") == "line,<mask>,<mask>"``.
    """
    if token1 == token2:
        return token1
    c1 = parse_components(token1, sep)
    c2 = parse_components(token2, sep)
    if not c1 or not c2 or c1[0] != c2[0]:
        return MASK
    result = [c1[0]]
    n = max(len(c1), len(c2))
    for i in range(1, n):
        a = c1[i] if i < len(c1) else None
        b = c2[i] if i < len(c2) else None
        result.append(a if a is not None and a == b else MASK)
    return sep.join(result)


def merge_consecutive_masks(tokens: Sequence[str]) -> List[str]:
    """Coalesce runs of the *whole-token* mask into a single ``<mask>``.

    Partial masks (``line,<mask>,<mask>``) are distinct tokens and are never
    merged -- only bare ``<mask>`` placeholders collapse.
    """
    merged: List[str] = []
    for tok in tokens:
        if tok == MASK and merged and merged[-1] == MASK:
            continue
        merged.append(tok)
    return merged


def generate_mask(original: Sequence[str], edited: Sequence[str],
                  sep: str = ",", merge: bool = True) -> List[str]:
    """Build the fine-grained locate mask aligning ``original`` to ``edited``.

    Returns the original token stream with unchanged tokens preserved, aligned
    differing tokens partial-masked component-wise, and pure insert/delete spans
    collapsed to ``<mask>``. When ``merge`` is true (default) adjacent whole-token
    masks are coalesced.
    """
    a = list(original)
    b = list(edited)
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    out: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(a[i1:i2])
        elif tag == "replace":
            left = a[i1:i2]
            right = b[j1:j2]
            paired = min(len(left), len(right))
            for k in range(paired):
                out.append(compare_tokens(left[k], right[k], sep))
            # unequal run lengths: the surplus is an add/remove -> one mask
            if len(left) != len(right):
                out.append(MASK)
        elif tag in ("delete", "insert"):
            out.append(MASK)
    return merge_consecutive_masks(out) if merge else out


def mask_span_count(masked: Sequence[str]) -> int:
    """Number of contiguous edit regions in a masked sequence.

    Counts maximal runs containing at least one masked token (bare ``<mask>`` or a
    partial ``...<mask>...`` token). A useful complexity signal for the edit
    (mirrors the downstream length/complexity filtering).
    """
    count = 0
    in_run = False
    for tok in masked:
        is_masked = tok == MASK or (MASK in parse_components(tok))
        if is_masked and not in_run:
            count += 1
            in_run = True
        elif not is_masked:
            in_run = False
    return count
