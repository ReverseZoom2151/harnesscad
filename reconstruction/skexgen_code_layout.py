"""SkexGen disentangled-codebook layout (topology | geometry | extrude).

SkexGen's headline idea is that a CAD model is summarised by a *short sequence
of discrete codes* drawn from three separate codebooks, one per disentangled
branch (``extract_code.py`` / ``train_code.py`` / ``sample.py``)::

    code sequence (length 10)
    +-------------------+-----------+-------------------+
    | topology  (4)     | geom (2)  | extrude (4)       |
    | codebook 500      | 1000      | 1000              |
    +-------------------+-----------+-------------------+

* the **topology** codes come from the VQ over the *command* stream (curve types
  and loop/face structure only -- no coordinates);
* the **geometry** codes come from the VQ over the *pixel/xy* stream (the actual
  coordinates);
* the **extrude** codes come from the VQ over the 19-token extrude stream.

Swapping one branch's codes while holding the others fixed is what produces the
paper's controlled interpolation / "same topology, new geometry" edits, so the
split and the recombination are worth having as a deterministic utility.

A sampled code row is *rejected* before decoding if any topology code falls
outside its (smaller) codebook -- the autoregressive code model shares one
softmax of size ``max(codebook)`` over all 10 positions, so it can emit codes
that no branch owns.  ``is_valid_code`` reproduces that filter.

The VQ nearest-code assignment itself lives in
``reconstruction/hnc_code_assignment``; this module is about the *layout* of the
code sequence, which is SkexGen-specific.

Deterministic, stdlib only.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

TOPOLOGY_LEN = 4
GEOMETRY_LEN = 2
EXTRUDE_LEN = 4
CODE_LEN = TOPOLOGY_LEN + GEOMETRY_LEN + EXTRUDE_LEN     # 10

TOPOLOGY_CODEBOOK = 500
GEOMETRY_CODEBOOK = 1000
EXTRUDE_CODEBOOK = 1000

BRANCHES = ("topology", "geometry", "extrude")
BRANCH_SLICE = {
    "topology": slice(0, TOPOLOGY_LEN),
    "geometry": slice(TOPOLOGY_LEN, TOPOLOGY_LEN + GEOMETRY_LEN),
    "extrude": slice(TOPOLOGY_LEN + GEOMETRY_LEN, CODE_LEN),
}
BRANCH_CODEBOOK = {
    "topology": TOPOLOGY_CODEBOOK,
    "geometry": GEOMETRY_CODEBOOK,
    "extrude": EXTRUDE_CODEBOOK,
}


def code_model_vocab() -> int:
    """Softmax size of the shared code model: the largest codebook."""
    return max(BRANCH_CODEBOOK.values())


def branch_of(position: int) -> str:
    """Which branch owns a position of the code sequence."""
    if not 0 <= position < CODE_LEN:
        raise ValueError("position out of range: %d" % position)
    for name, sl in BRANCH_SLICE.items():
        if sl.start <= position < sl.stop:
            return name
    raise AssertionError("unreachable")


def split_code(code: Sequence[int]) -> Dict[str, List[int]]:
    """Split a length-10 code row into its three branches."""
    if len(code) != CODE_LEN:
        raise ValueError("code must hold %d entries" % CODE_LEN)
    return {name: [int(c) for c in code[sl]] for name, sl in BRANCH_SLICE.items()}


def join_code(topology: Sequence[int], geometry: Sequence[int],
              extrude: Sequence[int]) -> List[int]:
    """Recombine three branches into one code row (branch swapping / editing)."""
    parts = ((topology, TOPOLOGY_LEN), (geometry, GEOMETRY_LEN),
             (extrude, EXTRUDE_LEN))
    for values, want in parts:
        if len(values) != want:
            raise ValueError("branch length mismatch: %d != %d" % (len(values), want))
    return [int(c) for c in topology] + [int(c) for c in geometry] + \
           [int(c) for c in extrude]


def swap_branch(code: Sequence[int], branch: str,
                values: Sequence[int]) -> List[int]:
    """Replace one branch of a code row, keeping the other two."""
    if branch not in BRANCH_SLICE:
        raise ValueError("unknown branch: %r" % (branch,))
    parts = split_code(code)
    if len(values) != len(parts[branch]):
        raise ValueError("branch length mismatch")
    parts[branch] = [int(v) for v in values]
    return join_code(parts["topology"], parts["geometry"], parts["extrude"])


def is_valid_code(code: Sequence[int]) -> bool:
    """Does every code fall inside the codebook that owns its position?"""
    if len(code) != CODE_LEN:
        return False
    for i, value in enumerate(code):
        v = int(value)
        if v < 0 or v >= BRANCH_CODEBOOK[branch_of(i)]:
            return False
    return True


def filter_valid(codes: Iterable[Sequence[int]]) -> List[List[int]]:
    """Drop sampled code rows the codebooks cannot decode (SkexGen ``sample.py``)."""
    return [[int(c) for c in code] for code in codes if is_valid_code(code)]


def unique_codes(codes: Iterable[Sequence[int]]) -> List[List[int]]:
    """De-duplicate code rows, keeping the first occurrence of each."""
    seen = set()
    out: List[List[int]] = []
    for code in codes:
        key: Tuple[int, ...] = tuple(int(c) for c in code)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def codebook_usage(codes: Sequence[Sequence[int]]) -> Dict[str, float]:
    """Fraction of each codebook that a set of code rows actually uses."""
    used = {name: set() for name in BRANCHES}
    for code in codes:
        parts = split_code(code)
        for name in BRANCHES:
            used[name].update(parts[name])
    return {name: len(used[name]) / BRANCH_CODEBOOK[name] for name in BRANCHES}


def branch_histogram(codes: Sequence[Sequence[int]], branch: str) -> Dict[int, int]:
    """Count how often each code of one branch is used."""
    if branch not in BRANCH_SLICE:
        raise ValueError("unknown branch: %r" % (branch,))
    hist: Dict[int, int] = {}
    for code in codes:
        for value in split_code(code)[branch]:
            hist[value] = hist.get(value, 0) + 1
    return hist
