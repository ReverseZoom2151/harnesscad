"""LDraw brick-assembly parser, validator and buildability metrics (Prompt-to-Parts).

Mined from *Prompt-to-Parts: Generative AI for Physical Assembly and Scalable
Instructions*. The paper targets **LDraw** as a "text-rich intermediate
representation" for discrete brick assemblies -- a finite parts vocabulary, explicit
spatial coordinates, and a sequential build order -- and evaluates generated builds
along three orthogonal dimensions:

*   **drawing accuracy** -- syntactic correctness of the LDraw output;
*   **structural validity** -- parts drawn from a known vocabulary; and
*   **instructional coherence** -- whether the build sequence is complete/ordered.

This module ports the deterministic pieces: an LDraw line parser/validator (type-1
sub-file/part lines carry a colour, a 3-vector position, a 3x3 rotation matrix and a
part name) plus the three buildability metrics. Stdlib-only, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "PartLine",
    "parse_line",
    "parse_model",
    "drawing_accuracy",
    "structural_validity",
    "instructional_coherence",
]


@dataclass(frozen=True)
class PartLine:
    """A parsed LDraw type-1 line: ``1 colour x y z a b c d e f g h i part.dat``."""

    colour: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, ...]  # row-major 3x3 (9 values)
    part: str

    def __post_init__(self) -> None:
        if len(self.rotation) != 9:
            raise ValueError("rotation must have 9 elements (3x3)")


def parse_line(line: str) -> Optional[PartLine]:
    """Parse one LDraw line.

    Returns a :class:`PartLine` for a valid type-1 line, ``None`` for a comment /
    meta line (type 0) or blank line, and raises ``ValueError`` for a malformed
    type-1 line.
    """
    toks = line.strip().split()
    if not toks:
        return None
    if toks[0] == "0":
        return None  # comment / meta
    if toks[0] != "1":
        raise ValueError(f"unsupported line type {toks[0]!r}")
    if len(toks) < 15:
        raise ValueError("type-1 line needs 15 tokens")
    try:
        colour = int(toks[1])
        nums = [float(t) for t in toks[2:14]]
    except ValueError as exc:
        raise ValueError(f"non-numeric field in type-1 line: {exc}") from None
    part = " ".join(toks[14:])
    return PartLine(
        colour=colour,
        position=(nums[0], nums[1], nums[2]),
        rotation=tuple(nums[3:12]),
        part=part,
    )


def parse_model(text: str) -> Tuple[List[PartLine], List[int]]:
    """Parse a whole LDraw model.

    Returns ``(part_lines, bad_line_numbers)``: the successfully parsed type-1
    part lines (in build order) and the 1-based line numbers that failed to parse.
    """
    parts: List[PartLine] = []
    bad: List[int] = []
    for i, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        try:
            pl = parse_line(raw)
        except ValueError:
            bad.append(i)
            continue
        if pl is not None:
            parts.append(pl)
    return parts, bad


def drawing_accuracy(text: str) -> float:
    """Fraction of non-blank lines that parse without error (syntactic correctness)."""
    total = 0
    ok = 0
    for raw in text.splitlines():
        if not raw.strip():
            continue
        total += 1
        try:
            parse_line(raw)
            ok += 1
        except ValueError:
            pass
    if total == 0:
        raise ValueError("model has no content lines")
    return ok / total


def structural_validity(
    parts: Sequence[PartLine], vocabulary: Sequence[str]
) -> float:
    """Fraction of placed parts whose part name is in the known vocabulary."""
    if not parts:
        raise ValueError("no parts to validate")
    vocab = set(vocabulary)
    return sum(1 for p in parts if p.part in vocab) / len(parts)


def instructional_coherence(
    build_steps: Sequence[Sequence[PartLine]], total_parts: int
) -> float:
    """Coherence of a stepwise build: every part placed exactly once, no step empty.

    ``build_steps`` is the ordered list of steps, each a group of parts placed on
    that page. Returns the fraction of the ideal (all parts placed once, no empty
    step); a perfectly coherent sequence scores 1.0.
    """
    if total_parts <= 0:
        raise ValueError("total_parts must be positive")
    placed = sum(len(step) for step in build_steps)
    empty_steps = sum(1 for step in build_steps if len(step) == 0)
    coverage = min(1.0, placed / total_parts)
    empty_penalty = empty_steps / len(build_steps) if build_steps else 1.0
    return max(0.0, coverage * (1.0 - empty_penalty))
