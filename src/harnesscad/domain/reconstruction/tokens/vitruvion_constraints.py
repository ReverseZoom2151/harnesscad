"""Constraint-token codec: the constraint hypergraph as a pointer stream.

This codec models the
constraint hypergraph *without* any graph machinery: it flattens it into a second token
stream whose reference tokens **point into the primitive token stream** produced by
``reconstruction.vitruvion_primitive_tokens``.  Nodes are primitives (or their
sub-points), hyperedges are constraints, and an edge is encoded as::

    <constraint type> <ref token> <ref token> ...

Value vocabulary (``len(Token) == 16``)::

    0  Pad      1  Start     2  Stop
    3  Coincident   4  Concentric   5  Equal        6  Fix        7  Horizontal
    8  Midpoint     9  Normal      10  Offset      11  Parallel   12  Perpendicular
    13 Quadrant    14  Tangent     15  Vertical

    16 + gather_idxs[node]   -- a reference token, i.e. a *pointer* to the position in
                               the primitive ``val`` stream where that node's token sits.

Three streams again (``val`` / ``coord`` / ``pos``), where:

  * ``coord`` distinguishes the *argument slot*: ``1`` (NON_COORD) for a constraint-type
    or control token, then ``2`` and ``3`` for the first and second reference.  There are
    **only two reference coord tokens**, so this encoding is implicitly limited to
    constraints of arity <= 2; the reference implementation slices
    ``CONSTRAINT_COORD_TOKENS[:len(refs)]`` and would silently desynchronise its streams
    on a higher-arity hyperedge.  This module raises instead (``max_arity``).
  * ``pos`` is a per-constraint group index: the type token and *all* of its references
    share one value, so the model can tell which reference belongs to which constraint.

Two rules that change what is learnable, and are easy to miss:

  * **External constraints are dropped.**  Any edge referencing node ``0`` (the sketch
    graph's "external" node, i.e. a constraint against another feature of the part) is
    skipped.
  * **References are emitted in sorted node order**, not in the designer's argument
    order.  Argument roles are therefore *not* recoverable from the token stream: for an
    asymmetric constraint (``Midpoint``: point vs. line) the model only learns the
    unordered member set.  Every constraint type kept here is either symmetric or
    has its roles inferable from the member types, so this is consistent -- but a decoder
    must not assume the first reference is the first argument.

Constraints whose label is not in the vocabulary (dimensional/valued constraints:
distance, angle, diameter, ...) are dropped: only the *categorical* constraints are
modelled.

Pure stdlib.
"""

from __future__ import annotations

import enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from harnesscad.domain.reconstruction.tokens.vitruvion_primitives import NON_COORD_TOKEN, pad_or_truncate

__all__ = [
    "ConstraintToken",
    "CONSTRAINT_COORD_TOKENS",
    "MAX_TOKEN_LENGTH",
    "ConstraintEdge",
    "reference_token",
    "reference_vocabulary_size",
    "tokenize_constraints",
    "constraints_from_tokens",
]


class ConstraintToken(enum.IntEnum):
    """Non-reference value tokens of the constraint model (categorical constraints)."""

    Pad = 0
    Start = 1
    Stop = 2
    Coincident = 3
    Concentric = 4
    Equal = 5
    Fix = 6
    Horizontal = 7
    Midpoint = 8
    Normal = 9
    Offset = 10
    Parallel = 11
    Perpendicular = 12
    Quadrant = 13
    Tangent = 14
    Vertical = 15


# Argument-slot ids for the first and second reference of a constraint.
CONSTRAINT_COORD_TOKENS = [NON_COORD_TOKEN + 1, NON_COORD_TOKEN + 2]  # [2, 3]

MAX_TOKEN_LENGTH = 130  # constraint-data configuration default

EXTERNAL_NODE = 0


class ConstraintEdge(tuple):
    """A hyperedge: ``(label, references)`` with node indices into ``gather_idxs``."""

    __slots__ = ()

    def __new__(cls, label: str, references: Sequence[int]) -> "ConstraintEdge":
        return super().__new__(cls, (str(label), tuple(int(r) for r in references)))

    @property
    def label(self) -> str:
        return self[0]

    @property
    def references(self) -> Tuple[int, ...]:
        return self[1]


def reference_token(gather_index: int) -> int:
    """The value token that points at position ``gather_index`` of the primitive stream."""
    return gather_index + len(ConstraintToken)


def reference_vocabulary_size(max_primitive_tokens: int) -> int:
    """Total constraint-model value vocabulary for a given primitive stream length."""
    return len(ConstraintToken) + max_primitive_tokens


def tokenize_constraints(
    edges: Iterable[ConstraintEdge],
    gather_idxs: Sequence[int],
    max_length: Optional[int] = None,
    max_arity: int = 2,
) -> Dict[str, List[int]]:
    """Tokenise the constraint hypergraph into ``val`` / ``coord`` / ``pos`` streams.

    ``gather_idxs`` is the second return value of
    ``reconstruction.vitruvion_primitive_tokens.tokenize_sketch``.  Edges with an unknown
    label, or referencing the external node, are dropped.  ``ValueError`` if an edge
    exceeds ``max_arity`` (the coord vocabulary cannot address a third argument) or names
    a node that has no gather index.
    """
    if max_arity > len(CONSTRAINT_COORD_TOKENS):
        raise ValueError("coord vocabulary addresses at most 2 references")

    val: List[int] = [int(ConstraintToken.Start)]
    coord: List[int] = [NON_COORD_TOKEN]
    pos_idx = 1  # 0 is reserved for padding
    pos: List[int] = [pos_idx]

    for edge in edges:
        label = edge.label if isinstance(edge, ConstraintEdge) else str(edge[0])
        refs = tuple(edge.references if isinstance(edge, ConstraintEdge) else edge[1])

        if label not in ConstraintToken.__members__:
            continue  # dimensional / unsupported constraint
        if EXTERNAL_NODE in refs:
            continue  # external constraint
        if not refs:
            continue
        if len(refs) > max_arity:
            raise ValueError(
                "constraint {!r} has arity {} > {}".format(label, len(refs), max_arity)
            )
        for ref in refs:
            if ref < 0 or ref >= len(gather_idxs):
                raise ValueError("reference {} has no gather index".format(ref))

        val.append(int(ConstraintToken[label]))
        coord.append(NON_COORD_TOKEN)
        pos_idx += 1
        pos.append(pos_idx)

        ordered = sorted(refs)  # node order, NOT argument order
        val.extend(reference_token(gather_idxs[ref]) for ref in ordered)
        coord.extend(CONSTRAINT_COORD_TOKENS[: len(ordered)])
        pos.extend([pos_idx] * len(ordered))

    val.append(int(ConstraintToken.Stop))
    coord.append(NON_COORD_TOKEN)
    pos.append(pos_idx + 1)

    return {
        "val": pad_or_truncate(val, max_length),
        "coord": pad_or_truncate(coord, max_length),
        "pos": pad_or_truncate(pos, max_length),
    }


def constraints_from_tokens(
    val: Sequence[int], gather_idxs: Sequence[int]
) -> List[ConstraintEdge]:
    """Invert :func:`tokenize_constraints`, mapping pointers back to node indices.

    Decoding stops at ``Stop`` / ``Pad``.  ``ValueError`` if a reference token points at a
    position that is not a gather index (i.e. not an addressable node).
    """
    position_to_node = {}
    for node, position in enumerate(gather_idxs):
        position_to_node.setdefault(position, node)

    edges: List[ConstraintEdge] = []
    label: Optional[str] = None
    refs: List[int] = []

    for token in val:
        token = int(token)
        if token == int(ConstraintToken.Start):
            continue
        if token in (int(ConstraintToken.Stop), int(ConstraintToken.Pad)):
            break
        if token < len(ConstraintToken):
            if label is not None:
                edges.append(ConstraintEdge(label, refs))
            label = ConstraintToken(token).name
            refs = []
            continue

        position = token - len(ConstraintToken)
        if position not in position_to_node:
            raise ValueError("reference token {} addresses no node".format(token))
        refs.append(position_to_node[position])

    if label is not None:
        edges.append(ConstraintEdge(label, refs))
    return edges
