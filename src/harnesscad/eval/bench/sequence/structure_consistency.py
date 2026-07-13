"""Structure-consistency metrics for GeoFusion-CAD hierarchical CAD trees.

GeoFusion-CAD argues (Sec. 5.3, Sec. E.1) that its *hierarchical tree*
representation is what preserves "topological consistency" across long command
sequences -- removing it ("w/o Tree") degrades every metric. This module gives
the deterministic structure-aware checks and metrics implied by that claim,
operating on the serialized token sequence and the typed tree from
:mod:`reconstruction.geofusion_hierarchy`:

* :func:`closure_valid` -- verifies the Table S1 *hierarchical closure*: end
  tokens (``ec`` for a curve, ``eloop``, ``eface``, ``esketch``, ``ee``,
  ``esolid``) nest correctly, like balanced brackets. A malformed nesting means
  the serialization is not reversible (the property the tree representation is
  designed to guarantee).
* :func:`valid_ratio` -- the batch-level fraction of well-formed sequences
  (the paper's "ValidRatios" family of well-formedness measures, Sec. C.2).
* :func:`structure_signature` -- a canonical, parameter-free descriptor of the
  tree shape (node-type counts + depth), so two trees with identical topology
  but different coordinates compare equal.
* :func:`structure_f1` -- precision/recall/F1 over the multiset of
  parent->child *type paths*, a topology-aware consistency score between a
  predicted and a reference tree (distinct from
  :mod:`bench.sketch_sequence_metrics`, which scores per-sketch primitive
  multisets, and from :mod:`reconstruction.cmt_topology_validity`, which scores
  B-Rep surface-edge adjacency).

Pure, deterministic, stdlib-only.
"""

from __future__ import annotations

from collections import Counter

from harnesscad.domain.reconstruction.tokens.geofusion import (
    Token, Solid, deserialize, count_nodes, tree_depth, type_paths,
    CLS, ESOLID, EC,
)

_CURVE_KINDS = ("line", "arc", "circle")


def closure_valid(tokens: tuple[Token, ...]) -> tuple[bool, str]:
    """Check that the token sequence is a well-formed hierarchical closure.

    Returns ``(ok, reason)``. ``reason`` is empty when valid, otherwise a short
    diagnostic. Cheap prechecks give a human-readable reason for the common
    failure modes (missing ``cls`` / ``esolid``, stray ``ec``); the authoritative
    verdict is delegated to
    :func:`reconstruction.geofusion_hierarchy.deserialize`, which enforces the
    full Table S1 nesting grammar. Never raises, so it can be mapped over a batch.
    """
    if not tokens:
        return False, "empty sequence"
    if not (tokens[0].kind == "ctl" and tokens[0].payload == CLS):
        return False, "missing cls start token"
    if not any(tok.kind == "ctl" and tok.payload == ESOLID for tok in tokens):
        return False, "missing esolid close token"
    # a curve token must be immediately followed by its ec closer.
    for idx, tok in enumerate(tokens):
        if tok.kind in _CURVE_KINDS:
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
            if not (nxt is not None and nxt.kind == "ctl" and nxt.payload == EC):
                return False, f"curve at {idx} not closed by ec"
    try:
        deserialize(tokens)
    except ValueError as exc:
        return False, str(exc)
    return True, ""


def valid_ratio(batch: tuple[tuple[Token, ...], ...]) -> float:
    """Fraction of sequences in ``batch`` that are well-formed closures."""
    if not batch:
        return 0.0
    good = sum(1 for seq in batch if closure_valid(seq)[0])
    return good / len(batch)


def structure_signature(solid: Solid) -> tuple:
    """Canonical parameter-free descriptor of a tree's *shape*.

    Two solids with the same topology but different coordinates yield the same
    signature; a different number of faces/loops/curves changes it.
    """
    counts = count_nodes(solid)
    ordered = tuple(sorted(counts.items()))
    return (tree_depth(solid), ordered)


def structure_match(pred: Solid, ref: Solid) -> bool:
    """True iff the two trees have identical structural signatures."""
    return structure_signature(pred) == structure_signature(ref)


def structure_f1(pred: Solid, ref: Solid) -> dict[str, float]:
    """Precision/recall/F1 over the multiset of parent->child type paths.

    Compares the topological skeleton (ignoring coordinate values but respecting
    node ordering and curve *kind*) of a predicted vs a reference tree.
    """
    p_paths = Counter(type_paths(pred))
    r_paths = Counter(type_paths(ref))
    inter = sum((p_paths & r_paths).values())
    np = sum(p_paths.values())
    nr = sum(r_paths.values())
    precision = inter / np if np else float(nr == 0)
    recall = inter / nr if nr else float(np == 0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}
