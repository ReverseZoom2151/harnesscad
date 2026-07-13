"""Octree split-signal encoding -- the deterministic core of OctFusion diffusion.

From "OctFusion: Octree-based Diffusion Models for 3D Shape Generation"
(Xiong et al., 2024), Sections 3.2 and the inference description on Page 5. The
diffusion network is learned, but the *representation the diffusion operates on*
is a deterministic 0/1 encoding of the octree structure:

  * "our key insight is to regard the splitting status of octree nodes as 0/1
    signals; ... 0 indicating no splitting and 1 indicating splitting."
  * "The predicted splitting signals are rounded to 0/1 and used to generate the
    octree structure."
  * "We store an 8-channel 0/1 signal for each octree node at depth 4, with which
    the octree nodes can be split to depth 6." -- i.e. each node carries one bit
    per child (8 children => 8 channels) marking which children are occupied.
  * "We force the octree to be full when the depth is less than 4" -- shallow
    levels are always fully subdivided; the split signals only matter from a
    configurable ``full_depth`` downward.

This module converts between an :class:`~geometry.octfusion_octree.Octree` and
its per-node 8-bit child-occupancy signals, and rebuilds a structure from
(possibly noisy) predicted signals by *rounding to 0/1* and growing the tree.
Round-tripping ``encode -> decode`` reproduces the occupied-leaf set exactly.

Stdlib-only, deterministic. Depends only on ``geometry.octfusion_octree``.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from harnesscad.domain.geometry.volumes.octree import Octree, OctreeNode

NodeKey = Tuple[int, int, int, int]  # (depth, ix, iy, iz)
# child bit c has x-bit c&1, y-bit (c>>1)&1, z-bit (c>>2)&1 (Morton, matches octree)


def _child_index(bx: int, by: int, bz: int) -> int:
    return (bz << 2) | (by << 1) | bx


def encode_split_signals(tree: Octree) -> Dict[NodeKey, List[int]]:
    """Map every *internal* node to its 8-channel child-occupancy signal.

    Channel ``c`` is 1 iff child ``c`` (Morton order) is occupied. A child is
    "occupied" when its subtree contains any occupied leaf -- i.e. when the
    diffusion would need to split into it.
    """
    signals: Dict[NodeKey, List[int]] = {}

    def occ(node: OctreeNode) -> bool:
        if node.is_leaf:
            return node.occupied
        return any(occ(ch) for ch in node.children)  # type: ignore[union-attr]

    def walk(node: OctreeNode) -> None:
        if node.is_leaf:
            return
        sig = [1 if occ(ch) else 0 for ch in node.children]  # type: ignore[union-attr]
        signals[node.key()] = sig
        for ch in node.children:  # type: ignore[union-attr]
            walk(ch)

    walk(tree.root)
    return signals


def round_signal(values: Sequence[float], threshold: float = 0.5) -> List[int]:
    """Round continuous/noisy channel values to 0/1 (``> threshold`` -> 1).

    Mirrors "The predicted splitting signals are rounded to 0/1". Length must be
    8 (one channel per child).
    """
    if len(values) != 8:
        raise ValueError("a split signal must have exactly 8 channels")
    return [1 if v > threshold else 0 for v in values]


def decode_split_signals(
    signals: Mapping[NodeKey, Sequence[float]],
    max_depth: int,
    full_depth: int = 0,
    threshold: float = 0.5,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: float = 1.0,
) -> Octree:
    """Rebuild an octree by rounding predicted child signals and growing nodes.

    * Nodes shallower than ``full_depth`` are forced full (all 8 children
      created) regardless of any signal -- the paper's "force the octree to be
      full when the depth is less than ``full_depth``".
    * At ``full_depth`` and below, a node's signal (looked up by its key and
      rounded via ``threshold``) decides which children are created; a child not
      created becomes an empty leaf, a created child at ``max_depth`` becomes an
      occupied leaf.

    ``signals`` values may be continuous (noisy diffusion output); rounding is
    applied here.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")
    if not (0 <= full_depth <= max_depth):
        raise ValueError("full_depth must satisfy 0 <= full_depth <= max_depth")

    def grow(depth: int, ix: int, iy: int, iz: int) -> OctreeNode:
        node = OctreeNode(depth, ix, iy, iz)
        if depth >= max_depth:
            node.occupied = True
            return node
        if depth < full_depth:
            mask = [1] * 8
        else:
            raw = signals.get((depth, ix, iy, iz))
            if raw is None:
                # no signal for this node -> it does not split (leaf)
                node.occupied = False
                return node
            mask = round_signal(raw, threshold)
        if not any(mask):
            node.occupied = False
            return node
        node.children = []
        any_occ = False
        for c in range(8):
            bx = c & 1
            by = (c >> 1) & 1
            bz = (c >> 2) & 1
            if mask[c]:
                child = grow(depth + 1, 2 * ix + bx, 2 * iy + by, 2 * iz + bz)
                any_occ = True
            else:
                child = OctreeNode(depth + 1, 2 * ix + bx, 2 * iy + by, 2 * iz + bz)
                child.occupied = False
            node.children.append(child)
        node.occupied = any_occ
        return node

    root = grow(0, 0, 0, 0)
    return Octree(root, max_depth, origin, size)


def full_octree_node_count(full_depth: int) -> int:
    """Number of nodes in a *full* octree down to ``full_depth`` (levels 0..d).

    ``sum_{k=0}^{d} 8**k`` -- used to size the always-present shallow levels the
    paper keeps full below ``full_depth``.
    """
    if full_depth < 0:
        raise ValueError("full_depth must be >= 0")
    total = 0
    for k in range(full_depth + 1):
        total += 8 ** k
    return total
