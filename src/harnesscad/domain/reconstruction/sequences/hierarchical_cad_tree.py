"""Hierarchical disentanglement of a JSON CAD construction tree (HierCAD).

Mined from *HierCAD: Hierarchical Text-to-CAD Design via Structure Alignment and
Parameter Grounding*. HierCAD trains an LLM, but its supervision reformulation is a
deterministic transformation of the CADmium-style JSON CAD sequence:

*   **Disentangle the construction tree** ``T = {P, F, L}`` (parts, faces, loops)
    from the flat sequence by abstracting away numerical attributes (paper Eq. 2).
*   **Object-level procedural reasoning**: the ordered ``(part, Boolean-operation)``
    trajectory (``part_1: NewBody``, ``part_2: Cut``).
*   **Part-level topology reasoning**: per loop, the primitive-type sequence
    (``line|arc|line|line``, ``circle``).
*   **Parameter grounding** via *structure-preserving perturbation*: mutate one
    numeric leaf while keeping the tree shape identical, yielding ranking negatives.

Input is a nested ``dict`` (parsed JSON). Everything is deterministic and stdlib-only.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "ConstructionTree",
    "PartNode",
    "FaceNode",
    "build_tree",
    "object_level_trajectory",
    "part_level_topology",
    "structure_preserving_perturbation",
    "OPERATION_ABBREV",
    "primitive_type",
]

#: Canonical short names for Fusion/CADmium Boolean feature operations.
OPERATION_ABBREV: Dict[str, str] = {
    "NewBodyFeatureOperation": "NewBody",
    "JoinFeatureOperation": "Join",
    "CutFeatureOperation": "Cut",
    "IntersectFeatureOperation": "Intersect",
}


def primitive_type(name: str) -> str:
    """Map a primitive key (``line_1``, ``arc_2``, ``circle_1``) to its type."""
    return name.rsplit("_", 1)[0].lower() if "_" in name else name.lower()


@dataclass(frozen=True)
class FaceNode:
    """One sketch face: ordered loops, each an ordered primitive-type list."""

    name: str
    loops: Tuple[Tuple[str, Tuple[str, ...]], ...]


@dataclass(frozen=True)
class PartNode:
    """One part: its Boolean operation and its faces."""

    name: str
    operation: str
    faces: Tuple[FaceNode, ...]


@dataclass(frozen=True)
class ConstructionTree:
    """The disentangled tree ``T = {P, F, L}`` (parts/faces/loops)."""

    parts: Tuple[PartNode, ...]

    def num_parts(self) -> int:
        return len(self.parts)

    def num_faces(self) -> int:
        return sum(len(p.faces) for p in self.parts)

    def num_loops(self) -> int:
        return sum(len(f.loops) for p in self.parts for f in p.faces)


def _is_part_key(key: str) -> bool:
    return key.lower().startswith("part")


def _is_face_key(key: str) -> bool:
    return key.lower().startswith("face")


def _is_loop_key(key: str) -> bool:
    return key.lower().startswith("loop")


def build_tree(cad: Dict) -> ConstructionTree:
    """Disentangle a nested JSON CAD dict into a :class:`ConstructionTree`.

    Numerical parameters are abstracted away; only the discrete hierarchical
    organisation is retained. Keys are visited in insertion order for determinism.
    """
    parts: List[PartNode] = []
    for pkey, pval in cad.items():
        if not _is_part_key(pkey) or not isinstance(pval, dict):
            continue
        operation = pval.get("operation", "")
        op = OPERATION_ABBREV.get(operation, operation)
        faces: List[FaceNode] = []
        for fkey, fval in pval.items():
            if not _is_face_key(fkey) or not isinstance(fval, dict):
                continue
            loops: List[Tuple[str, Tuple[str, ...]]] = []
            for lkey, lval in fval.items():
                if not _is_loop_key(lkey) or not isinstance(lval, dict):
                    continue
                prims = tuple(primitive_type(k) for k in lval)
                loops.append((lkey, prims))
            faces.append(FaceNode(name=fkey, loops=tuple(loops)))
        parts.append(PartNode(name=pkey, operation=op, faces=tuple(faces)))
    return ConstructionTree(parts=tuple(parts))


def object_level_trajectory(tree: ConstructionTree) -> List[str]:
    """Object-level procedural reasoning: one ``part: Operation`` line per part."""
    return [f"{p.name}: {p.operation}" for p in tree.parts]


def part_level_topology(tree: ConstructionTree) -> List[str]:
    """Part-level topology reasoning: ``loop: prim|prim|...`` for every loop."""
    out: List[str] = []
    for p in tree.parts:
        for f in p.faces:
            for lname, prims in f.loops:
                out.append(f"{lname}: {'|'.join(prims)}")
    return out


def _walk_numeric_leaves(node, path: Tuple = ()):  # type: ignore[no-untyped-def]
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_numeric_leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_numeric_leaves(v, path + (i,))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, node


def structure_preserving_perturbation(
    cad: Dict, delta: float = 1.0, index: int = 0
) -> Dict:
    """Return a copy with the ``index``-th numeric leaf perturbed by ``delta``.

    The tree *shape* is preserved exactly (same keys, same nesting); only one
    numeric value changes. This is the ranking negative HierCAD's parameter
    grounding contrasts against the text-grounded target (paper Sec. 4.2).
    """
    leaves = list(_walk_numeric_leaves(cad))
    if not leaves:
        raise ValueError("no numeric leaves to perturb")
    if not 0 <= index < len(leaves):
        raise IndexError(f"index {index} out of range for {len(leaves)} leaves")
    path, value = leaves[index]
    out = copy.deepcopy(cad)
    ref = out
    for key in path[:-1]:
        ref = ref[key]
    ref[path[-1]] = value + delta
    return out
