"""SldprtNet Encoder-txt feature-tree parametric representation.

SldprtNet (Li et al., ICRA 2026) ships an *encoder* that traverses the
SolidWorks ``.sldprt`` Feature Tree and emits a human- and machine-readable text
script (``Encoder.txt``). Unlike DeepCAD (six flat sketch/extrude command types,
see :mod:`reconstruction.deepcad_command_spec`) or HistCAD (a flat, unordered
constraint-aware sketch set, see :mod:`reconstruction.histcad_sequence`), the
SldprtNet script preserves the *hierarchical feature tree* of a real
feature-based part: features listed in modeling order, each carrying a name, a
type and an explicit parent/child relationship, followed by a per-feature
parameter block (Table II of the paper).

This module implements the deterministic schema and a lossless round-trip
serializer/parser for that format (no learned encoder/decoder, no SolidWorks COM
dependency). It supports the **13 representative feature types** the paper's tools
handle. Six are named explicitly in the paper (2D Sketch, Extrusion, Linear
Pattern, Mirror Pattern, Chamfer, Fillet) and ``RefPlane``/``ProfileFeature``
appear in Table II; the remaining standard SolidWorks feature-based operations
(Revolve, Sweep, Loft, Hole, Shell, Rib) round out the taxonomy.

The serialized layout mirrors Table II::

    Feature Tree:
    Top View (RefPlane)
    Sketch1 (ProfileFeature)
      Extrude1 (Extrusion)
    FeatureName: Top View   Type: RefPlane
      - Vertex1: -80.9, 0.0, -50.0
    FeatureName: Sketch1   Type: ProfileFeature
      - SketchPlane: Top View
    ...

Deterministic and stdlib-only: node order is preserved as given (the modeling
history), parameter order within a block is preserved, and ``from_text`` is the
exact inverse of ``to_text``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# The 13 supported feature types.
# ---------------------------------------------------------------------------
#: Canonical SolidWorks feature-tree type names the SldprtNet tools support.
FEATURE_TYPES: Tuple[str, ...] = (
    "RefPlane",        # reference plane (Table II: "Top View (RefPlane)")
    "ProfileFeature",  # a 2D sketch (Table II: "Sketch1 (ProfileFeature)")
    "Extrusion",       # extrude (named in paper)
    "Revolution",      # revolve
    "Sweep",           # sweep
    "Loft",            # loft
    "Fillet",          # fillet (named in paper)
    "Chamfer",         # chamfer (named in paper)
    "Hole",            # hole / hole-wizard
    "Shell",           # shell
    "Rib",             # rib
    "LinearPattern",   # linear pattern (named in paper)
    "MirrorPattern",   # mirror pattern (named in paper)
)
FEATURE_TYPE_SET = frozenset(FEATURE_TYPES)

#: A human-facing label per feature type (for statistics / captions).
FEATURE_LABELS: Dict[str, str] = {
    "RefPlane": "Reference Plane",
    "ProfileFeature": "2D Sketch",
    "Extrusion": "Extrusion",
    "Revolution": "Revolution",
    "Sweep": "Sweep",
    "Loft": "Loft",
    "Fillet": "Fillet",
    "Chamfer": "Chamfer",
    "Hole": "Hole",
    "Shell": "Shell",
    "Rib": "Rib",
    "LinearPattern": "Linear Pattern",
    "MirrorPattern": "Mirror Pattern",
}


@dataclass(frozen=True)
class FeatureNode:
    """One node in the SldprtNet feature tree.

    ``name`` is the (unique) feature name; ``ftype`` is one of
    :data:`FEATURE_TYPES`; ``parent`` names the parent feature (or ``None`` for a
    root feature); ``params`` is an *ordered* sequence of ``(key, value)`` string
    pairs forming the detail block; ``depends_on`` names sibling features this one
    references (e.g. an Extrusion referencing its Sketch).
    """

    name: str
    ftype: str
    parent: Optional[str] = None
    params: Tuple[Tuple[str, str], ...] = ()
    depends_on: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("feature name must be non-empty")
        if self.ftype not in FEATURE_TYPE_SET:
            raise ValueError(f"unknown feature type: {self.ftype!r}")
        for key, _val in self.params:
            if not key or not key.strip():
                raise ValueError("parameter key must be non-empty")

    def param(self, key: str, default: Optional[str] = None) -> Optional[str]:
        for k, v in self.params:
            if k == key:
                return v
        return default


@dataclass
class FeatureTree:
    """An ordered SldprtNet feature tree (modeling history)."""

    nodes: List[FeatureNode] = field(default_factory=list)

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        seen: set = set()
        for node in self.nodes:
            if node.name in seen:
                raise ValueError(f"duplicate feature name: {node.name!r}")
            seen.add(node.name)
        names = {n.name for n in self.nodes}
        # parents / dependencies must refer to features declared earlier
        declared: set = set()
        for node in self.nodes:
            if node.parent is not None and node.parent not in names:
                raise ValueError(f"parent {node.parent!r} of {node.name!r} not found")
            if node.parent is not None and node.parent not in declared:
                raise ValueError(
                    f"parent {node.parent!r} must precede child {node.name!r}"
                )
            for dep in node.depends_on:
                if dep not in names:
                    raise ValueError(f"dependency {dep!r} of {node.name!r} not found")
                if dep not in declared:
                    raise ValueError(
                        f"dependency {dep!r} must precede {node.name!r}"
                    )
            declared.add(node.name)

    def by_name(self, name: str) -> FeatureNode:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    def depth_of(self, name: str) -> int:
        """Indentation depth = number of ancestors via the ``parent`` chain."""
        depth = 0
        cur = self.by_name(name).parent
        guard = 0
        while cur is not None:
            depth += 1
            cur = self.by_name(cur).parent
            guard += 1
            if guard > len(self.nodes):
                raise ValueError("cycle detected in parent chain")
        return depth

    def feature_counts(self) -> Dict[str, int]:
        """Frequency of each feature type (sorted keys for determinism)."""
        counts: Dict[str, int] = {}
        for node in self.nodes:
            counts[node.ftype] = counts.get(node.ftype, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def num_features(self) -> int:
        return len(self.nodes)

    # -- serialization -----------------------------------------------------
    def to_text(self) -> str:
        """Serialize to the canonical Encoder-txt layout (see module docstring)."""
        self.validate()
        lines: List[str] = ["Feature Tree:"]
        for node in self.nodes:
            indent = "  " * self.depth_of(node.name)
            lines.append(f"{indent}{node.name} ({node.ftype})")
        for node in self.nodes:
            lines.append(f"FeatureName: {node.name}\tType: {node.ftype}")
            for dep in node.depends_on:
                lines.append(f"  - Depends: {dep}")
            for key, val in node.params:
                lines.append(f"  - {key}: {val}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_text(cls, text: str) -> "FeatureTree":
        """Parse the canonical Encoder-txt layout back into a :class:`FeatureTree`.

        Exact inverse of :meth:`to_text`.
        """
        raw_lines = text.split("\n")
        # strip a single trailing empty line from the final newline
        while raw_lines and raw_lines[-1] == "":
            raw_lines.pop()
        if not raw_lines or raw_lines[0].strip() != "Feature Tree:":
            raise ValueError("missing 'Feature Tree:' header")

        # Phase 1: parse the header block (name, type, indentation depth).
        header: List[Tuple[str, str, int]] = []  # (name, ftype, depth)
        i = 1
        while i < len(raw_lines):
            line = raw_lines[i]
            if line.startswith("FeatureName:"):
                break
            stripped = line.lstrip(" ")
            indent_len = len(line) - len(stripped)
            if indent_len % 2 != 0:
                raise ValueError(f"bad indentation on line: {line!r}")
            depth = indent_len // 2
            # form "Name (Type)"
            if not stripped.endswith(")") or "(" not in stripped:
                raise ValueError(f"bad tree line: {line!r}")
            paren = stripped.rfind("(")
            name = stripped[:paren].rstrip()
            ftype = stripped[paren + 1 : -1]
            header.append((name, ftype, depth))
            i += 1

        # Reconstruct parent from the depth stack.
        parents: Dict[str, Optional[str]] = {}
        stack: List[Tuple[str, int]] = []  # (name, depth)
        for name, _ftype, depth in header:
            while stack and stack[-1][1] >= depth:
                stack.pop()
            parents[name] = stack[-1][0] if stack else None
            stack.append((name, depth))

        # Phase 2: parse detail blocks.
        detail_params: Dict[str, List[Tuple[str, str]]] = {}
        detail_deps: Dict[str, List[str]] = {}
        detail_type: Dict[str, str] = {}
        order: List[str] = []
        current: Optional[str] = None
        while i < len(raw_lines):
            line = raw_lines[i]
            if line.startswith("FeatureName:"):
                rest = line[len("FeatureName:") :]
                if "\t" in rest:
                    name_part, type_part = rest.split("\t", 1)
                else:
                    # tolerate space-separated "Type:"
                    idx = rest.find("Type:")
                    name_part, type_part = rest[:idx], rest[idx:]
                name = name_part.strip()
                ftype = type_part.replace("Type:", "", 1).strip()
                current = name
                detail_type[name] = ftype
                detail_params.setdefault(name, [])
                detail_deps.setdefault(name, [])
                order.append(name)
            elif line.startswith("  - "):
                if current is None:
                    raise ValueError("parameter line before any FeatureName")
                body = line[len("  - ") :]
                if body.startswith("Depends: "):
                    detail_deps[current].append(body[len("Depends: ") :])
                else:
                    key, _sep, val = body.partition(": ")
                    detail_params[current].append((key, val))
            elif line.strip() == "":
                pass
            else:
                raise ValueError(f"unexpected line: {line!r}")
            i += 1

        nodes: List[FeatureNode] = []
        for name, ftype, _depth in header:
            nodes.append(
                FeatureNode(
                    name=name,
                    ftype=detail_type.get(name, ftype),
                    parent=parents[name],
                    params=tuple(detail_params.get(name, [])),
                    depends_on=tuple(detail_deps.get(name, [])),
                )
            )
        tree = cls(nodes=nodes)
        tree.validate()
        return tree


def is_supported_feature(ftype: str) -> bool:
    """Whether ``ftype`` is one of the 13 SldprtNet-supported feature types."""
    return ftype in FEATURE_TYPE_SET
