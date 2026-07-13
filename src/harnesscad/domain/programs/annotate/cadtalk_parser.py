"""CADTalk program parsing — hierarchical commentable-block segmentation.

CADTalk (Yuan et al.) frames semantic commenting of a CAD program as: parse the
program into a syntax tree, then identify *commentable code blocks* at multiple
compositional levels, each corresponding to a semantically meaningful shape part.
The paper's parsing rule (Sec. 3.3, Fig. 4) is:

  * Build the syntax tree of the program.
  * Traverse the tree *downward* (breadth-first) until reaching an **irreducible
    block** — a sequence of instructions that corresponds either to a single
    geometric primitive (cube, sphere, ...) or to a ``difference`` /
    ``intersection`` / ``hull`` operation on primitives. These are marked as the
    commentable **leaf** nodes.
  * Traverse *upward* to collect all commentable blocks as the nodes that are
    parents of commentable nodes (i.e. every ancestor of an irreducible leaf).

The key distinction versus :mod:`cadreview_blocks` (which only splits *top-level*
statements) is that this module builds a genuine nested tree and classifies
sub-trees as irreducible / compositional, yielding the *multi-level* commentable
blocks CADTalk needs. ``union`` and module boundaries are treated as
compositional (they group separate parts), whereas ``difference`` /
``intersection`` / ``hull`` / ``minkowski`` fuse primitives into one solid and so
are irreducible.

Pure stdlib; input is OpenSCAD text, output is data — nothing is executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Construct vocabularies (OpenSCAD), aligned with cadreview_blocks.
PRIMITIVES = {
    "cube", "sphere", "cylinder", "polyhedron",
    "square", "circle", "polygon", "text",
    "linear_extrude", "rotate_extrude", "surface", "import",
}
# Boolean ops that fuse primitives into a *single* solid (irreducible-forming).
SINGLE_SOLID_OPS = {"difference", "intersection", "hull", "minkowski"}
# Boolean op that composes *separate* parts (compositional).
COMPOSE_OPS = {"union"}
TRANSFORMS = {
    "translate", "rotate", "scale", "mirror", "resize",
    "color", "offset", "multmatrix",
}
CONTROL = {"for", "if", "else", "let", "intersection_for"}

_ASSIGN_RE = re.compile(r"^\s*\$?[A-Za-z_]\w*\s*=")
_DEF_RE = re.compile(r"^\s*(module|function)\s+([A-Za-z_]\w*)")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


@dataclass
class Node:
    """One node of the CAD program syntax tree.

    ``head`` is the construct name (call/keyword) or ``None`` for a bare group.
    ``kind`` is a coarse category: ``primitive`` / ``single_solid`` / ``compose``
    / ``transform`` / ``control`` / ``module`` / ``assignment`` / ``other``.
    ``children`` are nested nodes; ``text`` is the raw source span."""

    head: Optional[str]
    kind: str
    children: List["Node"] = field(default_factory=list)
    text: str = ""
    # Filled by :func:`identify_blocks`.
    irreducible: bool = False
    commentable: bool = False
    block_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "kind": self.kind,
            "irreducible": self.irreducible,
            "commentable": self.commentable,
            "block_id": self.block_id,
            "text": self.text,
            "children": [c.to_dict() for c in self.children],
        }


def _strip_comments(src: str) -> str:
    """Remove // and /* */ comments, respecting string bodies."""
    out: List[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


class _Parser:
    """Recursive-descent parser over brace/paren-balanced OpenSCAD source."""

    def __init__(self, src: str):
        self.s = src
        self.i = 0
        self.n = len(src)

    def _skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in " \t\r\n":
            self.i += 1

    def _balanced(self, open_c: str, close_c: str) -> str:
        """Consume a balanced ``open_c ... close_c`` region, returning inner text.

        Assumes ``self.s[self.i] == open_c``. Respects nested pairs and strings."""
        assert self.s[self.i] == open_c
        start = self.i + 1
        depth = 0
        i = self.i
        while i < self.n:
            c = self.s[i]
            if c == '"':
                i += 1
                while i < self.n and self.s[i] != '"':
                    if self.s[i] == "\\":
                        i += 1
                    i += 1
                i += 1
                continue
            if c == open_c:
                depth += 1
            elif c == close_c:
                depth -= 1
                if depth == 0:
                    inner = self.s[start:i]
                    self.i = i + 1
                    return inner
            i += 1
        # Unbalanced — take the rest.
        inner = self.s[start:self.n]
        self.i = self.n
        return inner

    def _read_ident(self) -> Optional[str]:
        m = _IDENT_RE.match(self.s, self.i)
        if not m:
            return None
        self.i = m.end()
        return m.group(0)

    def parse_seq(self) -> List[Node]:
        """Parse a sequence of statements until end / an unmatched ``}``."""
        nodes: List[Node] = []
        while True:
            self._skip_ws()
            if self.i >= self.n:
                break
            if self.s[self.i] == "}":
                break
            node = self.parse_statement()
            if node is not None:
                nodes.append(node)
        return nodes

    def parse_statement(self) -> Optional[Node]:
        self._skip_ws()
        if self.i >= self.n:
            return None
        c = self.s[self.i]
        if c == ";":
            self.i += 1
            return None
        if c == "{":
            inner = self._balanced("{", "}")
            children = _Parser(inner).parse_seq()
            return Node(head=None, kind="group", children=children,
                        text="{" + inner + "}")
        start = self.i
        rest = self.s[self.i:]
        # Module / function definition.
        m = _DEF_RE.match(rest)
        if m:
            return self._parse_def(m, start)
        # Assignment: ident = ... ;
        if _ASSIGN_RE.match(rest):
            j = self._find_semicolon(self.i)
            text = self.s[start:j].strip()
            self.i = j + 1 if j < self.n else self.n
            return Node(head=None, kind="assignment", text=text)
        # Call chain.
        return self._parse_call_chain(start)

    def _parse_def(self, m: "re.Match", start: int) -> Node:
        keyword = m.group(1)
        name = m.group(2)
        # advance past the header ident portion
        self.i += m.end()
        self._skip_ws()
        if self.i < self.n and self.s[self.i] == "(":
            self._balanced("(", ")")
        self._skip_ws()
        children: List[Node] = []
        if keyword == "module" and self.i < self.n and self.s[self.i] == "{":
            inner = self._balanced("{", "}")
            children = _Parser(inner).parse_seq()
        else:
            # single-statement module body or function '= expr ;'
            j = self._find_semicolon(self.i)
            self.i = j + 1 if j < self.n else self.n
        return Node(head=name, kind="module", children=children,
                    text=self.s[start:self.i].strip())

    def _parse_call_chain(self, start: int) -> Node:
        name = self._read_ident()
        if name is None:
            # Unparseable token — consume to next ; to stay total.
            j = self._find_semicolon(self.i)
            text = self.s[start:j].strip()
            self.i = j + 1 if j < self.n else self.n
            return Node(head=None, kind="other", text=text)
        self._skip_ws()
        if self.i < self.n and self.s[self.i] == "(":
            self._balanced("(", ")")
        self._skip_ws()
        children: List[Node] = []
        if self.i < self.n and self.s[self.i] == "{":
            inner = self._balanced("{", "}")
            children = _Parser(inner).parse_seq()
        elif self.i < self.n and self.s[self.i] == ";":
            self.i += 1
        elif self.i < self.n and self.s[self.i] == "}":
            pass
        else:
            # Chained modifier: the next statement is this node's single child.
            child = self.parse_statement()
            if child is not None:
                children = [child]
        kind = _classify_head(name)
        return Node(head=name, kind=kind, children=children,
                    text=self.s[start:self.i].strip())

    def _find_semicolon(self, i: int) -> int:
        depth_p = depth_b = 0
        while i < self.n:
            c = self.s[i]
            if c == '"':
                i += 1
                while i < self.n and self.s[i] != '"':
                    if self.s[i] == "\\":
                        i += 1
                    i += 1
                i += 1
                continue
            if c == "(":
                depth_p += 1
            elif c == ")":
                depth_p -= 1
            elif c == "{":
                depth_b += 1
            elif c == "}":
                depth_b -= 1
            elif c == ";" and depth_p <= 0 and depth_b <= 0:
                return i
            i += 1
        return self.n


def _classify_head(name: str) -> str:
    if name in PRIMITIVES:
        return "primitive"
    if name in SINGLE_SOLID_OPS:
        return "single_solid"
    if name in COMPOSE_OPS:
        return "compose"
    if name in TRANSFORMS:
        return "transform"
    if name in CONTROL:
        return "control"
    return "other"


def parse(src: str) -> List[Node]:
    """Parse OpenSCAD source into a list of top-level syntax-tree nodes."""
    code = _strip_comments(src)
    return _Parser(code).parse_seq()


def is_single_solid(node: Node) -> bool:
    """Whether ``node``'s subtree yields a single coherent solid (no separate
    parts). True for primitives and single-solid ops whose descendants are all
    single-solid; False for ``union`` / modules / control flow / unknown calls
    that may compose multiple parts."""
    if node.kind == "primitive":
        return True
    if node.kind in ("transform", "single_solid"):
        # A transform/fusing op is a single solid iff it has children and every
        # child subtree is itself a single solid.
        return bool(node.children) and all(
            is_single_solid(c) for c in node.children)
    return False


def _has_primitive(node: Node) -> bool:
    if node.kind == "primitive":
        return True
    return any(_has_primitive(c) for c in node.children)


def identify_blocks(nodes: List[Node]) -> List[Node]:
    """Mark irreducible + commentable blocks and assign block IDs.

    Implements CADTalk's two-pass rule: a downward pass marks the topmost
    single-solid nodes (that actually contain a primitive) as irreducible
    commentable *leaves*; an upward pass marks every ancestor of a commentable
    node as commentable. Assignments / module *definitions* are not commentable
    geometry. Returns the flat list of commentable nodes in pre-order, each with
    ``block_id`` assigned."""

    def down(node: Node) -> bool:
        """Return True if ``node`` (or a descendant) is commentable."""
        if is_single_solid(node) and _has_primitive(node):
            node.irreducible = True
            node.commentable = True
            return True
        child_flag = False
        for c in node.children:
            if down(c):
                child_flag = True
        if child_flag:
            node.commentable = True
        return child_flag

    for root in nodes:
        down(root)

    ordered: List[Node] = []

    def collect(node: Node) -> None:
        if node.commentable:
            ordered.append(node)
        for c in node.children:
            collect(c)

    for root in nodes:
        collect(root)

    for bid, node in enumerate(ordered):
        node.block_id = bid
    return ordered


def commentable_blocks(src: str) -> List[Node]:
    """Convenience: parse ``src`` and return its commentable blocks (pre-order)."""
    nodes = parse(src)
    return identify_blocks(nodes)


def annotate(src: str, labels: Optional[dict] = None, tbc: str = "TBC") -> str:
    """Re-emit ``src`` with a comment before each commentable block.

    Unlabelled blocks are marked ``// TBC`` (to-be-commented, per Fig. 2a).
    ``labels`` maps ``block_id -> label`` to emit ``// <label>`` instead."""
    labels = labels or {}
    blocks = commentable_blocks(src)
    out: List[str] = []
    for b in blocks:
        lab = labels.get(b.block_id, tbc)
        out.append(f"// [{b.block_id}] {lab}")
        out.append(b.text)
        out.append("")
    return "\n".join(out).rstrip() + "\n"
