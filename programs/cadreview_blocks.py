"""OpenSCAD block segmenter — parse a CAD program into ID'd code blocks.

CADReview reviews a CAD program by *component*: it splits the program into
"code blocks", each an irreducible unit that maps to one geometric component,
and comments a block ID before each so feedback can point at exactly the wrong
component ("the rotation in Block 2 is wrong"). The paper builds these blocks by
traversing the program's syntax tree top-down and treating macros, modules,
control flows (loops / conditionals), boolean operations (difference / union /
intersection) and geometric primitives (cube, cylinder, ...) as independent
blocks, with the leading macros/constants forming the first block.

This module reproduces that segmentation deterministically for OpenSCAD source,
using a brace/paren/string/comment-aware top-level statement splitter (no third
-party parser). Leading assignment/constant statements are grouped as Block 0
(the "macros" block); every subsequent top-level statement becomes its own
block. Each block is classified by kind (assignments / module / control_flow /
boolean / primitive / transform / other) and its salient tokens are extracted so
downstream detectors (:mod:`cadreview_detect`) can compare a program against its
reference. :func:`annotate` re-emits the source with ``// Block N`` comments,
matching the paper's block-ID commenting.

Pure stdlib; input is text, output is data — nothing is executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Construct vocabularies (OpenSCAD).
_PRIMITIVES = {
    "cube", "sphere", "cylinder", "polyhedron",
    "square", "circle", "polygon", "text",
    "linear_extrude", "rotate_extrude", "surface", "import",
}
_BOOLEANS = {"difference", "union", "intersection", "hull", "minkowski"}
_TRANSFORMS = {
    "translate", "rotate", "scale", "mirror", "resize",
    "color", "offset", "multmatrix",
}
_CONTROL = {"for", "if", "else", "let", "intersection_for"}

_ASSIGN_RE = re.compile(r"^\s*\$?[A-Za-z_]\w*\s*=")
_MODULE_RE = re.compile(r"^\s*(module|function)\s+([A-Za-z_]\w*)")
_CALL_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")


@dataclass
class Block:
    """One segmented code block.

    ``id`` is the 0-based block ID (Block 0 = leading macros). ``kind`` is the
    coarse category; ``head`` is the first construct name (primitive/transform/
    control/module keyword); ``calls`` are the call names in order; ``text`` is
    the raw source of the block (stripped)."""

    id: int
    kind: str
    head: Optional[str]
    calls: List[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "head": self.head,
            "calls": list(self.calls),
            "text": self.text,
        }


def _strip_comments(src: str) -> str:
    """Remove // line and /* */ block comments without touching string bodies."""
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


def split_statements(src: str) -> List[str]:
    """Split OpenSCAD source into top-level statements.

    A statement closes at brace/paren depth 0 either on ``;`` (a call/assignment)
    or when a ``}`` returns brace-depth to 0 (a construct with a body). Comments
    are stripped first; strings are respected. Deterministic and total."""
    code = _strip_comments(src)
    stmts: List[str] = []
    buf: List[str] = []
    depth_paren = 0
    depth_brace = 0
    saw_brace = False
    i, n = 0, len(code)
    while i < n:
        c = code[i]
        if c == '"':
            j = i + 1
            while j < n and code[j] != '"':
                if code[j] == "\\":
                    j += 1
                j += 1
            buf.append(code[i:j + 1])
            i = j + 1
            continue
        buf.append(c)
        if c == "(":
            depth_paren += 1
        elif c == ")":
            depth_paren = max(0, depth_paren - 1)
        elif c == "{":
            depth_brace += 1
            saw_brace = True
        elif c == "}":
            depth_brace = max(0, depth_brace - 1)
            if depth_brace == 0 and depth_paren == 0:
                text = "".join(buf).strip()
                if text:
                    stmts.append(text)
                buf = []
                saw_brace = False
        elif c == ";" and depth_paren == 0 and depth_brace == 0 and not saw_brace:
            text = "".join(buf).strip()
            if text:
                stmts.append(text)
            buf = []
        i += 1
    tail = "".join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


def _calls(stmt: str) -> List[str]:
    return _CALL_RE.findall(stmt)


def _classify(stmt: str, calls: List[str]) -> tuple:
    """Return ``(kind, head)`` for a single statement."""
    m = _MODULE_RE.match(stmt)
    if m:
        return "module", m.group(1)
    head = stmt.lstrip()
    # Leading keyword (control flow) — token before first '(' or '{'.
    lead = re.match(r"\s*([A-Za-z_]\w*)", stmt)
    lead_name = lead.group(1) if lead else None
    if lead_name in _CONTROL:
        return "control_flow", lead_name
    for name in calls:
        if name in _BOOLEANS:
            return "boolean", name
    for name in calls:
        if name in _PRIMITIVES:
            # A transform wrapping a primitive is still primitive-bearing, but
            # if a transform leads we tag it transform for the operation focus.
            if calls and calls[0] in _TRANSFORMS:
                return "transform", calls[0]
            return "primitive", name
    if calls and calls[0] in _TRANSFORMS:
        return "transform", calls[0]
    if _ASSIGN_RE.match(stmt):
        return "assignment", None
    return "other", (calls[0] if calls else None)


def segment(src: str) -> List[Block]:
    """Segment OpenSCAD source into ID'd :class:`Block` s.

    Leading contiguous assignment statements are merged into Block 0 (the
    macros/constants block, per the paper's "treat initial macros as the first
    block"). Every subsequent top-level statement is its own block."""
    stmts = split_statements(src)
    blocks: List[Block] = []

    # Gather leading assignments into Block 0.
    lead: List[str] = []
    idx = 0
    while idx < len(stmts) and _ASSIGN_RE.match(stmts[idx]) and not _MODULE_RE.match(stmts[idx]):
        lead.append(stmts[idx])
        idx += 1
    next_id = 0
    if lead:
        text = "\n".join(lead)
        blocks.append(Block(id=0, kind="assignment", head=None,
                            calls=[], text=text))
        next_id = 1

    for stmt in stmts[idx:]:
        calls = _calls(stmt)
        kind, head = _classify(stmt, calls)
        blocks.append(Block(id=next_id, kind=kind, head=head,
                            calls=calls, text=stmt.strip()))
        next_id += 1
    return blocks


def annotate(src: str) -> str:
    """Re-emit the source with a ``// Block N`` comment before each block."""
    blocks = segment(src)
    out: List[str] = []
    for b in blocks:
        out.append(f"// Block {b.id}")
        out.append(b.text)
        out.append("")
    return "\n".join(out).rstrip() + "\n"
