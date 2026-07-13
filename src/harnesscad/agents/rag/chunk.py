"""Structure-aware chunking for the RAG grounding layer (blueprint sec.7, sec.19 P2).

Standards and API docs are not flat prose — they have a *shape* (headings,
sections, code/API blocks). Naive fixed-window chunking shreds that shape and
strands facts from the heading that gives them meaning ("Table 3" under
"§4.2 Bolt torque" means nothing once the breadcrumb is lost). So we chunk
structurally:

  - **Markdown/text** splits on ATX headings (``#`` .. ``######``). Every chunk
    carries the *heading breadcrumb path* it lives under (e.g.
    ``["Fasteners", "M6 bolts", "Torque"]``) so retrieval preserves context.
  - **Fenced code / API blocks** (```` ``` ````) are emitted as **atomic units**
    — never split mid-block — because a code signature or a worked API example
    only means anything whole.

Dependency-free (stdlib only). ``chunk_document(text, source) -> list[Chunk]``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List

# ATX heading:  up to 6 '#', a space, then the title text.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# A fenced code block delimiter: ``` or ~~~ (optionally with an info string).
_FENCE = re.compile(r"^\s*(```+|~~~+)(.*)$")


@dataclass
class Chunk:
    """One retrievable unit of a document.

    - ``id``          : deterministic, content-derived id (stable across runs).
    - ``text``        : the chunk body (prose paragraph group or an atomic block).
    - ``source``      : where it came from (path or logical doc name).
    - ``heading_path``: breadcrumb of headings the chunk lives under (outer->inner).
    - ``kind``        : ``"text"`` | ``"code"`` — code blocks are atomic.
    - ``ordinal``     : position within the source document (stable ordering).
    """

    id: str
    text: str
    source: str
    heading_path: List[str] = field(default_factory=list)
    kind: str = "text"
    ordinal: int = 0

    @property
    def breadcrumb(self) -> str:
        """Human-readable heading trail, e.g. ``"Fasteners > M6 > Torque"``."""
        return " > ".join(self.heading_path)

    def with_context(self) -> str:
        """The chunk text prefixed with its breadcrumb — what you feed a model."""
        return f"{self.breadcrumb}\n{self.text}" if self.heading_path else self.text

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "heading_path": list(self.heading_path),
            "kind": self.kind,
            "ordinal": self.ordinal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            id=d["id"],
            text=d["text"],
            source=d.get("source", ""),
            heading_path=list(d.get("heading_path", [])),
            kind=d.get("kind", "text"),
            ordinal=d.get("ordinal", 0),
        )


def _mk_id(source: str, ordinal: int, text: str) -> str:
    """Deterministic id: source + ordinal + a short content hash (dedup-safe)."""
    h = hashlib.sha1(f"{source}\x00{ordinal}\x00{text}".encode("utf-8")).hexdigest()
    return f"{source}::{ordinal}:{h[:8]}"


def _update_heading_stack(stack: List[tuple], level: int, title: str) -> List[tuple]:
    """Pop siblings/deeper headings, then push the new (level, title)."""
    stack = [(lvl, t) for (lvl, t) in stack if lvl < level]
    stack.append((level, title))
    return stack


def chunk_document(text: str, source: str = "doc") -> List[Chunk]:
    """Split ``text`` into structure-aware chunks.

    Sections are delimited by Markdown headings; each chunk keeps the heading
    breadcrumb it sits under. Fenced code blocks are emitted whole as atomic
    ``kind="code"`` chunks. Consecutive prose (separated by blank lines only) is
    grouped into a single section chunk; a heading boundary always starts a new
    chunk.
    """
    chunks: List[Chunk] = []
    heading_stack: List[tuple] = []
    buf: List[str] = []
    ordinal = 0

    def flush_prose() -> None:
        nonlocal ordinal
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        path = [t for (_lvl, t) in heading_stack]
        chunks.append(Chunk(
            id=_mk_id(source, ordinal, body),
            text=body,
            source=source,
            heading_path=path,
            kind="text",
            ordinal=ordinal,
        ))
        ordinal += 1

    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        fence = _FENCE.match(line)
        if fence:
            # Atomic code/API block: flush prose, then consume to the closing fence.
            flush_prose()
            marker = fence.group(1)[0]  # '`' or '~'
            block = [line]
            i += 1
            while i < n:
                block.append(lines[i])
                close = _FENCE.match(lines[i])
                if close and close.group(1)[0] == marker:
                    i += 1
                    break
                i += 1
            body = "\n".join(block).strip()
            if body:
                path = [t for (_lvl, t) in heading_stack]
                chunks.append(Chunk(
                    id=_mk_id(source, ordinal, body),
                    text=body,
                    source=source,
                    heading_path=path,
                    kind="code",
                    ordinal=ordinal,
                ))
                ordinal += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            # A heading boundary closes the current section and opens a new one.
            flush_prose()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = _update_heading_stack(heading_stack, level, title)
            i += 1
            continue

        buf.append(line)
        i += 1

    flush_prose()
    return chunks


def chunk_documents(docs) -> List[Chunk]:
    """Chunk many ``(text, source)`` pairs into a single flat chunk list."""
    out: List[Chunk] = []
    for text, source in docs:
        out.extend(chunk_document(text, source))
    return out
