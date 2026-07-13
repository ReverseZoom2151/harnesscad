"""Validated, provider-neutral conditioning for sketch/image attachments.

The module prepares bytes for a caller-supplied encoder; it does not claim to
understand images. Paths are resolved under explicit roots, reads are bounded,
MIME types are verified from content, and provenance survives conditioning.
"""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence


class AttachmentKind(str, Enum):
    IMAGE = "image"
    SKETCH = "sketch"


@dataclass(frozen=True)
class AttachmentProvenance:
    origin: str
    author: Optional[str] = None
    captured_at: Optional[str] = None
    source_id: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.origin.strip():
            raise ValueError("provenance origin must be non-empty")


@dataclass(frozen=True)
class Attachment:
    kind: AttachmentKind
    provenance: AttachmentProvenance
    data: Optional[bytes] = None
    path: Optional[Path] = None
    declared_mime: Optional[str] = None
    expected_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.path is None):
            raise ValueError("provide exactly one of data or path")


@dataclass(frozen=True)
class ConditionedAttachment:
    kind: AttachmentKind
    mime: str
    size: int
    sha256: str
    provenance: AttachmentProvenance
    encoded: Any


class AttachmentEncoder(Protocol):
    """Provider adapter seam. Encoders receive only validated local bytes."""

    def encode(self, data: bytes, *, mime: str, kind: AttachmentKind) -> Any: ...


class DeterministicEncoder:
    """Test/offline encoder returning facts only; performs no vision inference."""

    def encode(self, data: bytes, *, mime: str, kind: AttachmentKind) -> Mapping[str, Any]:
        return {
            "kind": kind.value,
            "mime": mime,
            "byte_count": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }


_MIME_SIGNATURES = {
    "image/png": lambda b: b.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": lambda b: b.startswith(b"\xff\xd8\xff"),
    "image/webp": lambda b: len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP",
}
_ALLOWED = frozenset((*_MIME_SIGNATURES, "image/svg+xml"))


def condition_attachment(
    attachment: Attachment,
    encoder: AttachmentEncoder,
    *,
    allowed_roots: Sequence[Path] = (),
    max_bytes: int = 10 * 1024 * 1024,
) -> ConditionedAttachment:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    data, path = _load(attachment, allowed_roots, max_bytes)
    mime = _detect_mime(data)
    if attachment.declared_mime and attachment.declared_mime.lower() != mime:
        raise ValueError(
            f"declared MIME {attachment.declared_mime!r} does not match content {mime!r}"
        )
    if path is not None:
        guessed, _ = mimetypes.guess_type(path.name)
        if guessed in _ALLOWED and guessed != mime:
            raise ValueError(f"file extension MIME {guessed!r} does not match content {mime!r}")
    digest = hashlib.sha256(data).hexdigest()
    if attachment.expected_sha256:
        expected = attachment.expected_sha256.lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("expected_sha256 must be 64 lowercase/uppercase hex characters")
        if digest != expected:
            raise ValueError("attachment SHA-256 mismatch")
    encoded = encoder.encode(data, mime=mime, kind=attachment.kind)
    return ConditionedAttachment(
        attachment.kind, mime, len(data), digest, attachment.provenance, encoded
    )


def _load(
    attachment: Attachment, allowed_roots: Sequence[Path], max_bytes: int
) -> tuple[bytes, Optional[Path]]:
    if attachment.data is not None:
        data = bytes(attachment.data)
        if len(data) > max_bytes:
            raise ValueError(f"attachment exceeds {max_bytes} byte limit")
        return data, None

    assert attachment.path is not None
    if not allowed_roots:
        raise ValueError("path attachments require at least one allowed root")
    path = attachment.path.expanduser().resolve(strict=True)
    roots = [root.expanduser().resolve(strict=True) for root in allowed_roots]
    if not any(path == root or root in path.parents for root in roots):
        raise PermissionError("attachment path is outside allowed roots")
    if not path.is_file():
        raise ValueError("attachment path must be a regular file")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"attachment exceeds {max_bytes} byte limit")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"attachment exceeds {max_bytes} byte limit")
    return data, path


def _detect_mime(data: bytes) -> str:
    for mime, matches in _MIME_SIGNATURES.items():
        if matches(data):
            return mime
    stripped = data.lstrip()
    if stripped.startswith(b"<svg") or (
        stripped.startswith(b"<?xml") and b"<svg" in stripped[:1024]
    ):
        lowered = stripped.lower()
        # SVG is active XML. Permit simple local geometry only.
        forbidden = (b"<script", b"<!doctype", b"javascript:", b"file:", b"http:")
        if any(token in lowered for token in forbidden):
            raise ValueError("unsafe active or externally-referenced SVG")
        return "image/svg+xml"
    raise ValueError("unsupported or unrecognized attachment content")
