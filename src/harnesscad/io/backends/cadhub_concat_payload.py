"""Single-response artifact+metadata payload codec (deterministic, stdlib-only).

Ported from CadHub's render transport. Each language runner concatenates the
binary artifact, a sentinel, and a JSON metadata blob into one gzipped file::

    cat output.png /var/task/cadhub-concat-split metadata.json | gzip > output.gz

and the client (``splitGziped`` in ``helpers/cadPackages/common.ts``) splits on
the sentinel to recover both halves. The point is that a mesh/image *and* its
console output, its customizer parameter manifest and its camera summary travel
as one cacheable blob -- one fetch, one cache entry, no second round trip that
could disagree with the first.

The harness moves artifacts and metadata around as separate objects; this gives
it a deterministic, self-describing single-blob envelope with the same property.

Two deliberate hardenings over the original:

* the original splits on the FIRST sentinel occurrence, which a binary STL could
  contain by chance; this splits on the LAST occurrence (the metadata JSON is
  ASCII and cannot contain it once encoded), so binary payloads are safe;
* gzip is written with ``mtime=0`` so the same inputs always produce byte-identical
  output -- the blob can be content-addressed.

Deterministic: no clock, no randomness, no filesystem.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

SENTINEL = b"cadhub-concat-split"

# Metadata "type" values the pipeline understands (CadHub's artifact tags).
TYPE_STL = "stl"
TYPE_PNG = "png"


class PayloadError(ValueError):
    """Raised when a blob is not a well-formed concat payload."""


@dataclass(frozen=True)
class Payload:
    """A decoded response: the artifact bytes plus its metadata."""

    artifact: bytes
    metadata: Dict[str, Any]

    @property
    def artifact_type(self) -> Optional[str]:
        value = self.metadata.get("type")
        return value if isinstance(value, str) else None

    @property
    def console_message(self) -> str:
        value = self.metadata.get("consoleMessage")
        return value if isinstance(value, str) else ""

    @property
    def customizer_params(self) -> list:
        value = self.metadata.get("customizerParams")
        return value if isinstance(value, list) else []

    def is_mesh(self) -> bool:
        return self.artifact_type == TYPE_STL


def encode_metadata(metadata: Mapping[str, Any]) -> bytes:
    """Canonical (key-sorted, compact) JSON for the metadata half."""
    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    blob = text.encode("utf-8")
    if SENTINEL in blob:
        raise PayloadError("metadata may not contain the sentinel")
    return blob


def encode(artifact: bytes, metadata: Mapping[str, Any], *, compress: bool = True) -> bytes:
    """Concatenate artifact + sentinel + metadata, optionally gzipped."""
    if not isinstance(artifact, (bytes, bytearray)):
        raise PayloadError("artifact must be bytes")
    raw = bytes(artifact) + SENTINEL + encode_metadata(metadata)
    return gzip_bytes(raw) if compress else raw


def gzip_bytes(data: bytes) -> bytes:
    """Reproducible gzip: fixed mtime, fixed compression level."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def _is_gzip(data: bytes) -> bool:
    return len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B


def split(raw: bytes) -> Tuple[bytes, bytes]:
    """Split an *uncompressed* payload on the LAST sentinel occurrence."""
    index = raw.rfind(SENTINEL)
    if index < 0:
        raise PayloadError("sentinel not found")
    return raw[:index], raw[index + len(SENTINEL) :]


def decode(blob: bytes, *, strict: bool = True) -> Payload:
    """Decode a payload (gzipped or not) into artifact + metadata.

    With ``strict=False`` an undecodable metadata half degrades to ``{}`` -- the
    behaviour of the original ``splitGziped``, which never let a malformed tail
    lose the artifact.
    """
    raw = gzip.decompress(blob) if _is_gzip(blob) else bytes(blob)
    try:
        artifact, tail = split(raw)
    except PayloadError:
        if strict:
            raise
        return Payload(artifact=raw, metadata={})
    try:
        metadata = json.loads(tail.decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
    except (ValueError, UnicodeDecodeError):
        if strict:
            raise PayloadError("metadata is not a JSON object")
        metadata = {}
    return Payload(artifact=artifact, metadata=metadata)


def metadata_of(blob: bytes) -> Dict[str, Any]:
    """The metadata half only -- cheap when the artifact is not needed."""
    return decode(blob, strict=False).metadata


def roundtrip(artifact: bytes, metadata: Mapping[str, Any]) -> Payload:
    """encode -> decode, for tests and self-checks."""
    return decode(encode(artifact, metadata))
