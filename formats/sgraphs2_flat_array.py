"""Flat offset-indexed container format for large sketch corpora.

SketchGraphs ships 15M sketches.  Storing them as one pickle means the whole
corpus must be deserialised to touch one sketch; storing them as 15M files means
the filesystem becomes the bottleneck.  The reference implementation solves this
with a container format (``sketchgraphs/data/flat_array.py``) that the paper
never mentions but that every downstream loader depends on: a single contiguous
blob of ``[magic | version | count | offsets | payload]``, where the offset table
makes element *i* an O(1) slice and nothing but element *i* is ever decoded.
Because the blob is contiguous, it can be memory-mapped and shared across worker
processes without copying.

This module reimplements the container -- the part that is genuinely a *format* --
in the standard library, with two deliberate departures:

* **Payloads are opaque bytes, not pickles.**  The reference pickles each element
  into the payload region, which makes the file an arbitrary-code-execution
  vector and ties it to a Python version.  Here the container is agnostic: the
  caller encodes elements however it likes (JSON, a packed struct, ...) and the
  container only owns the framing.  This is strictly more useful and strictly
  safer.
* **The dictionary header is JSON with sorted keys, not a pickle.**  That makes
  packing *deterministic*: identical input produces byte-identical output, so
  corpus builds are reproducible and hashable.

Container layout (all integers little-endian unsigned 64-bit)
-------------------------------------------------------------
Array::

    b"SGFLAT2\\n" | version=2 | count=n | offsets[n+1] | payload

``offsets[i]..offsets[i+1]`` delimits element *i* within ``payload``.  There are
``n + 1`` offsets, so element sizes -- including the last -- are recoverable
without a length field, and an empty element is representable.

Dictionary (a set of related named blobs, e.g. a sketch array plus its
quantisation statistics)::

    b"SGFDICT1" | version=1 | header_len | header (JSON) | payload

where ``header`` maps ``name -> [offset, length]``.  A dictionary value may
itself be a packed array, which is how a corpus and its metadata live in one file.

Public API
----------
``pack_array(items)`` / ``unpack_array(blob)`` / ``FlatArray``.
``merge_arrays(blobs)``       -- concatenate shards without re-encoding payloads.
``pack_dict(mapping)`` / ``unpack_dict(blob)``.
``save_array`` / ``load_array`` / ``save_dict`` / ``load_dict``.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

__all__ = [
    "ARRAY_MAGIC",
    "DICT_MAGIC",
    "ARRAY_VERSION",
    "DICT_VERSION",
    "FlatArray",
    "pack_array",
    "unpack_array",
    "merge_arrays",
    "pack_dict",
    "unpack_dict",
    "save_array",
    "load_array",
    "save_dict",
    "load_dict",
]

ARRAY_MAGIC = b"SGFLAT2\n"
DICT_MAGIC = b"SGFDICT1"
ARRAY_VERSION = 2
DICT_VERSION = 1

_U64 = struct.Struct("<Q")
_U64_SIZE = _U64.size

BytesLike = Union[bytes, bytearray, memoryview]


def _read_u64(data: BytesLike, offset: int) -> Tuple[int, int]:
    if offset + _U64_SIZE > len(data):
        raise ValueError("truncated container: expected a 64-bit field")
    (value,) = _U64.unpack_from(data, offset)
    return value, offset + _U64_SIZE


# ---------------------------------------------------------------------------
# Array
# ---------------------------------------------------------------------------
def pack_array(items: Sequence[BytesLike]) -> bytes:
    """Pack a sequence of byte payloads into a flat array blob.

    Deterministic: the same items always produce the same bytes.
    """
    payloads = [bytes(item) for item in items]

    offsets: List[int] = [0]
    total = 0
    for payload in payloads:
        total += len(payload)
        offsets.append(total)

    out = bytearray()
    out += ARRAY_MAGIC
    out += _U64.pack(ARRAY_VERSION)
    out += _U64.pack(len(payloads))
    for offset in offsets:
        out += _U64.pack(offset)
    for payload in payloads:
        out += payload
    return bytes(out)


def _parse_array_header(blob: BytesLike) -> Tuple[List[int], int]:
    """Validate the header; return ``(offsets, payload_start)``."""
    view = memoryview(blob)
    if len(view) < len(ARRAY_MAGIC):
        raise ValueError("truncated container: missing magic")
    if bytes(view[: len(ARRAY_MAGIC)]) != ARRAY_MAGIC:
        raise ValueError("not a flat array: bad magic bytes")

    pos = len(ARRAY_MAGIC)
    version, pos = _read_u64(view, pos)
    if version != ARRAY_VERSION:
        raise ValueError(f"unsupported flat array version {version}")

    count, pos = _read_u64(view, pos)

    offsets: List[int] = []
    for _ in range(count + 1):
        value, pos = _read_u64(view, pos)
        offsets.append(value)

    payload_len = len(view) - pos
    if offsets and offsets[-1] > payload_len:
        raise ValueError("truncated container: payload shorter than offset table")
    for a, b in zip(offsets, offsets[1:]):
        if b < a:
            raise ValueError("corrupt container: offsets are not monotonic")

    return offsets, pos


class FlatArray:
    """A lazy, random-access view over a packed flat array.

    Behaves as an immutable sequence of ``bytes``.  Element *i* is sliced out of
    the backing buffer on access -- nothing is decoded up front, so opening a
    corpus is O(number of elements) in the offset table and O(1) in payload size.
    """

    __slots__ = ("_buffer", "_offsets", "_start")

    def __init__(self, blob: BytesLike) -> None:
        self._offsets, self._start = _parse_array_header(blob)
        self._buffer = memoryview(blob)

    def __len__(self) -> int:
        return max(len(self._offsets) - 1, 0)

    def __getitem__(self, index: int) -> bytes:
        length = len(self)
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(length))]
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError("flat array index out of range")
        lo = self._start + self._offsets[index]
        hi = self._start + self._offsets[index + 1]
        return bytes(self._buffer[lo:hi])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FlatArray(n={len(self)})"


def unpack_array(blob: BytesLike) -> List[bytes]:
    """Decode a flat array blob into a list of payloads (eager)."""
    return list(FlatArray(blob))


def merge_arrays(blobs: Iterable[BytesLike]) -> bytes:
    """Concatenate packed arrays into one, without re-encoding the payloads.

    Each shard's offsets are shifted by the running payload total, so building a
    corpus from per-worker shards costs a copy of the bytes and nothing more.
    """
    offsets: List[int] = [0]
    payloads: List[bytes] = []
    total = 0

    for blob in blobs:
        shard_offsets, start = _parse_array_header(blob)
        view = memoryview(blob)
        used = shard_offsets[-1] if shard_offsets else 0
        payloads.append(bytes(view[start : start + used]))
        for offset in shard_offsets[1:]:
            offsets.append(total + offset)
        total += used

    out = bytearray()
    out += ARRAY_MAGIC
    out += _U64.pack(ARRAY_VERSION)
    out += _U64.pack(len(offsets) - 1)
    for offset in offsets:
        out += _U64.pack(offset)
    for payload in payloads:
        out += payload
    return bytes(out)


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------
def pack_dict(mapping: Mapping[str, BytesLike]) -> bytes:
    """Pack a name -> bytes mapping into a flat dictionary blob.

    Keys are sorted, so packing is deterministic regardless of insertion order.
    """
    items = sorted((str(k), bytes(v)) for k, v in mapping.items())

    header: Dict[str, List[int]] = {}
    offset = 0
    for key, payload in items:
        header[key] = [offset, len(payload)]
        offset += len(payload)

    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")

    out = bytearray()
    out += DICT_MAGIC
    out += _U64.pack(DICT_VERSION)
    out += _U64.pack(len(header_bytes))
    out += header_bytes
    for _, payload in items:
        out += payload
    return bytes(out)


def unpack_dict(blob: BytesLike) -> Dict[str, bytes]:
    """Decode a flat dictionary blob into a name -> bytes mapping."""
    view = memoryview(blob)
    if len(view) < len(DICT_MAGIC) or bytes(view[: len(DICT_MAGIC)]) != DICT_MAGIC:
        raise ValueError("not a flat dictionary: bad magic bytes")

    pos = len(DICT_MAGIC)
    version, pos = _read_u64(view, pos)
    if version != DICT_VERSION:
        raise ValueError(f"unsupported flat dictionary version {version}")

    header_len, pos = _read_u64(view, pos)
    if pos + header_len > len(view):
        raise ValueError("truncated container: header runs past end of blob")

    header = json.loads(bytes(view[pos : pos + header_len]).decode("utf-8"))
    payload_start = pos + header_len

    out: Dict[str, bytes] = {}
    for key, (offset, length) in header.items():
        lo = payload_start + offset
        hi = lo + length
        if hi > len(view):
            raise ValueError(f"truncated container: entry {key!r} runs past end of blob")
        out[key] = bytes(view[lo:hi])
    return out


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def save_array(path: Union[str, os.PathLike], items: Sequence[BytesLike]) -> None:
    """Write a flat array to ``path``."""
    with open(path, "wb") as handle:
        handle.write(pack_array(items))


def load_array(path: Union[str, os.PathLike]) -> FlatArray:
    """Read a flat array from ``path`` as a lazy :class:`FlatArray`."""
    with open(path, "rb") as handle:
        return FlatArray(handle.read())


def save_dict(path: Union[str, os.PathLike], mapping: Mapping[str, BytesLike]) -> None:
    """Write a flat dictionary to ``path``."""
    with open(path, "wb") as handle:
        handle.write(pack_dict(mapping))


def load_dict(path: Union[str, os.PathLike]) -> Dict[str, bytes]:
    """Read a flat dictionary from ``path``."""
    with open(path, "rb") as handle:
        return unpack_dict(handle.read())
