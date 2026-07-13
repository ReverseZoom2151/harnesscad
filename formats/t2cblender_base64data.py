"""Deterministic URL-safe base64 codec with padding correction.

The Zoo / KittyCAD text-to-CAD API returns generated CAD files (glb, stl, obj,
...) as a single base64-encoded string embedded in the JSON operation result.
The ``text-to-cad-blender-addon`` decodes that payload with the peculiar
expression::

    base64.urlsafe_b64decode(outputs.strip("=") + "===")

which mirrors the official ``kittycad.py`` ``Base64Data`` model.  The reason is
that the transmitted string may carry incorrect, missing, or extra ``=``
padding; stripping all trailing ``=`` and then appending three ``=`` produces a
buffer whose first complete 4-character group re-establishes the correct
padding for any input length.

This module reimplements that behaviour as a self-contained, stdlib-only,
deterministic codec that does not depend on Blender, live network calls, or the
``kittycad`` package.  It is transferable to any consumer of a base64-encoded
binary asset that may arrive with malformed padding, and it works for both the
URL-safe alphabet (``-``/``_``) and the standard alphabet (``+``/``/``).

No randomness, no wall clock: the same bytes in always yield the same bytes out.
"""

from __future__ import annotations

import base64
import binascii

__all__ = [
    "Base64DecodeError",
    "canonical_padding",
    "normalise_padding",
    "decode_base64data",
    "encode_base64data",
    "is_urlsafe_alphabet",
]

# URL-safe base64 substitutes '-' for '+' and '_' for '/'.
_URLSAFE_ONLY = frozenset("-_")
_STD_ONLY = frozenset("+/")


class Base64DecodeError(ValueError):
    """Raised when a base64 payload cannot be decoded even after correction."""


def is_urlsafe_alphabet(text: str) -> bool:
    """Return True if *text* uses URL-safe characters (``-``/``_``).

    A string containing neither URL-safe nor standard specials is ambiguous;
    it is reported as URL-safe (the Zoo API default) since the two alphabets
    agree on ``[A-Za-z0-9]``.
    """
    chars = set(text)
    if chars & _STD_ONLY:
        return False
    return True


def canonical_padding(unpadded_len: int) -> int:
    """Return the number of ``=`` characters a stripped payload of length
    *unpadded_len* requires to become a valid base64 string.

    Base64 encodes 3 bytes as 4 characters, so a well-formed base64 body has a
    length that is a multiple of 4.  A remainder of 1 is impossible for genuine
    base64; the caller decides how to treat it (:func:`decode_base64data`
    raises).
    """
    if unpadded_len < 0:
        raise ValueError("length must be non-negative")
    return (-unpadded_len) % 4


def normalise_padding(text: str) -> str:
    """Strip every trailing ``=`` from *text* and re-append exactly the right
    number of ``=`` so the result is a validly-padded base64 string.

    This is the clean, canonical equivalent of the addon's ``strip("=") +
    "==="`` idiom.  Whitespace/newlines that transports sometimes inject are
    removed first so folded payloads decode correctly.
    """
    stripped = "".join(text.split()).rstrip("=")
    pad = canonical_padding(len(stripped))
    return stripped + ("=" * pad)


def decode_base64data(text: str, *, urlsafe: bool | None = None) -> bytes:
    """Decode a possibly mis-padded base64 *text* into raw bytes.

    * Surrounding/interior whitespace is discarded.
    * All existing ``=`` padding is stripped and recomputed.
    * The alphabet is auto-detected unless *urlsafe* is given explicitly.

    Raises :class:`Base64DecodeError` for an impossible length (``len % 4 == 1``)
    or genuinely corrupt input.
    """
    stripped = "".join(text.split()).rstrip("=")
    if len(stripped) % 4 == 1:
        raise Base64DecodeError(
            "invalid base64 length: %d characters cannot be padded" % len(stripped)
        )
    padded = stripped + ("=" * canonical_padding(len(stripped)))
    if urlsafe is None:
        urlsafe = is_urlsafe_alphabet(stripped)
    decoder = base64.urlsafe_b64decode if urlsafe else base64.b64decode
    try:
        return decoder(padded)
    except (binascii.Error, ValueError) as exc:  # pragma: no cover - defensive
        raise Base64DecodeError(str(exc)) from exc


def encode_base64data(data: bytes, *, urlsafe: bool = True, strip: bool = False) -> str:
    """Encode *data* to base64.

    With *urlsafe* the ``-``/``_`` alphabet is used.  With *strip* the trailing
    ``=`` padding is removed, producing exactly the transport form that
    :func:`decode_base64data` is designed to round-trip.
    """
    encoder = base64.urlsafe_b64encode if urlsafe else base64.b64encode
    text = encoder(data).decode("ascii")
    if strip:
        text = text.rstrip("=")
    return text
