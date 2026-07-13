"""ISO 10303-21 string control-directive (X-encoding) codec.

Part-21 string literals are restricted to the printable characters of the
STEP character set (ISO 10303-21 table 1).  Any character outside that set is
represented with reverse-solidus *control directives* embedded in the literal:

  * ``\\X\\HH``            - one ISO 8859-1 octet ``HH`` (two hex digits);
  * ``\\X2\\HHHH..\\X0\\`` - a run of UCS-2 code units (four hex digits each),
                            terminated by ``\\X0\\``;
  * ``\\X4\\HHHHHHHH..\\X0\\`` - a run of UCS-4 code points (eight hex digits);
  * ``\\S\\c``             - the character ``c`` shifted into the upper half of
                            the current 8-bit code page (code point ``c`` + 128);
  * ``\\\\``               - a literal reverse solidus.

:mod:`formats.stepllm_parser` reads the literal text verbatim: it correctly
handles the ``''`` apostrophe escape, but it does **not** interpret these
control directives, so a name such as ``'caf\\X2\\00E9\\X0\\'`` is returned with
the raw ``\\X2\\...`` sequence rather than the intended ``café``, and there is
no way to recover the Unicode text.  This module supplies the missing decode
(directives -> Python ``str``) and its inverse encode (``str`` -> a minimal
part-21 literal body), so callers can round-trip non-ASCII CAD labels.

Pure and deterministic; stdlib only.
"""

from __future__ import annotations


class XStringError(ValueError):
    """Raised on a malformed part-21 string control directive."""


_STEP_MIN = 0x20   # first printable code point kept verbatim
_STEP_MAX = 0x7E   # last printable code point kept verbatim


def decode(body: str) -> str:
    """Decode a part-21 string *body* (the text between the apostrophes,
    with ``''`` already collapsed to ``'``) into a Python ``str``."""

    out: list = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            raise XStringError("dangling reverse solidus")
        nxt = body[i + 1]
        if nxt == "\\":                       # \\  -> literal backslash
            out.append("\\")
            i += 2
            continue
        if nxt in "Ss":                       # \S\c  -> code point of c + 128
            if body[i + 2:i + 3] != "\\" or i + 3 >= n:
                raise XStringError("malformed \\S\\ directive")
            ch = body[i + 3]
            out.append(chr(ord(ch) + 128))
            i += 4
            continue
        if nxt in "Xx":
            marker = body[i + 2:i + 3]
            if marker == "\\":                # \X\HH  -> single ISO-8859-1 octet
                hexs = body[i + 3:i + 5]
                _need_hex(hexs, 2)
                out.append(chr(int(hexs, 16)))
                i += 5
                continue
            if marker in ("2", "4"):          # \X2\....\X0\  /  \X4\....\X0\
                width = 4 if marker == "2" else 8
                if body[i + 3:i + 4] != "\\":
                    raise XStringError("malformed \\X2\\/\\X4\\ directive")
                j = i + 4
                while True:
                    # End marker \X0\
                    if body[j:j + 4].upper() == "\\X0\\":
                        j += 4
                        break
                    hexs = body[j:j + width]
                    _need_hex(hexs, width)
                    out.append(chr(int(hexs, 16)))
                    j += width
                    if j > n:
                        raise XStringError("unterminated \\X2\\/\\X4\\ run")
                i = j
                continue
            raise XStringError(f"unknown \\X directive marker {marker!r}")
        raise XStringError(f"unknown control directive \\{nxt}")
    return "".join(out)


def _need_hex(s: str, width: int) -> None:
    if len(s) != width or any(ch not in "0123456789abcdefABCDEF" for ch in s):
        raise XStringError(f"expected {width} hex digits, got {s!r}")


def encode(text: str) -> str:
    """Encode a Python ``str`` into a minimal part-21 string *body*.

    ASCII-printable characters are emitted verbatim (a literal backslash as
    ``\\\\``); everything else uses the narrowest directive: ``\\X\\HH`` for
    Latin-1 octets and a ``\\X2\\``/``\\X4\\`` run for wider code points.
    """

    out: list = []
    i, n = 0, len(text)
    while i < n:
        cp = ord(text[i])
        if text[i] == "\\":
            out.append("\\\\")
            i += 1
            continue
        if _STEP_MIN <= cp <= _STEP_MAX:
            out.append(text[i])
            i += 1
            continue
        if cp <= 0xFF:
            out.append(f"\\X\\{cp:02X}")
            i += 1
            continue
        # Group a run of wide characters into one \X2\ or \X4\ block.
        width = 4 if cp <= 0xFFFF else 8
        marker = "2" if width == 4 else "4"
        run: list = []
        while i < n:
            cp = ord(text[i])
            if _STEP_MIN <= cp <= _STEP_MAX or cp <= 0xFF:
                break
            w = 4 if cp <= 0xFFFF else 8
            if w != width:
                break
            run.append(f"{cp:0{width}X}")
            i += 1
        out.append(f"\\X{marker}\\" + "".join(run) + "\\X0\\")
    return "".join(out)


def round_trip(text: str) -> str:
    """Convenience: ``decode(encode(text))`` (should equal ``text``)."""

    return decode(encode(text))
