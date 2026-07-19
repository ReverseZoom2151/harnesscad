"""Deterministic source-text editing operations for a CAD script editor.

A stdlib-only, toolkit-free implementation of the text transformations a code
editor widget performs: block comment toggling, block indent/unindent, and
end-of-line detection.  Rather than driving a rich-text document through cursor
operations, these algorithms work on plain Python line lists and byte strings so
they are deterministic and testable without a GUI.

All functions are pure: they take input and return new values without mutating
their arguments and without touching any global state.
"""

INDENT = "    "
COMMENT = "# "


def detect_eol(raw):
    """Detect the end-of-line convention used in a byte string.

    Mirrors the usual file-load EOL sniffing: a Windows ``\\r\\n`` wins over a
    lone ``\\r`` (classic Mac), which wins over the Unix ``\\n`` default.

    :param raw: file contents as ``bytes``.
    :returns: one of ``"\\r\\n"``, ``"\\r"`` or ``"\\n"``.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise TypeError("raw must be bytes")
    if b"\r\n" in raw:
        return "\r\n"
    if b"\r" in raw:
        return "\r"
    return "\n"


def _leading_ws(line):
    """Return the number of leading whitespace characters in *line*."""
    return len(line) - len(line.lstrip())


def indent_lines(lines):
    """Indent every line by one indent unit (four spaces).

    Blank lines (empty string) are left untouched, matching the editor
    behaviour of not indenting empty rows.

    :param lines: iterable of line strings (without EOL terminators).
    :returns: a new list of indented lines.
    """
    out = []
    for line in lines:
        if line == "":
            out.append(line)
        else:
            out.append(INDENT + line)
    return out


def unindent_lines(lines):
    """Remove one leading indent unit from every line that has one.

    Only a full four-space prefix is stripped; a line indented with fewer
    spaces (or a tab) is left unchanged: only an exact prefix match is stripped.

    :param lines: iterable of line strings.
    :returns: a new list of lines.
    """
    out = []
    for line in lines:
        if line.startswith(INDENT):
            out.append(line[len(INDENT):])
        else:
            out.append(line)
    return out


def _is_commented(line):
    stripped = line.strip()
    return bool(stripped) and stripped[0] == "#"


def toggle_comment_block(lines):
    """Toggle line comments across a block of lines.

    The toggle semantics are:

    * Blank lines are ignored entirely.
    * If the block mixes commented and uncommented (non-blank) lines, every
      non-blank line is commented by inserting ``"# "`` at column zero.
    * Otherwise (all non-blank lines share the same comment state) the block is
      toggled at the shared left-most indentation column: an existing ``"#"``
      (and one following space, if present) is removed, otherwise ``"# "`` is
      inserted there.
    * A block whose lines are all blank is returned unchanged.

    :param lines: iterable of line strings.
    :returns: a new list of lines with comments toggled.
    """
    lines = list(lines)
    nonblank = [ln for ln in lines if ln.strip() != ""]
    if not nonblank:
        return list(lines)

    leftmost = min(_leading_ws(ln) for ln in nonblank)
    has_commented = any(_is_commented(ln) for ln in nonblank)
    has_uncommented = any(not _is_commented(ln) for ln in nonblank)
    mixed = has_commented and has_uncommented

    out = []
    for line in lines:
        if line.strip() == "":
            out.append(line)
            continue
        if mixed:
            out.append(COMMENT + line)
            continue
        # uniform block: toggle at the shared left-most column
        if line[leftmost:leftmost + 1] == "#":
            rest = line[leftmost + 1:]
            if rest[:1] == " ":
                rest = rest[1:]
            out.append(line[:leftmost] + rest)
        else:
            out.append(line[:leftmost] + COMMENT + line[leftmost:])
    return out


def line_number_gutter_digits(line_count):
    """Return the digit width needed to render line numbers for *line_count*.

    Deterministic gutter-width digit count, done with pure integer arithmetic
    rather than a float-multiply loop.  At least one column is always reserved.

    :param line_count: number of lines in the document.
    :returns: number of decimal digits in ``max(1, line_count)``.
    """
    n = max(1, int(line_count))
    digits = 1
    while n >= 10:
        n //= 10
        digits += 1
    return digits
