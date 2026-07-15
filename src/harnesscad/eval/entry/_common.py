"""Shared plumbing for the eval entry-point wrappers.

One JSON document in (``--input FILE`` or ``-`` for stdin), one JSON document out.
Deterministic: no wall clock, no environment, no network. ``--selfcheck`` lets a
wrapper prove it is wired by running on a fixed in-code fixture, which is how CI
reaches these without a data file or a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Optional, Sequence


def read_json(path: Optional[str]) -> Any:
    """Load a JSON document from ``path`` (or stdin when ``path`` is ``-``/None)."""
    if path in (None, "-"):
        text = sys.stdin.read()
    else:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    if not text.strip():
        raise ValueError("no JSON on the input; pass --input FILE or pipe a "
                         "document on stdin, or use --selfcheck")
    return json.loads(text)


def _default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError("not JSON serialisable: %r" % (type(obj).__name__,))


def emit(obj: Any) -> None:
    """Write one JSON document to stdout, deterministically."""
    json.dump(obj, sys.stdout, indent=2, sort_keys=True, default=_default)
    sys.stdout.write("\n")


def build_main(
    prog: str,
    description: str,
    compute: Callable[[Any], Any],
    selfcheck_input: Any,
    extra: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> Callable[[Optional[Sequence[str]]], int]:
    """Assemble a uniform ``main`` for a wrapper.

    ``compute`` maps a parsed-JSON document to a JSON-able result. ``selfcheck_input``
    is the fixture ``--selfcheck`` feeds to ``compute``. The returned ``main`` reads
    input, computes, emits, and returns a POSIX exit code.
    """

    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--input", "-i", default="-",
                            help="JSON document to read (default: stdin)")
        parser.add_argument("--selfcheck", action="store_true",
                            help="ignore --input and run on a fixed in-code fixture; "
                                 "proves the wiring with no data file and no model")
        if extra is not None:
            extra(parser)

    def run(args: argparse.Namespace) -> int:
        doc = selfcheck_input if getattr(args, "selfcheck", False) \
            else read_json(getattr(args, "input", "-"))
        emit(compute(doc))
        return 0

    def main(argv: Optional[Sequence[str]] = None) -> int:
        parser = argparse.ArgumentParser(prog=prog, description=description)
        add_arguments(parser)
        return run(parser.parse_args(list(argv) if argv is not None else None))

    main.add_arguments = add_arguments   # type: ignore[attr-defined]
    main.run = run                       # type: ignore[attr-defined]
    return main
