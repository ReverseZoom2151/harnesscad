"""persistence — atomic, deterministic JSON writes for the memory subsystem.

Session continuity means a store written at the end of one run is byte-identical
to what the next run loads, and a crash mid-write never leaves a truncated file
that poisons the next session. Both stores (``MemoryStore``, ``SkillLibrary``,
``ErrorNotebook``, ``HarnessMemory``) share the same tiny discipline here:

  * write to a temporary file in the SAME directory as the target, then
    ``os.replace`` it into place -- an atomic rename on POSIX and on Windows, so
    a reader never observes a half-written file;
  * flush and ``fsync`` before the rename so the bytes are on disk first;
  * serialise with ``sort_keys=True`` and a fixed indent so the output is a pure
    function of the data -- no dict-ordering drift between runs, no wall clock.

Stdlib only. No global state.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

__all__ = ["atomic_write_text", "dump_json", "load_json"]


def atomic_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (temp file + fsync + os.replace)."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".mem-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Best-effort cleanup; never mask the original error.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def dump_json(obj: Any, path: str) -> None:
    """Serialise ``obj`` to ``path`` deterministically and atomically."""
    text = json.dumps(obj, indent=2, sort_keys=True)
    atomic_write_text(path, text)


def load_json(path: str) -> Any:
    """Read and parse a JSON file written by :func:`dump_json`."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
