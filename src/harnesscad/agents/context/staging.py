"""Context Staging Area — transparent, per-task context (HARNESS_BLUEPRINT sec.7).

The anti-RAG. Instead of a black-box retriever deciding what the model sees, a
per-task `task-context/` directory holds the context as plain files under version
control, and a `context.toml` manifest declares *exactly* what gets rendered into
the window each turn:

    task-context/
      context.toml      <- the manifest: what to include, and in what order
      01_BRIEF.md       <- intent + constraints (the spec)
      02_MODEL/         <- the current feature tree as text
      03_DOCS/          <- specs / standards / DFM rules

`StagingArea(root)` builds that skeleton, writes/reads files by relative path, and
`render_for_turn(manifest)` composes the selected files into one deterministic
context string. Explicit control, fully auditable — you can diff exactly what the
model saw on any turn.

TOML handling is stdlib-only: reading prefers `tomllib` (Python 3.11+) and falls
back to a minimal in-module parser covering the manifest subset we emit (tables,
string values, and arrays of strings). Writing uses a tiny in-module emitter
(the stdlib has no TOML writer) whose output both `tomllib` and the fallback
parser accept.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:  # Python 3.11+ ships a read-only TOML parser in the stdlib.
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    _tomllib = None


MANIFEST_NAME = "context.toml"
BRIEF = "01_BRIEF.md"
MODEL_DIR = "02_MODEL"
DOCS_DIR = "03_DOCS"


# Default manifest: brief + model tree + (no docs yet). Section `[manifest]`
# with keys `brief` (a path), `model` (a path), and `docs` (a list of paths).
_DEFAULT_MANIFEST: Dict[str, Any] = {
    "manifest": {
        "brief": BRIEF,
        "model": f"{MODEL_DIR}/tree.txt",
        "docs": [],
    }
}


class StagingArea:
    """A per-task `task-context/` staging area rooted at `root`.

    `root` is the parent directory; the staging area lives in
    `root/task-context/`. All read/write paths are relative to that folder.
    """

    def __init__(self, root: str, dirname: str = "task-context") -> None:
        self.root = os.path.abspath(root)
        self.dir = os.path.join(self.root, dirname)

    # --- filesystem paths -------------------------------------------------
    def path(self, relpath: str) -> str:
        """Absolute path for a `task-context/`-relative path (kept inside dir)."""
        p = os.path.normpath(os.path.join(self.dir, relpath))
        if os.path.commonpath([p, self.dir]) != self.dir:
            raise ValueError(f"path escapes the staging area: {relpath!r}")
        return p

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.dir, MANIFEST_NAME)

    # --- build / write / read --------------------------------------------
    def build(
        self,
        brief: str = "",
        model_tree: str = "",
        manifest: Optional[Dict[str, Any]] = None,
    ) -> "StagingArea":
        """Create the `task-context/` skeleton + a `context.toml` manifest.

        Lays down `01_BRIEF.md`, `02_MODEL/tree.txt`, an empty `03_DOCS/`, and the
        manifest. Idempotent for the directories; overwrites the seeded files with
        whatever `brief`/`model_tree` you pass (empty by default).
        """
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.path(MODEL_DIR), exist_ok=True)
        os.makedirs(self.path(DOCS_DIR), exist_ok=True)
        self.write(BRIEF, brief)
        self.write(f"{MODEL_DIR}/tree.txt", model_tree)
        self.write_manifest(manifest if manifest is not None else _DEFAULT_MANIFEST)
        return self

    def write(self, relpath: str, content: str) -> str:
        """Write `content` to a `task-context/`-relative path; returns abs path."""
        p = self.path(relpath)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return p

    def read(self, relpath: str) -> str:
        """Read a `task-context/`-relative file as text."""
        with open(self.path(relpath), "r", encoding="utf-8") as f:
            return f.read()

    def exists(self, relpath: str) -> bool:
        return os.path.exists(self.path(relpath))

    # --- manifest ---------------------------------------------------------
    def write_manifest(self, manifest: Dict[str, Any]) -> str:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(dumps_toml(manifest))
        return self.manifest_path

    def read_manifest(self) -> Dict[str, Any]:
        """Parse `context.toml` -> dict (prefers stdlib `tomllib`)."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            text = f.read()
        return loads_toml(text)

    # --- render -----------------------------------------------------------
    def render_for_turn(self, manifest: Optional[Dict[str, Any]] = None) -> str:
        """Compose the manifest-selected files into one deterministic string.

        Order is fixed and explicit: BRIEF, then MODEL, then each DOC in the
        manifest's `docs` list (in listed order). Each section gets a header so
        the model (and a human auditor) can see exactly what was staged. Missing
        files are skipped with an explicit `(missing)` marker rather than raising,
        so a partially-built task still renders.

        Pass `manifest` to render a specific selection; omit it to load
        `context.toml` from disk.
        """
        m = manifest if manifest is not None else self.read_manifest()
        sel = m.get("manifest", m)  # tolerate a bare dict or a wrapped table
        blocks: List[str] = []

        brief_path = sel.get("brief")
        if brief_path:
            blocks.append(self._section("BRIEF", brief_path))

        model_path = sel.get("model")
        if model_path:
            blocks.append(self._section("MODEL", model_path))

        for doc in sel.get("docs", []) or []:
            blocks.append(self._section(f"DOC: {doc}", doc))

        return "\n\n".join(blocks)

    def _section(self, title: str, relpath: str) -> str:
        header = f"# {title}"
        try:
            body = self.read(relpath)
        except FileNotFoundError:
            body = "(missing)"
        return f"{header}\n{body}".rstrip()


# --- minimal TOML I/O (stdlib-only) ----------------------------------------
def loads_toml(text: str) -> Dict[str, Any]:
    """Parse TOML text. Prefers stdlib `tomllib`; falls back to `_mini_loads`."""
    if _tomllib is not None:
        return _tomllib.loads(text)
    return _mini_loads(text)  # pragma: no cover - only on <3.11


def dumps_toml(data: Dict[str, Any]) -> str:
    """Emit the manifest subset (top-level tables of scalars / string arrays).

    Output is accepted by both `tomllib` and `_mini_loads`. Supports string,
    int, float, bool values and arrays of those; that covers every manifest we
    write and keeps the round-trip closed.
    """
    lines: List[str] = []
    # top-level scalars first (rare, but valid), then tables.
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    for k, v in scalars.items():
        lines.append(f"{k} = {_fmt_toml_value(v)}")
    for name, table in tables.items():
        if lines:
            lines.append("")
        lines.append(f"[{name}]")
        for k, v in table.items():
            lines.append(f"{k} = {_fmt_toml_value(v)}")
    return "\n".join(lines) + "\n"


def _fmt_toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return _fmt_toml_string(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value: {v!r}")


def _fmt_toml_string(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _mini_loads(text: str) -> Dict[str, Any]:  # pragma: no cover - <3.11 only
    """A minimal TOML parser: `[table]` sections, `key = scalar`, and arrays of
    scalars. Enough for the manifest subset we emit; used only when `tomllib` is
    unavailable (Python < 3.11).
    """
    root: Dict[str, Any] = {}
    cur = root
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            cur = root.setdefault(name, {})
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        cur[key.strip()] = _mini_parse_value(val.strip())
    return root


def _mini_parse_value(v: str) -> Any:  # pragma: no cover - <3.11 only
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_mini_parse_value(x.strip()) for x in _split_top_commas(inner)]
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1].replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
    if v == "true":
        return True
    if v == "false":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _split_top_commas(s: str) -> List[str]:  # pragma: no cover - <3.11 only
    out: List[str] = []
    depth = 0
    in_str = False
    buf: List[str] = []
    for ch in s:
        if ch == '"':
            in_str = not in_str
        if ch == "[" and not in_str:
            depth += 1
        elif ch == "]" and not in_str:
            depth -= 1
        if ch == "," and depth == 0 and not in_str:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out
