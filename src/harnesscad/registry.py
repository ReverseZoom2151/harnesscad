"""Capability registry: a static, lazy, deterministic index of every HarnessCAD module.

The repo carries ~1,100 product modules mined from papers and CAD repos. Most are
correct and tested but nothing imports them -- they are unreachable islands. This
module makes all of them *discoverable* and *dispatchable* without importing any of
them eagerly (some depend on OCCT and would blow up at import time).

How it works
------------
*   The index is built by walking ``src/harnesscad/**/*.py`` with :mod:`ast`. No
    module is ever executed to be indexed, so a broken/optional-dep module can
    still be catalogued.
*   For each module we record: dotted path, layer, package, name, the first line
    of the module docstring, its public top-level ``def``/``class`` names, and the
    set of intra-product modules it imports (used for the orphan computation).
*   Capability tags are derived deterministically from the package name, the
    module name and docstring keywords via :data:`KEYWORD_TAGS` below.
*   :func:`build_index` persists the whole thing to ``_capability_index.json``
    with sorted keys and no timestamps, so two builds are byte-identical.

Typical use::

    from harnesscad import registry

    for e in registry.find(tag="sdf"):
        print(e.dotted, e.summary)

    mod = registry.load("harnesscad.domain.geometry.sdf.primitives")
    mod.sphere_sdf(...)

Everything here is stdlib-only and deterministic (no wall clock, no randomness).
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ModuleEntry",
    "KEYWORD_TAGS",
    "PACKAGE_TAGS",
    "INDEX_PATH",
    "build_index",
    "index",
    "find",
    "search",
    "get",
    "load",
    "symbols",
    "stats",
    "orphans",
    "import_graph",
    "main",
]

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(ROOT, "_capability_index.json")
PACKAGE_ROOT = "harnesscad"
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Capability tagging tables.
#
# Tags are derived ONLY from evidence in the module's own name, its package, or
# its docstring. Nothing is inferred beyond these tables -- if a module is not
# named/documented for a capability, it does not get the tag.
# ---------------------------------------------------------------------------

# package (the directory under a layer) -> tags every module in it inherits.
PACKAGE_TAGS: Dict[str, Tuple[str, ...]] = {
    "geometry": ("geometry",),
    "numeric": ("numeric",),
    "procedural": ("procedural",),
    "reconstruction": ("reconstruction",),
    "drawings": ("drawings",),
    "editing": ("editing",),
    "fabrication": ("fabrication",),
    "library": ("library",),
    "sizing": ("sizing",),
    "skeleton": ("skeleton",),
    "spec": ("spec",),
    "standards": ("standards",),
    "vision": ("vision",),
    "formats": ("format", "io"),
    "ingestion": ("io",),
    "kernels": ("kernel",),
    "adapters": ("adapter",),
    "surfaces": ("surface",),
    "benchmarks": ("benchmark",),
    "quality": ("quality",),
    "verifiers": ("verify",),
    "reliability": ("reliability",),
    "loop": ("agent",),
    "llm": ("llm",),
    "generation": ("generation",),
    "rag": ("rag",),
    "memory": ("memory",),
    "protocols": ("protocol",),
    "engine": ("dataset",),
    "generators": ("dataset",),
    "security": ("security",),
    "research": ("research",),
    "closure": ("audit",),
}

# substring (matched against the module NAME, lowercased, and against the
# docstring words) -> tags. Ordered as a tuple-of-pairs so iteration is stable.
KEYWORD_TAGS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("sdf", ("sdf",)),
    ("signed_distance", ("sdf",)),
    ("implicit", ("sdf", "implicit")),
    ("marching_cubes", ("meshing", "isosurface")),
    ("marching_squares", ("meshing", "isosurface")),
    ("surface_nets", ("meshing", "isosurface")),
    ("dual_contour", ("meshing", "isosurface")),
    ("isosurface", ("meshing", "isosurface")),
    ("mesh", ("meshing",)),
    ("tessellat", ("meshing",)),
    ("triangulat", ("meshing",)),
    ("remesh", ("meshing",)),
    ("voxel", ("voxel",)),
    ("occupancy", ("voxel",)),
    ("bvh", ("acceleration",)),
    ("kdtree", ("acceleration",)),
    ("octree", ("acceleration",)),
    ("bounding", ("acceleration",)),
    ("stl", ("format", "io")),
    ("glb", ("format", "io")),
    ("gltf", ("format", "io")),
    ("obj_", ("format", "io")),
    ("step", ("format", "io")),
    ("iges", ("format", "io")),
    ("dxf", ("format", "io")),
    ("dwg", ("format", "io")),
    ("amf", ("format", "io")),
    ("3mf", ("format", "io")),
    ("svg", ("format", "io")),
    ("ply", ("format", "io")),
    ("off_", ("format", "io")),
    ("codec", ("format", "io")),
    ("parser", ("parsing",)),
    ("lexer", ("parsing",)),
    ("grammar", ("parsing",)),
    ("tokeniz", ("parsing",)),
    ("express", ("spec", "parsing")),
    ("schema", ("spec",)),
    ("bench", ("benchmark",)),
    ("metric", ("benchmark",)),
    ("score", ("benchmark",)),
    ("eval", ("benchmark",)),
    ("urdf", ("kinematics",)),
    ("srdf", ("kinematics",)),
    ("joint", ("kinematics",)),
    ("kinematic", ("kinematics",)),
    ("linkage", ("kinematics",)),
    ("ik_", ("kinematics",)),
    ("thread", ("mechanical",)),
    ("gear", ("mechanical",)),
    ("fastener", ("mechanical",)),
    ("bolt", ("mechanical",)),
    ("screw", ("mechanical",)),
    ("bearing", ("mechanical",)),
    ("spring", ("mechanical",)),
    ("csg", ("csg",)),
    ("boolean", ("csg",)),
    ("brep", ("brep",)),
    ("nurbs", ("curves",)),
    ("bspline", ("curves",)),
    ("bezier", ("curves",)),
    ("spline", ("curves",)),
    ("curve", ("curves",)),
    ("arc", ("curves",)),
    ("fillet", ("features",)),
    ("chamfer", ("features",)),
    ("extrude", ("features",)),
    ("revolve", ("features",)),
    ("loft", ("features",)),
    ("sweep", ("features",)),
    ("shell", ("features",)),
    ("draft", ("features",)),
    ("pattern", ("features",)),
    ("constraint", ("constraints",)),
    ("solver", ("solver",)),
    ("sketch", ("sketch",)),
    ("gcode", ("cam",)),
    ("toolpath", ("cam",)),
    ("cam_", ("cam",)),
    ("mill", ("cam",)),
    ("lathe", ("cam",)),
    ("slice", ("fabrication",)),
    ("print", ("fabrication",)),
    ("laser", ("fabrication",)),
    ("tolerance", ("tolerancing",)),
    ("gdt", ("tolerancing",)),
    ("dimension", ("drawings",)),
    ("drawing", ("drawings",)),
    ("projection", ("drawings",)),
    ("render", ("render",)),
    ("raster", ("render",)),
    ("shader", ("render",)),
    ("raytrac", ("render",)),
    ("material", ("material",)),
    ("texture", ("material",)),
    ("assembl", ("assembly",)),
    ("mate", ("assembly",)),
    ("simulat", ("simulation",)),
    ("fem", ("simulation",)),
    ("stress", ("simulation",)),
    ("physics", ("simulation",)),
    ("optimi", ("optimization",)),
    ("gradient", ("optimization",)),
    ("prompt", ("llm",)),
    ("llm", ("llm",)),
    ("agent", ("agent",)),
    ("retriev", ("rag",)),
    ("embedding", ("rag",)),
    ("dataset", ("dataset",)),
    ("synthes", ("dataset",)),
    ("verif", ("verify",)),
    ("validat", ("verify",)),
    ("check", ("verify",)),
    ("repair", ("repair",)),
    ("heal", ("repair",)),
    ("cache", ("infra",)),
    ("hash", ("infra",)),
    ("digest", ("infra",)),
    ("graph", ("graph",)),
    ("topolog", ("topology",)),
    ("point_cloud", ("pointcloud",)),
    ("pointcloud", ("pointcloud",)),
    ("scan", ("pointcloud",)),
    ("image", ("vision",)),
    ("sketch2", ("vision",)),
    ("param", ("parametric",)),
    ("feature_tree", ("parametric",)),
    ("history", ("parametric",)),
)


@dataclass(frozen=True)
class ModuleEntry:
    """One indexed product module. Purely static data -- nothing is imported."""

    dotted: str
    layer: str
    package: str
    name: str
    summary: str
    tags: Tuple[str, ...]
    symbols: Tuple[str, ...]
    imports: Tuple[str, ...]  # intra-product dotted paths this module imports

    def to_json(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["symbols"] = list(self.symbols)
        d["imports"] = list(self.imports)
        return d

    @staticmethod
    def from_json(d: dict) -> "ModuleEntry":
        return ModuleEntry(
            dotted=d["dotted"],
            layer=d["layer"],
            package=d["package"],
            name=d["name"],
            summary=d["summary"],
            tags=tuple(d["tags"]),
            symbols=tuple(d["symbols"]),
            imports=tuple(d["imports"]),
        )


# ---------------------------------------------------------------------------
# AST scanning
# ---------------------------------------------------------------------------

def _iter_source_files() -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            if fn == "__init__.py":
                continue
            out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _dotted_for(path: str) -> str:
    rel = os.path.relpath(path, ROOT)
    parts = rel.replace("\\", "/").split("/")
    parts[-1] = parts[-1][:-3]  # strip .py
    return ".".join([PACKAGE_ROOT] + parts)


def _summary_of(tree: ast.Module) -> str:
    doc = ast.get_docstring(tree) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _public_symbols(tree: ast.Module) -> Tuple[str, ...]:
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
    return tuple(sorted(set(names)))


def _product_imports(tree: ast.Module, dotted: str) -> Tuple[str, ...]:
    """Dotted paths of harnesscad modules this module imports (module-granular)."""
    found = set()
    pkg = dotted.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE_ROOT + "."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import -> resolve against this package
                base = pkg
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0] if "." in base else base
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            if not mod.startswith(PACKAGE_ROOT):
                continue
            found.add(mod)
            # `from pkg.sub import mod_name` -- the name may itself be a module.
            for alias in node.names:
                found.add(f"{mod}.{alias.name}")
    found.discard(dotted)
    return tuple(sorted(found))


def _tags_for(package: str, name: str, summary: str, subpath: str = "") -> Tuple[str, ...]:
    """Derive tags from the package, the module's PATH, and its docstring.

    ``subpath`` is the sub-package chain below the package (e.g. ``sdf`` for
    ``domain.geometry.sdf.primitives``). It must be part of the haystack: since
    modules are named by capability rather than provenance, the capability now
    lives in the FOLDER. ``sdf/primitives.py`` is an SDF module even though the
    word "sdf" no longer appears in its name.
    """
    tags = set(PACKAGE_TAGS.get(package, ()))
    haystack_name = f"{subpath}.{name}".lower() if subpath else name.lower()
    haystack_doc = summary.lower()
    for needle, add in KEYWORD_TAGS:
        if needle in haystack_name or needle in haystack_doc:
            tags.update(add)
    return tuple(sorted(tags))


def _scan(path: str) -> Optional[ModuleEntry]:
    dotted = _dotted_for(path)
    parts = dotted.split(".")
    if len(parts) < 3:
        # top-level modules like harnesscad.registry: layer/package unknown.
        layer, package = "", ""
    else:
        layer = parts[1]
        package = parts[2] if len(parts) >= 4 else ""
    name = parts[-1]
    # sub-package chain between the package and the module, e.g. "sdf" in
    # harnesscad.domain.geometry.sdf.primitives -- this is where the capability
    # now lives, since modules are named by capability rather than provenance.
    subpath = ".".join(parts[3:-1]) if len(parts) >= 5 else ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src, filename=path)
    except (OSError, SyntaxError, ValueError):
        return None
    summary = _summary_of(tree)
    return ModuleEntry(
        dotted=dotted,
        layer=layer,
        package=package,
        name=name,
        summary=summary,
        tags=_tags_for(package, name, summary, subpath),
        symbols=_public_symbols(tree),
        imports=_product_imports(tree, dotted),
    )


def scan_source_tree() -> List[ModuleEntry]:
    """Build the entry list from AST alone. Deterministic (sorted by dotted path)."""
    entries = [e for e in (_scan(p) for p in _iter_source_files()) if e is not None]
    entries.sort(key=lambda e: e.dotted)
    return entries


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def build_index(path: str = INDEX_PATH) -> List[ModuleEntry]:
    """Rescan the tree and write the deterministic JSON index. Returns the entries."""
    entries = scan_source_tree()
    payload = {
        "schema": SCHEMA_VERSION,
        "root": PACKAGE_ROOT,
        "modules": [e.to_json() for e in entries],
    }
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    global _CACHE
    _CACHE = entries
    return entries


def load_index(path: str = INDEX_PATH) -> Optional[List[ModuleEntry]]:
    """Read the persisted index, or None if it is absent/unreadable/stale-schema."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != SCHEMA_VERSION:
        return None
    try:
        return [ModuleEntry.from_json(d) for d in payload["modules"]]
    except (KeyError, TypeError):
        return None


_CACHE: Optional[List[ModuleEntry]] = None


def index(refresh: bool = False) -> List[ModuleEntry]:
    """All indexed modules (cached). Loads the JSON if present, else rescans."""
    global _CACHE
    if refresh:
        _CACHE = scan_source_tree()
        return _CACHE
    if _CACHE is None:
        _CACHE = load_index() or scan_source_tree()
    return _CACHE


# ---------------------------------------------------------------------------
# Query API
# ---------------------------------------------------------------------------

def _by_dotted() -> Dict[str, ModuleEntry]:
    return {e.dotted: e for e in index()}


def get(dotted: str) -> ModuleEntry:
    """The entry for one dotted path. Raises KeyError if unknown."""
    try:
        return _by_dotted()[dotted]
    except KeyError:
        raise KeyError(f"no such module in the capability index: {dotted!r}") from None


def find(
    tag: Optional[str] = None,
    layer: Optional[str] = None,
    package: Optional[str] = None,
    name: Optional[str] = None,
) -> List[ModuleEntry]:
    """Entries matching every supplied filter. `name` is a substring match."""
    out = []
    for e in index():
        if tag is not None and tag not in e.tags:
            continue
        if layer is not None and e.layer != layer:
            continue
        if package is not None and e.package != package:
            continue
        if name is not None and name.lower() not in e.name.lower():
            continue
        out.append(e)
    return out


def search(text: str) -> List[ModuleEntry]:
    """Substring match (case-insensitive) over dotted path, summary and symbols."""
    q = text.lower()
    out = []
    for e in index():
        if q in e.dotted.lower() or q in e.summary.lower():
            out.append(e)
            continue
        if any(q in s.lower() for s in e.symbols):
            out.append(e)
    return out


def load(dotted: str) -> ModuleType:
    """Lazily import and return the real module for `dotted` (must be indexed)."""
    get(dotted)  # validate against the index first
    return importlib.import_module(dotted)


def symbols(dotted: str) -> Tuple[str, ...]:
    """Public top-level callables/classes of a module (from the AST, no import)."""
    return get(dotted).symbols


def import_graph() -> Dict[str, List[str]]:
    """dotted -> the indexed modules it imports (edges to unknown targets dropped)."""
    known = set(_by_dotted())
    graph: Dict[str, List[str]] = {}
    for e in index():
        graph[e.dotted] = sorted(t for t in e.imports if t in known and t != e.dotted)
    return graph


def orphans() -> List[str]:
    """Indexed modules that no other indexed module imports -- the unreachable islands."""
    graph = import_graph()
    imported = set()
    for targets in graph.values():
        imported.update(targets)
    return sorted(d for d in graph if d not in imported)


def stats() -> dict:
    """Counts by layer / package / tag, plus the orphan list. Deterministic."""
    entries = index()
    by_layer: Dict[str, int] = {}
    by_package: Dict[str, int] = {}
    by_tag: Dict[str, int] = {}
    for e in entries:
        by_layer[e.layer] = by_layer.get(e.layer, 0) + 1
        key = f"{e.layer}/{e.package}" if e.package else (e.layer or "(root)")
        by_package[key] = by_package.get(key, 0) + 1
        for t in e.tags:
            by_tag[t] = by_tag.get(t, 0) + 1
    orph = orphans()
    return {
        "total_modules": len(entries),
        "total_symbols": sum(len(e.symbols) for e in entries),
        "by_layer": dict(sorted(by_layer.items())),
        "by_package": dict(sorted(by_package.items())),
        "by_tag": dict(sorted(by_tag.items(), key=lambda kv: (-kv[1], kv[0]))),
        "orphan_count": len(orph),
        "orphans": orph,
    }


# ---------------------------------------------------------------------------
# CLI (also reachable as `harnesscad capabilities ...` via core.cli)
# ---------------------------------------------------------------------------

def _fmt(e: ModuleEntry) -> str:
    tags = ",".join(e.tags) or "-"
    summary = e.summary or "(no docstring)"
    return f"{e.dotted}\n    tags: {tags}\n    {summary}"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Wire the `capabilities` flags onto an existing parser (used by core.cli)."""
    parser.add_argument("--list", action="store_true", help="list modules")
    parser.add_argument("--tag", default=None, help="filter by capability tag")
    parser.add_argument("--layer", default=None, help="filter by layer")
    parser.add_argument("--package", default=None, help="filter by package")
    parser.add_argument("--search", default=None, help="substring search name/doc/symbols")
    parser.add_argument("--show", default=None, help="show one module by dotted path")
    parser.add_argument("--stats", action="store_true", help="print index statistics")
    parser.add_argument("--orphans", action="store_true", help="list unimported modules")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the JSON index")
    parser.add_argument("--limit", type=int, default=0, help="cap listed results (0 = all)")


def run(args: argparse.Namespace) -> int:
    """Execute a parsed `capabilities` invocation. Returns a process exit code."""
    if getattr(args, "rebuild", False):
        entries = build_index()
        print(f"wrote {INDEX_PATH} ({len(entries)} modules)")
        return 0

    if getattr(args, "show", None):
        try:
            e = get(args.show)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"module:  {e.dotted}")
        print(f"layer:   {e.layer or '-'}")
        print(f"package: {e.package or '-'}")
        print(f"tags:    {','.join(e.tags) or '-'}")
        print(f"summary: {e.summary or '(no docstring)'}")
        print("symbols:")
        for s in e.symbols:
            print(f"  {s}")
        if not e.symbols:
            print("  (none)")
        return 0

    if getattr(args, "stats", False):
        st = stats()
        print(f"total modules:  {st['total_modules']}")
        print(f"total symbols:  {st['total_symbols']}")
        print(f"orphan modules: {st['orphan_count']}")
        print("by layer:")
        for k, v in st["by_layer"].items():
            print(f"  {k or '(root)':<12} {v}")
        print("by tag:")
        for k, v in st["by_tag"].items():
            print(f"  {k:<16} {v}")
        return 0

    if getattr(args, "orphans", False):
        for d in orphans():
            print(d)
        return 0

    if getattr(args, "search", None):
        results = search(args.search)
    else:
        results = find(tag=args.tag, layer=args.layer, package=args.package)

    limit = getattr(args, "limit", 0) or 0
    shown = results[:limit] if limit > 0 else results
    for e in shown:
        print(_fmt(e))
    print(f"-- {len(shown)} shown / {len(results)} matched / {len(index())} indexed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harnesscad capabilities",
        description="discover and dispatch HarnessCAD capability modules",
    )
    add_arguments(parser)
    return run(parser.parse_args(list(argv) if argv is not None else None))


if __name__ == "__main__":
    raise SystemExit(main())
