"""Real-part geometry fixtures mined from resources/cad_repos, with provenance.

Every corpus in the harness before this package was procedurally generated
(hardcorpus factories, hand-written op streams, synthetic meshes). This package
imports the highest-value REAL fixtures found in the resources tree
(audit/inventory_fixtures_benchmarks.md), one loader module per source repo:

* :mod:`.brepnet_steps` -- BRepNet's curated known-good / known-bad STEP sets
  (CC BY-NC-SA 4.0: manifest + resources-path only, nothing vendored).
* :mod:`.cadgenbench_pose` -- cadgenbench rotation/translation twin STEPs +
  the ``open_shell.step`` invalid-solid oracle (Apache-2.0, vendored).
* :mod:`.manifold_meshes` -- manifold's crash-reproducer OBJ meshes and
  triangulation polygon corpus (Apache-2.0, small files vendored).
* :mod:`.cad_coder_heldout` -- CAD-Coder's 100-part GT-STEP + reference-code
  held-out set (Apache-2.0, manifest + resources-path only by design).
* :mod:`.birdhouse_nversion` -- curated-code-cad's birdhouse, the same part in
  8 languages: a natural cross-backend differential oracle (MIT, vendored).
* :mod:`.cadclaw_bom` -- CADCLAW's BOM sextet, 1 good + 5 labeled-wrong: a
  fleet-audit-style precision corpus for BOM verifiers (MIT, vendored).
* :mod:`.step_canaries` -- pythonocc-core and ruststep STEP/BREP parse canaries
  (LGPL / permissive: manifest-only, SHA-256 verified, nothing vendored).
* :mod:`.adversarial_code` -- an attack/benign/gap corpus for the pre-execution
  code-safety gate, whose ``--selfcheck`` RUNS ``check_cad_code`` over every
  case (spatialhero taxonomy, reimplemented: MIT-declared but no LICENSE file,
  nothing vendored verbatim).

Every loader is reachable through this hub: :data:`LOADERS` names them,
:func:`loader` returns one by name, and :func:`availability` is the per-source
manifest census. (``eval/bench/imports`` is the external-benchmark sibling of
this package and carries the same hub surface over the same dataclasses.)

Discipline shared by every loader:

* a ``MANIFEST.json`` in the data directory records, for EVERY file, the
  resources-relative source path, SHA-256, byte count and role -- whether the
  file is vendored or not;
* vendored copies are preferred, the resources tree is the fallback, and a
  missing file degrades to ``path=None`` rather than an exception (NO kernel,
  NO hard dependency on the resources checkout);
* each loader is runnable: ``python -m harnesscad.eval.corpus.fixtures.<mod>
  --selfcheck`` validates the manifest, counts and vendored hashes and exits 0.

Stdlib only. Deterministic. No geometry kernel anywhere in this package.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "FixtureEntry",
    "LOADERS",
    "Manifest",
    "availability",
    "fixtures_dir",
    "loader",
    "resources_root",
    "sha256_of",
    "load_manifest",
]

#: Loader module names, in the package docstring's order. Each module exposes
#: ``manifest()`` plus its own case accessors, and ``main(argv)`` with
#: ``--selfcheck``. Bound to real modules at the bottom of this file.
LOADERS: Tuple[str, ...] = (
    "brepnet_steps",
    "cadgenbench_pose",
    "manifold_meshes",
    "cad_coder_heldout",
    "birdhouse_nversion",
    "cadclaw_bom",
    "step_canaries",
    "adversarial_code",
)


def fixtures_dir() -> Path:
    """The directory this package (and its per-source data dirs) lives in."""
    return Path(__file__).resolve().parent


def resources_root() -> Optional[Path]:
    """The repo's ``resources/`` tree, or ``None`` when not checked out.

    Walks up from this file looking for a parent that contains
    ``resources/cad_repos`` (the repo root has ``src/`` and ``resources/`` as
    siblings). Installed wheels have no resources tree: that is the clean
    not-present degrade every loader must survive.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "resources" / "cad_repos"
        if candidate.is_dir():
            return parent / "resources"
    return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FixtureEntry:
    """One manifest row: where a fixture file comes from and how to find it."""

    name: str
    role: str
    vendored: Optional[str]      # path relative to the source's data dir
    resource: Optional[str]      # path relative to resources/
    sha256: str
    bytes: int
    format: str

    def resolve(self, data_dir: Path) -> Optional[Path]:
        """Prefer the vendored copy; fall back to resources/; else ``None``."""
        if self.vendored:
            p = data_dir / self.vendored
            if p.is_file():
                return p
        if self.resource:
            root = resources_root()
            if root is not None:
                p = root / self.resource
                if p.is_file():
                    return p
        return None


@dataclass(frozen=True)
class Manifest:
    source_repo: str
    source_path: str
    license: str
    attribution: str
    entries: Tuple[FixtureEntry, ...]
    data_dir: Path

    def by_role(self, role: str) -> List[FixtureEntry]:
        return [e for e in self.entries if e.role == role]

    def by_name(self, name: str) -> Optional[FixtureEntry]:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def resolve(self, entry: FixtureEntry) -> Optional[Path]:
        return entry.resolve(self.data_dir)

    def verify_vendored(self) -> List[str]:
        """SHA-check every vendored file; return a list of problems (empty=ok)."""
        problems: List[str] = []
        for e in self.entries:
            if not e.vendored:
                continue
            p = self.data_dir / e.vendored
            if not p.is_file():
                problems.append("missing vendored file: %s" % e.vendored)
            elif sha256_of(p) != e.sha256:
                problems.append("sha256 mismatch: %s" % e.vendored)
        return problems

    def availability(self) -> Dict[str, int]:
        """How many entries resolve to a real file, and how."""
        vendored = present = 0
        for e in self.entries:
            p = self.resolve(e)
            if p is None:
                continue
            present += 1
            if e.vendored and p == self.data_dir / e.vendored:
                vendored += 1
        return {"total": len(self.entries), "present": present,
                "vendored": vendored, "absent": len(self.entries) - present}


def load_manifest(source: str) -> Manifest:
    """Load ``fixtures/<source>/MANIFEST.json`` into a :class:`Manifest`."""
    data_dir = fixtures_dir() / source
    raw = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = tuple(
        FixtureEntry(
            name=e["name"],
            role=e["role"],
            vendored=e.get("vendored"),
            resource=e.get("resource"),
            sha256=e["sha256"],
            bytes=int(e["bytes"]),
            format=e.get("format", ""),
        )
        for e in raw["entries"]
    )
    return Manifest(
        source_repo=raw.get("source_repo", ""),
        source_path=raw.get("source_path", ""),
        license=raw.get("license", ""),
        attribution=raw.get("attribution", ""),
        entries=entries,
        data_dir=data_dir,
    )


def loader(name: str):
    """Return one fixture loader module by its :data:`LOADERS` name."""
    try:
        return _MODULES[name]
    except KeyError:
        raise KeyError(
            "no such fixture loader: %r (known: %s)"
            % (name, ", ".join(LOADERS))) from None


def availability() -> Dict[str, Dict[str, int]]:
    """Per-source manifest availability -- what actually resolves on this box.

    The honest census behind an empty case list: it tells "no resources
    checkout" apart from "loader broken". Never raises, so it is safe on a
    bare wheel where the manifest-mode sources resolve nothing.
    """
    out: Dict[str, Dict[str, int]] = {}
    for name in LOADERS:
        try:
            out[name] = loader(name).manifest().availability()
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            out[name] = {"total": 0, "present": 0, "vendored": 0, "absent": 0}
    return out


# --------------------------------------------------------------------------- #
# The loader routes.
#
# Bottom-of-module on purpose: every loader does
# ``from harnesscad.eval.corpus.fixtures import Manifest, load_manifest``, so
# the cycle only resolves once this module has defined them. Real ``import``
# statements rather than ``importlib`` lookups, so the capability index's AST
# scan sees reachable modules. No loader touches a kernel at import time and
# each resolves its data lazily, so binding them costs nothing and keeps the
# resources/ tree optional.
# --------------------------------------------------------------------------- #

from harnesscad.eval.corpus.fixtures import adversarial_code     # noqa: E402
from harnesscad.eval.corpus.fixtures import birdhouse_nversion   # noqa: E402
from harnesscad.eval.corpus.fixtures import brepnet_steps        # noqa: E402
from harnesscad.eval.corpus.fixtures import cad_coder_heldout    # noqa: E402
from harnesscad.eval.corpus.fixtures import cadclaw_bom          # noqa: E402
from harnesscad.eval.corpus.fixtures import cadgenbench_pose     # noqa: E402
from harnesscad.eval.corpus.fixtures import manifold_meshes      # noqa: E402
from harnesscad.eval.corpus.fixtures import step_canaries        # noqa: E402

#: name -> loader module, for :func:`loader`. Keys are exactly :data:`LOADERS`.
_MODULES: Dict[str, object] = {
    "brepnet_steps": brepnet_steps,
    "cadgenbench_pose": cadgenbench_pose,
    "manifold_meshes": manifold_meshes,
    "cad_coder_heldout": cad_coder_heldout,
    "birdhouse_nversion": birdhouse_nversion,
    "cadclaw_bom": cadclaw_bom,
    "step_canaries": step_canaries,
    "adversarial_code": adversarial_code,
}

assert tuple(_MODULES) == LOADERS, "LOADERS and the route table disagree"
