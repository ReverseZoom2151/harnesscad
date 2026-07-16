"""External text-to-CAD benchmarks imported into harness shapes, with provenance.

The harness's corpora are procedurally generated; this package imports the
highest-value EXTERNAL benchmark/task sets found in the resources tree
(audit/inventory_fixtures_benchmarks.md), one loader module per source, each
mapped onto shapes the harness already grades with -- the corpus Brief fields
(``eval/corpus/spec.py``), the Measured Geometric Contract
(``eval/hardcorpus/contract_grader.py``) and the judge checklist surface
(``eval/judge/``):

* :mod:`.graphcad_cadbench` -- Graph-CAD's ``CADBench.jsonl``, 700
  rubric-annotated tasks (NO LICENSE in the repo: manifest + resources-path
  only, nothing vendored).
* :mod:`.agentscad_tasks` -- AgentSCAD's 14 machine-checkable tasks with
  expected_bbox / required_features / tolerances (MIT, vendored), emitted as
  brief records plus contract-style oracle predicates.
* :mod:`.zoo_kcl_manifest` -- Zoo modeling-app's ``kcl-samples/manifest.json``,
  100 human-described real parts (MIT; the small manifest is vendored, the KCL
  sources are NEVER vendored -- referenced by resources path only).
* :mod:`.cadam_textcad_briefs` -- 23 graded design briefs with reference
  OpenSCAD solutions: text-to-cad's 10 (MIT, briefs vendored) and CADAM's 13
  (GPL-3.0: manifest + resources-path only, nothing vendored). The ``.scad``
  references are ALWAYS by path, never vendored.
* :mod:`.intentforge_refusals` -- IntentForge's prompt -> expected-REJECTION
  oracle pairs (Apache-2.0, vendored): refusal / underspecification canaries
  complementing ``eval/hardcorpus/ambiguous.py``.
* :mod:`.cadjudge_prompts` -- cad-judge's three-abstraction-tier prompts per
  part (Apache-2.0; ``prompts.json`` vendored, the ``.pth`` weights are
  skipped entirely): prompt-robustness cases.

Discipline shared by every loader (same as ``eval/corpus/fixtures``):

* a ``MANIFEST.json`` per source records, for EVERY file, the
  resources-relative source path, SHA-256, byte count and role -- whether the
  file is vendored or not; vendored data dirs additionally carry a
  ``LICENSE-NOTICE`` naming the license and attribution;
* vendoring happens ONLY for licenses that permit redistribution (MIT,
  Apache-2.0, BSD, CC0, public domain). GPL / no-license sources are never
  copied into ``src/``: their loaders resolve against ``resources/`` at run
  time and DEGRADE CLEANLY (empty result with a stated reason) when absent;
* each loader is runnable: ``python -m harnesscad.eval.bench.imports.<mod>
  --selfcheck`` validates manifests, counts and shapes and exits 0. NO
  geometry kernel and NO model anywhere in this package.

WHY THESE ARE NOT ``eval.corpus.spec.Brief`` INSTANCES. A corpus ``Brief``
requires a closed-form volume, a stated bbox, a citation and a hand-written
reference op stream -- ground truth that is declared and not-us. An imported
benchmark task carries only what its authors wrote down (a prompt, sometimes a
bbox, sometimes a rubric). Promoting one to a strict ``Brief`` would mean
INVENTING the missing numbers, which is exactly the contamination ``spec.py``
exists to refuse. So loaders emit :class:`ImportedBrief` records carrying the
Brief-shaped fields they honestly have, and :meth:`ImportedBrief.to_case`
feeds them to ``contract_grader.contract_for_case``, whose unbound-predicate
path handles the missing measurables without guessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from harnesscad.eval.corpus.fixtures import (
    FixtureEntry,
    Manifest,
    resources_root,
    sha256_of,
)

__all__ = [
    "LOADERS",
    "ImportedBrief",
    "imports_dir",
    "load_manifest",
    "resources_root",
    "sha256_of",
    "FixtureEntry",
    "Manifest",
]

#: Loader module names, in the inventory's ranked order. The hub can iterate
#: these; each module exposes ``main(argv)`` with ``--selfcheck``.
LOADERS: Tuple[str, ...] = (
    "graphcad_cadbench",
    "agentscad_tasks",
    "zoo_kcl_manifest",
    "cadam_textcad_briefs",
    "intentforge_refusals",
    "cadjudge_prompts",
)


def imports_dir() -> Path:
    """The directory this package (and its per-source data dirs) lives in."""
    return Path(__file__).resolve().parent


def load_manifest(source: str) -> Manifest:
    """Load ``imports/<source>/MANIFEST.json`` into a fixtures ``Manifest``.

    Same schema and same dataclasses as ``eval/corpus/fixtures`` -- only the
    data directory differs, so the two fixture families read identically.
    """
    data_dir = imports_dir() / source
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


@dataclass(frozen=True)
class ImportedBrief:
    """One external benchmark task, carried in Brief-shaped fields.

    Deliberately NOT an ``eval.corpus.spec.Brief`` (see the package
    docstring): fields the source states are filled, fields it does not state
    stay ``None`` / empty, and nothing is invented. ``to_case()`` returns the
    mapping shape ``eval.hardcorpus.contract_grader.contract_for_case``
    accepts, so every imported task is contract-gradable today -- unstated
    measurables become unbound ``[NEEDS CLARIFICATION]`` predicates there.
    """

    id: str
    source_repo: str
    license: str
    text: str
    difficulty: str = ""
    #: exact (dx, dy, dz) envelope in mm WHEN THE SOURCE STATES ONE, else None.
    bbox: Optional[Tuple[float, float, float]] = None
    volume: Optional[float] = None
    genus: Optional[int] = None
    categories: Tuple[str, ...] = ()
    #: reference-solution paths (resources-relative or vendored-relative);
    #: existence proofs, never parsed here and never required to be present.
    reference_paths: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("imported brief has no id")
        if not self.text.strip():
            raise ValueError("imported brief %r has no prompt text" % self.id)
        if not self.source_repo.strip():
            raise ValueError("imported brief %r names no source repo" % self.id)
        if self.bbox is not None:
            if len(self.bbox) != 3 or any(float(v) <= 0.0 for v in self.bbox):
                raise ValueError(
                    "imported brief %r states a malformed bbox %r"
                    % (self.id, self.bbox))

    def to_case(self) -> Dict[str, Any]:
        """The mapping ``contract_grader.contract_for_case`` grades against."""
        return {
            "id": self.id,
            "text": self.text,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "volume": self.volume,
            "genus": self.genus,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_repo": self.source_repo,
            "license": self.license,
            "text": self.text,
            "difficulty": self.difficulty,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "volume": self.volume,
            "genus": self.genus,
            "categories": list(self.categories),
            "reference_paths": list(self.reference_paths),
            "tags": list(self.tags),
            "note": self.note,
        }
