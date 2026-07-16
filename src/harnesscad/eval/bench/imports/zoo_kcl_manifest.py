"""Zoo modeling-app's kcl-samples: 100 human-described real parts as briefs.

Source: modeling-app by The Zoo Authors (resources/cad_repos/modeling-app-main/
modeling-app-main), ``public/kcl-samples/manifest.json``. Each entry is a real
mechanical part -- ball bearing, axial fan, car wheel assembly, bone plate --
with a human-written title + description and one or more categories. This is
the diversity the harness's procedural corpora lack: 100 realistic parts whose
descriptions read like actual design briefs, each backed by a working KCL
solution as an EXISTENCE PROOF that the brief is buildable.

LICENSE: The MIT License (MIT), Copyright (c) 2023 The Zoo Authors --
redistribution permitted. The small ``manifest.json`` is VENDORED (as
``zoo_kcl/kcl_samples_manifest.json``, provenance in ``zoo_kcl/MANIFEST.json``,
attribution in ``zoo_kcl/LICENSE-NOTICE``). The KCL SOURCES are deliberately
NEVER vendored, per the import work order: they are reference solutions, not
data this harness should ship; each one's resources-relative path, SHA-256 and
byte count is manifested (role ``kcl_source``) so the reference is verifiable,
and :func:`kcl_path` resolves it at run time, returning ``None`` when the
resources checkout is absent.

Mapping onto harness shapes:

* :func:`briefs` -- each sample as an
  :class:`~harnesscad.eval.bench.imports.ImportedBrief`: ``text`` is the
  human description (the only thing a model should be shown), the categories
  are carried, and the KCL solution path rides in ``reference_paths`` as an
  existence proof. The manifest states no bbox/volume/genus, so those stay
  ``None`` and contract grading falls to unbound predicates -- honestly.

Stdlib only, deterministic, no kernel, no model, no KCL parsing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from harnesscad.eval.bench.imports import ImportedBrief, Manifest, load_manifest

__all__ = [
    "SOURCE_REPO",
    "LICENSE",
    "EXPECTED_SAMPLES",
    "KclSample",
    "manifest",
    "load",
    "briefs",
    "kcl_path",
    "main",
]

_SOURCE = "zoo_kcl"
SOURCE_REPO = "modeling-app (Zoo)"
LICENSE = "MIT"

#: Entries in the vendored kcl-samples manifest; the selfcheck re-counts.
EXPECTED_SAMPLES = 100

_KCL_ROOT = "cad_repos/modeling-app-main/modeling-app-main/public/kcl-samples"


@dataclass(frozen=True)
class KclSample:
    """One kcl-samples manifest entry: a real part with a human description."""

    id: str                      # the sample's project directory name
    title: str
    description: str
    categories: Tuple[str, ...]
    entry_file: str              # path from the project dir to the first file
    files: Tuple[str, ...]
    multiple_files: bool


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _manifest_path() -> Optional[Path]:
    m = manifest()
    e = m.by_name("manifest.json")
    return m.resolve(e) if e is not None else None


def load() -> List[KclSample]:
    """Every kcl-samples entry, from the vendored manifest."""
    path = _manifest_path()
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: List[KclSample] = []
    for s in raw:
        entry_file = str(s.get("pathFromProjectDirectoryToFirstFile", ""))
        samples.append(KclSample(
            id=entry_file.split("/")[0] if "/" in entry_file else entry_file,
            title=str(s.get("title", "")),
            description=str(s.get("description", "")),
            categories=tuple(str(c) for c in s.get("categories", [])),
            entry_file=entry_file,
            files=tuple(str(f) for f in s.get("files", [])),
            multiple_files=bool(s.get("multipleFiles", False)),
        ))
    return samples


def briefs() -> List[ImportedBrief]:
    """Each sample as an ImportedBrief; the KCL solution is a reference path."""
    out: List[ImportedBrief] = []
    for s in load():
        text = s.description.strip() or s.title.strip()
        if not s.id or not text:
            continue
        out.append(ImportedBrief(
            id="zoo_kcl_%s" % s.id,
            source_repo=SOURCE_REPO,
            license=LICENSE,
            text=text,
            categories=s.categories,
            reference_paths=("%s/%s" % (_KCL_ROOT, s.entry_file),),
            tags=("real_part", "kcl_reference")
                 + (("multi_file",) if s.multiple_files else ()),
            note="Zoo kcl-samples part %r; the KCL source is an existence "
                 "proof, referenced by resources path and never vendored"
                 % s.title,
        ))
    return out


def kcl_path(sample: KclSample) -> Optional[Path]:
    """The sample's KCL entry file resolved from resources/, or ``None``.

    Never vendored: an absent resources checkout degrades to ``None``, which
    callers must treat as "reference not present", never as a failure of the
    brief itself.
    """
    m = manifest()
    e = m.by_name(sample.id)
    if e is None or e.role != "kcl_source":
        return None
    return m.resolve(e)


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "MIT", m.license
    problems = m.verify_vendored()
    assert not problems, "vendored data drifted: %s" % problems
    # Policy check, executable: only the small manifest may be vendored; no
    # KCL source may ever be.
    for e in m.entries:
        if e.role == "kcl_source":
            assert not e.vendored, (
                "policy violation: KCL source vendored: %s" % e.name)
            assert e.resource, e.name
        assert len(e.sha256) == 64, e.name

    samples = load()
    assert len(samples) == EXPECTED_SAMPLES, (
        "expected %d samples, loaded %d" % (EXPECTED_SAMPLES, len(samples)))
    ids = {s.id for s in samples}
    assert len(ids) == len(samples), "duplicate sample ids"
    for s in samples:
        assert s.title.strip(), "sample %s has no title" % s.id
        assert s.description.strip(), "sample %s has no description" % s.id
        assert s.entry_file.endswith(".kcl"), s.id
    # A few tutorial samples legitimately carry no category; they must still
    # carry a description (asserted above), so they remain usable briefs.
    uncategorised = sum(1 for s in samples if not s.categories)
    assert uncategorised <= 5, "unexpectedly many uncategorised samples"

    bs = briefs()
    assert len(bs) == EXPECTED_SAMPLES, "briefs dropped samples: %d" % len(bs)
    assert all(b.bbox is None and b.volume is None for b in bs), (
        "a zoo brief invented a measurable")

    resolved = sum(1 for s in samples if kcl_path(s) is not None)
    if resolved == 0:
        print("SELFCHECK OK: %d samples from the vendored manifest; "
              "resources/ absent, KCL references degrade to None as designed"
              % len(samples))
    else:
        print("SELFCHECK OK: %d samples, %d/%d KCL reference solutions "
              "resolve from resources/" % (len(samples), resolved, len(samples)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zoo kcl-samples loader (100 human-described real parts, "
                    "MIT; manifest vendored, KCL sources referenced by path "
                    "only).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate vendored hash, counts and shapes; KCL "
                             "references degrade cleanly without resources/.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.selfcheck:
        parser.print_help()
        return 0
    try:
        return _selfcheck()
    except AssertionError as exc:
        print("SELFCHECK FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
