"""STEP/BREP parse canaries from pythonocc-core and ruststep test suites.

Sources (both under ``resources/cad_repos``):

* pythonocc-core (https://github.com/tpaviot/pythonocc-core), ``test/test_io/``
  -- the I/O regression set of THE reference OpenCascade Python binding: six
  STEP files spanning AP203 and AP214 (single solids, the classic NIST/PDES
  ``as1`` assembly in both protocols, a multi-root file, a dual-schema
  ``FILE_SCHEMA`` header) plus one native OCC BREP (``Motor-c.brep``). A STEP
  reader that chokes on files OpenCascade itself round-trips has a bug.
* ruststep (https://github.com/ricosjp/ruststep), ``ruststep/tests/steps/`` --
  the single real-world ABC Dataset part the ruststep exchange parser is
  integration-tested against (their ``abc_dataset.rs``): a pure grammar-level
  parse canary, valuable precisely because a from-scratch Rust parser was
  hardened on it.

LICENSE VERDICTS (strict; NOTHING is vendored from either source):

* pythonocc-core is LGPL-3.0 (repo ``LICENSE`` verified): the vendoring policy
  treats copyleft data as manifest-only. Paths, SHA-256 sums and byte counts
  are recorded in ``step_canaries/MANIFEST.json``; files resolve from the
  ``resources/`` tree at run time and DEGRADE CLEANLY to ``path=None`` when it
  is absent. Attribution: Thomas Paviot et al., pythonocc-core (LGPL-3.0).
* ruststep is Apache-2.0 (repo ``LICENSE`` + crate metadata verified), which
  WOULD permit vendoring -- but its ``tests/steps/README.md`` states verbatim
  "This directory is not a part of ruststep project. See the original licenses
  for each files": the fixture is an ABC Dataset part
  (https://deep-geometry.github.io/abc-dataset/) under the dataset's own
  terms, not the crate's Apache grant. Not clearly permissive, therefore
  manifest-only as well. Attribution: RICOS Co. Ltd., ruststep (Apache-2.0);
  fixture from the ABC Dataset.

Cross-source bonus oracle: the ruststep ABC part and pythonocc-core's
``stp_multiple_shp_at_root.stp`` are BYTE-IDENTICAL (same SHA-256) -- two
independent parser test suites converged on the same real part, and the
manifest records that duplication honestly instead of hiding one copy.

Corpus shape: :class:`StepCanary` mirrors ``brepnet_steps.StepCase`` (and,
transitively, ``fleet_audit.Case``): ``name`` / ``why`` plus a ``path`` in
place of an op stream. Every entry here is a PARSE canary -- "does the reader
load it sanely?" -- so instead of a good/bad bit each carries ``source`` and a
``category`` tag (``ap203_assembly``, ``ap214_assembly``, ``multi_root``,
``dual_schema``, ``native_brep``, ``abc_real_part``, ...) so harnesses can
slice by protocol or shape of nastiness. Wiring an actual STEP importer to
them lives with the hub, not here (no kernel in this package).

Stdlib only, deterministic, ASCII-only, degrades to empty when resources/ is
absent.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest

__all__ = [
    "StepCanary",
    "manifest",
    "pythonocc_cases",
    "ruststep_cases",
    "all_cases",
    "parse_canaries",
    "main",
]

_SOURCE = "step_canaries"

#: Manifest-declared counts; the selfcheck asserts the manifest still says so.
EXPECTED_PYTHONOCC = 7   # 6 STEP + 1 BREP
EXPECTED_RUSTSTEP = 1

#: The known byte-identical pair (same ABC part in both test suites).
_DUPLICATE_PAIR = ("stp_multiple_shp_at_root", "abc_00000050")

#: Per-entry category tags: what kind of parser nastiness each file probes.
_CATEGORY: Dict[str, str] = {
    "io1_ug_214": "ap214_single_part",        # smallest STEP here (18 KB)
    "test_ocaf": "dual_schema",               # two names in one FILE_SCHEMA
    "eight_cyl": "short_schema_id",           # FILE_SCHEMA(('AP214IS'))
    "stp_multiple_shp_at_root": "multi_root",  # several shapes at root
    "as1_pe_203": "ap203_assembly",           # classic as1, AP203 MIM_LF
    "as1_oc_214": "ap214_assembly",           # classic as1, AP214, largest
    "motor_c": "native_brep",                 # OCC 'CASCADE Topology V1'
    "abc_00000050": "abc_real_part",          # ruststep's grammar canary
}

_WHY: Dict[str, str] = {
    "io1_ug_214": (
        "AP214 part exported by Unigraphics; the smallest STEP in the set, "
        "so the cheapest smoke canary: a reader that cannot load THIS loads "
        "nothing."),
    "test_ocaf": (
        "FILE_SCHEMA declares TWO schema names ('CONFIG_CONTROL_DESIGN', "
        "'SHAPE_APPEARANCE_LAYER_MIM'); readers that assume exactly one "
        "schema string crash on the header before touching geometry."),
    "eight_cyl": (
        "eight cylinders under the terse legacy schema id 'AP214IS'; probes "
        "schema-name normalisation rather than geometry."),
    "stp_multiple_shp_at_root": (
        "multiple independent shapes at document root; importers that assume "
        "one root product silently drop bodies -- the failure mode this "
        "canary exists to catch."),
    "as1_pe_203": (
        "the classic NIST 'as1' bolted-plate assembly in AP203 (config-"
        "controlled design): nested assemblies, shared subassemblies, the "
        "canonical STEP interop torture part since the 1990s."),
    "as1_oc_214": (
        "the same 'as1' assembly exported as AP214 by OpenCascade; the "
        "largest STEP here, and paired with as1_pe_203 it is a cross-"
        "protocol differential oracle for one identical product."),
    "motor_c": (
        "native OpenCascade BREP ('CASCADE Topology V1'), NOT a STEP file: "
        "a format-dispatch canary -- a STEP reader handed this must reject "
        "it EXPLICITLY, never parse garbage."),
    "abc_00000050": (
        "real ABC Dataset part that ruststep's from-scratch exchange parser "
        "is integration-tested against line by line; byte-identical to "
        "pythonocc-core's stp_multiple_shp_at_root.stp, so two independent "
        "test suites vouch for it."),
}


@dataclass(frozen=True)
class StepCanary:
    """One parse-canary fixture: a real file plus why it earns its keep."""

    name: str
    path: Optional[Path]     # None when resources/ is not checked out
    source: str              # "pythonocc-core" or "ruststep"
    format: str              # "step" or "brep"
    category: str
    why: str
    sha256: str

    @property
    def available(self) -> bool:
        return self.path is not None


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _cases(role: str, source: str) -> List[StepCanary]:
    m = manifest()
    return [
        StepCanary(
            name=e.name,
            path=m.resolve(e),
            source=source,
            format=e.format,
            category=_CATEGORY.get(e.name, "uncategorised"),
            why=_WHY.get(e.name, ""),
            sha256=e.sha256,
        )
        for e in m.by_role(role)
    ]


def pythonocc_cases() -> List[StepCanary]:
    """pythonocc-core's test_io set: 6 STEP files + 1 native OCC BREP."""
    return _cases("pythonocc_step", "pythonocc-core") + \
        _cases("pythonocc_brep", "pythonocc-core")


def ruststep_cases() -> List[StepCanary]:
    """ruststep's single ABC Dataset integration-test STEP."""
    return _cases("ruststep_step", "ruststep")


def all_cases() -> List[StepCanary]:
    return pythonocc_cases() + ruststep_cases()


def parse_canaries() -> List[Path]:
    """Every AVAILABLE fixture path: feed them to any STEP/BREP reader.

    Empty when the resources tree is absent -- callers must treat an empty
    list as "corpus not present", never as "corpus passed".
    """
    return [c.path for c in all_cases() if c.path is not None]


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "LGPL-3.0-only AND LicenseRef-ABC-Dataset", m.license
    # NOTHING may be vendored: pythonocc-core is LGPL-3.0 and ruststep's
    # fixture is third-party ABC Dataset content its own README excludes from
    # the Apache-2.0 grant. This is the policy check, executable.
    vendored = [e.name for e in m.entries if e.vendored]
    assert not vendored, "policy violation: vendored files %s" % vendored

    py = m.by_role("pythonocc_step") + m.by_role("pythonocc_brep")
    rs = m.by_role("ruststep_step")
    assert len(py) == EXPECTED_PYTHONOCC, (
        "manifest declares %d pythonocc entries, expected %d"
        % (len(py), EXPECTED_PYTHONOCC))
    assert len(rs) == EXPECTED_RUSTSTEP, (
        "manifest declares %d ruststep entries, expected %d"
        % (len(rs), EXPECTED_RUSTSTEP))
    for e in m.entries:
        assert e.resource, "entry %s has no resources path" % e.name
        assert len(e.sha256) == 64, "entry %s has no sha256" % e.name
        assert e.name in _CATEGORY, "entry %s has no category tag" % e.name
        assert e.name in _WHY, "entry %s has no why" % e.name

    # The documented byte-identical pair must really share a hash.
    a = m.by_name(_DUPLICATE_PAIR[0])
    b = m.by_name(_DUPLICATE_PAIR[1])
    assert a is not None and b is not None
    assert a.sha256 == b.sha256, (
        "manifest no longer records %s and %s as byte-identical"
        % _DUPLICATE_PAIR)

    avail = m.availability()
    cases = all_cases()
    assert len(cases) == EXPECTED_PYTHONOCC + EXPECTED_RUSTSTEP
    assert avail["vendored"] == 0
    present = [c for c in cases if c.available]
    if avail["present"] == 0:
        print("SELFCHECK OK: manifest valid (%d entries); resources/ absent, "
              "corpus degrades to empty as designed" % avail["total"])
        return 0
    # When resources are present, verify that resolved files really are the
    # manifested bytes (full sha over every resolved file; largest is 2.5 MB).
    from harnesscad.eval.corpus.fixtures import sha256_of
    for c in present:
        entry = m.by_name(c.name)
        assert entry is not None
        actual = sha256_of(c.path)
        assert actual == entry.sha256, (
            "resources file drifted from manifest: %s" % c.name)
    print("SELFCHECK OK: %d/%d entries present from resources/ "
          "(%d pythonocc-core, %d ruststep), all SHA-256 verified, "
          "nothing vendored"
          % (avail["present"], avail["total"],
             sum(1 for c in present if c.source == "pythonocc-core"),
             sum(1 for c in present if c.source == "ruststep")))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="pythonocc-core + ruststep STEP/BREP parse canaries "
                    "(manifest + resources-path loader; nothing vendored).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate the manifest, counts and hashes; "
                             "degrades cleanly when resources/ is absent.")
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
