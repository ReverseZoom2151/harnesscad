"""BRepNet's curated known-good / known-bad STEP sets as harness corpora.

Source: Autodesk AI Lab's BRepNet (resources/cad_repos/BRepNet-master,
https://github.com/AutodeskAILab/BRepNet), ``tests/test_data/``:

* ``simple_solids/*.step`` -- 16 curated solids the BRepNet pipeline is KNOWN
  to process correctly (blocks, fillets, holes, cones, a 35-gon, concave and
  elliptical edges, plus two real ABC parts). Known-good: a STEP ingester that
  fails on any of these has a bug, full stop.
* ``issues_16/*.step`` -- 10 real ABC-dataset parts that BROKE the BRepNet
  pipeline (their issue #16). Known-bad-for-pipelines: each one is a real part
  that some real STEP consumer choked on, which makes the set the strongest
  STEP-ingest stress corpus in the resources tree.

LICENSE: BRepNet is CC BY-NC-SA 4.0, which the vendoring policy does not
accept for redistribution inside ``src/``. NOTHING from the repo is vendored.
``brepnet/MANIFEST.json`` records every file's resources-relative path,
SHA-256 and byte count; this loader resolves them against ``resources/`` at
run time and DEGRADES CLEANLY (``path=None``, empty canary list) when the
resources checkout is absent. Attribution: Autodesk AI Lab, BRepNet
(CC BY-NC-SA 4.0).

Corpus shape: :class:`StepCase` deliberately mirrors the fields of
``eval/selftest/fleet_audit.Case`` (``name`` / ``good`` / ``why``) so the two
corpora read the same way, with one honest difference: a fleet-audit ``Case``
carries an op stream (``ops``) the verifier fleet can judge, while a STEP file
carries none -- so ``StepCase`` carries a ``path`` instead, and these rows feed
STEP-INGEST canaries (does the importer load / reject sanely?) rather than the
verifier fleet itself. Wiring an actual STEP importer to them lives with the
hub, not here (no kernel in this package).

Stdlib only, deterministic, degrades to empty when resources/ is absent.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest

__all__ = [
    "StepCase",
    "manifest",
    "known_good_cases",
    "known_bad_cases",
    "all_cases",
    "ingest_canaries",
    "main",
]

_SOURCE = "brepnet"

#: Manifest-declared counts; the selfcheck asserts the manifest still says so.
EXPECTED_KNOWN_GOOD = 16
EXPECTED_KNOWN_BAD = 10


@dataclass(frozen=True)
class StepCase:
    """One STEP fixture: mirrors ``fleet_audit.Case`` minus the op stream."""

    name: str
    path: Optional[Path]     # None when resources/ is not checked out
    good: bool
    why: str
    sha256: str

    @property
    def available(self) -> bool:
        return self.path is not None


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _why_good(name: str) -> str:
    return (
        "curated BRepNet simple_solids part %r: the reference pipeline "
        "processes it correctly, so a STEP ingester that fails on it has a "
        "bug. An ERROR raised on this part is a FALSE POSITIVE." % name
    )


def _why_bad(name: str) -> str:
    return (
        "real ABC part %r from BRepNet issue #16: it broke the BRepNet "
        "pipeline in the wild. An ingester should either load it or reject "
        "it EXPLICITLY -- crashing or silently mangling it is the failure "
        "this canary exists to catch." % name
    )


def known_good_cases() -> List[StepCase]:
    m = manifest()
    return [
        StepCase(e.name, m.resolve(e), True, _why_good(e.name), e.sha256)
        for e in m.by_role("known_good")
    ]


def known_bad_cases() -> List[StepCase]:
    m = manifest()
    return [
        StepCase(e.name, m.resolve(e), False, _why_bad(e.name), e.sha256)
        for e in m.by_role("known_bad")
    ]


def all_cases() -> List[StepCase]:
    return known_good_cases() + known_bad_cases()


def ingest_canaries() -> List[Path]:
    """Every AVAILABLE STEP path, good and bad: feed them to any STEP reader.

    Empty when the resources tree is absent -- callers must treat an empty
    list as "corpus not present", never as "corpus passed".
    """
    return [c.path for c in all_cases() if c.path is not None]


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "CC-BY-NC-SA-4.0", m.license
    good = m.by_role("known_good")
    bad = m.by_role("known_bad")
    assert len(good) == EXPECTED_KNOWN_GOOD, (
        "manifest declares %d known-good, expected %d"
        % (len(good), EXPECTED_KNOWN_GOOD))
    assert len(bad) == EXPECTED_KNOWN_BAD, (
        "manifest declares %d known-bad, expected %d"
        % (len(bad), EXPECTED_KNOWN_BAD))
    # NOTHING may be vendored from a CC BY-NC-SA repo. This is the policy
    # check, executable.
    vendored = [e.name for e in m.entries if e.vendored]
    assert not vendored, "policy violation: vendored files %s" % vendored
    for e in m.entries:
        assert e.resource, "entry %s has no resources path" % e.name
        assert len(e.sha256) == 64, "entry %s has no sha256" % e.name

    avail = m.availability()
    cases = all_cases()
    assert len(cases) == EXPECTED_KNOWN_GOOD + EXPECTED_KNOWN_BAD
    present = [c for c in cases if c.available]
    if avail["present"] == 0:
        print("SELFCHECK OK: manifest valid (%d entries); resources/ absent, "
              "corpus degrades to empty as designed" % avail["total"])
        return 0
    # When resources are present, spot-verify that resolved files really are
    # the manifested bytes (full sha over every resolved STEP; they are small).
    from harnesscad.eval.corpus.fixtures import sha256_of
    for c in present:
        entry = m.by_name(c.name)
        assert entry is not None
        actual = sha256_of(c.path)
        assert actual == entry.sha256, (
            "resources file drifted from manifest: %s" % c.name)
    print("SELFCHECK OK: %d/%d entries present from resources/ "
          "(%d known-good, %d known-bad), all SHA-256 verified"
          % (avail["present"], avail["total"],
             sum(1 for c in present if c.good),
             sum(1 for c in present if not c.good)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="BRepNet known-good/known-bad STEP corpora "
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
