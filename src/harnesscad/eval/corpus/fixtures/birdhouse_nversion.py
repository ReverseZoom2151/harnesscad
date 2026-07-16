"""curated-code-cad's birdhouse: one part, eight languages -- an N-version
differential oracle.

Source: curated-code-cad (resources/cad_repos/curated-code-cad-main,
https://github.com/Irev-Dev/curated-code-cad, Copyright (c) 2020 Kurt Hutten,
MIT License), ``birdhouse/``: the SAME birdhouse modeled independently in
CadQuery, build123d, FreeCAD, OpenSCAD, JSCAD, CascadeStudio, DeclaraCAD and
sdfx (Go). Vendored under ``fixtures/birdhouse/sources/`` with SHA-256
provenance in ``MANIFEST.json`` and attribution in ``LICENSE-NOTICE.txt``.

Why this is an oracle: ``eval/selftest/differential.py`` runs ONE op stream
through the harness's six engines and calls any spread a finding, because the
implementations are independent and the intent is identical. The birdhouse is
the same argument with the INDEPENDENCE turned all the way up -- eight authors,
eight languages, eight kernels, one intended part. Where their built solids
disagree (volume, bbox, genus, watertightness -- differential.py's exact
signature), at least one implementation or one backend is wrong, and no
ground-truth label was ever needed. It is the natural cross-backend
differential CASE the harness's synthetic streams cannot be: a real part
written by people, not generated from a single canonical plan.

What the loader does and does not do: it exposes each version's source text,
language and (where one exists) the harness backend that can execute it --
``cadquery``, ``freecad``, ``openscad`` today; the other five versions are
comparison material for external runners. Executing any of them takes a
kernel, so this module stops at the case description
(:func:`differential_case`) and the hub wires execution, exactly as with
every other loader in this package.

Stdlib only, deterministic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest

__all__ = [
    "NVersion",
    "manifest",
    "versions",
    "harness_executable_versions",
    "differential_case",
    "main",
]

_SOURCE = "birdhouse"

#: file stem -> (language, harness backend able to run it, or None)
_LANGUAGES = {
    "CadQuery": ("python/cadquery", "cadquery"),
    "build123d": ("python/build123d", None),
    "FreeCad": ("python/freecad", "freecad"),
    "OpenSCAD": ("openscad", "openscad"),
    "JSCAD": ("javascript/jscad", None),
    "CascadeStudio": ("javascript/cascadestudio", None),
    "DeclaraCAD": ("enaml/declaracad", None),
    "sdfx": ("go/sdfx", None),
}

EXPECTED_VERSIONS = 8


@dataclass(frozen=True)
class NVersion:
    """One independent implementation of the birdhouse."""

    name: str
    language: str
    harness_backend: Optional[str]   # differential.py backend name, if any
    path: Optional[Path]
    sha256: str

    @property
    def available(self) -> bool:
        return self.path is not None

    def source(self) -> str:
        if self.path is None:
            raise FileNotFoundError(
                "version %r is not vendored and resources/ is absent"
                % self.name)
        return self.path.read_text(encoding="utf-8", errors="replace")


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def versions() -> List[NVersion]:
    m = manifest()
    out: List[NVersion] = []
    for e in m.by_role("n_version"):
        language, backend = _LANGUAGES.get(e.name, (e.format, None))
        out.append(NVersion(e.name, language, backend, m.resolve(e), e.sha256))
    return sorted(out, key=lambda v: v.name.lower())


def harness_executable_versions() -> List[NVersion]:
    """The versions a harness backend could execute directly (3 of 8)."""
    return [v for v in versions() if v.harness_backend]


def differential_case() -> dict:
    """The birdhouse as one differential-oracle case description.

    Mirrors the vocabulary of ``eval/selftest/differential.py``: N independent
    builds of one intended part; compare the built solids' signature (volume,
    bbox, genus, watertight) and report clusters. Disagreement means at least
    one implementation or kernel is wrong -- no ground truth required, and the
    same caveat applies verbatim: agreement is evidence, not proof.
    """
    vs = versions()
    return {
        "name": "birdhouse_nversion",
        "intent": "the same birdhouse, independently modeled 8 times",
        "signature": ("volume", "bbox", "genus", "watertight"),
        "versions": [
            {"name": v.name, "language": v.language,
             "harness_backend": v.harness_backend,
             "available": v.available, "sha256": v.sha256}
            for v in vs
        ],
        "harness_executable": [v.name for v in vs if v.harness_backend],
        "external": [v.name for v in vs if not v.harness_backend],
        "reading": "any structural spread across versions means at least one "
                   "implementation or kernel is wrong; consensus is a signal, "
                   "not truth",
    }


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "MIT", m.license
    problems = m.verify_vendored()
    assert not problems, "; ".join(problems)
    vs = versions()
    assert len(vs) == EXPECTED_VERSIONS, (
        "expected %d versions, got %d" % (EXPECTED_VERSIONS, len(vs)))
    assert all(v.name in _LANGUAGES for v in vs), (
        "unmapped version present: %s"
        % [v.name for v in vs if v.name not in _LANGUAGES])

    available = [v for v in vs if v.available]
    if not available:
        print("SELFCHECK OK: manifest valid (%d entries); no source "
              "resolvable, corpus degrades to empty as designed"
              % len(m.entries))
        return 0
    for v in available:
        text = v.source()
        assert len(text) > 200, "%s: suspiciously short source" % v.name
    runnable = harness_executable_versions()
    assert {v.harness_backend for v in runnable} == {
        "cadquery", "freecad", "openscad"}, runnable
    case = differential_case()
    assert len(case["versions"]) == EXPECTED_VERSIONS
    assert len(case["harness_executable"]) == 3
    print("SELFCHECK OK: %d/%d birdhouse versions present, %d executable on "
          "harness backends (%s), differential case well-formed"
          % (len(available), len(vs), len(runnable),
             ", ".join(sorted(v.harness_backend for v in runnable))))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="birdhouse N-version set: one part in 8 languages, a "
                    "cross-backend differential oracle (MIT, vendored).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate manifest/hashes, sources and the "
                             "differential case shape.")
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
