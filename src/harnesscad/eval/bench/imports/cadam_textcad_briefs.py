"""23 graded design briefs with reference OpenSCAD: text-to-cad + CADAM.

Two sibling benchmark suites, each an ordered ladder of increasingly hard
text-to-CAD briefs with a working reference solution:

* text-to-cad (resources/cad_repos/text-to-cad-main/text-to-cad-main),
  ``benchmarks/01..10-*.md`` -- 10 briefs, each a precise engineering prompt
  (exact mm dimensions) PLUS a machine-checkable "Test Cases" table (STEP
  import, solid count, bounding box, hole locations, negative checks).
  LICENSE: MIT (Copyright (c) 2026 earthtojake) -- the brief markdown files
  are VENDORED under ``textcad/`` (provenance in ``textcad/MANIFEST.json``,
  attribution in ``textcad/LICENSE-NOTICE``).
* CADAM (resources/cad_repos/CADAM-master/CADAM-master),
  ``benchmarks/01..13-*.{md,scad}`` -- 13 briefs from a twisted vase up to a
  V8 engine, each with a fully parametric reference ``.scad``.
  LICENSE: GPL-3.0 -- the vendoring policy does not accept GPL for
  redistribution inside ``src/``, so NOTHING from CADAM is copied.
  ``cadam/MANIFEST.json`` records every brief's and every reference .scad's
  resources-relative path, SHA-256 and byte count; the loader resolves them
  against ``resources/`` at run time and DEGRADES CLEANLY (the CADAM half of
  the corpus is simply absent, with the reason stated) when the checkout is
  not present.

Reference ``.scad`` solutions are ALWAYS referenced by path (existence
proofs), never vendored and never parsed here -- for BOTH suites, per the
import work order.

Mapping onto harness shapes:

* :func:`briefs` -- every brief as an
  :class:`~harnesscad.eval.bench.imports.ImportedBrief`. ``difficulty`` is
  the suite's own ordinal (both suites are explicitly ordered easy -> hard);
  the text-to-cad test-case rows ride on each :class:`GradedBrief` as
  checklist grading language (same role as CADBench rubric rows). Neither
  suite states a closed-form bbox/volume/genus OUTSIDE the prose, and this
  package never re-derives numbers from prose -- so the measurables stay
  ``None`` and contract grading falls to unbound predicates, honestly.

Stdlib only, deterministic, no kernel, no model, no OpenSCAD parsing.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from harnesscad.eval.bench.imports import ImportedBrief, Manifest, load_manifest

__all__ = [
    "EXPECTED_TEXTCAD",
    "EXPECTED_CADAM",
    "GradedBrief",
    "textcad_manifest",
    "cadam_manifest",
    "load_textcad",
    "load_cadam",
    "load",
    "briefs",
    "main",
]

EXPECTED_TEXTCAD = 10
EXPECTED_CADAM = 13

_TEXTCAD_LICENSE = "MIT"
_CADAM_LICENSE = "GPL-3.0"

#: "# 1. Title" (text-to-cad) or "# 1 <dash> Title" (CADAM; the dash in the
#: source files is U+2014). Written with escapes to keep this file dash-clean.
_TITLE_RE = re.compile("^#\\s*(\\d+)\\s*[.\\-\\u2013\\u2014]*\\s*(.+?)\\s*$")


@dataclass(frozen=True)
class GradedBrief:
    """One graded design brief: prompt, ordinal difficulty, reference paths."""

    id: str
    source_repo: str
    license: str
    ordinal: int              # the suite's own 1-based easy -> hard position
    title: str
    prompt: str
    #: text-to-cad's machine-checkable rows as (test, expected_result);
    #: empty for CADAM, which grades by its parametric reference instead.
    checks: Tuple[Tuple[str, str], ...]
    #: resources-relative path of the reference .scad (None for text-to-cad,
    #: which publishes test cases rather than a .scad in its benchmarks dir).
    scad_resource: Optional[str]


def textcad_manifest() -> Manifest:
    return load_manifest("textcad")


def cadam_manifest() -> Manifest:
    return load_manifest("cadam")


def _parse_title(lines: List[str]) -> Tuple[int, str]:
    for line in lines:
        m = _TITLE_RE.match(line.strip())
        if m:
            return int(m.group(1)), m.group(2).strip()
    return 0, ""


def _parse_textcad(text: str) -> Tuple[int, str, str, Tuple[Tuple[str, str], ...]]:
    """(ordinal, title, prompt, checks) from a text-to-cad benchmark .md."""
    lines = text.splitlines()
    ordinal, title = _parse_title(lines)

    def section(name: str) -> List[str]:
        out: List[str] = []
        inside = False
        for line in lines:
            if line.strip().lower().startswith("## " + name):
                inside = True
                continue
            if inside and line.startswith("## "):
                break
            if inside:
                out.append(line)
        return out

    prompt = "\n".join(
        l for l in (s.rstrip() for s in section("prompt")) if l).strip()

    checks: List[Tuple[str, str]] = []
    for line in section("test cases"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= {"-"} or cells[0].lower() == "test":
            continue
        checks.append((cells[0], cells[1]))
    return ordinal, title, prompt, tuple(checks)


def _parse_cadam(text: str) -> Tuple[int, str, str]:
    """(ordinal, title, prompt) from a CADAM benchmark .md.

    The prompt lives in a blockquote opened by a line containing
    ``**Prompt**``; the quoted lines that follow are the prompt body.
    """
    lines = text.splitlines()
    ordinal, title = _parse_title(lines)
    prompt_lines: List[str] = []
    seen_marker = False
    for line in lines:
        stripped = line.strip()
        if not seen_marker:
            if stripped.startswith(">") and "**Prompt**" in stripped:
                seen_marker = True
            continue
        if not stripped.startswith(">"):
            if prompt_lines:
                break
            continue
        body = stripped.lstrip(">").strip()
        if body:
            prompt_lines.append(body)
    return ordinal, title, " ".join(prompt_lines).strip()


def load_textcad() -> List[GradedBrief]:
    """The 10 text-to-cad briefs from the vendored copies."""
    m = textcad_manifest()
    out: List[GradedBrief] = []
    for e in m.by_role("brief"):
        path = m.resolve(e)
        if path is None:
            continue
        ordinal, title, prompt, checks = _parse_textcad(
            path.read_text(encoding="utf-8"))
        out.append(GradedBrief(
            id="textcad_%s" % e.name,
            source_repo="text-to-cad",
            license=_TEXTCAD_LICENSE,
            ordinal=ordinal,
            title=title,
            prompt=prompt,
            checks=checks,
            scad_resource=None,
        ))
    out.sort(key=lambda b: b.ordinal)
    return out


def load_cadam() -> List[GradedBrief]:
    """The 13 CADAM briefs, read from resources/ (GPL: nothing vendored).

    Empty when the resources checkout is absent; callers must treat that as
    "corpus not present", never as "corpus passed".
    """
    m = cadam_manifest()
    scad_by_stem = {e.name: e for e in m.by_role("reference_scad")}
    out: List[GradedBrief] = []
    for e in m.by_role("brief"):
        path = m.resolve(e)
        if path is None:
            continue
        ordinal, title, prompt = _parse_cadam(path.read_text(encoding="utf-8"))
        scad = scad_by_stem.get(e.name)
        out.append(GradedBrief(
            id="cadam_%s" % e.name,
            source_repo="CADAM",
            license=_CADAM_LICENSE,
            ordinal=ordinal,
            title=title,
            prompt=prompt,
            checks=(),
            scad_resource=scad.resource if scad is not None else None,
        ))
    out.sort(key=lambda b: b.ordinal)
    return out


def load() -> List[GradedBrief]:
    """Both ladders: up to 23 graded briefs (10 vendored + 13 from resources/)."""
    return load_textcad() + load_cadam()


def briefs() -> List[ImportedBrief]:
    out: List[ImportedBrief] = []
    for b in load():
        if not b.prompt:
            continue
        tags = ["graded_ladder"]
        if b.checks:
            tags.append("machine_checkable")
        if b.scad_resource:
            tags.append("scad_reference")
        out.append(ImportedBrief(
            id=b.id,
            source_repo=b.source_repo,
            license=b.license,
            text=b.prompt,
            difficulty="%02d" % b.ordinal,
            reference_paths=(b.scad_resource,) if b.scad_resource else (),
            tags=tuple(tags),
            note="%s benchmark %02d %r%s" % (
                b.source_repo, b.ordinal, b.title,
                "; %d test-case rows" % len(b.checks) if b.checks else ""),
        ))
    return out


def _selfcheck() -> int:
    # --- text-to-cad: vendored, MIT -------------------------------------- #
    tm = textcad_manifest()
    assert tm.license == "MIT", tm.license
    problems = tm.verify_vendored()
    assert not problems, "vendored data drifted: %s" % problems
    tc = load_textcad()
    assert len(tc) == EXPECTED_TEXTCAD, (
        "expected %d text-to-cad briefs, parsed %d" % (EXPECTED_TEXTCAD, len(tc)))
    assert [b.ordinal for b in tc] == list(range(1, EXPECTED_TEXTCAD + 1))
    for b in tc:
        assert b.title, b.id
        assert len(b.prompt) > 80, "brief %s: prompt did not parse" % b.id
        assert len(b.checks) >= 5, (
            "brief %s: test-case table did not parse (%d rows)"
            % (b.id, len(b.checks)))

    # --- CADAM: manifest-only, GPL-3.0 ----------------------------------- #
    cm = cadam_manifest()
    assert cm.license == "GPL-3.0", cm.license
    # Policy check, executable: a GPL repo may vendor NOTHING.
    vendored = [e.name for e in cm.entries if e.vendored]
    assert not vendored, "policy violation: vendored files %s" % vendored
    assert len(cm.by_role("brief")) == EXPECTED_CADAM
    assert len(cm.by_role("reference_scad")) == EXPECTED_CADAM
    for e in cm.entries:
        assert e.resource, e.name
        assert len(e.sha256) == 64, e.name

    ca = load_cadam()
    if not ca:
        assert len(briefs()) == EXPECTED_TEXTCAD
        print("SELFCHECK OK: %d text-to-cad briefs (vendored, SHA verified); "
              "resources/ absent, CADAM half degrades to empty as designed"
              % len(tc))
        return 0

    assert len(ca) == EXPECTED_CADAM, (
        "expected %d CADAM briefs, parsed %d" % (EXPECTED_CADAM, len(ca)))
    assert [b.ordinal for b in ca] == list(range(1, EXPECTED_CADAM + 1))
    for b in ca:
        assert b.title, b.id
        assert len(b.prompt) > 60, "brief %s: prompt did not parse" % b.id
        assert b.scad_resource, "brief %s: no reference .scad manifested" % b.id

    bs = briefs()
    assert len(bs) == EXPECTED_TEXTCAD + EXPECTED_CADAM, len(bs)
    print("SELFCHECK OK: %d graded briefs (%d text-to-cad vendored + %d CADAM "
          "from resources/), all prompts and reference paths parsed"
          % (len(bs), len(tc), len(ca)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="text-to-cad (MIT, vendored) + CADAM (GPL-3.0, "
                    "manifest-only) graded design-brief ladders, 23 briefs "
                    "with reference OpenSCAD by path.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate manifests, hashes, parsed prompts and "
                             "test tables; CADAM degrades cleanly without "
                             "resources/.")
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
