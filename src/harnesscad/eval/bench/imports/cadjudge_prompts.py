"""cad-judge's abstraction-tier prompts: prompt-robustness cases per part.

Source: cad-judge (resources/cad_repos/cad-judge-main/cad-judge-main),
``data/prompts.json`` -- 5 DeepCAD parts, each described at THREE abstraction
tiers by the paper's protocol:

* ``abstract`` -- one sentence, shape words only ("a rectangular block with a
  flat top and bottom");
* ``beginner`` -- approximate dimensions in plain language;
* ``expert``   -- a full modelling walkthrough (coordinate system, sketch,
  loop, extrude) precise enough to reproduce the part.

All three tiers of one part name THE SAME geometry, which is exactly what a
prompt-robustness case needs: a system graded on these should converge to one
part as the tier gets more precise, and a grader (or a generator) whose
verdict swings wildly across tiers of the same part is being steered by
phrasing, not geometry. Each part also carries the DeepCAD ``description``
and ``keywords``, carried through as metadata.

LICENSE: Apache License 2.0 -- redistribution permitted, so ``prompts.json``
is VENDORED under ``cadjudge/`` (provenance in ``cadjudge/MANIFEST.json``,
attribution in ``cadjudge/LICENSE-NOTICE``). The repo's ``data/cad_seq``
``.pth`` model weights are SKIPPED ENTIRELY, per the import work order --
they appear in no manifest and are never read.

Mapping onto harness shapes:

* :func:`triples` -- the 5 parts with their three tier prompts side by side
  (the robustness pairing structure).
* :func:`cases` / :func:`briefs` -- the 15 (part, tier) prompts flattened,
  each as an :class:`~harnesscad.eval.bench.imports.ImportedBrief` tagged
  with its tier and part id so a run can be grouped back into triples.
  cad-judge states no bbox/volume/genus per part, so the measurables stay
  ``None``.

Stdlib only, deterministic, no kernel, no model, no .pth anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from harnesscad.eval.bench.imports import ImportedBrief, Manifest, load_manifest

__all__ = [
    "SOURCE_REPO",
    "LICENSE",
    "TIERS",
    "EXPECTED_PARTS",
    "PromptTiers",
    "RobustnessCase",
    "manifest",
    "triples",
    "cases",
    "briefs",
    "main",
]

_SOURCE = "cadjudge"
SOURCE_REPO = "cad-judge"
LICENSE = "Apache-2.0"

#: The three abstraction tiers, least to most specified.
TIERS: Tuple[str, ...] = ("abstract", "beginner", "expert")

EXPECTED_PARTS = 5


@dataclass(frozen=True)
class PromptTiers:
    """One part's three abstraction-tier prompts, plus DeepCAD metadata."""

    part_id: str          # the DeepCAD id, e.g. "0035/00359148"
    abstract: str
    beginner: str
    expert: str
    description: str = ""
    keywords: Tuple[str, ...] = ()

    def prompt(self, tier: str) -> str:
        if tier not in TIERS:
            raise ValueError("unknown tier %r; tiers are %s" % (tier, TIERS))
        return getattr(self, tier)


@dataclass(frozen=True)
class RobustnessCase:
    """One (part, tier) prompt: 15 of these make the robustness battery."""

    part_id: str
    tier: str
    prompt: str


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def triples() -> List[PromptTiers]:
    """The 5 parts with all three tiers, in stable part-id order."""
    m = manifest()
    e = m.by_name("prompts.json")
    path = m.resolve(e) if e is not None else None
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[PromptTiers] = []
    for part_id in sorted(raw):
        r = raw[part_id]
        kw = r.get("keywords", [])
        if isinstance(kw, str):
            kw = [k.strip() for k in kw.split(",") if k.strip()]
        out.append(PromptTiers(
            part_id=str(part_id),
            abstract=str(r.get("abstract", "")),
            beginner=str(r.get("beginner", "")),
            expert=str(r.get("expert", "")),
            description=str(r.get("description", "")),
            keywords=tuple(str(k) for k in kw),
        ))
    return out


def cases() -> List[RobustnessCase]:
    """The 15 (part, tier) prompts, tier-major within each part."""
    return [
        RobustnessCase(t.part_id, tier, t.prompt(tier))
        for t in triples()
        for tier in TIERS
    ]


def briefs() -> List[ImportedBrief]:
    """Each (part, tier) prompt as an ImportedBrief, groupable by part id."""
    out: List[ImportedBrief] = []
    for c in cases():
        if not c.prompt.strip():
            continue
        out.append(ImportedBrief(
            id="cadjudge_%s_%s" % (c.part_id.replace("/", "_"), c.tier),
            source_repo=SOURCE_REPO,
            license=LICENSE,
            text=c.prompt,
            difficulty=c.tier,
            tags=("prompt_robustness", "tier_%s" % c.tier,
                  "part_%s" % c.part_id.replace("/", "_")),
            note="cad-judge abstraction-tier prompt: all tiers of part %s "
                 "describe the same geometry" % c.part_id,
        ))
    return out


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "Apache-2.0", m.license
    problems = m.verify_vendored()
    assert not problems, "vendored data drifted: %s" % problems
    # Policy check, executable: exactly one vendored file, and no .pth in any
    # manifest role.
    assert [e.name for e in m.entries] == ["prompts.json"], (
        "unexpected manifest entries: %s" % [e.name for e in m.entries])
    assert all((e.vendored or "").endswith(".json") for e in m.entries)

    ts = triples()
    assert len(ts) == EXPECTED_PARTS, (
        "expected %d parts, loaded %d" % (EXPECTED_PARTS, len(ts)))
    for t in ts:
        for tier in TIERS:
            assert t.prompt(tier).strip(), (
                "part %s tier %s is empty" % (t.part_id, tier))
        # The tiers must actually be an abstraction ladder: strictly more
        # words as the tier gets more specified, for every part.
        lengths = [len(t.prompt(tier).split()) for tier in TIERS]
        assert lengths[0] < lengths[1] < lengths[2], (
            "part %s tiers are not an abstraction ladder: %s"
            % (t.part_id, lengths))

    cs = cases()
    assert len(cs) == EXPECTED_PARTS * len(TIERS), len(cs)
    bs = briefs()
    assert len(bs) == len(cs), "briefs dropped cases: %d" % len(bs)
    assert len({b.id for b in bs}) == len(bs), "duplicate brief ids"

    print("SELFCHECK OK: %d parts x %d tiers = %d prompt-robustness cases, "
          "word-count ladder verified per part, no .pth touched"
          % (len(ts), len(TIERS), len(cs)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="cad-judge abstraction-tier prompt loader (5 parts x 3 "
                    "tiers, Apache-2.0, vendored; .pth weights skipped).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate vendored hash, tier counts and the "
                             "abstraction ladder per part.")
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
