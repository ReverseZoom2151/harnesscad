"""CAD-Coder's 100-part held-out set: GT STEP + reference CadQuery code pairs.

Source: CAD-Coder (resources/cad_repos/CAD-Coder-main,
https://github.com/anniedoris/CAD-Coder, Apache-2.0), ``inference/``:
``test100_gt_steps/*.step`` (100 ground-truth solids) joined to
``cadquery_test_data_subset100.jsonl`` (per part: question id, render image
name, prompt text, and the reference CadQuery program as ``ground_truth``).
The join key is the 8-digit part id -- the image name's prefix matches the
STEP stem for all 100 records (verified at import time and re-verified by the
selfcheck).

BY DESIGN nothing is vendored: 100 STEPs belong in resources/, not in
``src/``. ``cad_coder/MANIFEST.json`` records every file's resources-relative
path, SHA-256 and byte count (101 entries: 100 STEPs + the JSONL), and this
loader resolves them at run time, degrading to an empty pair list when the
resources checkout is absent.

What the pairs are FOR -- geometry-similarity evaluation in the shape of
``eval/corpus/shape.py``: that module's primitive (``iou_of_backends``)
compares two BUILT solids, and its ``iou_of_ops`` convenience compares two op
streams. A :class:`HeldoutPair` supplies the two sides' raw material -- the
GT STEP file (import it) and the reference CadQuery program (execute it, or
hand a model its prompt and compare the model's output against either side).
Executing CadQuery or importing STEP takes a kernel, which this package must
not touch, so the loader stops exactly at the inputs: paths, code, prompt,
provenance. The hub wires the kernels.

Stdlib only, deterministic, degrades to empty when resources/ is absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest

__all__ = [
    "HeldoutPair",
    "manifest",
    "pairs",
    "main",
]

_SOURCE = "cad_coder"
EXPECTED_PAIRS = 100


@dataclass(frozen=True)
class HeldoutPair:
    """One held-out part: ground-truth STEP + reference CadQuery program."""

    part_id: str                 # 8-digit ABC-style id, the join key
    question_id: int
    prompt: str                  # the image-conditioned generation prompt
    reference_code: str          # ground-truth CadQuery program
    gt_step: Optional[Path]     # None when resources/ is absent
    gt_step_sha256: str
    image_name: str              # render the prompt refers to (not manifested)

    @property
    def available(self) -> bool:
        return self.gt_step is not None


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _jsonl_path(m: Manifest) -> Optional[Path]:
    e = m.by_name("cadquery_test_data_subset100")
    return m.resolve(e) if e is not None else None


def pairs() -> List[HeldoutPair]:
    """All 100 pairs, or ``[]`` when the resources tree is not present.

    An empty list means CORPUS NOT PRESENT; callers must not read it as a
    passing (or empty) evaluation.
    """
    m = manifest()
    jsonl = _jsonl_path(m)
    if jsonl is None:
        return []
    out: List[HeldoutPair] = []
    with open(jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image = rec.get("image", "")
            part_id = image.split("_", 1)[0]
            entry = m.by_name(part_id)
            out.append(HeldoutPair(
                part_id=part_id,
                question_id=int(rec.get("question_id", -1)),
                prompt=rec.get("text", ""),
                reference_code=rec.get("ground_truth", ""),
                gt_step=m.resolve(entry) if entry is not None else None,
                gt_step_sha256=entry.sha256 if entry is not None else "",
                image_name=image,
            ))
    return out


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "Apache-2.0", m.license
    # BY DESIGN nothing is vendored; executable policy check.
    vendored = [e.name for e in m.entries if e.vendored]
    assert not vendored, "vendored files present: %s" % vendored
    steps = m.by_role("gt_step")
    assert len(steps) == EXPECTED_PAIRS, (
        "manifest declares %d GT steps, expected %d"
        % (len(steps), EXPECTED_PAIRS))
    assert m.by_role("reference_code"), "JSONL entry missing from manifest"
    for e in m.entries:
        assert e.resource and len(e.sha256) == 64, e.name

    got = pairs()
    if not got:
        print("SELFCHECK OK: manifest valid (%d entries); resources/ absent, "
              "corpus degrades to empty as designed" % len(m.entries))
        return 0
    assert len(got) == EXPECTED_PAIRS, "parsed %d pairs" % len(got)
    ids = {p.part_id for p in got}
    assert len(ids) == EXPECTED_PAIRS, "duplicate part ids"
    # The join must be total: every JSONL record finds its GT STEP.
    missing = [p.part_id for p in got if not p.available]
    assert not missing, "pairs without a resolvable GT STEP: %s" % missing[:5]
    for p in got:
        assert "cadquery" in p.reference_code.lower(), (
            "%s: reference code does not look like CadQuery" % p.part_id)
        assert p.prompt, "%s: empty prompt" % p.part_id
    # Spot-verify bytes against the manifest on a deterministic sample (full
    # 100-file hashing is what the manifest generator already did once).
    from harnesscad.eval.corpus.fixtures import sha256_of
    sample = sorted(got, key=lambda p: p.part_id)[::20]
    for p in sample:
        assert sha256_of(p.gt_step) == p.gt_step_sha256, (
            "resources file drifted from manifest: %s" % p.part_id)
    print("SELFCHECK OK: %d GT-STEP + reference-code pairs joined "
          "(join total, %d spot hashes verified)" % (len(got), len(sample)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CAD-Coder 100-part held-out set: GT-STEP + reference "
                    "CadQuery pairs (manifest + resources-path only).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate the manifest, the JSONL-to-STEP join "
                             "and spot hashes; degrades cleanly when "
                             "resources/ is absent.")
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
