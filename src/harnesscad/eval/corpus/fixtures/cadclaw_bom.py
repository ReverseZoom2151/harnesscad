"""CADCLAW's BOM sextet: 1 known-good + 5 labeled-wrong bills of materials.

Source: CADCLAW (resources/cad_repos/CADCLAW-main,
https://github.com/sunnyday-technologies/CADCLAW, Copyright (c) 2026 Sunnyday
Technologies, MIT License), ``tests/fixtures/m3_crete/``: the M3-CRETE CNC
machine's BOM audit fixtures. Vendored under ``fixtures/cadclaw/`` (six BOM
JSONs plus the ``cadclaw_m3.yaml`` rule file they are graded against), with
SHA-256 provenance in ``MANIFEST.json`` and attribution in
``LICENSE-NOTICE.txt``.

This is a fleet-audit-shaped corpus for a BOM VERIFIER, in exactly the sense
of ``eval/selftest/fleet_audit.py``: a known-good artifact where any finding
is a FALSE POSITIVE, and known-bad artifacts each carrying a STATED defect
where silence is a FALSE NEGATIVE. The defect labels are not invented here --
they are lifted from CADCLAW's own acceptance tests (``tests/
test_bom_audit.py``), which assert the exact finding code per variant:

* ``bom_good``               -- passes the audit; every qty / mfg_type /
                                description matches the rule file.
* ``bom_stale_connectors``   -- qty 16 vs expected (``bom.qty_mismatch``,
                                rule 5) plus stale design language
                                ("maximum rigidity", "primary stiffness").
* ``bom_stale_inserts``      -- stale bonded-insert design surviving in
                                descriptions: "JB Weld", "West System",
                                "custom 2m cut" (all forbidden terms).
* ``bom_stale_motor_mounts`` -- mfg_type "buy" where rule 41 requires a
                                printed part (``bom.mfg_type_mismatch``).
* ``bom_wrong_belts``        -- X belt described as 10mm where 6mm is
                                required and 10mm forbidden (rule 30), Y/Z
                                belt described as 6mm where forbidden
                                (rule 31).
* ``bom_wrong_wheels``       -- V-wheel qty 24 vs expected
                                (``bom.qty_mismatch``, rule 12, got=24).

A BOM verifier here is any ``callable(bom) -> bool`` over the parsed JSON
payload (a list of item dicts), returning ``True`` when it judges the BOM
CLEAN. :func:`score_verifier` runs one over the sextet and reports the same
confusion-matrix vocabulary fleet_audit uses: tp / fp / fn / tn, precision,
recall, F1, with named false positives and false negatives. Verifiers that
also read the rule file get it via :func:`rule_file_path` (raw text -- no
YAML dependency is taken here).

Stdlib only, deterministic.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from harnesscad.eval.corpus.fixtures import Manifest, load_manifest

__all__ = [
    "BomCase",
    "manifest",
    "known_good_cases",
    "known_bad_cases",
    "all_cases",
    "rule_file_path",
    "score_verifier",
    "main",
]

_SOURCE = "cadclaw"

#: name -> (expected finding codes from CADCLAW's own tests, why)
_DEFECTS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "bom_good": (
        (),
        "the reference BOM: every qty, mfg_type and description matches the "
        "rule file. ANY finding raised here is a false positive, full stop."),
    "bom_stale_connectors": (
        ("bom.qty_mismatch", "bom.forbidden_term_present"),
        "connector qty 16 where the rule file expects otherwise (rule 5), "
        "and descriptions still carry the superseded design's language: "
        "'maximum rigidity', 'primary stiffness'. A stale BOM lying about "
        "the current design."),
    "bom_stale_inserts": (
        ("bom.forbidden_term_present",),
        "the abandoned bonded-insert design survives in the text: 'JB Weld', "
        "'West System', 'custom 2m cut'. The parts list was updated, the "
        "descriptions were not."),
    "bom_stale_motor_mounts": (
        ("bom.mfg_type_mismatch",),
        "Z motor mount listed as mfg_type 'buy' where rule 41 requires a "
        "printed part. Wrong sourcing for a known part."),
    "bom_wrong_belts": (
        ("bom.forbidden_term_present", "bom.required_term_missing"),
        "belt widths swapped: the X-gantry belt described as 10mm where rule "
        "30 requires '6mm' and forbids '10mm'; the Y/Z belt described as 6mm "
        "where rule 31 forbids it."),
    "bom_wrong_wheels": (
        ("bom.qty_mismatch",),
        "V-wheel qty 24 against the rule file's expected count (rule 12, "
        "got=24). A count that would leave the gantry short."),
}


@dataclass(frozen=True)
class BomCase:
    """One BOM fixture: mirrors ``fleet_audit.Case`` (name / good / why),
    carrying the parsed JSON payload instead of an op stream."""

    name: str
    good: bool
    why: str
    expected_finding_codes: Tuple[str, ...]
    path: Optional[Path]
    sha256: str

    @property
    def available(self) -> bool:
        return self.path is not None

    def payload(self) -> List[dict]:
        if self.path is None:
            raise FileNotFoundError(
                "BOM %r is not vendored and resources/ is absent" % self.name)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("BOM %r is not a list of items" % self.name)
        return data


def manifest() -> Manifest:
    return load_manifest(_SOURCE)


def _case(m: Manifest, name: str, good: bool) -> BomCase:
    codes, why = _DEFECTS[name]
    e = m.by_name(name)
    return BomCase(
        name=name,
        good=good,
        why=why,
        expected_finding_codes=codes,
        path=m.resolve(e) if e is not None else None,
        sha256=e.sha256 if e is not None else "",
    )


def known_good_cases() -> List[BomCase]:
    m = manifest()
    return [_case(m, e.name, True) for e in m.by_role("known_good")]


def known_bad_cases() -> List[BomCase]:
    m = manifest()
    return [_case(m, e.name, False) for e in m.by_role("known_bad")]


def all_cases() -> List[BomCase]:
    return known_good_cases() + known_bad_cases()


def rule_file_path() -> Optional[Path]:
    """The CADCLAW rule file the sextet is graded against (raw YAML text)."""
    m = manifest()
    e = m.by_name("cadclaw_m3")
    return m.resolve(e) if e is not None else None


def score_verifier(verifier: Callable[[List[dict]], bool]) -> dict:
    """Confusion matrix for one BOM verifier over the sextet.

    ``verifier(payload) -> True`` means "this BOM is clean". Scoring follows
    fleet_audit's reading: firing (returning False) on a known-good BOM is a
    FALSE POSITIVE; staying silent (returning True) on a known-bad one is a
    FALSE NEGATIVE. A verifier crash on a case is charged as an error and
    excluded from the matrix.
    """
    tp = fp = fn = tn = 0
    false_positives: List[str] = []
    false_negatives: List[str] = []
    errors: Dict[str, str] = {}
    skipped: List[str] = []
    for case in all_cases():
        if not case.available:
            skipped.append(case.name)
            continue
        try:
            clean = bool(verifier(case.payload()))
        except Exception as exc:  # noqa: BLE001 - crash is a scored outcome
            errors[case.name] = repr(exc)
            continue
        if case.good:
            if clean:
                tn += 1
            else:
                fp += 1
                false_positives.append(case.name)
        else:
            if clean:
                fn += 1
                false_negatives.append(case.name)
            else:
                tp += 1
    fired = tp + fp
    positives = tp + fn
    precision = tp / float(fired) if fired else None
    recall = tp / float(positives) if positives else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "errors": errors, "skipped": skipped}


def _selfcheck() -> int:
    m = manifest()
    assert m.license == "MIT", m.license
    problems = m.verify_vendored()
    assert not problems, "; ".join(problems)
    good, bad = known_good_cases(), known_bad_cases()
    assert len(good) == 1 and good[0].name == "bom_good", good
    assert len(bad) == 5, "expected 5 known-bad, got %d" % len(bad)
    assert {c.name for c in bad} == set(_DEFECTS) - {"bom_good"}

    available = [c for c in all_cases() if c.available]
    if not available:
        print("SELFCHECK OK: manifest valid (%d entries); no BOM "
              "resolvable, corpus degrades to empty as designed"
              % len(m.entries))
        return 0
    for c in available:
        items = c.payload()
        assert items and all(
            isinstance(i, dict) and "name" in i for i in items), c.name

    # Score two degenerate verifiers to prove the matrix arithmetic: the
    # always-clean verifier misses all 5 defects; the always-dirty one has
    # perfect recall and a false alarm on bom_good -- fleet_audit's
    # "fires on everything" pathology, reproduced in miniature.
    lenient = score_verifier(lambda bom: True)
    assert lenient["fn"] == 5 and lenient["tn"] == 1, lenient
    strict = score_verifier(lambda bom: False)
    assert strict["tp"] == 5 and strict["fp"] == 1, strict
    assert strict["false_positives"] == ["bom_good"], strict

    rules = rule_file_path()
    if rules is not None:
        assert "bom_audit" in rules.read_text(encoding="utf-8")
    print("SELFCHECK OK: %d/6 BOMs present (1 good, %d labeled-bad), "
          "confusion arithmetic verified on degenerate verifiers, rule "
          "file %s" % (len(available),
                       sum(1 for c in available if not c.good),
                       "present" if rules is not None else "absent"))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="CADCLAW BOM sextet: fleet-audit-style precision corpus "
                    "for BOM verifiers (MIT, vendored).")
    parser.add_argument("--selfcheck", action="store_true",
                        help="validate manifest/hashes, payloads and the "
                             "confusion-matrix scoring.")
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
