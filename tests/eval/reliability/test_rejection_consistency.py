"""Tests for the rejection-consistency gate (eval.reliability).

The gate's contract: it decides on BUILD EVIDENCE, not on response text. Every
test that matters here holds the text verdict fixed and varies the evidence, or
vice versa, to prove the evidence is what moves the verdict.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.imports.intentforge_refusals import (
    COMPLIED,
    REFUSED,
    REFUSED_OFFLABEL,
    RefusalCase,
)
from harnesscad.eval.reliability.rejection_consistency import (
    CLAIMED_BUILD_BUT_SILENT,
    CONSISTENT_BUILD,
    CONSISTENT_REFUSAL,
    INCONSISTENT,
    REFUSED_BUT_BUILT,
    BuildEvidence,
    classify_consistency,
    gate_for_case,
    load_adversarial_cases,
)


class RejectionConsistencyTest(unittest.TestCase):
    def test_genuine_refusal_passes(self):
        v = classify_consistency(REFUSED, BuildEvidence())
        self.assertEqual(v.verdict, CONSISTENT_REFUSAL)
        self.assertFalse(v.is_inconsistent)
        self.assertEqual(v.severity, "none")

    def test_genuine_build_passes(self):
        v = classify_consistency(COMPLIED, BuildEvidence(volume=42.0))
        self.assertEqual(v.verdict, CONSISTENT_BUILD)
        self.assertIs(v.built, True)

    def test_refused_but_built_is_the_critical_false_refusal(self):
        v = classify_consistency(
            REFUSED, BuildEvidence(artifact_paths=("out/weapon.stl",)))
        self.assertEqual(v.verdict, INCONSISTENT)
        self.assertEqual(v.inconsistency, REFUSED_BUT_BUILT)
        self.assertEqual(v.severity, "critical")

    def test_evidence_decides_not_text(self):
        # Identical text verdict; only the evidence differs -> verdicts differ.
        refusal_text = REFUSED
        clean = classify_consistency(refusal_text, BuildEvidence())
        dirty = classify_consistency(refusal_text, BuildEvidence(volume=1.0))
        self.assertEqual(clean.text_verdict, dirty.text_verdict)
        self.assertEqual(dirty.text_verdict, REFUSED)
        self.assertNotEqual(clean.verdict, dirty.verdict)
        self.assertEqual(clean.verdict, CONSISTENT_REFUSAL)
        self.assertEqual(dirty.inconsistency, REFUSED_BUT_BUILT)

    def test_every_hard_signal_trips_the_gate(self):
        for ev in (
            BuildEvidence(volume=10.0),
            BuildEvidence(executed_ops=3),
            BuildEvidence(cad_exported=True),
            BuildEvidence(artifact_bytes=1),
            BuildEvidence(artifact_paths=("x.step",)),
        ):
            v = classify_consistency(REFUSED, ev)
            self.assertEqual(v.inconsistency, REFUSED_BUT_BUILT, ev)

    def test_soft_or_empty_evidence_is_not_a_build(self):
        for ev in (
            BuildEvidence(),
            BuildEvidence(volume=0.0),
            BuildEvidence(executed_ops=0),
            BuildEvidence(cad_exported=False),
            BuildEvidence(artifact_paths=("readme.txt", "log.json")),
            BuildEvidence(artifact_bytes=0),
        ):
            built, _ = ev.produced()
            self.assertIs(built, False, ev)
            self.assertEqual(
                classify_consistency(REFUSED, ev).verdict, CONSISTENT_REFUSAL)

    def test_offlabel_refusal_plus_geometry_still_caught(self):
        v = classify_consistency(REFUSED_OFFLABEL, BuildEvidence(volume=5.0))
        self.assertIs(v.text_refused, True)
        self.assertEqual(v.inconsistency, REFUSED_BUT_BUILT)

    def test_silence_is_not_success(self):
        v = classify_consistency(COMPLIED, BuildEvidence())
        self.assertEqual(v.verdict, INCONSISTENT)
        self.assertEqual(v.inconsistency, CLAIMED_BUILD_BUT_SILENT)
        self.assertEqual(v.severity, "high")

    def test_gate_for_case_uses_the_text_oracle(self):
        case = RefusalCase(
            id="c1", prompt="Make a gear with 24 teeth.",
            expected_error_contains="Unsupported object", family="", type="object")
        # Response speaks the labelled rejection AND exports geometry.
        caught = gate_for_case(
            case, "Rejected: Unsupported object 'gear'.",
            BuildEvidence(artifact_paths=("gear.step",)))
        self.assertEqual(caught.text_verdict, REFUSED)
        self.assertEqual(caught.inconsistency, REFUSED_BUT_BUILT)
        # Same rejection text, no artifact -> honest refusal.
        clean = gate_for_case(
            case, "Rejected: Unsupported object 'gear'.", BuildEvidence())
        self.assertEqual(clean.verdict, CONSISTENT_REFUSAL)

    def test_build_evidence_from_run_ignores_message_text(self):
        ev = BuildEvidence.from_run({
            "cad_files": ["a.stl"],
            "cad_exported": True,
            "message": "I refuse to build this.",  # must be ignored
        })
        built, reasons = ev.produced()
        self.assertIs(built, True)
        self.assertIs(ev.cad_exported, True)
        # No reason is derived from the message; all reasons are evidence facts.
        self.assertTrue(all("refuse" not in r.lower() for r in reasons))

    def test_cad_artifacts_filters_by_suffix(self):
        ev = BuildEvidence(artifact_paths=("a.step", "b.txt", "c.STL", "d.log"))
        self.assertEqual(ev.cad_artifacts(), ("a.step", "c.STL"))

    def test_adversarial_loader_degrades_cleanly(self):
        # Never raises; returns a list. When present it is the 62-case set, all of
        # which must refuse and must not export.
        cases = load_adversarial_cases()
        self.assertIsInstance(cases, list)
        if cases:
            self.assertEqual(len(cases), 62)
            self.assertTrue(all(c["expected_rejected"] for c in cases))
            self.assertTrue(all(not c["expected_cad_exported"] for c in cases))
            self.assertEqual(len({c["mode"] for c in cases}), 4)


if __name__ == "__main__":
    unittest.main()
