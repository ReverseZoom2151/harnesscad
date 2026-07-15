"""The Tier-2 evidence runner: it measures, and it does not overclaim.

COST NOTE, and why this suite is narrow. Every measurement here drives a real
geometry engine. ``freecad``, ``openscad`` and ``blender`` each FORK A PROCESS
and the OCCT-backed engines can abort at interpreter teardown -- which is why the
repo's suite runs per-module in the first place. So these tests never reach an
external engine: they run on the two IN-PROCESS backends (``stub`` and ``frep``)
over one or two short streams. The full six-engine sweep is what
``python -m harnesscad.eval.gallery.tier2`` is for, and it is a report a human
asks for, not a unit test.

The most important test in this file is
:meth:`TestHonesty.test_the_report_says_what_it_does_not_prove` -- because the
entire value of Tier 2 depends on nobody mistaking "the engines agreed" for "the
part is correct", and a module that made that mistake would be worse than no
module at all.
"""

from __future__ import annotations

import json
import unittest

from harnesscad.eval.gallery import complex_parts, tier2
from harnesscad.eval.selftest.probe import available


#: The engines that cost nothing: in-process, no fork, no OCCT teardown.
IN_PROCESS = ("frep",)


class TestHonesty(unittest.TestCase):
    """Tier 2 is a bug DETECTOR, not a correctness PROOF. Say so, in the artifact."""

    def test_the_report_says_what_it_does_not_prove(self):
        blob = json.dumps(tier2.Tier2Report().to_dict()).lower()
        self.assertIn("does_not_prove", blob)
        # The three ways agreement can be worthless, all named in the artifact.
        self.assertIn("share a bug", blob)
        self.assertIn("many-to-one", blob)
        self.assertIn("no ground truth", blob)

    def test_it_does_not_claim_correctness(self):
        proves = " ".join(tier2.PROVES).lower()
        self.assertNotIn("correct", proves)
        self.assertNotIn("proof", proves)
        self.assertIn("did not disagree", proves)

    def test_agreed_is_not_spelled_correct(self):
        """The field is `agreed`, not `passed`/`valid`/`correct`. Words matter."""
        ev = tier2.StreamEvidence(name="x")
        self.assertTrue(hasattr(ev, "agreed"))
        self.assertFalse(hasattr(ev, "correct"))

    def test_does_not_prove_mentions_intent(self):
        text = " ".join(tier2.DOES_NOT_PROVE).lower()
        self.assertIn("brief", text)


class TestStreamEvidence(unittest.TestCase):

    def test_agreed_is_false_when_anything_was_found(self):
        ev = tier2.StreamEvidence(name="x")
        self.assertTrue(ev.agreed)
        ev.disagreements = [{"metric": "volume", "structural": True}]
        self.assertFalse(ev.agreed)

    def test_a_crash_is_a_finding_not_a_silence(self):
        """An engine that BLEW UP must not hide inside a '0 disagreements' headline."""
        ev = tier2.StreamEvidence(name="x")
        ev.crashed = {"cadquery": "BackendUnavailable"}
        self.assertFalse(ev.agreed)

    def test_a_broken_law_is_a_finding(self):
        ev = tier2.StreamEvidence(name="x")
        ev.violations = [{"property": "scale_is_cubic", "backend": "frep"}]
        self.assertFalse(ev.agreed)

    def test_a_refusal_is_not_a_finding(self):
        """A capability gap is an engine being HONEST, not an engine being wrong."""
        ev = tier2.StreamEvidence(name="x")
        ev.refused = {"openscad": "rejected fillet (unsupported-op)"}
        self.assertTrue(ev.agreed)

    def test_structural_disagreements_are_separated(self):
        ev = tier2.StreamEvidence(name="x")
        ev.disagreements = [{"metric": "volume", "structural": False},
                            {"metric": "genus", "structural": True}]
        self.assertEqual(len(ev.structural), 1)

    def test_round_trips_through_json(self):
        ev = tier2.StreamEvidence(name="x", depth=3, engines=["frep"])
        back = tier2._from_dict(json.loads(json.dumps(ev.to_dict())))
        self.assertEqual(back.name, "x")
        self.assertEqual(back.depth, 3)
        self.assertEqual(back.engines, ["frep"])


class TestMeasure(unittest.TestCase):
    """One real stream, on the in-process engines only. Slow-ish but honest."""

    def test_needs_two_engines_to_differentiate(self):
        """One engine cannot disagree with anybody, and must not pretend to."""
        ev = tier2.measure_stream("shell-and-holes", backends=("frep",),
                                  gate_parts=False)
        self.assertIn("differential", ev.error)

    @unittest.skipUnless(available(IN_PROCESS) == list(IN_PROCESS),
                         "frep backend unavailable")
    def test_a_real_stream_measures(self):
        ev = tier2.measure_stream("shell-and-holes", backends=("frep", "stub"),
                                  gate_parts=False)
        # stub carries no geometry, so it never joins a geometric cluster; the
        # point of this test is that the runner produces a well-formed verdict.
        self.assertEqual(ev.name, "shell-and-holes")
        self.assertEqual(ev.depth, 7)
        self.assertIn("shell", ev.ops)
        self.assertIn("hole", ev.ops)
        json.dumps(ev.to_dict())


class TestReportFormatting(unittest.TestCase):

    def test_format_text_states_the_limit_of_the_claim(self):
        report = tier2.Tier2Report(engines=["frep", "blender"],
                                   coverage=complex_parts.coverage_report())
        report.streams.append(tier2.StreamEvidence(name="x", depth=9))
        text = tier2.format_text(report)
        self.assertIn("DOES NOT PROVE", text)
        self.assertIn("bug DETECTOR, not a proof", text)
        self.assertIn("no engine disagreed", text)

    def test_findings_counts_every_kind(self):
        report = tier2.Tier2Report()
        ev = tier2.StreamEvidence(name="x")
        ev.disagreements = [{"metric": "volume"}]
        ev.crashed = {"cadquery": "boom"}
        ev.violations = [{"property": "scale_is_cubic"}]
        ev.gate = {"frep": {"ok": False, "failures": ["not-watertight"]}}
        report.streams.append(ev)
        self.assertEqual(report.findings, 4)
        self.assertEqual(len(report.gate_failures), 1)


if __name__ == "__main__":
    unittest.main()
