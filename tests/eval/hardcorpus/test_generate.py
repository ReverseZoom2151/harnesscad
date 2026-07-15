"""Every generated reference must BUILD and pass BOTH its own oracle and the gate.

A brief whose own reference solution does not build is the exact bug that
contaminated v1 (the shell briefs probed a point on the outer face and only passed
because of a backend bug). This is asserted per brief, on both the dev seed and the
held-out seed, because the two are the same factories with different numbers and a
factory that is sound at one seed and not another is a factory with a bug.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.corpus.spec import Split
from harnesscad.eval.hardcorpus import generate, oracle
from harnesscad.io import gate


class TestGenerate(unittest.TestCase):

    def test_two_prompt_styles_per_part(self):
        briefs = generate.all_briefs(1, Split.DEV)
        self.assertEqual(len(briefs), 2 * len(generate.FACTORIES))
        plains = [b for b in briefs if b.id.endswith("_plain")]
        procs = [b for b in briefs if b.id.endswith("_proc")]
        self.assertEqual(len(plains), len(procs))
        # a plain and its proc twin share every ground-truth number.
        for p, q in zip(plains, procs):
            self.assertNotEqual(p.text, q.text)
            self.assertEqual(p.volume, q.volume)
            self.assertEqual(p.bbox, q.bbox)
            self.assertEqual(p.reference, q.reference)

    def test_every_dev_reference_solves_its_own_oracle(self):
        for b in generate.all_briefs(1, Split.DEV):
            s = oracle.grade_reference(b)
            self.assertTrue(s.built, "%s: reference did not build: %s"
                            % (b.id, s.reasons))
            self.assertTrue(s.solved, "%s: reference fails its own oracle: %s"
                            % (b.id, s.reasons))

    def test_every_heldout_reference_solves_its_own_oracle(self):
        for b in generate.all_briefs(7919, Split.HELDOUT):
            s = oracle.grade_reference(b)
            self.assertTrue(s.solved, "%s: %s" % (b.id, s.reasons))

    def test_every_reference_passes_the_gate(self):
        for b in generate.all_briefs(1, Split.DEV):
            built = oracle.occt.build(b.reference)
            self.assertTrue(built, "%s did not build" % b.id)
            report = gate.check(built.engine, source=built.engine)
            self.assertTrue(report.ok, "%s failed the gate: %s"
                            % (b.id, [f.check for f in report.failures]))

    def test_deep_chains_are_deep(self):
        briefs = {b.id: b for b in generate.all_briefs(1, Split.DEV)}
        for key in ("deep_chamfer_holes", "deep_fillet_holes"):
            b = briefs["gen_%s_s1_plain" % key]
            self.assertGreaterEqual(len(b.reference), 10,
                                    "%s is only %d ops -- not a deep chain"
                                    % (key, len(b.reference)))

    def test_the_generator_is_deterministic(self):
        a = generate.all_briefs(1, Split.DEV)
        b = generate.all_briefs(1, Split.DEV)
        self.assertEqual([x.to_dict() for x in a], [x.to_dict() for x in b])

    def test_dev_and_heldout_differ(self):
        dev = {b.text for b in generate.all_briefs(1, Split.DEV)}
        held = {b.text for b in generate.all_briefs(7919, Split.HELDOUT)}
        self.assertEqual(dev & held, set(),
                         "dev and held-out share a brief -- the split is not held out")

    def test_draft_is_dropped_and_named(self):
        self.assertIn("draft", generate.DROPPED_OPS)
        self.assertNotIn("draft", generate.FACTORIES)


if __name__ == "__main__":
    unittest.main()
