"""Every few-shot exemplar must BUILD. A worked example that does not build is a
false diagnostic that fires on every single brief."""

from __future__ import annotations

import unittest

from harnesscad.agents.context import cisp_exemplars as bank
from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.frep import FRepBackend


class TestExemplarsAreVerified(unittest.TestCase):
    def test_the_bank_is_not_empty(self):
        self.assertGreaterEqual(len(bank.EXEMPLARS), 3)

    def test_every_exemplar_parses_and_builds_on_the_real_kernel(self):
        for ex in bank.EXEMPLARS:
            with self.subTest(exemplar=ex.name):
                ops = [parse_op(dict(o)) for o in ex.ops]
                session = HarnessSession(FRepBackend())
                result = session.apply_ops(ops)
                self.assertTrue(
                    result.ok,
                    "exemplar %r does not build: rejected %r" % (
                        ex.name, result.rejected))
                self.assertEqual(result.applied, len(ops))


class TestSelection(unittest.TestCase):
    def test_selection_is_deterministic(self):
        brief = "A round flange, 80 mm diameter, 8 mm thick, with a 30 mm bore."
        a = [e.name for e in bank.select(brief, 3)]
        b = [e.name for e in bank.select(brief, 3)]
        self.assertEqual(a, b)

    def test_selection_prefers_the_exemplar_that_tiles_the_brief(self):
        chosen = [e.name for e in bank.select(
            "A round flange: an 80 mm diameter disc with bolt holes on a bolt "
            "circle.", 1)]
        self.assertEqual(chosen, ["round_flange_bolt_circle"])

    def test_k_is_honoured_and_never_over_delivers(self):
        self.assertEqual(len(bank.select("a plate", 2)), 2)
        self.assertEqual(bank.select("a plate", 0), [])

    def test_an_empty_brief_still_yields_a_format_demonstration(self):
        # The failure few-shot fixes is FORMAT, so a brief that tiles nothing must
        # still get worked examples rather than an empty block.
        self.assertEqual(len(bank.select("", 3)), 3)

    def test_few_shot_block_renders_json_ops(self):
        block = bank.few_shot_block("a 50x30 plate with rounded corners", 2)
        self.assertIn("WORKED EXAMPLES", block)
        self.assertIn('"op": "extrude"', block)


if __name__ == "__main__":
    unittest.main()
