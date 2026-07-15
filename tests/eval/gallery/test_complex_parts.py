"""The Tier-2 corpus is what it claims to be, and it stays that way.

These tests are cheap and hermetic: they read the corpus and the pre-existing
corpora and compare them. Nothing here drives a geometry engine -- that is
:mod:`tests.eval.gallery.test_tier2`, which is deliberately restricted to the
two in-process backends (see the note there).

The tests that matter are the ones that would go RED if the corpus quietly
regressed to plates:

* :meth:`test_no_part_has_a_closed_form` -- the entry condition for Tier 2. A
  part whose volume can be written down belongs in ``golden``, where it can be
  proved CORRECT; putting it here would let a weak claim masquerade as a strong
  one.
* :meth:`test_corpus_is_deeper_than_golden` -- golden's deepest stream is 8 ops.
* :meth:`test_novel_pairs_are_actually_novel` -- recomputed against the live
  ``golden`` and gallery corpora, so it cannot rot into a lie.
"""

from __future__ import annotations

import unittest

from harnesscad.core.cisp.ops import Op, _REGISTRY
from harnesscad.eval.gallery import complex_parts
from harnesscad.eval.selftest import golden


class TestCorpusShape(unittest.TestCase):

    def test_corpus_is_not_empty(self):
        self.assertGreaterEqual(len(complex_parts.CORPUS), 10)

    def test_names_are_unique(self):
        names = complex_parts.names()
        self.assertEqual(len(names), len(set(names)))

    def test_every_stream_parses_to_real_ops(self):
        """Every raw dict round-trips through ``parse_op`` into a typed Op."""
        for part in complex_parts.CORPUS:
            ops = part.ops
            self.assertEqual(len(ops), part.depth, part.name)
            for op in ops:
                self.assertIsInstance(op, Op, part.name)
                self.assertIn(op.OP, _REGISTRY, part.name)

    def test_no_part_has_a_closed_form(self):
        """THE ENTRY CONDITION FOR TIER 2.

        Tier 2 buys evidence about shapes we cannot check arithmetically, and it
        pays for it with a weaker claim (agreement, not correctness). A part that
        HAS a closed form must go to ``golden`` instead and be proved exactly --
        keeping it here would launder a strong claim into a weak one for no
        reason.
        """
        for part in complex_parts.CORPUS:
            self.assertFalse(part.closed_form, part.name)

    def test_corpus_is_deeper_than_golden(self):
        """Golden's deepest stream is 8 ops; the monoculture warning named that."""
        golden_max = max(len(p.ops) for p in golden.PARTS)
        mine_max = max(p.depth for p in complex_parts.CORPUS)
        self.assertLessEqual(golden_max, 8)
        self.assertGreater(mine_max, 2 * golden_max)

    def test_a_part_stacks_at_least_eight_ops(self):
        deep = [p for p in complex_parts.CORPUS if p.depth >= 8]
        self.assertGreaterEqual(len(deep), 4)

    def test_get_rejects_an_unknown_name(self):
        with self.assertRaises(KeyError):
            complex_parts.get("no-such-part")


class TestOpCoverage(unittest.TestCase):
    """The 'nothing in the repo had ever done this' claims, re-derived live."""

    def test_novel_ops_are_actually_novel(self):
        novel = complex_parts.novel_ops()
        # mirror / add_instance / mate are in the CISP op set, in the registry,
        # and in NO corpus, golden part or gallery part anywhere in the repo.
        for op in ("mirror", "add_instance", "mate"):
            self.assertIn(op, novel, "%s should be new to this corpus" % op)

    def test_novel_pairs_are_actually_novel(self):
        """Recomputed against the LIVE corpora, so the claim cannot go stale."""
        baseline = complex_parts.baseline_pairs()
        novel = complex_parts.novel_pairs()
        self.assertTrue(novel.isdisjoint(baseline))
        # The pairs that break: rounding a block is fine and drilling a block is
        # fine, but nothing had ever rounded a block and THEN drilled it.
        for pair in (("fillet", "hole"), ("hole", "shell"), ("fillet", "shell"),
                     ("hole", "revolve")):
            self.assertIn(tuple(sorted(pair)), novel,
                          "%s+%s should be a newly-exercised pair" % pair)

    def test_the_corpus_meaningfully_widens_coverage(self):
        cov = complex_parts.coverage_report()
        self.assertGreaterEqual(cov["novel_pair_count"], 25)
        self.assertGreaterEqual(len(cov["novel_ops"]), 4)

    def test_capability_gap_streams_are_present(self):
        """`loft` and `draft` are in the vocabulary and realised by no backend.

        They stay in the corpus BECAUSE they fail. A gap that is measured every
        run is a known gap; a gap that lives in somebody's memory is a surprise
        waiting for a customer.
        """
        gaps = {op for p in complex_parts.CORPUS for op in p.expect_refusal}
        self.assertIn("loft", gaps)
        self.assertIn("draft", gaps)

    def test_declared_gaps_are_in_the_op_registry(self):
        """A 'capability gap' must name a REAL op, or it is just a typo."""
        for op in complex_parts.CAPABILITY_GAPS:
            self.assertIn(op, _REGISTRY, op)

    def test_coverage_report_is_serialisable(self):
        import json
        json.dumps(complex_parts.coverage_report())
        for part in complex_parts.CORPUS:
            json.dumps(part.to_dict())


class TestDeterminism(unittest.TestCase):

    def test_streams_are_stable_across_calls(self):
        """No randomness, no wall clock: the corpus is a literal."""
        a = [(n, [o.to_dict() for o in ops]) for n, ops in complex_parts.streams()]
        b = [(n, [o.to_dict() for o in ops]) for n, ops in complex_parts.streams()]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
