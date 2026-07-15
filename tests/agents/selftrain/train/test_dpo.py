"""DPO is the least forgiving stage and it runs on a corpus with a MEASURED
false-positive problem, so the two guards that keep it honest are pure python and
tested without a GPU: it refuses an invented label_smoothing (the robust loss is
pointless with a guessed epsilon), and it flags the sub-200-pair count the recipe
names as the binding constraint. The trainer body SKIPs cleanly when the training
stack is absent."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import train
from harnesscad.agents.selftrain.train import dpo as D


class TestDPOGuards(unittest.TestCase):

    def test_min_pairs_constant_matches_recipe(self):
        # The recipe's stop condition is "fewer than ~200 separating pairs is not a
        # DPO dataset". The constant must encode that, not a softer number.
        self.assertEqual(D.MIN_PAIRS_FOR_DPO, 200)

    def test_label_smoothing_is_required_and_range_checked(self):
        # A flip-probability >= 0.5 inverts the preference; a negative one is
        # nonsense. Either must raise BEFORE any GPU work -- and the check lives
        # above require(), so it is reachable on a core-only box.
        for bad in (-0.01, 0.5, 0.9, 1.0):
            with self.assertRaises(ValueError):
                D.train_dpo("x.jsonl", "out", label_smoothing=bad)

    def test_valid_label_smoothing_passes_range_check_then_hits_require(self):
        # A valid epsilon must get PAST the range check; on a core-only machine the
        # next thing it hits is require(), which raises RuntimeError (not
        # ValueError). Where the stack is present this would proceed to load a model,
        # which we do not do in a unit test -- so assert only the core-only path.
        if train.MISSING:
            with self.assertRaises(RuntimeError):
                D.train_dpo("x.jsonl", "out", label_smoothing=0.1)


class TestDPOResult(unittest.TestCase):

    def test_result_serialises_and_carries_pair_warning_shape(self):
        res = D.DPOResult(records=66, briefs=6, label_smoothing=0.1,
                          pair_count_ok=False,
                          warnings=["only 66 separating pairs"])
        d = res.to_dict()
        self.assertEqual(d["stage"], "dpo-robust")
        self.assertEqual(d["loss_type"], "robust")
        self.assertFalse(d["pair_count_ok"])
        self.assertTrue(d["warnings"])


if __name__ == "__main__":
    unittest.main()
