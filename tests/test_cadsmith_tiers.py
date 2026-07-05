import unittest

from generation.cadsmith_tiers import (
    classify, tier_spec, op_count_tier, TIERS, T1, T2, T3,
)


class TestTierSpecs(unittest.TestCase):
    def test_three_tiers(self):
        self.assertEqual([t.tier for t in TIERS], ["T1", "T2", "T3"])

    def test_bands(self):
        self.assertEqual((T1.op_min, T1.op_max), (1, 3))
        self.assertEqual((T2.op_min, T2.op_max), (3, 8))
        self.assertEqual((T3.op_min, T3.op_max), (5, 15))

    def test_lookup(self):
        self.assertIs(tier_spec("T2"), T2)
        with self.assertRaises(KeyError):
            tier_spec("T9")


class TestClassify(unittest.TestCase):
    def test_primitive_is_t1(self):
        self.assertEqual(classify(["box"]).tier, "T1")
        self.assertEqual(classify(["cylinder", "extrude"]).tier, "T1")

    def test_boolean_is_t2(self):
        self.assertEqual(classify(["box", "cut", "hole"]).tier, "T2")

    def test_complex_is_t3(self):
        self.assertEqual(classify(["box", "loft", "shell"]).tier, "T3")

    def test_complex_dominates_boolean(self):
        c = classify(["box", "cut", "sweep"])
        self.assertEqual(c.tier, "T3")

    def test_many_primitives_promote_to_t2(self):
        # 5 primitive ops, no boolean/complex -> exceeds T1 band.
        c = classify(["box", "cylinder", "cone", "torus", "prism"])
        self.assertEqual(c.tier, "T2")

    def test_op_count_recorded(self):
        self.assertEqual(classify(["box", "cut", "hole"]).op_count, 3)

    def test_reason_present(self):
        self.assertTrue(classify(["loft"]).reason)


class TestOpCountTier(unittest.TestCase):
    def test_unambiguous_low(self):
        self.assertEqual(op_count_tier(1), "T1")
        self.assertEqual(op_count_tier(2), "T1")

    def test_unambiguous_high(self):
        self.assertEqual(op_count_tier(9), "T3")   # only T3 spans 9

    def test_ambiguous_none(self):
        self.assertIsNone(op_count_tier(5))        # both T2 and T3


if __name__ == "__main__":
    unittest.main()
