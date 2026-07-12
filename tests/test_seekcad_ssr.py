import unittest

from geometry.seekcad_ssr import (
    BOOLEAN_OPS,
    REFINEMENT_FEATURES,
    SKETCH_FEATURES,
    Refinement,
    SSRModel,
    SSRTriple,
)


class TestRefinement(unittest.TestCase):
    def test_valid(self):
        r = Refinement("fillet", 2.0, ("START:e1",))
        self.assertEqual(r.kind, "fillet")
        self.assertEqual(r.entities, ("START:e1",))

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            Refinement("mirror", 1.0)

    def test_nonpositive_param(self):
        with self.assertRaises(ValueError):
            Refinement("chamfer", 0.0)


class TestSSRTriple(unittest.TestCase):
    def test_command_count(self):
        t = SSRTriple(4, "extrude", [Refinement("fillet", 1.0), Refinement("shell", 2.0)])
        # 1 sketch + 4 curves + 1 feature + 2 refinements
        self.assertEqual(t.command_count(), 8)

    def test_empty_refinements(self):
        t = SSRTriple(3, "revolve")
        self.assertEqual(t.command_count(), 5)
        self.assertEqual(t.refinements, [])

    def test_bad_feature(self):
        with self.assertRaises(ValueError):
            SSRTriple(2, "loft")

    def test_zero_curves(self):
        with self.assertRaises(ValueError):
            SSRTriple(0, "extrude")

    def test_to_dict(self):
        t = SSRTriple(2, "extrude", [Refinement("chamfer", 0.5, ("END:f0",))])
        d = t.to_dict()
        self.assertEqual(d["feature"], "extrude")
        self.assertEqual(d["sketch"]["curves"], 2)
        self.assertEqual(d["refinements"][0]["entities"], ["END:f0"])


class TestSSRModel(unittest.TestCase):
    def setUp(self):
        self.t1 = SSRTriple(4, "extrude", [Refinement("fillet", 1.0)])
        self.t2 = SSRTriple(1, "revolve")

    def test_ops_length_mismatch(self):
        with self.assertRaises(ValueError):
            SSRModel([self.t1, self.t2], [])

    def test_bad_op(self):
        with self.assertRaises(ValueError):
            SSRModel([self.t1, self.t2], ["Merge"])

    def test_empty_model(self):
        with self.assertRaises(ValueError):
            SSRModel([], [])

    def test_single_triple(self):
        m = SSRModel([self.t1], [])
        self.assertEqual(len(m), 1)
        # 1 + 4 + 1 + 1 = 7 commands, no ops
        self.assertEqual(m.command_count(), 7)

    def test_command_count_with_ops(self):
        m = SSRModel([self.t1, self.t2], ["Cut"])
        # t1=7, t2=1+1+1=3, +1 op = 11
        self.assertEqual(m.command_count(), 11)

    def test_complexity_bands(self):
        small = SSRModel([SSRTriple(1, "extrude")], [])
        self.assertEqual(small.complexity_band(), "Low")
        mid_triples = [SSRTriple(10, "extrude") for _ in range(3)]
        mid = SSRModel(mid_triples, ["Union", "Union"])
        # each triple 1+10+1=12 -> 36 + 2 ops = 38 -> Medium
        self.assertEqual(mid.command_count(), 38)
        self.assertEqual(mid.complexity_band(), "Medium")
        big_triples = [SSRTriple(25, "extrude") for _ in range(3)]
        big = SSRModel(big_triples, ["Union", "Union"])
        # each triple 1+25+1=27 -> 81 + 2 ops = 83 -> High
        self.assertEqual(big.command_count(), 83)
        self.assertEqual(big.complexity_band(), "High")

    def test_refinement_kinds_order(self):
        m = SSRModel(
            [
                SSRTriple(1, "extrude", [Refinement("shell", 1.0), Refinement("fillet", 1.0)]),
                SSRTriple(1, "extrude", [Refinement("fillet", 2.0), Refinement("chamfer", 1.0)]),
            ],
            ["Union"],
        )
        self.assertEqual(m.refinement_kinds(), ("shell", "fillet", "chamfer"))

    def test_to_json_dict(self):
        m = SSRModel([self.t1, self.t2], ["Intersect"])
        d = m.to_json_dict()
        self.assertEqual(d["paradigm"], "SSR")
        self.assertEqual(d["n_units"], 2)
        self.assertNotIn("op", d["sequence"][0])
        self.assertEqual(d["sequence"][1]["op"], "Intersect")

    def test_constants(self):
        self.assertIn("extrude", SKETCH_FEATURES)
        self.assertIn("shell", REFINEMENT_FEATURES)
        self.assertIn("Union", BOOLEAN_OPS)


if __name__ == "__main__":
    unittest.main()
