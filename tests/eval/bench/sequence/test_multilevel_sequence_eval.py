import unittest

from harnesscad.eval.bench.sequence import multilevel_sequence_eval as e
from harnesscad.domain.reconstruction.tokens import image2cadseq as g


def _cylinder(r=0.5, d=0.5):
    return g.build_feature_matrix(
        [g.add_sketch(0), g.add_circle(0.0, 0.0, r), g.add_extrude(d)])


def _triprism():
    return g.build_feature_matrix(
        [g.add_sketch(0), g.add_line(0.5, 0.0), g.add_line(0.5, 0.5),
         g.add_line(0.0, 0.0), g.add_extrude(0.5)])


class TestLevenshtein(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(e.levenshtein([1, 2, 3], [1, 2, 3]), 0)

    def test_substitution(self):
        self.assertEqual(e.levenshtein([1, 2, 3], [1, 9, 3]), 1)

    def test_insertion_deletion(self):
        self.assertEqual(e.levenshtein([1, 2], [1, 2, 3]), 1)
        self.assertEqual(e.levenshtein([1, 2, 3], [1, 3]), 1)

    def test_classic_kitten_sitting(self):
        self.assertEqual(e.levenshtein("kitten", "sitting"), 3)


class TestH1Sequence(unittest.TestCase):
    def test_acp_perfect_when_identical(self):
        cyl = _cylinder()
        self.assertEqual(e.accuracy_cad_programs([cyl], [cyl]), 1.0)

    def test_acp_zero_when_params_differ_beyond_tolerance(self):
        pred = _cylinder(r=0.5, d=0.5)
        gt = _cylinder(r=0.5, d=-0.9)
        self.assertEqual(e.accuracy_cad_programs([pred], [gt], eta=3), 0.0)

    def test_acp_tolerance_absorbs_small_diff(self):
        pred = _cylinder(d=0.5)
        gt = _cylinder(d=0.5)
        self.assertEqual(e.accuracy_cad_programs([pred], [gt], eta=3), 1.0)

    def test_asot_matches_op_type_sequence(self):
        cyl, tri = _cylinder(), _triprism()
        self.assertEqual(e.accuracy_seq_op_types([cyl], [cyl]), 1.0)
        self.assertEqual(e.accuracy_seq_op_types([cyl], [tri]), 0.0)

    def test_edsot_zero_for_identical(self):
        cyl = _cylinder()
        self.assertEqual(e.edit_distance_seq_op_types([cyl], [cyl]), 0.0)

    def test_edsot_positive_for_different(self):
        self.assertGreater(
            e.edit_distance_seq_op_types([_cylinder()], [_triprism()]), 0.0)


class TestH2(unittest.TestCase):
    def test_aot_full_when_types_match(self):
        cyl = _cylinder()
        self.assertEqual(e.accuracy_op_types([cyl], [cyl]), 1.0)

    def test_aot_partial(self):
        # cylinder vs triprism share only some op-type positions
        val = e.accuracy_op_types([_cylinder()], [_triprism()])
        self.assertGreater(val, 0.0)
        self.assertLess(val, 1.0)

    def test_ap1_perfect_for_identical(self):
        cyl = _cylinder()
        self.assertEqual(e.accuracy_parameter_ordered([cyl], [cyl]), 1.0)

    def test_ap1_penalises_param_error(self):
        pred = _cylinder(r=0.1)
        gt = _cylinder(r=0.9)
        self.assertLess(e.accuracy_parameter_ordered([pred], [gt], eta=3), 1.0)


class TestH3(unittest.TestCase):
    def test_tanimoto_identical(self):
        self.assertAlmostEqual(e.tanimoto([1, 2, 3], [1, 2, 3]), 1.0)

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(e.cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_cosine_identical(self):
        self.assertAlmostEqual(e.cosine_similarity([1, 2], [2, 4]), 1.0)

    def test_msot_identical_programs(self):
        cyl = _cylinder()
        m = e.multiset_similarity_op_types([cyl], [cyl])
        self.assertAlmostEqual(m["tc"], 1.0)
        self.assertAlmostEqual(m["cs"], 1.0)

    def test_msot_order_agnostic(self):
        # same op-type multiset, permuted order -> similarity 1.0
        a = g.build_feature_matrix([g.add_sketch(0), g.add_line(0.1, 0.1),
                                    g.add_circle(0.0, 0.0, 0.5), g.add_extrude(0.5)])
        b = g.build_feature_matrix([g.add_sketch(0), g.add_circle(0.0, 0.0, 0.5),
                                    g.add_line(0.1, 0.1), g.add_extrude(0.5)])
        m = e.multiset_similarity_op_types([a], [b])
        self.assertAlmostEqual(m["tc"], 1.0)

    def test_ap2_unordered_matches_repeated_types(self):
        cyl = _cylinder()
        self.assertEqual(e.accuracy_parameter_unordered([cyl], [cyl]), 1.0)


class TestBaselineAndReport(unittest.TestCase):
    def test_random_baseline_formula(self):
        self.assertAlmostEqual(e.random_baseline_ap1(0), 256 / 65536.0)
        # matches (-eta^2 + 511 eta + 256)/65536
        self.assertAlmostEqual(e.random_baseline_ap1(3),
                               (-9 + 1533 + 256) / 65536.0)

    def test_baseline_monotonic_region(self):
        self.assertLess(e.random_baseline_ap1(0), e.random_baseline_ap1(10))

    def test_evaluate_report_structure(self):
        cyl = _cylinder()
        rep = e.evaluate([cyl], [cyl])
        self.assertEqual(set(rep["H1"]), {"ACP", "ASOT", "EDSOT"})
        self.assertEqual(set(rep["H2"]), {"AOT", "AP1"})
        self.assertEqual(set(rep["H3"]), {"MSOT_TC", "MSOT_CS", "AP2"})
        self.assertEqual(rep["H1"]["ACP"], 1.0)

    def test_evaluate_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            e.evaluate([_cylinder()], [])


if __name__ == "__main__":
    unittest.main()
