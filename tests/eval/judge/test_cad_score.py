import unittest

from harnesscad.eval.judge.cad_score import (
    cad_score,
    renormalize_edit_shape,
    shape_similarity,
    topology_match,
)


class CadScoreTests(unittest.TestCase):
    def test_topology_match_identical(self):
        self.assertAlmostEqual(topology_match((1, 2, 0), (1, 2, 0)), 1.0)

    def test_topology_worked_example_b0(self):
        # GT (1,0,0) vs (2,0,0) -> (2/3)**2 = 0.444...
        self.assertAlmostEqual(topology_match((2, 0, 0), (1, 0, 0)), (2 / 3) ** 2, places=6)

    def test_topology_worked_example_b1(self):
        # GT (1,2,0) vs (1,4,0) -> (3/5)**2 = 0.36
        self.assertAlmostEqual(topology_match((1, 4, 0), (1, 2, 0)), 0.36, places=6)

    def test_topology_is_symmetric_in_min_max(self):
        self.assertAlmostEqual(
            topology_match((1, 4, 0), (1, 2, 0)),
            topology_match((1, 2, 0), (1, 4, 0)),
        )

    def test_topology_product_collapses_on_one_wrong_axis(self):
        s = topology_match((2, 4, 1), (1, 2, 0))
        self.assertLess(s, 0.36)

    def test_topology_requires_triples(self):
        with self.assertRaises(ValueError):
            topology_match((1, 2), (1, 2, 0))

    def test_shape_similarity_mean(self):
        self.assertAlmostEqual(shape_similarity(0.8, 0.6), 0.7)

    def test_shape_similarity_bounds(self):
        with self.assertRaises(ValueError):
            shape_similarity(1.2, 0.5)

    def test_renormalize_maps_noop_to_zero(self):
        self.assertAlmostEqual(renormalize_edit_shape(0.9, 0.9), 0.0)

    def test_renormalize_perfect_stays_one(self):
        self.assertAlmostEqual(renormalize_edit_shape(1.0, 0.9), 1.0)

    def test_renormalize_below_baseline_clamps_zero(self):
        self.assertAlmostEqual(renormalize_edit_shape(0.5, 0.9), 0.0)

    def test_renormalize_midpoint(self):
        self.assertAlmostEqual(renormalize_edit_shape(0.95, 0.9), 0.5)

    def test_cad_score_validity_gate(self):
        b = cad_score(is_valid=False, shape=1.0, interface=1.0, topology=1.0)
        self.assertEqual(b.cad_score, 0.0)
        self.assertFalse(b.is_valid)

    def test_cad_score_generation_weights(self):
        b = cad_score(is_valid=True, shape=1.0, interface=0.0, topology=0.0)
        self.assertAlmostEqual(b.cad_score, 0.4)
        b2 = cad_score(is_valid=True, shape=0.0, interface=0.0, topology=1.0)
        self.assertAlmostEqual(b2.cad_score, 0.2)

    def test_cad_score_editing_weights(self):
        b = cad_score(is_valid=True, shape=1.0, interface=0.0, topology=0.0, editing=True)
        self.assertAlmostEqual(b.cad_score, 0.6)

    def test_cad_score_editing_noop_cap(self):
        # A no-op edit: renormalised shape = 0, interface/topology raw high.
        s = renormalize_edit_shape(0.9, 0.9)
        b = cad_score(is_valid=True, shape=s, interface=1.0, topology=1.0, editing=True)
        self.assertAlmostEqual(b.cad_score, 0.4)  # 0.3 + 0.1 cap

    def test_interface_example_from_docs(self):
        # cad_score = 0.4*0.89 + 0.4*0.0 + 0.2*1.0 = 0.556
        b = cad_score(is_valid=True, shape=0.89, interface=0.0, topology=1.0)
        self.assertAlmostEqual(b.cad_score, 0.556, places=6)


if __name__ == "__main__":
    unittest.main()
