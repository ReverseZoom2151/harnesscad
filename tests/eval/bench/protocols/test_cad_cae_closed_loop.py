import math
import unittest

from harnesscad.eval.bench.protocols import cad_cae_closed_loop as cl


class TestExtractor(unittest.TestCase):
    def test_displacement_magnitude(self):
        self.assertAlmostEqual(cl.displacement_magnitude((3.0, 4.0, 0.0)), 5.0)

    def test_max_displacement(self):
        self.assertAlmostEqual(
            cl.max_displacement([(1.0, 0.0, 0.0), (3.0, 4.0, 0.0)]), 5.0)

    def test_max_displacement_empty(self):
        self.assertEqual(cl.max_displacement([]), 0.0)

    def test_von_mises_uniaxial(self):
        # Pure uniaxial stress sxx=100 -> von Mises = 100
        self.assertAlmostEqual(cl.von_mises((100.0, 0, 0, 0, 0, 0)), 100.0)

    def test_von_mises_pure_shear(self):
        # Pure shear txy=t -> von Mises = sqrt(3)*t
        self.assertAlmostEqual(cl.von_mises((0, 0, 0, 10.0, 0, 0)),
                               math.sqrt(3) * 10.0)

    def test_von_mises_hydrostatic_is_zero(self):
        # Hydrostatic stress produces zero von Mises
        self.assertAlmostEqual(cl.von_mises((50.0, 50.0, 50.0, 0, 0, 0)), 0.0)

    def test_von_mises_bad_len(self):
        with self.assertRaises(ValueError):
            cl.von_mises((1.0, 2.0, 3.0))

    def test_max_von_mises(self):
        self.assertAlmostEqual(
            cl.max_von_mises([(10.0, 0, 0, 0, 0, 0), (100.0, 0, 0, 0, 0, 0)]),
            100.0)


class TestCost(unittest.TestCase):
    def test_mass_price_chain(self):
        # rho=7900, V=0.001 m3, price=6 -> 47.4
        self.assertAlmostEqual(cl.material_cost(0.001, 7900.0, 6.0), 47.4)


class TestFeasibility(unittest.TestCase):
    def test_all_satisfied(self):
        f = cl.evaluate_feasibility(50.0, 100.0, 8.0,
                                    delta=60.0, sigma_allow=167.0, kappa=10.0)
        self.assertTrue(f.feasible)
        self.assertEqual(f.n_satisfied, 3)

    def test_partial(self):
        f = cl.evaluate_feasibility(70.0, 100.0, 8.0,
                                    delta=60.0, sigma_allow=167.0, kappa=10.0)
        self.assertFalse(f.displacement_ok)
        self.assertEqual(f.n_satisfied, 2)


class TestReward(unittest.TestCase):
    def test_constraint_reward_levels(self):
        self.assertEqual(cl.constraint_reward(0), 0.0)
        self.assertEqual(cl.constraint_reward(1), 0.2)
        self.assertEqual(cl.constraint_reward(2), 0.5)
        self.assertEqual(cl.constraint_reward(3), 1.0)

    def test_stop_penalty_clamped(self):
        # 10 events * 0.02 = 0.20, clamped to lam_max 0.10
        self.assertAlmostEqual(cl.feasible_then_stop_penalty(10), -0.10)
        self.assertAlmostEqual(cl.feasible_then_stop_penalty(2), -0.04)
        self.assertEqual(cl.feasible_then_stop_penalty(0), 0.0)

    def test_format_reward(self):
        self.assertEqual(cl.format_reward(True), 0.10)
        self.assertEqual(cl.format_reward(False), 0.0)

    def test_total_feasible_immediate_stop(self):
        # N=3, no extra events, consistent JSON -> 1.0 + 0 + 0.1
        self.assertAlmostEqual(
            cl.total_reward(3, 0, True), 1.10)

    def test_total_no_feasible_no_stop_penalty(self):
        # feasible never reached -> Rstop=0
        self.assertAlmostEqual(
            cl.total_reward(1, 5, False, feasible_reached=False), 0.2)


if __name__ == "__main__":
    unittest.main()
