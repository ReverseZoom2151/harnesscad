"""Unit tests for quality.llmdesign_first_order_performance.

Deterministic, stdlib-only. Verifies the first-order performance scorers
against hand-computed reference values and their monotonicity/logic.
"""

import unittest

from quality import llmdesign_first_order_performance as perf


class ChairTest(unittest.TestCase):

    # A nominal, comfortably-safe chair configuration used as the baseline that
    # individual tests then perturb around one threshold at a time.
    BASE = dict(
        weight_kg=80.0,
        leg_area_m2=4e-4,
        leg_yield_pa=250e6,
        seat_thickness_m=0.02,
        seat_length_m=0.4,
        seat_width_m=0.4,
        seat_bending_strength_pa=40e6,
        back_height_m=0.4,
        back_width_m=0.4,
        back_strength_pa=10e6,
    )

    def test_leg_compressive_stress_hand_value(self):
        # 80 kg / 4 legs * 9.81 = 196.2 N; / 4e-4 m^2 = 490500 Pa.
        sigma = perf.chair_leg_compressive_stress(80.0, 4e-4)
        self.assertAlmostEqual(sigma, 490500.0, places=3)

    def test_leg_zero_area_raises(self):
        with self.assertRaises(ValueError):
            perf.chair_leg_compressive_stress(80.0, 0.0)

    def test_leg_failure_toggles_around_threshold(self):
        sigma = perf.chair_leg_compressive_stress(self.BASE["weight_kg"],
                                                  self.BASE["leg_area_m2"])
        # Just above the computed stress -> safe; just below -> fails.
        safe = perf.will_chair_break(**dict(self.BASE, leg_yield_pa=sigma + 1.0))
        fail = perf.will_chair_break(**dict(self.BASE, leg_yield_pa=sigma - 1.0))
        self.assertFalse(safe.leg_fails)
        self.assertTrue(fail.leg_fails)

    def test_seat_failure_toggles_around_threshold(self):
        sigma = perf.seat_bending_stress(80.0, 0.02, 0.4, 0.4)
        safe = perf.will_chair_break(**dict(self.BASE, seat_bending_strength_pa=sigma + 1.0))
        fail = perf.will_chair_break(**dict(self.BASE, seat_bending_strength_pa=sigma - 1.0))
        self.assertFalse(safe.seat_fails)
        self.assertTrue(fail.seat_fails)

    def test_back_failure_toggles_around_threshold(self):
        sigma = perf.back_stress(80.0, 0.4, 0.4)
        safe = perf.will_chair_break(**dict(self.BASE, back_strength_pa=sigma + 1.0))
        fail = perf.will_chair_break(**dict(self.BASE, back_strength_pa=sigma - 1.0))
        self.assertFalse(safe.back_fails)
        self.assertTrue(fail.back_fails)

    def test_overall_will_break_is_or_of_modes(self):
        # Baseline is safe on every mode.
        res = perf.will_chair_break(**self.BASE)
        self.assertFalse(res.will_break)
        # Force only the back mode to fail: overall must become True.
        only_back = perf.will_chair_break(**dict(self.BASE, back_strength_pa=1.0))
        self.assertTrue(only_back.back_fails)
        self.assertFalse(only_back.leg_fails)
        self.assertFalse(only_back.seat_fails)
        self.assertTrue(only_back.will_break)

    def test_can_support_is_inverse_of_will_break(self):
        res = perf.will_chair_break(**self.BASE)
        self.assertEqual(perf.can_support(**self.BASE), not res.will_break)
        weak = dict(self.BASE, leg_yield_pa=1.0)
        self.assertFalse(perf.can_support(**weak))
        self.assertTrue(perf.will_chair_break(**weak).will_break)

    def test_back_stress_uses_one_third_fraction(self):
        # sigma = (1/3 * 80 * 9.81) / (0.4 * 0.4)
        expected = (80.0 * 9.81 / 3.0) / (0.4 * 0.4)
        self.assertAlmostEqual(perf.back_stress(80.0, 0.4, 0.4), expected, places=6)


class CabinetTest(unittest.TestCase):

    def test_storage_capacity_exact_with_shelf(self):
        # H=1.0 W=0.6 D=0.4 t=0.02, 1 shelf.
        # interior = 0.56*0.36*0.96 = 0.193536
        # shelf    = 0.56*0.36*0.02 = 0.004032
        cap = perf.cabinet_storage_capacity(1.0, 0.6, 0.4, 0.02, num_shelves=1)
        self.assertAlmostEqual(cap, 0.193536 - 0.004032, places=9)

    def test_storage_capacity_no_shelf(self):
        cap = perf.cabinet_storage_capacity(1.0, 0.6, 0.4, 0.02, num_shelves=0)
        self.assertAlmostEqual(cap, 0.193536, places=9)

    def test_storage_capacity_nonpositive_inner_raises(self):
        with self.assertRaises(ValueError):
            # 2*t = 0.6 >= width 0.6 -> inner width non-positive.
            perf.cabinet_storage_capacity(1.0, 0.6, 0.4, 0.3)

    def test_material_cost_exact(self):
        # exterior = 1.0*0.6*0.4 = 0.24 ; interior = 0.193536
        # wall = 0.046464 ; shelf = 0.004032 ; sum = 0.050496 ; *100 = 5.0496
        cost = perf.cabinet_material_cost(1.0, 0.6, 0.4, 0.02, 100.0, num_shelves=1)
        self.assertAlmostEqual(cost, 5.0496, places=6)

    def test_sagulator_hand_value(self):
        # L=1 b=0.3 t=0.02 E=1e10 delta=0.005
        # I = 0.3*0.02^3/12 = 2e-7
        # w = 384*1e10*2e-7*0.005/(5*1) = 768 (per length) ; total = 768*1 = 768 N
        load = perf.shelf_sag_load_capacity(1.0, 0.3, 0.02, 1e10, 0.005)
        self.assertAlmostEqual(load, 768.0, places=6)

    def test_sagulator_monotonic_in_modulus_and_thickness(self):
        base = perf.shelf_sag_load_capacity(1.0, 0.3, 0.02, 1e10, 0.005)
        stiffer = perf.shelf_sag_load_capacity(1.0, 0.3, 0.02, 2e10, 0.005)
        thicker = perf.shelf_sag_load_capacity(1.0, 0.3, 0.03, 1e10, 0.005)
        self.assertGreater(stiffer, base)
        self.assertGreater(thicker, base)

    def test_sagulator_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            perf.shelf_sag_load_capacity(0.0, 0.3, 0.02, 1e10, 0.005)
        with self.assertRaises(ValueError):
            perf.shelf_sag_load_capacity(1.0, 0.3, 0.02, 1e10, 0.0)

    def test_accessibility_shorter_scores_higher(self):
        tall = perf.cabinet_wheelchair_accessibility_score(1.6, 0.5)
        short = perf.cabinet_wheelchair_accessibility_score(0.9, 0.5)
        self.assertGreater(short, tall)

    def test_accessibility_deeper_scores_higher(self):
        shallow = perf.cabinet_wheelchair_accessibility_score(1.0, 0.3)
        deep = perf.cabinet_wheelchair_accessibility_score(1.0, 0.6)
        self.assertGreater(deep, shallow)

    def test_accessibility_bounds_clamped(self):
        # Extreme short+deep saturates high; extreme tall+shallow floors low.
        hi = perf.cabinet_wheelchair_accessibility_score(0.1, 5.0)
        lo = perf.cabinet_wheelchair_accessibility_score(50.0, 0.001)
        self.assertLessEqual(hi, 10.0)
        self.assertGreaterEqual(hi, 0.0)
        self.assertLessEqual(lo, 10.0)
        self.assertGreaterEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 10.0, places=6)
        # Extreme tall + near-zero depth floors close to 0 (depth term ~ 0).
        self.assertLess(lo, 0.05)


class QuadcopterTest(unittest.TestCase):

    def test_hover_time_hand_value(self):
        # 5000 mAh -> 5 Ah ; *0.8 = 4 Ah ; / 20 A = 0.2 h ; *60 = 12 min.
        t = perf.copter_hover_time_min(5000.0, 11.1, 20.0)
        self.assertAlmostEqual(t, 12.0, places=6)

    def test_hover_time_invalid_current_raises(self):
        with self.assertRaises(ValueError):
            perf.copter_hover_time_min(5000.0, 11.1, 0.0)

    def test_max_range_exact(self):
        # 12 min -> 0.2 h ; *30 km/h = 6 km.
        r = perf.copter_max_range_km(12.0, 30.0)
        self.assertAlmostEqual(r, 6.0, places=6)

    def test_max_range_negative_raises(self):
        with self.assertRaises(ValueError):
            perf.copter_max_range_km(-1.0, 30.0)


if __name__ == "__main__":
    unittest.main()
