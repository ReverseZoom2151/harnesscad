"""Tests for gear-ratio rotational driver coupling and train propagation."""

import unittest

from geometry.codetocad2_gear_coupling import (
    EXTERNAL,
    INTERNAL,
    SHAFT,
    GearCoupling,
    GearError,
    GearTrain,
    ratio_from_teeth,
)


class TestGearCoupling(unittest.TestCase):
    def test_ratio_from_teeth(self):
        self.assertEqual(ratio_from_teeth(10, 20), 0.5)
        with self.assertRaises(GearError):
            ratio_from_teeth(0, 10)

    def test_external_mesh_reverses(self):
        coupling = GearCoupling("a", "b", ratio=2.0)
        self.assertEqual(coupling.sign, -1.0)
        self.assertEqual(coupling.drive(1.0), -2.0)
        self.assertEqual(coupling.signed_ratio, -2.0)

    def test_internal_mesh_keeps_sense(self):
        coupling = GearCoupling("a", "b", ratio=2.0, mesh=INTERNAL)
        self.assertEqual(coupling.drive(1.0), 2.0)

    def test_shaft_coupling(self):
        coupling = GearCoupling.on_shaft("a", "b")
        self.assertEqual(coupling.mesh, SHAFT)
        self.assertEqual(coupling.drive(1.5), 1.5)
        with self.assertRaises(GearError):
            GearCoupling("a", "b", ratio=2.0, mesh=SHAFT)

    def test_from_teeth(self):
        coupling = GearCoupling.from_teeth("a", "b", 10, 40)
        self.assertEqual(coupling.drive(4.0), -1.0)

    def test_back_drive_inverts(self):
        coupling = GearCoupling.from_teeth("a", "b", 12, 30)
        self.assertAlmostEqual(coupling.back_drive(coupling.drive(0.7)), 0.7, places=12)

    def test_inverse(self):
        inverse = GearCoupling("a", "b", ratio=4.0).inverse()
        self.assertEqual(inverse.driver, "b")
        self.assertEqual(inverse.driven, "a")
        self.assertAlmostEqual(inverse.ratio, 0.25, places=12)

    def test_expression_matches_upstream_form(self):
        coupling = GearCoupling("gearA", "gearB", ratio=3.0)
        self.assertEqual(coupling.expression(), "-3.0 * gearARotation")

    def test_validation(self):
        with self.assertRaises(GearError):
            GearCoupling("a", "a")
        with self.assertRaises(GearError):
            GearCoupling("a", "b", mesh="magnetic")
        with self.assertRaises(GearError):
            GearCoupling("a", "b", ratio=-1.0)

    def test_equality_and_hash(self):
        self.assertEqual(GearCoupling("a", "b"), GearCoupling("a", "b"))
        self.assertEqual(hash(GearCoupling("a", "b")), hash(GearCoupling("a", "b")))


class TestGearTrain(unittest.TestCase):
    def simple_train(self):
        # 10 -> 20 -> 40 teeth, external meshes.
        train = GearTrain()
        train.mesh("a", "b", 10, 20)
        train.mesh("b", "c", 20, 40)
        return train

    def test_structure(self):
        train = self.simple_train()
        self.assertEqual(train.gears(), ["a", "b", "c"])
        self.assertEqual(train.roots(), ["a"])
        self.assertEqual(train.leaves(), ["c"])
        self.assertEqual(len(train), 2)

    def test_propagate(self):
        train = self.simple_train()
        angles = train.propagate("a", 8.0)
        self.assertEqual(angles["a"], 8.0)
        self.assertEqual(angles["b"], -4.0)
        self.assertEqual(angles["c"], 2.0)

    def test_propagate_from_middle(self):
        train = self.simple_train()
        angles = train.propagate("b", 4.0)
        self.assertNotIn("a", angles)
        self.assertEqual(angles["c"], -2.0)

    def test_effective_ratio(self):
        train = self.simple_train()
        self.assertEqual(train.effective_ratio("a"), 1.0)
        self.assertEqual(train.effective_ratio("b"), -0.5)
        self.assertEqual(train.effective_ratio("c"), 0.25)

    def test_idler_cancels_from_magnitude_but_not_sign(self):
        # driver 10 -> idler 15 -> output 30: ratio |1/3|, sign +.
        train = GearTrain()
        train.mesh("driver", "idler", 10, 15)
        train.mesh("idler", "output", 15, 30)
        direct = GearTrain()
        direct.mesh("driver", "output", 10, 30)
        self.assertAlmostEqual(abs(train.effective_ratio("output")), 1.0 / 3.0, places=12)
        self.assertGreater(train.effective_ratio("output"), 0.0)
        self.assertLess(direct.effective_ratio("output"), 0.0)

    def test_compound_train_via_shaft(self):
        # stage 1: 10 -> 30 (1/3); shaft carries 12 -> 36 (1/3). Total 1/9.
        train = GearTrain()
        train.mesh("in", "s1", 10, 30)
        train.couple_shaft("s1", "s2")
        train.mesh("s2", "out", 12, 36)
        self.assertAlmostEqual(train.effective_ratio("out"), 1.0 / 9.0, places=12)
        angles = train.propagate("in", 9.0)
        self.assertAlmostEqual(angles["out"], 1.0, places=12)
        self.assertAlmostEqual(angles["s1"], -3.0, places=12)
        self.assertAlmostEqual(angles["s2"], -3.0, places=12)

    def test_branching(self):
        train = GearTrain()
        train.mesh("a", "b", 10, 10)
        train.mesh("a", "c", 10, 20)
        angles = train.propagate("a", 2.0)
        self.assertEqual(angles["b"], -2.0)
        self.assertEqual(angles["c"], -1.0)

    def test_one_driver_per_gear(self):
        train = self.simple_train()
        with self.assertRaises(GearError):
            train.mesh("a", "c", 10, 40)

    def test_cycle_rejected(self):
        train = self.simple_train()
        with self.assertRaises(GearError):
            train.mesh("c", "a", 40, 10)
        self.assertEqual(len(train), 2)

    def test_torques_conserve_power(self):
        train = self.simple_train()
        torques = train.torques("a", 1.0)
        speeds = train.speeds("a", 1.0)
        for name in train.gears():
            self.assertAlmostEqual(
                torques[name] * speeds[name], 1.0, places=12
            )

    def test_path_and_reduction(self):
        train = self.simple_train()
        path = train.path_to("c")
        self.assertEqual([c.driven for c in path], ["b", "c"])
        self.assertEqual(train.reduction("b", "c"), -0.5)
        with self.assertRaises(GearError):
            train.reduction("c", "b")

    def test_driver_expressions(self):
        train = self.simple_train()
        expressions = train.drivers_expressions()
        self.assertEqual(expressions["b"], "-0.5 * aRotation")
        self.assertEqual(list(expressions), ["b", "c"])

    def test_unknown_gear(self):
        train = self.simple_train()
        with self.assertRaises(GearError):
            train.propagate("zzz", 1.0)
        with self.assertRaises(GearError):
            train.torques("zzz", 1.0)

    def test_determinism(self):
        train = self.simple_train()
        self.assertEqual(train.propagate("a", 1.0), train.propagate("a", 1.0))
        self.assertEqual(train.couplings[0].mesh, EXTERNAL)


if __name__ == "__main__":
    unittest.main()
