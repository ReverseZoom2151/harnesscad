import math
import unittest

from harnesscad.domain.library.gear_train import (
    gear_geometry,
    mesh_pair,
    snap_module,
)


class SnapModuleTests(unittest.TestCase):
    def test_snaps_to_primary(self):
        self.assertEqual(snap_module(2.4), 2.5)
        self.assertEqual(snap_module(1.0), 1.0)

    def test_falls_back_to_secondary(self):
        # 1.8 is closer to secondary 1.75 than within tolerance of primary 2.
        self.assertEqual(snap_module(1.8), 1.75)

    def test_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            snap_module(0.0)


class GearGeometryTests(unittest.TestCase):
    def test_spur_diameters(self):
        g = gear_geometry(2.0, 20)
        self.assertAlmostEqual(g.pitch_diameter, 40.0)
        self.assertAlmostEqual(g.outside_diameter, 44.0)
        self.assertAlmostEqual(g.root_diameter, 35.0)
        self.assertAlmostEqual(g.base_diameter, 40.0 * math.cos(math.radians(20)))

    def test_helical_pitch_diameter_grows(self):
        g = gear_geometry(1.0, 20, helix_angle=15.0)
        self.assertAlmostEqual(g.pitch_diameter, 20.0 / math.cos(math.radians(15)))
        self.assertGreater(g.pitch_diameter, 20.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            gear_geometry(0.0, 20)
        with self.assertRaises(ValueError):
            gear_geometry(1.0, 0)
        with self.assertRaises(ValueError):
            gear_geometry(1.0, 20, helix_angle=95.0)


class MeshPairTests(unittest.TestCase):
    def test_matching_gears_mesh(self):
        a = gear_geometry(2.0, 20)
        b = gear_geometry(2.0, 40)
        r = mesh_pair(a, b)
        self.assertTrue(r.meshes)
        self.assertAlmostEqual(r.gear_ratio, 2.0)
        self.assertAlmostEqual(r.center_distance, 60.0)

    def test_module_mismatch_does_not_mesh(self):
        a = gear_geometry(2.0, 20)
        b = gear_geometry(3.0, 20)
        r = mesh_pair(a, b)
        self.assertFalse(r.meshes)
        self.assertTrue(any("module" in x for x in r.reasons))

    def test_helix_mismatch_does_not_mesh(self):
        a = gear_geometry(1.0, 20, helix_angle=15.0)
        b = gear_geometry(1.0, 30, helix_angle=20.0)
        self.assertFalse(mesh_pair(a, b).meshes)


if __name__ == "__main__":
    unittest.main()
