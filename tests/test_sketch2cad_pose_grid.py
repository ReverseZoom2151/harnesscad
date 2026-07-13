"""Tests for the Sketch2CAD camera-pose ID grid."""

import math
import unittest

from harnesscad.domain.vision import sketch2cad_pose_grid as pg


class TestGrid(unittest.TestCase):
    def test_sixty_poses(self):
        self.assertEqual(pg.num_poses(), 60)
        self.assertEqual(pg.N_ELEVATION, 5)
        self.assertEqual(pg.N_AZIMUTH, 12)

    def test_elevation_range(self):
        self.assertEqual(pg.ELEVATIONS[0], -15.0)
        self.assertEqual(pg.ELEVATIONS[-1], 45.0)
        # every 15 deg
        for a, b in zip(pg.ELEVATIONS, pg.ELEVATIONS[1:]):
            self.assertAlmostEqual(b - a, 15.0)

    def test_azimuth_step(self):
        for a, b in zip(pg.AZIMUTHS, pg.AZIMUTHS[1:]):
            self.assertAlmostEqual(b - a, 30.0)
        self.assertNotIn(180.0, pg.AZIMUTHS)  # +180 folds to -180


class TestRoundtrip(unittest.TestCase):
    def test_id_pose_roundtrip(self):
        for pid in range(pg.num_poses()):
            p = pg.id_to_pose(pid)
            self.assertEqual(pg.pose_to_id(p.azimuth, p.elevation), pid)

    def test_ids_unique_and_dense(self):
        seen = {pg.pose_to_id(p.azimuth, p.elevation) for _, p in pg.all_poses()}
        self.assertEqual(seen, set(range(60)))

    def test_azimuth_wrap(self):
        # +180 wraps to the -180 slot
        pid = pg.pose_to_id(180.0, 0.0)
        self.assertEqual(pg.id_to_pose(pid).azimuth, -180.0)

    def test_out_of_range_id(self):
        with self.assertRaises(ValueError):
            pg.id_to_pose(60)

    def test_offgrid_elevation(self):
        with self.assertRaises(ValueError):
            pg.pose_to_id(0.0, 10.0)


class TestViewDirection(unittest.TestCase):
    def test_unit_length(self):
        for pid in range(pg.num_poses()):
            d = pg.view_direction(pid)
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in d)), 1.0, places=6)

    def test_positive_elevation_looks_down(self):
        # a camera at +45 elevation must look downward (negative z view dir)
        pid = pg.pose_to_id(0.0, 45.0)
        d = pg.view_direction(pid)
        self.assertLess(d[2], 0.0)

    def test_azimuth_zero_looks_toward_neg_x(self):
        pid = pg.pose_to_id(0.0, 0.0)
        d = pg.view_direction(pid)
        self.assertAlmostEqual(d[0], -1.0, places=6)
        self.assertAlmostEqual(d[1], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
