import unittest
from math import sqrt

from harnesscad.eval.bench.geometry.vlmcadcode_metrics import (
    UNIT_CUBE_DIAGONAL, bounding_box, normalize_unit_cube,
    point_cloud_distance, hausdorff_distance, iogt, evaluate_object, aggregate,
)


def _cube(shift=0.0, scale=1.0):
    pts = []
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                pts.append((x * scale + shift, y * scale + shift, z * scale + shift))
    return pts


class TestNormalization(unittest.TestCase):
    def test_normalize_fits_unit_cube(self):
        norm = normalize_unit_cube(_cube(shift=5.0, scale=4.0))
        lo, hi = bounding_box(norm)
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (1.0, 1.0, 1.0))

    def test_degenerate_cloud_maps_to_origin(self):
        norm = normalize_unit_cube([(2, 2, 2), (2, 2, 2)])
        self.assertEqual(norm, [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)])

    def test_empty_cloud(self):
        self.assertEqual(normalize_unit_cube([]), [])


class TestDistances(unittest.TestCase):
    def test_identical_clouds_zero_distance(self):
        c = _cube()
        self.assertAlmostEqual(point_cloud_distance(c, c), 0.0)
        self.assertAlmostEqual(hausdorff_distance(c, c), 0.0)

    def test_point_cloud_distance_symmetric(self):
        a = _cube()
        b = [(p[0] + 0.1, p[1], p[2]) for p in a]
        self.assertAlmostEqual(point_cloud_distance(a, b),
                               point_cloud_distance(b, a))

    def test_hausdorff_is_max_nn(self):
        a = [(0, 0, 0)]
        b = [(0, 0, 0), (3, 0, 0)]
        # directed a->b = 0, b->a = 3 -> max = 3
        self.assertAlmostEqual(hausdorff_distance(a, b), 3.0)

    def test_pcd_known_value(self):
        a = [(0, 0, 0)]
        b = [(2, 0, 0)]
        # forward = 2/(2*1)=1, backward=2/(2*1)=1 -> 2
        self.assertAlmostEqual(point_cloud_distance(a, b), 2.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            point_cloud_distance([], [(0, 0, 0)])


class TestIoGT(unittest.TestCase):
    def test_identical_boxes(self):
        c = _cube()
        self.assertAlmostEqual(iogt(c, c), 1.0)

    def test_half_overlap(self):
        gt = _cube(scale=2.0)  # bbox 2x2x2 volume 8
        gen = [(1, 0, 0), (3, 2, 2)]  # intersect x in [1,2], y,z [0,2] -> 1*2*2=4
        self.assertAlmostEqual(iogt(gen, gt), 0.5)

    def test_degenerate_gt(self):
        self.assertEqual(iogt(_cube(), [(0, 0, 0), (0, 0, 0)]), 0.0)


class TestEvaluate(unittest.TestCase):
    def test_compile_failure_penalty(self):
        r = evaluate_object(_cube(), _cube(), compiled=False)
        self.assertFalse(r["compiled"])
        self.assertAlmostEqual(r["point_cloud_distance"], sqrt(3))
        self.assertAlmostEqual(r["hausdorff_distance"], UNIT_CUBE_DIAGONAL)
        self.assertEqual(r["iogt"], 0.0)

    def test_compiled_normalizes(self):
        # Same shape, different scale/offset -> after normalization identical.
        r = evaluate_object(_cube(shift=3, scale=5), _cube(), compiled=True)
        self.assertTrue(r["compiled"])
        self.assertAlmostEqual(r["point_cloud_distance"], 0.0)
        self.assertAlmostEqual(r["iogt"], 1.0)


class TestAggregate(unittest.TestCase):
    def test_median_iqr_and_compile_rate(self):
        rows = [
            {"compiled": True, "point_cloud_distance": 0.1, "hausdorff_distance": 0.4, "iogt": 0.9},
            {"compiled": True, "point_cloud_distance": 0.2, "hausdorff_distance": 0.5, "iogt": 0.8},
            {"compiled": False, "point_cloud_distance": sqrt(3), "hausdorff_distance": sqrt(3), "iogt": 0.0},
        ]
        agg = aggregate(rows)
        self.assertEqual(agg["n"], 3)
        self.assertAlmostEqual(agg["compile_rate"], 2 / 3)
        self.assertAlmostEqual(agg["point_cloud_distance"]["median"], 0.2)
        self.assertIn("iqr", agg["hausdorff_distance"])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            aggregate([])


if __name__ == "__main__":
    unittest.main()
