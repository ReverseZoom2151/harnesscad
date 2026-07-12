import unittest

from drawings.t2cadtd_projection_convention import (
    FRONT, TOP, SIDE, THIRD_ANGLE, FIRST_ANGLE,
    view_placements, convert_layout, infer_convention,
    dimensions_covered, views_sufficient, SufficiencyResult,
)


class TestPlacements(unittest.TestCase):
    def test_third_angle_top_above_side_right(self):
        p = view_placements(THIRD_ANGLE)
        self.assertGreater(p[TOP][1], p[FRONT][1])   # top above front
        self.assertGreater(p[SIDE][0], p[FRONT][0])  # side right of front

    def test_first_angle_top_below_side_left(self):
        p = view_placements(FIRST_ANGLE)
        self.assertLess(p[TOP][1], p[FRONT][1])
        self.assertLess(p[SIDE][0], p[FRONT][0])

    def test_unknown_convention(self):
        with self.assertRaises(ValueError):
            view_placements("bogus")


class TestConvert(unittest.TestCase):
    def test_convert_third_to_first(self):
        third = view_placements(THIRD_ANGLE)
        converted = convert_layout(third)
        self.assertEqual(infer_convention(converted), FIRST_ANGLE)

    def test_convert_is_involution(self):
        third = view_placements(THIRD_ANGLE)
        self.assertEqual(convert_layout(convert_layout(third)), third)

    def test_front_preserved(self):
        third = view_placements(THIRD_ANGLE)
        self.assertEqual(convert_layout(third)[FRONT], third[FRONT])


class TestInfer(unittest.TestCase):
    def test_infer_third(self):
        self.assertEqual(infer_convention(view_placements(THIRD_ANGLE)),
                         THIRD_ANGLE)

    def test_infer_first(self):
        self.assertEqual(infer_convention(view_placements(FIRST_ANGLE)),
                         FIRST_ANGLE)

    def test_infer_tolerates_spacing(self):
        # Arbitrary grid spacing keeps the third-angle classification.
        p = {FRONT: (0, 0), TOP: (0, 7), SIDE: (4, 0)}
        self.assertEqual(infer_convention(p), THIRD_ANGLE)

    def test_mixed_is_inconsistent(self):
        # top above (third) but side left (first) -> inconsistent.
        p = {FRONT: (0, 0), TOP: (0, 1), SIDE: (-1, 0)}
        self.assertEqual(infer_convention(p), "inconsistent")

    def test_missing_view_is_inconsistent(self):
        self.assertEqual(infer_convention({FRONT: (0, 0), TOP: (0, 1)}),
                         "inconsistent")


class TestSufficiency(unittest.TestCase):
    def test_single_view_insufficient(self):
        res = views_sufficient([FRONT])
        self.assertIsInstance(res, SufficiencyResult)
        self.assertFalse(res.sufficient)
        self.assertIn("depth", res.missing)

    def test_two_views_sufficient(self):
        self.assertTrue(views_sufficient([FRONT, TOP]).sufficient)

    def test_three_views_sufficient(self):
        res = views_sufficient([FRONT, TOP, SIDE])
        self.assertTrue(res.sufficient)
        self.assertEqual(set(res.covered), {"width", "height", "depth"})

    def test_front_side_covers_all(self):
        self.assertTrue(views_sufficient([FRONT, SIDE]).sufficient)

    def test_dimensions_covered(self):
        self.assertEqual(dimensions_covered([FRONT]), {"width", "height"})
        self.assertEqual(dimensions_covered([TOP]), {"width", "depth"})

    def test_to_dict(self):
        d = views_sufficient([FRONT]).to_dict()
        self.assertFalse(d["sufficient"])
        self.assertIn("depth", d["missing"])


if __name__ == "__main__":
    unittest.main()
