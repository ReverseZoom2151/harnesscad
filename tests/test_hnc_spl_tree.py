"""Tests for the HNC-CAD Solid-Profile-Loop tree (reconstruction.hnc_spl_tree)."""

import unittest

from reconstruction.hnc_spl_tree import (
    ARC,
    CIRCLE,
    LINE,
    QUANT_LEVELS,
    SEP_TOKEN,
    TOKEN_DIM,
    Curve,
    Loop,
    ProfileBBox,
    SolidNode,
    curve_type,
    dequantize6,
    loop_bbox,
    loop_token_indices,
    profile_from_loops,
    quantize6,
    solid_from_profiles,
    sort_loop_curves,
    token_onehot,
)


class TestQuantization(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(quantize6(0.0), 0)
        self.assertEqual(quantize6(1.0), QUANT_LEVELS - 1)
        self.assertEqual(quantize6(-5.0), 0)
        self.assertEqual(quantize6(9.0), QUANT_LEVELS - 1)

    def test_midpoint_roundtrip(self):
        code = quantize6(0.5)
        self.assertTrue(0 <= code < QUANT_LEVELS)
        self.assertAlmostEqual(dequantize6(code), 0.5, delta=0.02)

    def test_dequantize_range_check(self):
        with self.assertRaises(ValueError):
            dequantize6(QUANT_LEVELS)

    def test_onehot(self):
        vec = token_onehot(3)
        self.assertEqual(len(vec), TOKEN_DIM)
        self.assertEqual(sum(vec), 1)
        self.assertEqual(vec[3], 1)
        sep = token_onehot(SEP_TOKEN)
        self.assertEqual(sep[SEP_TOKEN], 1)
        with self.assertRaises(ValueError):
            token_onehot(SEP_TOKEN + 1)


class TestCurveType(unittest.TestCase):
    def test_type_by_point_count(self):
        self.assertEqual(curve_type(2), LINE)
        self.assertEqual(curve_type(3), ARC)
        self.assertEqual(curve_type(4), CIRCLE)

    def test_invalid_count(self):
        with self.assertRaises(ValueError):
            curve_type(5)

    def test_curve_property(self):
        self.assertEqual(Curve(((0, 0), (1, 1))).type, LINE)
        self.assertEqual(Curve(((0, 0), (0.5, 1), (1, 0))).type, ARC)
        with self.assertRaises(ValueError):
            Curve(((0, 0),))


class TestLoopOrdering(unittest.TestCase):
    def test_rotate_to_smallest_start(self):
        # Three connected lines forming a triangle; smallest start is (0.0, 0.0).
        c0 = Curve(((0.5, 0.5), (0.9, 0.1)))
        c1 = Curve(((0.9, 0.1), (0.0, 0.0)))
        c2 = Curve(((0.0, 0.0), (0.5, 0.5)))
        canon = sort_loop_curves(Loop((c0, c1, c2)))
        self.assertEqual(canon.curves[0].start, (0.0, 0.0))
        # rotation preserves the cyclic sequence
        self.assertEqual(canon.curves, (c2, c0, c1))

    def test_single_curve_unchanged(self):
        circle = Loop((Curve(((0, 0), (1, 0), (1, 1), (0, 1))),))
        self.assertEqual(sort_loop_curves(circle), circle)


class TestLoopTokens(unittest.TestCase):
    def test_sep_between_curves(self):
        c0 = Curve(((0.0, 0.0), (1.0, 0.0)))
        c1 = Curve(((1.0, 0.0), (1.0, 1.0)))
        toks = loop_token_indices(Loop((c0, c1)))
        # 2 pts * 2 coords = 4, SEP, 4 = 9 tokens
        self.assertEqual(len(toks), 9)
        self.assertEqual(toks[4], SEP_TOKEN)
        self.assertNotIn(SEP_TOKEN, toks[:4])


class TestBBoxAbstraction(unittest.TestCase):
    def test_loop_bbox(self):
        lp = Loop((Curve(((0.2, 0.3), (0.8, 0.3))),
                   Curve(((0.8, 0.3), (0.5, 0.9)))))
        x, y, w, h = loop_bbox(lp)
        self.assertAlmostEqual(x, 0.2)
        self.assertAlmostEqual(y, 0.3)
        self.assertAlmostEqual(w, 0.6)
        self.assertAlmostEqual(h, 0.6)

    def test_profile_boxes_sorted(self):
        far = Loop((Curve(((0.6, 0.6), (0.7, 0.6))), Curve(((0.7, 0.6), (0.6, 0.7))),
                    Curve(((0.6, 0.7), (0.6, 0.6)))))
        near = Loop((Curve(((0.1, 0.1), (0.2, 0.1))), Curve(((0.2, 0.1), (0.1, 0.2))),
                     Curve(((0.1, 0.2), (0.1, 0.1)))))
        prof = profile_from_loops((far, near))
        # sorted ascending by bottom-left corner -> near box first
        self.assertLess(prof.boxes[0][0], prof.boxes[1][0])
        self.assertAlmostEqual(prof.boxes[0][0], 0.1)

    def test_solid_from_profiles(self):
        p0 = profile_from_loops((Loop((Curve(((0.0, 0.0), (0.4, 0.0))),
                                       Curve(((0.4, 0.0), (0.0, 0.4))),
                                       Curve(((0.0, 0.4), (0.0, 0.0))))),))
        p1 = profile_from_loops((Loop((Curve(((0.5, 0.5), (0.9, 0.5))),
                                       Curve(((0.9, 0.5), (0.5, 0.9))),
                                       Curve(((0.5, 0.9), (0.5, 0.5))))),))
        solid = solid_from_profiles((p1, p0), ((0.3, 0.2), (0.0, 0.5)))
        self.assertEqual(len(solid.boxes), 2)
        # sorted by bottom-left (x,y,z) ascending -> p0-derived box (x=0.0) first
        self.assertAlmostEqual(solid.boxes[0][0], 0.0)
        # each box is a 6-tuple (x,y,z,w,h,d)
        self.assertEqual(len(solid.boxes[0]), 6)

    def test_solid_extrusion_count_mismatch(self):
        p0 = profile_from_loops((Loop((Curve(((0.0, 0.0), (0.4, 0.0))),
                                       Curve(((0.4, 0.0), (0.0, 0.4))),
                                       Curve(((0.0, 0.4), (0.0, 0.0))))),))
        with self.assertRaises(ValueError):
            solid_from_profiles((p0,), ())


if __name__ == "__main__":
    unittest.main()
