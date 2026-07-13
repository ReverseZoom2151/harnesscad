"""Tests for the ComplexGen probabilistic-complex post-process (gating, NMS, extraction)."""

import unittest

from harnesscad.domain.reconstruction.brep import chain_complex as cc
from harnesscad.domain.reconstruction.brep import chain_complex_nms as nms
from tests.domain.reconstruction.brep.test_chain_complex import cube_complex


def prob_cube(extra_corner=False, duplicate_patch=False):
    """A probabilistic complex whose confident part is exactly the unit cube."""
    cx = cube_complex()
    corners = list(cx.corners)
    corner_valid = [0.99] * cx.n_corners
    curves = list(cx.curves)
    curve_valid = [0.98] * cx.n_curves
    curve_closed = [0.01] * cx.n_curves
    patches = list(cx.patches)
    patch_valid = [0.97] * cx.n_patches
    ev = [[0.95 if v else 0.02 for v in row] for row in cx.curve_corner]
    fe = [[0.95 if v else 0.02 for v in row] for row in cx.patch_curve]

    if extra_corner:                       # a near-duplicate of corner 0
        corners.append((0.001, 0.0, 0.0))
        corner_valid.append(0.9)
        for i in range(len(curves)):
            ev[i].append(ev[i][0])
    if duplicate_patch:                    # an exact duplicate of patch 0
        patches.append(cc.Patch(patches[0].points))
        patch_valid.append(0.9)
        fe.append(list(fe[0]))

    return nms.ProbabilisticComplex(
        corners=corners, corner_valid=corner_valid,
        curves=curves, curve_valid=curve_valid, curve_closed_prob=curve_closed,
        patches=patches, patch_valid=patch_valid,
        curve_corner_prob=ev, patch_curve_prob=fe)


class TestConstruction(unittest.TestCase):
    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            nms.ProbabilisticComplex(
                corners=[(0.0, 0.0, 0.0)], corner_valid=[],
                curves=[], curve_valid=[], curve_closed_prob=[],
                patches=[], patch_valid=[],
                curve_corner_prob=[], patch_curve_prob=[])


class TestGating(unittest.TestCase):
    def test_invalid_cell_zeroes_incidences(self):
        pc = prob_cube()
        pc.corner_valid[0] = 0.0
        gated = nms.gate_similarities(pc)
        for i in range(len(pc.curves)):
            self.assertAlmostEqual(gated.curve_corner_prob[i][0], 0.0)

    def test_closed_curve_has_no_corner_incidence(self):
        pc = prob_cube()
        pc.curve_closed_prob[3] = 1.0
        gated = nms.gate_similarities(pc)
        self.assertTrue(all(v == 0.0 for v in gated.curve_corner_prob[3]))

    def test_suppressed_patch_zeroes_row(self):
        pc = prob_cube()
        pc.suppressed_patches.add(2)
        gated = nms.gate_similarities(pc)
        self.assertTrue(all(v == 0.0 for v in gated.patch_curve_prob[2]))

    def test_input_unmodified(self):
        pc = prob_cube()
        before = [row[:] for row in pc.curve_corner_prob]
        nms.gate_similarities(pc)
        self.assertEqual(pc.curve_corner_prob, before)


class TestNMS(unittest.TestCase):
    def test_duplicate_corner_suppressed(self):
        pc = prob_cube(extra_corner=True)
        out = nms.nms(pc)
        self.assertIn(8, out.suppressed_corners)
        self.assertNotIn(0, out.suppressed_corners)

    def test_duplicate_patch_suppressed(self):
        pc = prob_cube(duplicate_patch=True)
        out = nms.nms(pc)
        self.assertEqual(out.suppressed_patches, {6})

    def test_no_duplicates_no_suppression(self):
        out = nms.nms(prob_cube())
        self.assertEqual(out.suppressed_corners, set())
        self.assertEqual(out.suppressed_curves, set())
        self.assertEqual(out.suppressed_patches, set())

    def test_duplicate_curve_suppressed(self):
        pc = prob_cube()
        pc.curves.append(pc.curves[0])
        pc.curve_valid.append(0.9)
        pc.curve_closed_prob.append(0.01)
        pc.curve_corner_prob.append(list(pc.curve_corner_prob[0]))
        for row in pc.patch_curve_prob:
            row.append(0.02)
        self.assertIn(12, nms.nms_curves(pc))


class TestMergeDuplicatedCorners(unittest.TestCase):
    def test_over_connected_curve_drops_duplicate(self):
        pc = prob_cube(extra_corner=True)
        gated = nms.gate_similarities(pc)
        # curve 0 now claims corners 0, 1 and the duplicate 8
        dropped = nms.merge_duplicated_corners(gated)
        self.assertIn(8, dropped)

    def test_clean_complex_drops_nothing(self):
        gated = nms.gate_similarities(prob_cube())
        self.assertEqual(nms.merge_duplicated_corners(gated), set())


class TestExtraction(unittest.TestCase):
    def test_extract_recovers_cube(self):
        pc = prob_cube()
        ext = nms.extract(pc)
        self.assertEqual(ext.corner_ids, tuple(range(8)))
        self.assertEqual(ext.curve_ids, tuple(range(12)))
        self.assertEqual(ext.patch_ids, tuple(range(6)))
        diag = cc.check(ext.complex)
        self.assertTrue(diag.valid, diag.violations)
        self.assertEqual(cc.euler_characteristic(ext.complex), 2)

    def test_low_validity_cells_dropped_and_reindexed(self):
        pc = prob_cube()
        pc.patch_valid[5] = 0.1
        ext = nms.extract(pc)
        self.assertEqual(ext.patch_ids, (0, 1, 2, 3, 4))
        self.assertEqual(ext.complex.n_patches, 5)
        self.assertFalse(cc.is_watertight(ext.complex))

    def test_extract_after_nms_of_duplicates(self):
        pc = prob_cube(extra_corner=True, duplicate_patch=True)
        ext = nms.extract(nms.nms(pc))
        self.assertEqual(ext.complex.n_corners, 8)
        self.assertEqual(ext.complex.n_patches, 6)
        diag = cc.check(ext.complex)
        self.assertTrue(diag.valid, diag.violations)

    def test_extraction_is_deterministic(self):
        pc = prob_cube(extra_corner=True)
        a = nms.extract(nms.nms(pc)).complex
        b = nms.extract(nms.nms(pc)).complex
        self.assertEqual(a, b)


class TestRepair(unittest.TestCase):
    def test_repair_drops_spurious_corner(self):
        cx = cube_complex()
        ev = [list(r) for r in cx.curve_corner]
        ev[0][6] = 1                        # curve 0 wrongly claims the far corner 6
        broken = cc.make_complex(cx.corners, cx.curves, cx.patches, ev, cx.patch_curve)
        self.assertFalse(cc.is_valid(broken))
        fixed = nms.repair_extraction(broken)
        self.assertEqual(cc.corners_of_curve(fixed, 0), (0, 1))
        self.assertTrue(cc.is_valid(fixed))

    def test_repair_strips_closed_curve_corners(self):
        cx = cube_complex()
        curves = list(cx.curves)
        curves[0] = cc.Curve(curves[0].points, True)
        broken = cc.make_complex(cx.corners, curves, cx.patches,
                                 cx.curve_corner, cx.patch_curve)
        fixed = nms.repair_extraction(broken)
        self.assertEqual(cc.corners_of_curve(fixed, 0), ())

    def test_repair_limits_patches_per_curve(self):
        cx = cube_complex()
        fe = [list(r) for r in cx.patch_curve]
        fe[1][0] = 1                       # top face wrongly claims bottom edge 0
        broken = cc.make_complex(cx.corners, cx.curves, cx.patches, cx.curve_corner, fe)
        self.assertEqual(len(cc.patches_of_curve(broken, 0)), 3)
        fixed = nms.repair_extraction(broken)
        self.assertEqual(len(cc.patches_of_curve(fixed, 0)), 2)
        self.assertTrue(cc.is_valid(fixed))


if __name__ == "__main__":
    unittest.main()
