"""Tests for the ComplexGen B-Rep chain-complex structure and its validity rules."""

import unittest

from harnesscad.domain.reconstruction import complexgen_chain_complex as cc


CUBE_CORNERS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),          # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),          # top
    (0, 4), (1, 5), (2, 6), (3, 7),          # verticals
]
CUBE_FACES = {
    "bottom": (0, 1, 2, 3),
    "top": (4, 5, 6, 7),
    "front": (0, 9, 4, 8),
    "right": (1, 10, 5, 9),
    "back": (2, 11, 6, 10),
    "left": (3, 8, 7, 11),
}


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _edge_points(a, b, n=9):
    return [_lerp(a, b, i / (n - 1)) for i in range(n)]


def _face_grid(face_corners, n=5):
    """Bilinear sample grid of a quad face given its 4 corner points in order."""
    p0, p1, p2, p3 = face_corners
    pts = []
    for i in range(n):
        u = i / (n - 1)
        for j in range(n):
            v = j / (n - 1)
            top = _lerp(p0, p1, u)
            bottom = _lerp(p3, p2, u)
            pts.append(_lerp(top, bottom, v))
    return pts


FACE_CORNER_ORDER = {
    "bottom": (0, 1, 2, 3),
    "top": (4, 5, 6, 7),
    "front": (0, 1, 5, 4),
    "right": (1, 2, 6, 5),
    "back": (2, 3, 7, 6),
    "left": (3, 0, 4, 7),
}


def cube_complex():
    curves = []
    for (a, b) in CUBE_EDGES:
        curves.append(cc.Curve(tuple(_edge_points(CUBE_CORNERS[a], CUBE_CORNERS[b])), False))
    patches = []
    names = list(CUBE_FACES)
    for name in names:
        corner_ids = FACE_CORNER_ORDER[name]
        pts = _face_grid([CUBE_CORNERS[i] for i in corner_ids])
        patches.append(cc.Patch(tuple(pts)))
    ev = [[1 if j in edge else 0 for j in range(8)] for edge in CUBE_EDGES]
    fe = [[1 if i in CUBE_FACES[name] else 0 for i in range(12)] for name in names]
    return cc.make_complex(CUBE_CORNERS, curves, patches, ev, fe)


class TestConstruction(unittest.TestCase):
    def test_counts(self):
        cx = cube_complex()
        self.assertEqual((cx.n_corners, cx.n_curves, cx.n_patches), (8, 12, 6))

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            cc.ChainComplex(corners=((0.0, 0.0, 0.0),), curves=(), patches=(),
                            curve_corner=((1,),), patch_curve=())

    def test_bad_row_width(self):
        with self.assertRaises(ValueError):
            cc.ChainComplex(
                corners=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                curves=(cc.Curve(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),),
                patches=(),
                curve_corner=((1,),),
                patch_curve=())


class TestDerivedStructure(unittest.TestCase):
    def setUp(self):
        self.cx = cube_complex()

    def test_corners_of_curve(self):
        self.assertEqual(cc.corners_of_curve(self.cx, 0), (0, 1))

    def test_patches_of_curve(self):
        for i in range(self.cx.n_curves):
            self.assertEqual(len(cc.patches_of_curve(self.cx, i)), 2)

    def test_corner_degree(self):
        for j in range(8):
            self.assertEqual(cc.corner_degree(self.cx, j), 3)

    def test_patch_corner_incidence(self):
        fc = cc.patch_corner_incidence(self.cx)
        self.assertEqual(sum(fc[0]), 4)          # bottom face has 4 corners
        for row in fc:
            self.assertEqual(sum(row), 4)
            self.assertTrue(all(v in (0, 1) for v in row))

    def test_patch_loops(self):
        for k in range(self.cx.n_patches):
            loops = cc.patch_loops(self.cx, k)
            self.assertEqual(len(loops), 1)
            self.assertEqual(len(loops[0]), 4)

    def test_euler_characteristic(self):
        self.assertEqual(cc.euler_characteristic(self.cx), 2)

    def test_watertight_and_connected(self):
        self.assertTrue(cc.is_watertight(self.cx))
        self.assertTrue(cc.is_connected(self.cx))


class TestValidity(unittest.TestCase):
    def test_cube_is_valid(self):
        diag = cc.check(cube_complex())
        self.assertTrue(diag.valid, diag.violations)
        self.assertTrue(bool(diag))
        self.assertEqual(diag.violations, [])

    def test_missing_curve_breaks_watertightness(self):
        cx = cube_complex()
        fe = [list(r) for r in cx.patch_curve]
        fe[0][0] = 0                                   # drop a curve from the bottom face
        broken = cc.make_complex(cx.corners, cx.curves, cx.patches, cx.curve_corner, fe)
        diag = cc.check(broken)
        self.assertFalse(diag.valid)
        self.assertTrue(any("bounds 1 patches" in v for v in diag.violations))

    def test_open_curve_with_one_corner_is_invalid(self):
        cx = cube_complex()
        ev = [list(r) for r in cx.curve_corner]
        ev[0][1] = 0
        broken = cc.make_complex(cx.corners, cx.curves, cx.patches, ev, cx.patch_curve)
        diag = cc.check(broken)
        self.assertFalse(diag.valid)
        self.assertTrue(any("expected 2" in v for v in diag.violations))

    def test_odd_incidence_product_detected(self):
        cx = cube_complex()
        ev = [list(r) for r in cx.curve_corner]
        ev[0] = [0] * 8
        ev[0][0] = 1
        ev[0][2] = 1                                    # curve 0 now spans 0-2, breaking the loop
        broken = cc.make_complex(cx.corners, cx.curves, cx.patches, ev, cx.patch_curve)
        diag = cc.check(broken)
        self.assertFalse(diag.valid)
        self.assertTrue(any("odd" in v for v in diag.violations))
        with self.assertRaises(ValueError):
            cc.patch_corner_incidence(broken)

    def test_closed_curve_must_have_no_corners(self):
        curve = cc.Curve(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)), True)
        cx = cc.make_complex([(0.0, 0.0, 0.0)], [curve], [], [[1]], [])
        diag = cc.check(cx, min_corner_degree=1, require_watertight=False)
        self.assertFalse(diag.valid)
        self.assertTrue(any("closed" in v for v in diag.violations))

    def test_min_corner_degree(self):
        cx = cube_complex()
        self.assertTrue(cc.is_valid(cx, min_corner_degree=3))
        self.assertFalse(cc.is_valid(cx, min_corner_degree=4))

    def test_cylinder_like_closed_curves(self):
        """Two closed curves + two caps + a lateral patch: no corners at all."""
        bottom = cc.Curve(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0),
                           (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)), True)
        top = cc.Curve(((1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (-1.0, 0.0, 1.0),
                        (0.0, -1.0, 1.0), (1.0, 0.0, 1.0)), True)
        patches = [cc.Patch(bottom.points), cc.Patch(top.points),
                   cc.Patch(bottom.points + top.points)]
        fe = [[1, 0], [0, 1], [1, 1]]
        cx = cc.make_complex([], [bottom, top], patches, [[], []], fe)
        diag = cc.check(cx)
        self.assertTrue(diag.valid, diag.violations)
        self.assertEqual(cc.euler_characteristic(cx), 1)
        self.assertEqual(cc.patch_loops(cx, 2), ((0,), (1,)))


class TestGeometryConsistency(unittest.TestCase):
    def test_cube_geometry_consistent(self):
        diag = cc.check_geometry(cube_complex(), tol=0.2)
        self.assertTrue(diag.valid, diag.violations)

    def test_displaced_corner_detected(self):
        cx = cube_complex()
        corners = list(cx.corners)
        corners[0] = (5.0, 5.0, 5.0)
        broken = cc.make_complex(corners, cx.curves, cx.patches,
                                 cx.curve_corner, cx.patch_curve)
        diag = cc.check_geometry(broken, tol=0.2)
        self.assertFalse(diag.valid)
        self.assertTrue(any("corner 0" in v for v in diag.violations))

    def test_displaced_curve_detected(self):
        cx = cube_complex()
        curves = list(cx.curves)
        shifted = tuple((p[0], p[1], p[2] + 3.0) for p in curves[0].points)
        curves[0] = cc.Curve(shifted, False)
        broken = cc.make_complex(cx.corners, curves, cx.patches,
                                 cx.curve_corner, cx.patch_curve)
        diag = cc.check_geometry(broken, tol=0.2)
        self.assertFalse(diag.valid)
        self.assertTrue(any("curve 0" in v for v in diag.violations))


class TestSimilarity(unittest.TestCase):
    def test_geometric_similarity(self):
        self.assertAlmostEqual(cc.geometric_similarity(0.0, 0.2), 1.0)
        self.assertLess(cc.geometric_similarity(0.4, 0.2), 0.02)
        with self.assertRaises(ValueError):
            cc.geometric_similarity(1.0, 0.0)

    def test_curve_corner_similarity_recovers_incidence(self):
        cx = cube_complex()
        sim = cc.curve_corner_similarity(cx, sigma=0.2)
        ev = cc.threshold_incidence(sim, 0.5)
        self.assertEqual(ev, cx.curve_corner)

    def test_patch_curve_similarity_recovers_incidence(self):
        cx = cube_complex()
        sim = cc.patch_curve_similarity(cx, sigma=0.1)
        fe = cc.threshold_incidence(sim, 0.5)
        self.assertEqual(fe, cx.patch_curve)


if __name__ == "__main__":
    unittest.main()
