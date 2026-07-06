import unittest

from reconstruction.img2cadsvg_loi_align import (
    psi,
    loi_sample,
    decoupled_groups,
    loi_align_score,
    filter_false_positives,
)


class PsiTest(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(psi(0.0, (0, 0), (4, 2)), (0, 0))
        self.assertEqual(psi(1.0, (0, 0), (4, 2)), (4, 2))

    def test_midpoint(self):
        self.assertEqual(psi(0.5, (0, 0), (4, 2)), (2.0, 1.0))

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            psi(1.5, (0, 0), (1, 1))


class SampleTest(unittest.TestCase):
    def test_even_spacing_includes_endpoints(self):
        pts = loi_sample(((0, 0), (4, 0)), 5)
        self.assertEqual(len(pts), 5)
        self.assertEqual(pts[0], (0, 0))
        self.assertEqual(pts[-1], (4, 0))
        self.assertEqual(pts[2], (2.0, 0.0))

    def test_min_two(self):
        with self.assertRaises(ValueError):
            loi_sample(((0, 0), (1, 1)), 1)


class DecoupledTest(unittest.TestCase):
    def test_three_groups(self):
        g = decoupled_groups((0, 0), (2, 0), (0, 4), (0, 2))
        self.assertEqual(g.endpoints, ((0, 0), (0, 4)))
        self.assertEqual(g.mid_x, (1.0, 0.0))
        self.assertEqual(g.mid_y, (0.0, 3.0))


class AlignTest(unittest.TestCase):
    def test_perfect_alignment(self):
        seg = ((0, 0), (4, 0))
        self.assertAlmostEqual(loi_align_score(seg, ((0, 0), (4, 0))), 1.0)

    def test_orientation_invariant(self):
        seg = ((0, 0), (4, 0))
        s = loi_align_score(seg, ((4, 0), (0, 0)))  # swapped nodes
        self.assertAlmostEqual(s, 1.0)

    def test_decays_with_distance(self):
        seg = ((0, 0), (4, 0))
        near = loi_align_score(seg, ((0.1, 0), (4.1, 0)))
        far = loi_align_score(seg, ((3, 3), (7, 3)))
        self.assertGreater(near, far)

    def test_bad_scale(self):
        with self.assertRaises(ValueError):
            loi_align_score(((0, 0), (1, 0)), ((0, 0), (1, 0)), scale=0.0)


class FilterTest(unittest.TestCase):
    def test_filters_false_positives(self):
        props = [((0, 0), (4, 0)), ((0, 0), (0.1, 5.0))]
        nodes = [((0, 0), (4, 0))]
        kept = filter_false_positives(props, nodes, threshold=0.9, scale=1.0)
        self.assertEqual(kept, [((0, 0), (4, 0))])

    def test_threshold_range(self):
        with self.assertRaises(ValueError):
            filter_false_positives([], [], threshold=2.0)


if __name__ == "__main__":
    unittest.main()
