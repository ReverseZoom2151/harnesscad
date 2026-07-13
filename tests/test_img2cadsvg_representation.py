import unittest

from harnesscad.domain.reconstruction.fitting.img2cadsvg_representation import (
    Segment,
    Wireframe,
    build_wireframe,
    validity,
    normalise,
    bounding_box,
    total_length,
)


class BuildTest(unittest.TestCase):
    def test_shared_junctions_at_corner(self):
        # unit square as 4 segments; corners shared -> 4 junctions, 4 segments
        raw = [
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (0, 1)),
            ((0, 1), (0, 0)),
        ]
        wf = build_wireframe(raw)
        self.assertEqual(len(wf.junctions), 4)
        self.assertEqual(len(wf.segments), 4)
        self.assertEqual(wf.degrees(), [2, 2, 2, 2])

    def test_merge_within_eps(self):
        raw = [((0, 0), (1, 0)), ((1.0000001, 0), (1, 1))]
        wf = build_wireframe(raw, eps=1e-4)
        self.assertEqual(len(wf.junctions), 3)  # the two ~(1,0) merged

    def test_drops_zero_length_and_duplicates(self):
        raw = [((0, 0), (0, 0)), ((0, 0), (1, 0)), ((1, 0), (0, 0))]
        wf = build_wireframe(raw)
        self.assertEqual(len(wf.segments), 1)  # zero-length + duplicate removed

    def test_negative_eps_raises(self):
        with self.assertRaises(ValueError):
            build_wireframe([], eps=-1.0)


class ValidityTest(unittest.TestCase):
    def test_valid_square(self):
        raw = [
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (0, 1)),
            ((0, 1), (0, 0)),
        ]
        v = validity(build_wireframe(raw))
        self.assertTrue(v.ok)
        self.assertEqual(v.n_segments, 4)
        self.assertEqual(v.isolated_junctions, 0)

    def test_out_of_range(self):
        wf = Wireframe(junctions=[(0, 0), (1, 0)], segments=[Segment(0, 5)])
        v = validity(wf)
        self.assertFalse(v.ok)
        self.assertEqual(v.out_of_range, 1)

    def test_isolated_junction(self):
        wf = Wireframe(
            junctions=[(0, 0), (1, 0), (5, 5)], segments=[Segment(0, 1)]
        )
        v = validity(wf)
        self.assertFalse(v.ok)
        self.assertEqual(v.isolated_junctions, 1)

    def test_duplicate_detected(self):
        wf = Wireframe(
            junctions=[(0, 0), (1, 0)], segments=[Segment(0, 1), Segment(1, 0)]
        )
        v = validity(wf)
        self.assertEqual(v.duplicates, 1)
        self.assertFalse(v.ok)


class NormaliseTest(unittest.TestCase):
    def test_normalise_range(self):
        raw = [((0, 0), (4, 0)), ((4, 0), (4, 2)), ((4, 2), (0, 0))]
        wf = normalise(build_wireframe(raw))
        (minx, miny), (maxx, maxy) = bounding_box(wf)
        self.assertAlmostEqual(maxx, 1.0)
        self.assertAlmostEqual(minx, -1.0)
        # aspect preserved: y half-extent < 1
        self.assertLessEqual(maxy, 1.0 + 1e-9)
        self.assertLess(maxy, 0.6)

    def test_segments_preserved(self):
        raw = [((0, 0), (2, 0)), ((2, 0), (2, 2))]
        wf = build_wireframe(raw)
        norm = normalise(wf)
        self.assertEqual(
            [s.key() for s in norm.segments], [s.key() for s in wf.segments]
        )


class LengthTest(unittest.TestCase):
    def test_total_length(self):
        raw = [((0, 0), (3, 0)), ((3, 0), (3, 4))]
        self.assertAlmostEqual(total_length(build_wireframe(raw)), 7.0)


if __name__ == "__main__":
    unittest.main()
