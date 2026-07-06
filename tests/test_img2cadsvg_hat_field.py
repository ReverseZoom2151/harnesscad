import math
import unittest

from geometry.img2cadsvg_hat_field import (
    HatVector,
    encode,
    decode,
    build_field,
    decode_field,
    point_segment_distance2,
)


def _close(seg_a, seg_b, tol=1e-9):
    (a1, a2), (b1, b2) = seg_a, seg_b
    return all(
        abs(u - v) < tol
        for pa, pb in ((a1, b1), (a2, b2))
        for u, v in zip(pa, pb)
    )


class RoundTripTest(unittest.TestCase):
    def test_encode_decode_off_line(self):
        seg = ((0.0, 0.0), (4.0, 0.0))
        p = (2.0, 3.0)
        vec = encode(p, seg)
        self.assertAlmostEqual(vec.d, 3.0)
        rec = decode(p, vec)
        self.assertTrue(_close(rec, seg))

    def test_encode_decode_various_points(self):
        seg = ((1.0, 1.0), (5.0, 4.0))
        for p in [(0, 0), (3, 10), (-2, 2), (5, -5)]:
            vec = encode(p, seg)
            self.assertTrue(_close(decode(p, vec), seg), p)

    def test_point_on_line(self):
        seg = ((0.0, 0.0), (4.0, 0.0))
        p = (1.0, 0.0)  # on the line
        vec = encode(p, seg)
        self.assertAlmostEqual(vec.d, 0.0)
        self.assertTrue(_close(decode(p, vec), seg))

    def test_zero_length_raises(self):
        with self.assertRaises(ValueError):
            encode((1, 1), ((2, 2), (2, 2)))


class DistanceTest(unittest.TestCase):
    def test_clamped_distance(self):
        seg = ((0.0, 0.0), (4.0, 0.0))
        self.assertAlmostEqual(point_segment_distance2((2, 3), seg), 9.0)
        # beyond endpoint clamps to nearest endpoint
        self.assertAlmostEqual(point_segment_distance2((6, 0), seg), 4.0)


class FieldTest(unittest.TestCase):
    def test_build_and_decode_single_segment(self):
        seg = ((0.0, 0.0), (4.0, 0.0))
        pixels = [(0, 1), (1, 2), (2, -1), (3, 5)]
        field = build_field(pixels, [seg])
        recovered = decode_field(pixels, field)
        self.assertEqual(len(recovered), 1)
        self.assertTrue(_close(recovered[0], seg))

    def test_nearest_assignment(self):
        s1 = ((0.0, 0.0), (0.0, 4.0))  # vertical at x=0
        s2 = ((10.0, 0.0), (10.0, 4.0))  # vertical at x=10
        pixels = [(1, 2), (9, 2)]
        field = build_field(pixels, [s1, s2])
        recs = decode_field(pixels, field)
        self.assertEqual(len(recs), 2)

    def test_empty_segments_raises(self):
        with self.assertRaises(ValueError):
            build_field([(0, 0)], [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            decode_field([(0, 0)], [])


if __name__ == "__main__":
    unittest.main()
