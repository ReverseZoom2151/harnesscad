import unittest

from harnesscad.domain.reconstruction.skexgen_canonical_order import (
    bottom_left, canonicalize_sketch, circle_rim_points, curve_bbox, endpoints,
    face_bbox, flip_curve, loop_bbox, point_key, sort_curves, sort_faces,
    sort_loops,
)


def _line(a, b):
    return {"type": "line", "start": a, "end": b}


def _square(x0=0.0, y0=0.0, s=1.0):
    p = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s)]
    return [_line(p[i], p[(i + 1) % 4]) for i in range(4)]


class TestBBox(unittest.TestCase):
    def test_line(self):
        self.assertEqual(curve_bbox(_line((1.0, 2.0), (-1.0, 0.0))),
                         (-1.0, 1.0, 0.0, 2.0))

    def test_arc_uses_three_points(self):
        arc = {"type": "arc", "start": (-1.0, 0.0), "mid": (0.0, 1.0),
               "end": (1.0, 0.0)}
        self.assertEqual(curve_bbox(arc), (-1.0, 1.0, 0.0, 1.0))

    def test_circle_from_center_radius(self):
        circ = {"type": "circle", "center": (1.0, 1.0), "radius": 2.0}
        self.assertEqual(curve_bbox(circ), (-1.0, 3.0, -1.0, 3.0))

    def test_circle_rim_points(self):
        self.assertEqual(circle_rim_points((0.0, 0.0), 1.0),
                         ((0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0)))

    def test_bad_type(self):
        self.assertRaises(ValueError, curve_bbox, {"type": "nurbs"})

    def test_loop_and_face_bbox(self):
        self.assertEqual(loop_bbox(_square()), (0.0, 1.0, 0.0, 1.0))
        self.assertEqual(face_bbox([_square(), _square(2.0, 3.0)]),
                         (0.0, 3.0, 0.0, 4.0))

    def test_bottom_left(self):
        self.assertEqual(bottom_left(_line((1.0, 5.0), (3.0, 2.0))), (1.0, 2.0))


class TestSorting(unittest.TestCase):
    def test_faces_sorted_by_min_corner(self):
        far = [_square(5.0, 0.0)]
        near = [_square(0.0, 1.0)]
        sketch = sort_faces([far, near])
        self.assertEqual(face_bbox(sketch[0])[0], 0.0)

    def test_faces_tie_broken_by_y(self):
        a = [_square(0.0, 4.0)]
        b = [_square(0.0, 1.0)]
        self.assertEqual(face_bbox(sort_faces([a, b])[0])[2], 1.0)

    def test_loops_outer_first(self):
        outer = _square(0.0, 0.0, 10.0)
        inner_far = _square(6.0, 6.0)
        inner_near = _square(1.0, 1.0)
        face = sort_loops([outer, inner_far, inner_near])
        self.assertIs(face[0][0], outer[0])
        self.assertEqual(loop_bbox(face[1])[0], 1.0)

    def test_single_loop(self):
        self.assertEqual(len(sort_loops([_square()])), 1)

    def test_empty_face(self):
        self.assertRaises(ValueError, sort_loops, [])


class TestSortCurves(unittest.TestCase):
    def test_chain_from_bottom_left(self):
        loop = _square()
        shuffled = [loop[2], loop[0], loop[3], loop[1]]
        out = sort_curves(shuffled)
        self.assertEqual(len(out), 4)
        # closed chain
        for i, c in enumerate(out):
            self.assertEqual(endpoints(c)[1], endpoints(out[(i + 1) % 4])[0])
        # starts at the bottom-left curve
        self.assertEqual(bottom_left(out[0]), (0.0, 0.0))

    def test_increasing_x_direction(self):
        out = sort_curves(_square())
        # first curve is the bottom edge; second must move in +x, i.e. the
        # right edge rather than the left edge
        self.assertEqual(endpoints(out[0])[0], (0.0, 0.0))
        self.assertEqual(endpoints(out[1])[0], (1.0, 0.0))

    def test_flipped_input_is_repaired(self):
        loop = _square()
        loop[1] = flip_curve(loop[1])
        out = sort_curves(loop)
        for i, c in enumerate(out):
            self.assertEqual(endpoints(c)[1], endpoints(out[(i + 1) % 4])[0])

    def test_circle_loop(self):
        circ = {"type": "circle", "center": (0.0, 0.0), "radius": 1.0}
        self.assertEqual(sort_curves([circ])[0]["type"], "circle")
        self.assertRaises(ValueError, sort_curves, [circ, _line((0.0, 0.0), (1.0, 0.0))])
        self.assertRaises(ValueError, endpoints, circ)

    def test_two_curve_loop(self):
        arc = {"type": "arc", "start": (-1.0, 0.0), "mid": (0.0, 1.0),
               "end": (1.0, 0.0)}
        line = _line((-1.0, 0.0), (1.0, 0.0))
        out = sort_curves([arc, line])
        self.assertEqual(endpoints(out[0])[1], endpoints(out[1])[0])
        self.assertEqual(endpoints(out[1])[1], endpoints(out[0])[0])

    def test_broken_loop(self):
        curves = [_line((0.0, 0.0), (1.0, 0.0)), _line((5.0, 5.0), (6.0, 5.0))]
        self.assertRaises(ValueError, sort_curves, curves)

    def test_empty(self):
        self.assertRaises(ValueError, sort_curves, [])

    def test_single_line_loop_rejected(self):
        self.assertRaises(ValueError, sort_curves, [_line((0.0, 0.0), (1.0, 0.0))])


class TestCanonicalize(unittest.TestCase):
    def test_full(self):
        f1 = [_square(3.0, 0.0)]
        f2 = [_square(0.0, 0.0, 10.0), _square(2.0, 2.0)]
        out = canonicalize_sketch([f1, f2])
        self.assertEqual(face_bbox(out[0])[0], 0.0)      # big face first
        self.assertEqual(len(out[0]), 2)                 # outer + inner
        self.assertEqual(loop_bbox(out[0][0])[1], 10.0)  # outer loop kept first
        self.assertEqual(len(out[1][0]), 4)

    def test_idempotent(self):
        sketch = [[_square(1.0, 1.0)]]
        once = canonicalize_sketch(sketch)
        twice = canonicalize_sketch(once)
        self.assertEqual(once, twice)

    def test_point_key_rounding(self):
        self.assertEqual(point_key((1.0000000001, 2.0)), (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
