import math
import unittest

from drawings.sympoint_primitive_points import (
    ARC,
    CIRCLE,
    COMMANDS,
    ELLIPSE,
    LINE,
    Primitive,
    anchor_points,
    command_id,
    ellipse_axes,
    evaluate,
    instance_boxes,
    primitive_length,
    primitive_point,
    primitive_record,
    sample_args,
    to_point_set,
)


class TestPrimitiveValidation(unittest.TestCase):
    def test_command_vocabulary_order(self):
        self.assertEqual(COMMANDS, (LINE, ARC, CIRCLE, ELLIPSE))
        self.assertEqual(command_id(LINE), 0)
        self.assertEqual(command_id(ELLIPSE), 3)

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            Primitive("spline", (0, 0))

    def test_bad_arity(self):
        with self.assertRaises(ValueError):
            Primitive(LINE, (0, 0, 1))

    def test_negative_radius(self):
        with self.assertRaises(ValueError):
            Primitive(CIRCLE, (0, 0, -1))
        with self.assertRaises(ValueError):
            Primitive(ELLIPSE, (0, 0, 1, -2))


class TestSampling(unittest.TestCase):
    def test_line_args_are_thirds(self):
        args = sample_args(Primitive(LINE, (0, 0, 3, 6)))
        self.assertEqual(len(args), 8)
        for got, want in zip(args, (0, 0, 1, 2, 2, 4, 3, 6)):
            self.assertAlmostEqual(got, want)

    def test_circle_args_are_cardinal(self):
        args = sample_args(Primitive(CIRCLE, (1, 1, 2)))
        pts = anchor_points(args)
        for got, want in zip(pts, [(3, 1), (1, 3), (-1, 1), (1, -1)]):
            self.assertAlmostEqual(got[0], want[0])
            self.assertAlmostEqual(got[1], want[1])

    def test_ellipse_uses_major_axis_first(self):
        wide = anchor_points(sample_args(Primitive(ELLIPSE, (0, 0, 4, 1))))
        tall = anchor_points(sample_args(Primitive(ELLIPSE, (0, 0, 1, 4))))
        self.assertEqual(wide, tall)
        self.assertAlmostEqual(wide[0][0], 4.0)
        self.assertAlmostEqual(wide[1][1], 1.0)

    def test_arc_evaluate_endpoints(self):
        prim = Primitive(ARC, (0, 0, 1, 0.0, math.pi / 2))
        x0, y0 = evaluate(prim, 0.0)
        x1, y1 = evaluate(prim, 1.0)
        self.assertAlmostEqual(x0, 1.0)
        self.assertAlmostEqual(y0, 0.0)
        self.assertAlmostEqual(x1, 0.0)
        self.assertAlmostEqual(y1, 1.0)

    def test_evaluate_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            evaluate(Primitive(LINE, (0, 0, 1, 1)), 1.5)


class TestLengths(unittest.TestCase):
    def test_line_length(self):
        self.assertAlmostEqual(primitive_length(Primitive(LINE, (0, 0, 3, 4))), 5.0)

    def test_circle_length(self):
        self.assertAlmostEqual(primitive_length(Primitive(CIRCLE, (0, 0, 2))), 4 * math.pi)

    def test_arc_length(self):
        prim = Primitive(ARC, (0, 0, 2, 0.0, math.pi))
        self.assertAlmostEqual(primitive_length(prim), 2 * math.pi)

    def test_ellipse_approximation(self):
        prim = Primitive(ELLIPSE, (0, 0, 5, 2))
        self.assertAlmostEqual(primitive_length(prim), 2 * math.pi * 2 + 4 * 3)

    def test_ellipse_degenerates_to_circle(self):
        self.assertAlmostEqual(primitive_length(Primitive(ELLIPSE, (0, 0, 3, 3))),
                               primitive_length(Primitive(CIRCLE, (0, 0, 3))))

    def test_ellipse_axes(self):
        self.assertEqual(ellipse_axes(1.0, 4.0), (4.0, 1.0))
        self.assertEqual(ellipse_axes(4.0, 1.0), (4.0, 1.0))


class TestPointCollapse(unittest.TestCase):
    def test_circle_point_is_center(self):
        args = sample_args(Primitive(CIRCLE, (5, -2, 3)))
        px, py = primitive_point(args)
        self.assertAlmostEqual(px, 5.0)
        self.assertAlmostEqual(py, -2.0)

    def test_line_point_is_mean_of_anchors(self):
        args = sample_args(Primitive(LINE, (0, 0, 3, 3)))
        px, py = primitive_point(args)
        self.assertAlmostEqual(px, 1.5)
        self.assertAlmostEqual(py, 1.5)

    def test_anchor_points_rejects_bad_length(self):
        with self.assertRaises(ValueError):
            anchor_points((0, 1, 2))

    def test_primitive_record(self):
        rec = primitive_record(Primitive(LINE, (0, 0, 0, 2)))
        self.assertEqual(rec["command"], 0)
        self.assertAlmostEqual(rec["length"], 2.0)
        self.assertAlmostEqual(rec["point"][1], 1.0)


class TestPointSet(unittest.TestCase):
    def setUp(self):
        self.prims = [
            Primitive(LINE, (0, 0, 2, 0)),
            Primitive(CIRCLE, (1, 1, 1)),
            Primitive(ARC, (0, 0, 1, 0.0, math.pi)),
        ]

    def test_to_point_set_shapes(self):
        ps = to_point_set(self.prims)
        self.assertEqual(ps["commands"], [0, 2, 1])
        self.assertEqual(len(ps["args"]), 3)
        self.assertEqual(len(ps["points"]), 3)
        self.assertTrue(all(len(a) == 8 for a in ps["args"]))

    def test_deterministic(self):
        self.assertEqual(to_point_set(self.prims), to_point_set(self.prims))


class TestInstanceBoxes(unittest.TestCase):
    def test_boxes_from_anchors(self):
        args = [
            sample_args(Primitive(LINE, (0, 0, 2, 0))),
            sample_args(Primitive(LINE, (2, 0, 2, 4))),
            sample_args(Primitive(LINE, (10, 10, 12, 10))),
        ]
        out = instance_boxes(args, [3, 3, -1], [5, 5, 34])
        self.assertEqual(len(out), 1)
        rec = out[0]
        self.assertEqual(rec["instance_id"], 3)
        self.assertEqual(rec["semantic_id"], 5)
        self.assertEqual(rec["box"], (0.0, 0.0, 2.0, 4.0))
        self.assertEqual(rec["center"], (1.0, 2.0))

    def test_sorted_and_grouped(self):
        args = [sample_args(Primitive(CIRCLE, (0, 0, 1))),
                sample_args(Primitive(CIRCLE, (10, 0, 1)))]
        out = instance_boxes(args, [7, 2], [1, 1])
        self.assertEqual([r["instance_id"] for r in out], [2, 7])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            instance_boxes([(0,) * 8], [0, 1], [0])


if __name__ == "__main__":
    unittest.main()
