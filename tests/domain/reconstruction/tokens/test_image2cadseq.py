import math
import unittest

from harnesscad.domain.reconstruction.tokens import image2cadseq as g


class TestVector7(unittest.TestCase):
    def test_sketch_vector_uses_plane_slot(self):
        vec = g.to_vector7(g.add_sketch(2))
        self.assertEqual(vec[g.SLOT_INDEX["t"]], float(g.ADD_SKETCH))
        self.assertEqual(vec[g.SLOT_INDEX["I"]], 2.0)
        self.assertEqual(vec[g.SLOT_INDEX["x"]], g.UNUSED)

    def test_line_vector_populates_xy_only(self):
        vec = g.to_vector7(g.add_line(0.5, -0.25))
        self.assertEqual(vec[g.SLOT_INDEX["x"]], 0.5)
        self.assertEqual(vec[g.SLOT_INDEX["y"]], -0.25)
        self.assertEqual(vec[g.SLOT_INDEX["alpha"]], g.UNUSED)
        self.assertEqual(vec[g.SLOT_INDEX["r"]], g.UNUSED)

    def test_circle_vector_populates_radius(self):
        vec = g.to_vector7(g.add_circle(0.1, 0.2, 0.5))
        self.assertEqual(vec[g.SLOT_INDEX["r"]], 0.5)

    def test_extrude_vector_populates_depth(self):
        vec = g.to_vector7(g.add_extrude(-0.4))
        self.assertEqual(vec[g.SLOT_INDEX["d"]], -0.4)

    def test_vector_length_is_seven(self):
        self.assertEqual(g.VEC_LEN, 7)
        self.assertEqual(len(g.to_vector7(g.add_line(0, 0))), 7)


class TestQuantisation(unittest.TestCase):
    def test_quantize_endpoints(self):
        self.assertEqual(g.quantize(-1.0, -1.0, 1.0), 0)
        self.assertEqual(g.quantize(1.0, -1.0, 1.0), 255)
        self.assertEqual(g.quantize(0.0, -1.0, 1.0), 128)

    def test_quantize_clamps_out_of_range(self):
        self.assertEqual(g.quantize(5.0, -1.0, 1.0), 255)
        self.assertEqual(g.quantize(-5.0, -1.0, 1.0), 0)

    def test_quantize_dequantize_roundtrip(self):
        for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
            q = g.quantize(v, -1.0, 1.0)
            self.assertAlmostEqual(g.dequantize(q, -1.0, 1.0), v, places=2)

    def test_vector7_quantise_keeps_discrete_and_marks_unused(self):
        q = g.quantize_vector7(g.to_vector7(g.add_line(0.0, 1.0)))
        self.assertEqual(q[g.SLOT_INDEX["t"]], g.ADD_LINE)
        self.assertEqual(q[g.SLOT_INDEX["x"]], 128)
        self.assertEqual(q[g.SLOT_INDEX["y"]], 255)
        self.assertEqual(q[g.SLOT_INDEX["alpha"]], g.UNUSED_LEVEL)

    def test_vector7_dequantise_roundtrip(self):
        raw = g.to_vector7(g.add_circle(0.5, -0.5, 0.25))
        q = g.quantize_vector7(raw)
        back = g.dequantize_vector7(q)
        self.assertAlmostEqual(back[g.SLOT_INDEX["x"]], 0.5, places=2)
        self.assertAlmostEqual(back[g.SLOT_INDEX["r"]], 0.25, places=2)
        self.assertEqual(back[g.SLOT_INDEX["alpha"]], g.UNUSED)

    def test_sweep_degrees(self):
        self.assertEqual(g.sweep_degrees(0.5), 90.0)
        self.assertEqual(g.sweep_degrees(1.0), 180.0)


class TestFeatureMatrix(unittest.TestCase):
    def test_cylinder_matrix_shape_and_markers(self):
        ops = [g.add_sketch(0), g.add_circle(0.0, 0.0, 0.5), g.add_extrude(0.5)]
        mat = g.build_feature_matrix(ops)
        self.assertEqual(len(mat), g.NC_DEFAULT)
        self.assertTrue(all(len(row) == 7 for row in mat))
        types = g.op_type_sequence(mat)
        self.assertEqual(types[0], g.SOP)
        self.assertEqual(types[1], g.ADD_SKETCH)
        self.assertEqual(types[4], g.EOP)      # after sketch/circle/extrude
        self.assertEqual(types[-1], g.EOP)     # padding

    def test_matrix_rejects_overlong_program(self):
        ops = [g.add_line(0.0, 0.0)] * 20
        with self.assertRaises(ValueError):
            g.build_feature_matrix(ops)


class TestArcCenter(unittest.TestCase):
    def test_quarter_circle_center_at_origin(self):
        cx, cy = g.arc_center((1.0, 0.0), (0.0, 1.0), 90.0)
        self.assertAlmostEqual(cx, 0.0, places=6)
        self.assertAlmostEqual(cy, 0.0, places=6)

    def test_negative_sweep_flips_side(self):
        pos = g.arc_center((1.0, 0.0), (0.0, 1.0), 90.0)
        neg = g.arc_center((1.0, 0.0), (0.0, 1.0), -90.0)
        self.assertNotAlmostEqual(pos[0], neg[0])
        # centre equidistant from both endpoints in each case
        for c in (pos, neg):
            r1 = math.dist(c, (1.0, 0.0))
            r2 = math.dist(c, (0.0, 1.0))
            self.assertAlmostEqual(r1, r2, places=6)

    def test_degenerate_arc_raises(self):
        with self.assertRaises(ValueError):
            g.arc_center((0.0, 0.0), (0.0, 0.0), 90.0)
        with self.assertRaises(ValueError):
            g.arc_center((1.0, 0.0), (0.0, 1.0), 360.0)


class TestParsing(unittest.TestCase):
    def test_start_points_inherited_from_precedent_curve(self):
        ops = [g.add_sketch(0), g.add_line(0.5, 0.0), g.add_line(0.5, 0.5),
               g.add_extrude(0.5)]
        parsed = g.parse_feature_matrix(g.build_feature_matrix(ops))
        # markers dropped, sketch + 2 lines + extrude
        types = [p.type for p in parsed]
        self.assertEqual(types, [g.ADD_SKETCH, g.ADD_LINE, g.ADD_LINE, g.ADD_EXTRUDE])
        first_line = parsed[1]
        second_line = parsed[2]
        # first line starts at origin (default), second inherits first's end
        self.assertAlmostEqual(first_line.start[0], 0.0, places=6)
        self.assertAlmostEqual(first_line.start[1], 0.0, places=6)
        self.assertAlmostEqual(second_line.start[0], first_line.end[0], places=2)
        self.assertAlmostEqual(second_line.start[1], first_line.end[1], places=2)

    def test_arc_center_reconstructed_during_parse(self):
        ops = [g.add_sketch(0), g.add_arc(0.0, 0.5, 0.5), g.add_extrude(0.5)]
        parsed = g.parse_feature_matrix(g.build_feature_matrix(ops))
        arc = parsed[1]
        self.assertEqual(arc.type, g.ADD_ARC)
        self.assertIsNotNone(arc.center)
        self.assertAlmostEqual(arc.sweep_deg, 90.0, places=0)

    def test_circle_center_and_radius(self):
        ops = [g.add_sketch(1), g.add_circle(0.25, -0.25, 0.5), g.add_extrude(0.5)]
        parsed = g.parse_feature_matrix(g.build_feature_matrix(ops))
        circle = parsed[1]
        self.assertAlmostEqual(circle.center[0], 0.25, places=2)
        self.assertAlmostEqual(circle.radius, 0.5, places=2)
        self.assertEqual(parsed[0].plane, 1)


class TestDslRendering(unittest.TestCase):
    def test_sketch_uses_plane_name(self):
        self.assertEqual(g.op_to_dsl(g.add_sketch(2)), 'add_sketch("YZ")')

    def test_extrude_uses_boolean_name(self):
        self.assertIn('"add"', g.op_to_dsl(g.add_extrude(0.5)))

    def test_program_to_dsl_lines(self):
        ops = [g.add_sketch(0), g.add_circle(0.0, 0.0, 0.5), g.add_extrude(0.5)]
        text = g.program_to_dsl(ops)
        self.assertEqual(len(text.splitlines()), 3)
        self.assertTrue(text.startswith("add_sketch"))


if __name__ == "__main__":
    unittest.main()
