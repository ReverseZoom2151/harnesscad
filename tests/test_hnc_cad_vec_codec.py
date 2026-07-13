import unittest

from harnesscad.domain.reconstruction.tokens.hnc_cad_vec_codec import (
    SKETCH_END, FACE_END, LOOP_END, LINE, ARC, CIRCLE,
    PARAM_WIDTH, SENTINEL,
    OP_ADD, OP_CUT, OP_INTERSECT,
    set_op_code,
    quantize,
    normalize_diagonal,
    normalize_max_extent,
    center_of,
    circle_cardinal_points,
    encode_line, encode_arc, encode_circle,
    encode_sketch,
    encode_extrude,
)


class TestQuantize(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        self.assertEqual(quantize(-1.0, 8, -1.0, 1.0), 0)
        self.assertEqual(quantize(1.0, 8, -1.0, 1.0), 255)
        self.assertEqual(quantize(0.0, 8, -1.0, 1.0), 127)

    def test_clip_out_of_range(self):
        self.assertEqual(quantize(-5.0, 8, -1.0, 1.0), 0)
        self.assertEqual(quantize(5.0, 8, -1.0, 1.0), 255)

    def test_uses_two_pow_bits_minus_one(self):
        # 6-bit -> max level 63
        self.assertEqual(quantize(1.0, 6, -1.0, 1.0), 63)


class TestNormalization(unittest.TestCase):
    def test_two_schemes_differ(self):
        verts = [(-1.0, -1.0), (1.0, 1.0)]  # centred, extents 2x2
        # diagonal: 0.5*sqrt(4+4) = sqrt(2)
        self.assertAlmostEqual(normalize_diagonal(verts), 2 ** 0.5)
        # max extent: max(2,2)/2 = 1.0
        self.assertAlmostEqual(normalize_max_extent(verts), 1.0)

    def test_diagonal_vs_max_on_non_square(self):
        verts = [(0.0, 0.0), (4.0, 3.0)]
        self.assertAlmostEqual(normalize_diagonal(verts), 2.5)   # 0.5*5
        self.assertAlmostEqual(normalize_max_extent(verts), 2.0)  # 4/2

    def test_center(self):
        self.assertEqual(center_of([(0.0, 0.0), (4.0, 2.0)]), (2.0, 1.0))


class TestCircleCardinal(unittest.TestCase):
    def test_nsew_order(self):
        n, s, e, w = circle_cardinal_points((0.0, 0.0), 2.0)
        self.assertEqual(n, (0.0, 2.0))
        self.assertEqual(s, (0.0, -2.0))
        self.assertEqual(e, (2.0, 0.0))
        self.assertEqual(w, (-2.0, 0.0))


class TestCurveEncoding(unittest.TestCase):
    def test_line_start_only_padded(self):
        cmd, p = encode_line((0.0, 0.0), (0.0, 0.0), 1.0)
        self.assertEqual(cmd, LINE)
        self.assertEqual(len(p), PARAM_WIDTH)
        self.assertEqual(p[0], 127)
        self.assertEqual(p[1], 127)
        self.assertEqual(p[2:], [SENTINEL] * 6)  # implicit endpoint => sentinels

    def test_arc_start_and_mid(self):
        cmd, p = encode_arc((0.0, 0.0), (1.0, 1.0), (0.0, 0.0), 1.0)
        self.assertEqual(cmd, ARC)
        self.assertEqual(p[0:4], [127, 127, 255, 255])
        self.assertEqual(p[4:], [SENTINEL] * 4)

    def test_circle_four_points_fill_all_slots(self):
        cmd, p = encode_circle((0.0, 0.0), 1.0, (0.0, 0.0), 1.0)
        self.assertEqual(cmd, CIRCLE)
        self.assertNotIn(SENTINEL, p)  # all 8 slots used
        # N=(0,1)->(127,255), S=(0,-1)->(127,0), E=(1,0)->(255,127), W=(-1,0)->(0,127)
        self.assertEqual(p, [127, 255, 127, 0, 255, 127, 0, 127])


class TestSketchStream(unittest.TestCase):
    def test_structural_tokens_and_order(self):
        faces = [
            [  # face 0
                [{"type": "line", "start": (0.0, 0.0)},
                 {"type": "arc", "start": (0.5, 0.5), "mid": (0.7, 0.7)}],  # loop 0
                [{"type": "circle", "center": (0.0, 0.0), "radius": 0.5}],   # loop 1
            ],
        ]
        cmds, params = encode_sketch(faces, (0.0, 0.0), 1.0)
        # line, arc, LOOP_END, circle, LOOP_END, FACE_END, SKETCH_END
        self.assertEqual(cmds, [LINE, ARC, LOOP_END, CIRCLE, LOOP_END,
                                FACE_END, SKETCH_END])
        self.assertEqual(len(cmds), len(params))
        for p in params:
            self.assertEqual(len(p), PARAM_WIDTH)
        # structural tokens carry all-sentinel params
        self.assertEqual(params[2], [SENTINEL] * PARAM_WIDTH)
        self.assertEqual(params[-1], [SENTINEL] * PARAM_WIDTH)


class TestExtrude(unittest.TestCase):
    def test_set_op_mapping(self):
        self.assertEqual(set_op_code("JoinFeatureOperation"), OP_ADD)
        self.assertEqual(set_op_code("NewBodyFeatureOperation"), OP_ADD)
        self.assertEqual(set_op_code("CutFeatureOperation"), OP_CUT)
        self.assertEqual(set_op_code("IntersectFeatureOperation"), OP_INTERSECT)
        with self.assertRaises(ValueError):
            set_op_code("BogusOperation")

    def test_11_slot_layout(self):
        vec = encode_extrude(
            center=(0.0, 0.0, 0.0), scale=1.0, ext_values=(1.0, -1.0),
            t_orig=(0.0, 0.0, 0.0),
            t_x=(1, 0, 0), t_y=(0, 1, 0), t_z=(0, 0, 1),
            set_op="JoinFeatureOperation")
        self.assertEqual(len(vec), 11)
        # center 3 -> 127, scale -> 255, ext_v (1,-1)->(255,0),
        # t_orig 3 -> 127, rot_idx (identity frame=20), op=1
        self.assertEqual(vec[0:3], [127, 127, 127])
        self.assertEqual(vec[3], 255)
        self.assertEqual(vec[4:6], [255, 0])
        self.assertEqual(vec[6:9], [127, 127, 127])
        self.assertEqual(vec[9], 20)     # identity orientation index
        self.assertEqual(vec[10], OP_ADD)

    def test_rejects_bad_orientation(self):
        with self.assertRaises(ValueError):
            encode_extrude((0.0, 0.0, 0.0), 1.0, (1.0, 0.0), (0.0, 0.0, 0.0),
                           (0, 0, 0), (0, 0, 0), (0, 0, 0), "CutFeatureOperation")

    def test_validates_lengths(self):
        with self.assertRaises(ValueError):
            encode_extrude((0.0, 0.0), 1.0, (1.0, 0.0), (0.0, 0.0, 0.0),
                           (1, 0, 0), (0, 1, 0), (0, 0, 1), "CutFeatureOperation")
        with self.assertRaises(ValueError):
            encode_extrude((0.0, 0.0, 0.0), 1.0, (1.0,), (0.0, 0.0, 0.0),
                           (1, 0, 0), (0, 1, 0), (0, 0, 1), "CutFeatureOperation")


if __name__ == "__main__":
    unittest.main()
