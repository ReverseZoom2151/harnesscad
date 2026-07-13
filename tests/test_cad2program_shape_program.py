import unittest

from harnesscad.domain.reconstruction.translate.shape_program import (
    Bbox, PrimitiveInstance, ShapeProgram, make_instance, normalize_model_id,
    model_id_bracketed, serialize_python, parse_python, serialize_yaml,
    parse_yaml, program_bounds, normalize_to_first_octant, translate,
)

# Listing 1 from the paper (verbatim values).
LISTING_1 = """
bbox_0 = Bbox(507, 185, 805, 1014, 370, 50, 0)
model_0 = <model_57761062>()
bbox_1 = Bbox(25, 185, 390, 50, 370, 780, 0)
model_1 = <model_57758898>()
bbox_2 = Bbox(532, 195, 390, 964, 350, 780, 0)
model_2 = <model_115813862>(N=1, NKA=928, DBXX=1, BT=18)
bbox_3 = Bbox(532, 185, 390, 928, 330, 18, 0)
model_3 = <model_57253481>()
bbox_4 = Bbox(291, 11, 390, 478, 18, 776, 0)
model_4 = <model_82289390>(openDirection=0, uCove=18, dCover=18, lCover=18, rCover=18)
"""


class NormalizeIdTest(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(normalize_model_id("<model_57761062>"), "model_57761062")
        self.assertEqual(normalize_model_id("model_57761062"), "model_57761062")
        self.assertEqual(normalize_model_id("57761062"), "model_57761062")
        self.assertEqual(model_id_bracketed("57761062"), "<model_57761062>")


class ParsePythonTest(unittest.TestCase):
    def test_parse_listing1(self):
        prog = parse_python(LISTING_1)
        self.assertEqual(len(prog), 5)
        self.assertEqual(prog.instances[0].model_id, "model_57761062")
        self.assertEqual(prog.instances[0].bbox,
                         Bbox(507, 185, 805, 1014, 370, 50, 0))
        self.assertEqual(prog.instances[0].params, ())

    def test_model_specific_params(self):
        prog = parse_python(LISTING_1)
        self.assertEqual(prog.instances[2].param_dict(),
                         {"N": 1, "NKA": 928, "DBXX": 1, "BT": 18})
        self.assertEqual(prog.instances[4].param_dict()["openDirection"], 0)
        self.assertEqual(len(prog.instances[4].params), 5)

    def test_whitespace_padding(self):
        # Extracted PDFs pad numbers with spaces.
        prog = parse_python("bbox_0 = Bbox(507,  185,  805,  1014,  370,  50,  0)\n"
                            "model_0 = <model_57761062>()")
        self.assertEqual(prog.instances[0].bbox.position_x, 507)

    def test_six_number_box_defaults_angle(self):
        prog = parse_python("bbox_0 = Bbox(1, 2, 3, 4, 5, 6)\n"
                            "model_0 = <model_9>()")
        self.assertEqual(prog.instances[0].bbox.angle_z, 0)

    def test_missing_model_raises(self):
        with self.assertRaises(ValueError):
            parse_python("bbox_0 = Bbox(1,2,3,4,5,6,0)")


class PythonRoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        prog = parse_python(LISTING_1)
        text = serialize_python(prog)
        prog2 = parse_python(text)
        self.assertEqual(len(prog), len(prog2))
        for a, b in zip(prog.instances, prog2.instances):
            self.assertEqual(a.model_id, b.model_id)
            self.assertEqual(a.bbox, b.bbox)
            self.assertEqual(a.param_dict(), b.param_dict())

    def test_float_no_trailing_zero(self):
        prog = ShapeProgram([make_instance("m1", (407.5, 280, 1079, 779, 520, 18, 0))])
        text = serialize_python(prog)
        self.assertIn("407.5", text)
        self.assertNotIn("407.50", text)
        self.assertEqual(parse_python(text).instances[0].bbox.position_x, 407.5)


class YamlRoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        prog = parse_python(LISTING_1)
        text = serialize_yaml(prog)
        prog2 = parse_yaml(text)
        self.assertEqual(len(prog2), 5)
        for a, b in zip(prog.instances, prog2.instances):
            self.assertEqual(a.model_id, b.model_id)
            self.assertEqual(a.bbox, b.bbox)
            self.assertEqual(a.param_dict(), b.param_dict())

    def test_yaml_shape(self):
        prog = ShapeProgram([make_instance("m1", (1, 2, 3, 4, 5, 6, 0),
                                           {"N": 2, "BT": 9})])
        text = serialize_yaml(prog)
        self.assertIn("- id: 0", text)
        self.assertIn("position_x: 1", text)
        self.assertIn("model_id: <model_m1>", text)
        self.assertIn("N: 2", text)

    def test_cross_language_equivalence(self):
        prog = parse_python(LISTING_1)
        from_yaml = parse_yaml(serialize_yaml(prog))
        from_py = parse_python(serialize_python(prog))
        for a, b in zip(from_yaml.instances, from_py.instances):
            self.assertEqual(a.bbox, b.bbox)
            self.assertEqual(a.param_dict(), b.param_dict())


class CanonicalPoseTest(unittest.TestCase):
    def test_bounds(self):
        prog = ShapeProgram([make_instance("m", (10, 10, 10, 4, 4, 4, 0))])
        lo, hi = program_bounds(prog)
        self.assertEqual(lo, (8, 8, 8))
        self.assertEqual(hi, (12, 12, 12))

    def test_first_octant(self):
        prog = ShapeProgram([
            make_instance("a", (10, 10, 10, 4, 4, 4, 0)),
            make_instance("b", (20, 5, 30, 2, 2, 2, 0)),
        ])
        norm = normalize_to_first_octant(prog)
        lo, _ = program_bounds(norm)
        self.assertAlmostEqual(lo[0], 0.0)
        self.assertAlmostEqual(lo[1], 0.0)
        self.assertAlmostEqual(lo[2], 0.0)

    def test_translate_preserves_size(self):
        prog = ShapeProgram([make_instance("m", (1, 2, 3, 4, 5, 6, 0))])
        moved = translate(prog, (10, 0, -5))
        self.assertEqual(moved.instances[0].bbox.size, (4, 5, 6))
        self.assertEqual(moved.instances[0].bbox.position, (11, 2, -2))

    def test_empty_bounds_raises(self):
        with self.assertRaises(ValueError):
            program_bounds(ShapeProgram())


if __name__ == "__main__":
    unittest.main()
