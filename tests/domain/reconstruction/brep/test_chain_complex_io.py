"""Tests for the ComplexGen .complex file reader/writer."""

import os
import tempfile
import unittest

from harnesscad.domain.reconstruction.brep import chain_complex as cc
from harnesscad.domain.reconstruction.brep import chain_complex_io as io
from tests.domain.reconstruction.brep.test_chain_complex import cube_complex


SMALL = """\
2 1 1
0.0 0.0 0.0
1.0 0.0 0.0
Line 0.0 0.0 0.0 0.0 0.5 0.0 0.0 1.0 0.0 0.0
Plane 0.0 0.0 0.0 1.0 0.0 0.0 0.5 0.5 0.0
1.0 1.0
1.0
"""


class TestParse(unittest.TestCase):
    def test_parse_small(self):
        cf = io.parse_complex(SMALL)
        self.assertEqual(len(cf.corners), 2)
        self.assertEqual(cf.curve_types, ("Line",))
        self.assertEqual(cf.patch_types, ("Plane",))
        self.assertEqual(len(cf.curve_points[0]), 3)
        self.assertEqual(len(cf.patch_points[0]), 3)
        self.assertEqual(cf.curve_corner, ((1.0, 1.0),))
        self.assertEqual(cf.patch_curve, ((1.0,),))

    def test_comments_and_blank_lines_ignored(self):
        cf = io.parse_complex("# header\n\n" + SMALL + "\n")
        self.assertEqual(len(cf.corners), 2)

    def test_empty(self):
        with self.assertRaises(ValueError):
            io.parse_complex("")

    def test_bad_header(self):
        with self.assertRaises(ValueError):
            io.parse_complex("2 1\n")

    def test_truncated(self):
        with self.assertRaises(ValueError):
            io.parse_complex("2 1 1\n0.0 0.0 0.0\n")

    def test_unknown_curve_type(self):
        bad = SMALL.replace("Line", "Helix")
        with self.assertRaises(ValueError):
            io.parse_complex(bad)

    def test_unknown_patch_type(self):
        bad = SMALL.replace("Plane", "Quadric")
        with self.assertRaises(ValueError):
            io.parse_complex(bad)

    def test_bad_incidence_width(self):
        bad = SMALL.replace("1.0 1.0\n1.0\n", "1.0\n1.0\n")
        with self.assertRaises(ValueError):
            io.parse_complex(bad)

    def test_non_multiple_of_three(self):
        bad = SMALL.replace("Line 0.0 0.0 0.0 0.0 0.5 0.0 0.0 1.0 0.0 0.0",
                            "Line 0.0 0.0 0.0 0.0 0.5")
        with self.assertRaises(ValueError):
            io.parse_complex(bad)


class TestVocabularies(unittest.TestCase):
    def test_type_ids_match_reference(self):
        self.assertEqual(io.CURVE_TYPE_ID["Circle"], 0)
        self.assertEqual(io.CURVE_TYPE_ID["Ellipse"], 3)
        self.assertEqual(io.PATCH_TYPE_ID["Cylinder"], 0)
        self.assertEqual(io.PATCH_TYPE_ID["Sphere"], 5)


class TestRoundTrip(unittest.TestCase):
    def test_text_round_trip(self):
        cf = io.parse_complex(SMALL)
        again = io.parse_complex(io.serialize_complex(cf))
        self.assertEqual(cf, again)

    def test_cube_round_trip(self):
        cx = cube_complex()
        cf = io.from_chain_complex(cx, ["Line"] * 12, ["Plane"] * 6)
        text = io.serialize_complex(cf)
        parsed = io.parse_complex(text)
        self.assertEqual(parsed.curve_types, cf.curve_types)
        self.assertEqual(parsed.patch_types, cf.patch_types)
        rebuilt = parsed.to_chain_complex()
        self.assertEqual(rebuilt.curve_corner, cx.curve_corner)
        self.assertEqual(rebuilt.patch_curve, cx.patch_curve)
        self.assertTrue(cc.is_valid(rebuilt))
        self.assertEqual(cc.euler_characteristic(rebuilt), 2)

    def test_closed_flag_preserved(self):
        curve = cc.Curve(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)), True)
        cx = cc.make_complex([], [curve], [cc.Patch(curve.points)], [[]], [[1]])
        cf = io.from_chain_complex(cx, ["Circle"], ["Cylinder"])
        rebuilt = io.parse_complex(io.serialize_complex(cf)).to_chain_complex()
        self.assertTrue(rebuilt.curves[0].closed)

    def test_bad_type_list(self):
        cx = cube_complex()
        with self.assertRaises(ValueError):
            io.from_chain_complex(cx, ["Line"], None)
        with self.assertRaises(ValueError):
            io.from_chain_complex(cx, ["Spiral"] * 12, None)

    def test_file_io(self):
        cf = io.parse_complex(SMALL)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.complex")
            io.save_complex(path, cf)
            self.assertEqual(io.load_complex(path), cf)


class TestToChainComplex(unittest.TestCase):
    def test_threshold_applied(self):
        text = SMALL.replace("1.0 1.0\n", "0.9 0.2\n")
        cx = io.parse_complex(text).to_chain_complex(threshold=0.5)
        self.assertEqual(cx.curve_corner, ((1, 0),))


if __name__ == "__main__":
    unittest.main()
