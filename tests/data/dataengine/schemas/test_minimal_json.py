"""Tests for the Text2CAD minimal-JSON schema (dataengine.t2c3_minimal_json_schema)."""

import math
import unittest

from harnesscad.data.dataengine.schemas.minimal_json import (
    MinimalJsonError,
    build_minimal_json,
    coordinate_system_json,
    curve_json,
    extrusion_json,
    float_round,
    loop_json,
    parse_curve,
    parse_minimal_json,
    sketch_dimension,
    sketch_json,
)


def _square():
    return [
        {"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)},
        {"type": "line", "start": (1.0, 0.0), "end": (1.0, 2.0)},
        {"type": "line", "start": (1.0, 2.0), "end": (0.0, 0.0)},
    ]


def _part():
    return {
        "coordinate_system": {"origin": (0.1, 0.2, 0.3), "euler": (math.pi / 2, 0.0, 0.0)},
        "sketch": [[_square()]],
        "extrusion": {"extent_one": 0.25, "extent_two": 0.125,
                      "sketch_size": 0.5, "boolean": 1},
    }


class TestFloatRound(unittest.TestCase):
    def test_rounds_to_four_decimals(self):
        self.assertEqual(float_round(0.123456), 0.1235)

    def test_negative_zero_normalised(self):
        self.assertEqual(float_round(-0.00001), 0.0)
        self.assertEqual(repr(float_round(-0.0)), "0.0")


class TestSerialisation(unittest.TestCase):
    def test_coordinate_system_in_degrees(self):
        js = coordinate_system_json((1.0, 2.0, 3.0), (math.pi, 0.0, -math.pi / 2))
        self.assertEqual(js["Euler Angles"], [180.0, 0.0, -90.0])
        self.assertEqual(js["Translation Vector"], [1.0, 2.0, 3.0])

    def test_coordinate_system_arity(self):
        with self.assertRaises(MinimalJsonError):
            coordinate_system_json((1.0, 2.0), (0.0, 0.0, 0.0))

    def test_curve_json_keys(self):
        line = curve_json({"type": "line", "start": (0, 0), "end": (1, 1)})
        self.assertEqual(sorted(line), ["End Point", "Start Point"])
        arc = curve_json({"type": "arc", "start": (0, 0), "mid": (1, 1), "end": (2, 0)})
        self.assertIn("Mid Point", arc)
        circle = curve_json({"type": "circle", "center": (0, 0), "radius": 3})
        self.assertEqual(circle, {"Centre": [0.0, 0.0], "Radius": 3.0})

    def test_unknown_curve_rejected(self):
        with self.assertRaises(MinimalJsonError):
            curve_json({"type": "spline"})

    def test_curves_numbered_per_type(self):
        loop = loop_json(_square() + [{"type": "arc", "start": (0, 0),
                                       "mid": (1, 1), "end": (2, 2)}])
        self.assertEqual(list(loop), ["line_1", "line_2", "line_3", "arc_1"])

    def test_faces_and_loops_one_based(self):
        js = sketch_json([[_square(), _square()], [_square()]])
        self.assertEqual(list(js), ["face_1", "face_2"])
        self.assertEqual(list(js["face_1"]), ["loop_1", "loop_2"])

    def test_extrusion_uses_operation_name(self):
        js = extrusion_json({"extent_one": 0.25, "extent_two": 0.0,
                             "sketch_size": 0.5, "boolean": 2})
        self.assertEqual(js["operation"], "CutFeatureOperation")
        self.assertEqual(js["extrude_depth_towards_normal"], 0.25)
        self.assertEqual(js["sketch_scale"], 0.5)

    def test_bad_boolean(self):
        with self.assertRaises(MinimalJsonError):
            extrusion_json({"extent_one": 0, "extent_two": 0,
                            "sketch_size": 1, "boolean": 7})

    def test_sketch_dimension(self):
        self.assertEqual(sketch_dimension([[_square()]]), (1.0, 2.0))

    def test_sketch_dimension_circle(self):
        sketch = [[[{"type": "circle", "center": (1.0, 1.0), "radius": 2.0}]]]
        self.assertEqual(sketch_dimension(sketch), (4.0, 4.0))


class TestDocument(unittest.TestCase):
    def test_top_level_shape(self):
        doc = build_minimal_json([_part()], final_name="bracket")
        self.assertEqual(list(doc), ["final_name", "final_shape", "parts"])
        self.assertEqual(doc["final_name"], "bracket")
        self.assertEqual(doc["final_shape"], "")
        self.assertEqual(list(doc["parts"]), ["part_1"])
        part = doc["parts"]["part_1"]
        self.assertEqual(list(part),
                         ["coordinate_system", "sketch", "extrusion", "description"])

    def test_description_dimensions(self):
        desc = build_minimal_json([_part()])["parts"]["part_1"]["description"]
        self.assertEqual(desc["length"], 1.0)
        self.assertEqual(desc["width"], 2.0)
        self.assertEqual(desc["height"], 0.375)   # extent_one + extent_two
        self.assertEqual(desc["name"], "")

    def test_empty_model_rejected(self):
        with self.assertRaises(MinimalJsonError):
            build_minimal_json([])

    def test_multi_part_numbering(self):
        doc = build_minimal_json([_part(), _part()])
        self.assertEqual(list(doc["parts"]), ["part_1", "part_2"])


class TestRoundTrip(unittest.TestCase):
    def test_parse_inverts_build(self):
        model = [_part()]
        parsed = parse_minimal_json(build_minimal_json(model))
        self.assertEqual(len(parsed), 1)
        got = parsed[0]
        self.assertEqual(got["extrusion"]["boolean"], 1)
        self.assertAlmostEqual(got["extrusion"]["extent_one"], 0.25)
        self.assertAlmostEqual(got["coordinate_system"]["euler"][0], math.pi / 2)
        self.assertEqual(got["coordinate_system"]["origin"], (0.1, 0.2, 0.3))
        self.assertEqual(got["sketch"][0][0][0]["start"], (0.0, 0.0))
        self.assertEqual([c["type"] for c in got["sketch"][0][0]], ["line"] * 3)

    def test_circle_round_trip(self):
        part = _part()
        part["sketch"] = [[[{"type": "circle", "center": (1.0, 1.0), "radius": 0.5}]]]
        got = parse_minimal_json(build_minimal_json([part]))[0]
        circle = got["sketch"][0][0][0]
        self.assertEqual(circle["center"], (1.0, 1.0))
        self.assertEqual(circle["radius"], 0.5)

    def test_parts_sorted_numerically_not_lexically(self):
        doc = build_minimal_json([_part()] * 11)
        self.assertEqual(len(parse_minimal_json(doc)), 11)
        self.assertEqual(list(doc["parts"])[10], "part_11")

    def test_parse_curve_key(self):
        curve = parse_curve("arc_2", {"Start Point": [0, 0], "Mid Point": [1, 1],
                                      "End Point": [2, 0]})
        self.assertEqual(curve["type"], "arc")
        self.assertEqual(curve["mid"], (1, 1))

    def test_unknown_operation_rejected(self):
        doc = build_minimal_json([_part()])
        doc["parts"]["part_1"]["extrusion"]["operation"] = "FilletFeature"
        with self.assertRaises(MinimalJsonError):
            parse_minimal_json(doc)

    def test_missing_parts_rejected(self):
        with self.assertRaises(MinimalJsonError):
            parse_minimal_json({"final_name": ""})


if __name__ == "__main__":
    unittest.main()
