"""Tests for formats.sgraphs2_onshape_json."""

import unittest

from harnesscad.io.formats.onshape_json import (
    Arc,
    Circle,
    EntityType,
    GenericEntity,
    Line,
    Point,
    SubnodeType,
    entity_to_dict,
    inspect_entity_type,
    parameter_layout,
    parse_entity,
    parse_sketch,
    sketch_to_dict,
    subnode_ids,
)


def _line_dict(entity_id="l0"):
    return {
        "type": 155,
        "typeName": "BTMSketchCurveSegment",
        "message": {
            "entityId": entity_id,
            "startPointId": entity_id + ".start",
            "endPointId": entity_id + ".end",
            "isConstruction": False,
            "startParam": -1.0,
            "endParam": 1.0,
            "geometry": {
                "type": 117,
                "typeName": "BTCurveGeometryLine",
                "message": {"dirX": 1.0, "dirY": 0.0, "pntX": 0.0, "pntY": 2.0},
            },
        },
    }


def _circle_dict(entity_id="c0"):
    return {
        "type": 4,
        "typeName": "BTMSketchCurve",
        "message": {
            "entityId": entity_id,
            "centerId": entity_id + ".center",
            "isConstruction": True,
            "geometry": {
                "type": 115,
                "typeName": "BTCurveGeometryCircle",
                "message": {
                    "xCenter": 1.0,
                    "yCenter": 2.0,
                    "xDir": 1.0,
                    "yDir": 0.0,
                    "radius": 3.0,
                    "clockwise": False,
                },
            },
        },
    }


def _arc_dict(entity_id="a0"):
    return {
        "type": 155,
        "typeName": "BTMSketchCurveSegment",
        "message": {
            "entityId": entity_id,
            "centerId": entity_id + ".center",
            "startPointId": entity_id + ".start",
            "endPointId": entity_id + ".end",
            "isConstruction": False,
            "startParam": 0.0,
            "endParam": 1.5,
            "geometry": {
                "type": 115,
                "typeName": "BTCurveGeometryCircle",
                "message": {
                    "xCenter": 0.0,
                    "yCenter": 0.0,
                    "xDir": 1.0,
                    "yDir": 0.0,
                    "radius": 2.0,
                    "clockwise": True,
                },
            },
        },
    }


def _point_dict(entity_id="p0"):
    return {
        "type": 158,
        "typeName": "BTMSketchPoint",
        "message": {"entityId": entity_id, "isConstruction": False, "x": 4.0, "y": -5.0},
    }


class TestInspectEntityType(unittest.TestCase):
    def test_point(self):
        self.assertIs(inspect_entity_type(_point_dict()), EntityType.Point)

    def test_line(self):
        self.assertIs(inspect_entity_type(_line_dict()), EntityType.Line)

    def test_circle_vs_arc_distinguished_by_container(self):
        # Identical BTCurveGeometryCircle geometry; only the container differs.
        self.assertIs(inspect_entity_type(_circle_dict()), EntityType.Circle)
        self.assertIs(inspect_entity_type(_arc_dict()), EntityType.Arc)

    def test_missing_geometry_is_unknown(self):
        blob = {
            "type": 999,
            "typeName": "BTMSketchTextEntity",
            "message": {"entityId": "t0", "isConstruction": False},
        }
        self.assertIs(inspect_entity_type(blob), EntityType.Unknown)

    def test_ellipse_and_elliptical_arc(self):
        base = {
            "typeName": "BTMSketchCurve",
            "message": {
                "entityId": "e0",
                "isConstruction": False,
                "geometry": {"typeName": "BTCurveGeometryEllipse", "type": 116, "message": {}},
            },
        }
        self.assertIs(inspect_entity_type(base), EntityType.Ellipse)
        base["typeName"] = "BTMSketchCurveSegment"
        self.assertIs(inspect_entity_type(base), EntityType.Unknown)

    def test_spline_requires_type_117(self):
        blob = {
            "typeName": "BTMSketchCurve",
            "message": {
                "entityId": "s0",
                "isConstruction": False,
                "geometry": {"typeName": "BTCurveGeometrySpline", "type": 117, "message": {}},
            },
        }
        self.assertIs(inspect_entity_type(blob), EntityType.Spline)
        blob["message"]["geometry"]["type"] = 118
        self.assertIs(inspect_entity_type(blob), EntityType.Unknown)

    def test_unrecognised_geometry_is_unknown(self):
        blob = {
            "typeName": "BTMSketchCurve",
            "message": {
                "entityId": "x0",
                "isConstruction": False,
                "geometry": {"typeName": "BTCurveGeometryHelix", "type": 1, "message": {}},
            },
        }
        self.assertIs(inspect_entity_type(blob), EntityType.Unknown)


class TestParse(unittest.TestCase):
    def test_parse_point(self):
        point = parse_entity(_point_dict())
        self.assertIsInstance(point, Point)
        self.assertEqual(point.entityId, "p0")
        self.assertEqual((point.x, point.y), (4.0, -5.0))
        self.assertFalse(point.isConstruction)

    def test_parse_line_parameters(self):
        line = parse_entity(_line_dict())
        self.assertIsInstance(line, Line)
        self.assertEqual((line.pntX, line.pntY), (0.0, 2.0))
        self.assertEqual((line.dirX, line.dirY), (1.0, 0.0))
        self.assertEqual((line.startParam, line.endParam), (-1.0, 1.0))

    def test_parse_circle_flags(self):
        circle = parse_entity(_circle_dict())
        self.assertIsInstance(circle, Circle)
        self.assertTrue(circle.isConstruction)
        self.assertEqual(circle.radius, 3.0)
        self.assertFalse(circle.clockwise)

    def test_parse_arc_carries_params_and_clockwise(self):
        arc = parse_entity(_arc_dict())
        self.assertIsInstance(arc, Arc)
        self.assertTrue(arc.clockwise)
        self.assertEqual((arc.startParam, arc.endParam), (0.0, 1.5))
        self.assertEqual(arc.radius, 2.0)

    def test_unknown_entity_becomes_generic(self):
        blob = {
            "type": 999,
            "typeName": "BTMSketchTextEntity",
            "message": {"entityId": "t0", "isConstruction": True},
        }
        entity = parse_entity(blob)
        self.assertIsInstance(entity, GenericEntity)
        self.assertIs(entity.type, EntityType.Unknown)
        self.assertTrue(entity.isConstruction)

    def test_point_from_dict_rejects_wrong_container(self):
        with self.assertRaises(ValueError):
            Point.from_dict(_line_dict())


class TestRoundTrip(unittest.TestCase):
    def test_entity_round_trip(self):
        for source in (_point_dict(), _line_dict(), _circle_dict(), _arc_dict()):
            with self.subTest(typeName=source["typeName"]):
                out = entity_to_dict(parse_entity(source))
                self.assertEqual(out["type"], source["type"])
                self.assertEqual(out["typeName"], source["typeName"])
                self.assertEqual(out["message"], source["message"])

    def test_generic_entity_round_trips_verbatim(self):
        blob = {
            "type": 999,
            "typeName": "BTMSketchTextEntity",
            "message": {"entityId": "t0", "isConstruction": False, "payload": [1, 2]},
        }
        self.assertEqual(entity_to_dict(parse_entity(blob)), blob)

    def test_sketch_round_trip_preserves_order(self):
        source = {
            "entities": [_point_dict("p0"), _line_dict("l0"), _arc_dict("a0")],
            "constraints": [{"typeName": "BTMSketchConstraint", "message": {"identifier": "k0"}}],
        }
        sketch = parse_sketch(source)
        self.assertEqual(list(sketch.entities), ["p0", "l0", "a0"])
        self.assertEqual(len(sketch), 3)
        self.assertEqual(sketch_to_dict(sketch), source)

    def test_of_type_filter(self):
        sketch = parse_sketch({"entities": [_point_dict("p0"), _circle_dict("c0")]})
        self.assertEqual([e.entityId for e in sketch.of_type(EntityType.Circle)], ["c0"])
        self.assertEqual(sketch.constraints, [])


class TestParameterLayout(unittest.TestCase):
    def test_layouts(self):
        self.assertEqual(parameter_layout(EntityType.Point), (("x", "y"), ("isConstruction",)))
        self.assertEqual(
            parameter_layout(EntityType.Line)[0],
            ("dirX", "dirY", "pntX", "pntY", "startParam", "endParam"),
        )
        self.assertEqual(
            parameter_layout(EntityType.Circle),
            (
                ("xCenter", "yCenter", "xDir", "yDir", "radius"),
                ("isConstruction", "clockwise"),
            ),
        )
        self.assertEqual(len(parameter_layout(EntityType.Arc)[0]), 7)

    def test_layout_unknown_type_raises(self):
        with self.assertRaises(KeyError):
            parameter_layout(EntityType.Spline)

    def test_parameters_follow_layout_order(self):
        arc = parse_entity(_arc_dict())
        params = arc.parameters()
        self.assertEqual(
            list(params),
            [
                "xCenter",
                "yCenter",
                "xDir",
                "yDir",
                "radius",
                "startParam",
                "endParam",
                "isConstruction",
                "clockwise",
            ],
        )
        self.assertEqual(params["radius"], 2.0)
        self.assertIs(params["clockwise"], True)


class TestSubnodes(unittest.TestCase):
    def test_line_subnodes(self):
        line = parse_entity(_line_dict("l7"))
        self.assertEqual(subnode_ids(line), ("l7.start", "l7.end"))
        self.assertEqual(
            Line.subnode_types(), (SubnodeType.SN_Start, SubnodeType.SN_End)
        )

    def test_circle_subnodes(self):
        self.assertEqual(subnode_ids(parse_entity(_circle_dict("c1"))), ("c1.center",))

    def test_arc_subnodes(self):
        self.assertEqual(
            subnode_ids(parse_entity(_arc_dict("a1"))),
            ("a1.center", "a1.start", "a1.end"),
        )

    def test_point_has_no_subnodes(self):
        self.assertEqual(subnode_ids(parse_entity(_point_dict())), ())

    def test_subnode_ids_match_serialised_ids(self):
        arc = parse_entity(_arc_dict("a2"))
        message = entity_to_dict(arc)["message"]
        self.assertEqual(
            (message["centerId"], message["startPointId"], message["endPointId"]),
            subnode_ids(arc),
        )


if __name__ == "__main__":
    unittest.main()
