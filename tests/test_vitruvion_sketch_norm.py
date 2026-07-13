"""Tests for geometry.vitruvion_sketch_norm."""

import math
import unittest

from geometry.vitruvion_sketch_norm import (
    NUM_PARAMS,
    VArc,
    VCircle,
    VLine,
    VPoint,
    center_sketch,
    entity_bbox,
    entity_from_params,
    normalize_sketch,
    parameterize_entity,
    rescale_sketch,
    sketch_bbox,
)


def _line(sx, sy, ex, ey):
    return entity_from_params([sx, sy, ex, ey])


class TestEntityBBox(unittest.TestCase):
    def test_point_bbox_is_degenerate(self):
        self.assertEqual(entity_bbox(VPoint(x=1.0, y=-2.0)), ((1.0, -2.0), (1.0, -2.0)))

    def test_circle_bbox_is_full_box(self):
        box = entity_bbox(VCircle(xCenter=1.0, yCenter=2.0, radius=3.0))
        self.assertEqual(box, ((-2.0, -1.0), (4.0, 5.0)))

    def test_line_bbox_orders_corners(self):
        box = entity_bbox(_line(3.0, -1.0, -1.0, 2.0))
        (x0, y0), (x1, y1) = box
        self.assertAlmostEqual(x0, -1.0)
        self.assertAlmostEqual(y0, -1.0)
        self.assertAlmostEqual(x1, 3.0)
        self.assertAlmostEqual(y1, 2.0)

    def test_quarter_arc_bbox_is_endpoints_only(self):
        # Unit arc from angle 0 to pi/2: sweeps quadrant 1 only, touches no axis
        # extremum, so the bbox is exactly the endpoints' box (NOT the circle box).
        arc = VArc(xCenter=0.0, yCenter=0.0, radius=1.0, startParam=0.0,
                   endParam=math.pi / 2)
        (x0, y0), (x1, y1) = entity_bbox(arc)
        self.assertAlmostEqual(x0, 0.0)
        self.assertAlmostEqual(y0, 0.0)
        self.assertAlmostEqual(x1, 1.0)
        self.assertAlmostEqual(y1, 1.0)

    def test_arc_crossing_plus_y_axis_keeps_top_of_circle(self):
        # From 45 deg to 135 deg: crosses the +y axis (quadrant 1 -> 2), so y1 = r
        # while x0/x1 shrink to the endpoints.
        arc = VArc(radius=1.0, startParam=math.pi / 4, endParam=3 * math.pi / 4)
        (x0, y0), (x1, y1) = entity_bbox(arc)
        self.assertAlmostEqual(y1, 1.0)
        self.assertAlmostEqual(x1, math.sqrt(0.5))
        self.assertAlmostEqual(x0, -math.sqrt(0.5))
        self.assertAlmostEqual(y0, math.sqrt(0.5))

    def test_clockwise_arc_swaps_traversal(self):
        # Clockwise negates the parameters, so the same numbers describe the mirrored
        # arc: it runs -45 deg -> -135 deg, sweeps the -y axis, and its box touches
        # y0 = -r while the other three sides shrink to the endpoints.
        arc = VArc(radius=1.0, clockwise=True, startParam=math.pi / 4,
                   endParam=3 * math.pi / 4)
        (x0, y0), (x1, y1) = entity_bbox(arc)
        self.assertAlmostEqual(y0, -1.0)
        self.assertAlmostEqual(x1, math.sqrt(0.5))
        self.assertAlmostEqual(x0, -math.sqrt(0.5))
        self.assertAlmostEqual(y1, -math.sqrt(0.5))

    def test_analytic_bbox_is_not_the_sampled_bbox(self):
        # A near-full arc's analytic box touches the circle box exactly; a 32-segment
        # sampled box would under-report. Check exactness at the +y extremum.
        arc = VArc(radius=2.0, startParam=0.1, endParam=math.pi - 0.1)
        (_, _), (_, y1) = entity_bbox(arc)
        self.assertAlmostEqual(y1, 2.0)

    def test_unsupported_entity_is_none(self):
        self.assertIsNone(entity_bbox(object()))


class TestSketchBBox(unittest.TestCase):
    def test_empty_sketch_bounds_origin(self):
        self.assertEqual(sketch_bbox([]), ((0.0, 0.0), (0.0, 0.0)))

    def test_union_of_entities(self):
        box = sketch_bbox([VPoint(x=-3.0, y=0.0), VCircle(xCenter=1.0, radius=1.0)])
        self.assertEqual(box, ((-3.0, -1.0), (2.0, 1.0)))


class TestNormalization(unittest.TestCase):
    def test_center_puts_bbox_center_on_origin(self):
        entities = [VPoint(x=0.0, y=0.0), VPoint(x=4.0, y=2.0)]
        center_sketch(entities)
        self.assertEqual(sketch_bbox(entities), ((-2.0, -1.0), (2.0, 1.0)))

    def test_rescale_requires_centered_sketch(self):
        with self.assertRaises(ValueError):
            rescale_sketch([VPoint(x=1.0, y=1.0), VPoint(x=3.0, y=3.0)])

    def test_normalize_makes_long_axis_one(self):
        entities = [_line(0.0, 0.0, 4.0, 2.0)]
        factor = normalize_sketch(entities)
        self.assertAlmostEqual(factor, 4.0)
        (x0, y0), (x1, y1) = sketch_bbox(entities)
        self.assertAlmostEqual(x1 - x0, 1.0)
        self.assertAlmostEqual(y1 - y0, 0.5)
        # Every parameter now lies inside the quantiser domain.
        for value in parameterize_entity(entities[0]):
            self.assertLessEqual(abs(value), 0.5 + 1e-9)

    def test_normalize_rescales_line_params(self):
        entities = [_line(0.0, 0.0, 2.0, 0.0)]
        normalize_sketch(entities)
        self.assertAlmostEqual(entities[0].startParam, -0.5)
        self.assertAlmostEqual(entities[0].endParam, 0.5)

    def test_normalize_rescales_radius(self):
        entities = [VCircle(xCenter=0.0, yCenter=0.0, radius=5.0)]
        factor = normalize_sketch(entities)
        self.assertAlmostEqual(factor, 10.0)
        self.assertAlmostEqual(entities[0].radius, 0.5)

    def test_zero_extent_sketch_returns_sentinel(self):
        self.assertEqual(normalize_sketch([VPoint(x=2.0, y=2.0)]), -1.0)

    def test_arc_angles_are_not_rescaled(self):
        arc = VArc(xCenter=0.0, yCenter=0.0, radius=4.0, startParam=0.25, endParam=1.25)
        normalize_sketch([arc])
        self.assertAlmostEqual(arc.startParam, 0.25)
        self.assertAlmostEqual(arc.endParam, 1.25)


class TestParameterization(unittest.TestCase):
    def test_param_counts(self):
        self.assertEqual(len(parameterize_entity(VArc())), NUM_PARAMS[VArc])
        self.assertEqual(len(parameterize_entity(VCircle())), NUM_PARAMS[VCircle])
        self.assertEqual(len(parameterize_entity(VLine())), NUM_PARAMS[VLine])
        self.assertEqual(len(parameterize_entity(VPoint())), NUM_PARAMS[VPoint])

    def test_clockwise_arc_parameterization_swaps_endpoints(self):
        ccw = VArc(radius=1.0, startParam=0.0, endParam=math.pi / 2)
        cw = VArc(radius=1.0, clockwise=True, startParam=0.0, endParam=math.pi / 2)
        p_ccw = parameterize_entity(ccw)
        p_cw = parameterize_entity(cw)
        # Clockwise swaps start/end so the vector always reads counter-clockwise.
        self.assertAlmostEqual(p_cw[0], math.cos(-math.pi / 2), places=6)
        self.assertAlmostEqual(p_ccw[0], 1.0)

    def test_line_roundtrip(self):
        line = entity_from_params([1.0, 1.0, 3.0, 5.0])
        params = parameterize_entity(line)
        for got, want in zip(params, [1.0, 1.0, 3.0, 5.0]):
            self.assertAlmostEqual(got, want)

    def test_circle_roundtrip(self):
        circle = entity_from_params([0.25, -0.25, 0.5])
        self.assertIsInstance(circle, VCircle)
        self.assertAlmostEqual(circle.radius, 0.5)

    def test_arc_roundtrip_via_circumcenter(self):
        arc = VArc(xCenter=0.2, yCenter=-0.1, radius=0.4, startParam=0.2, endParam=2.0)
        params = parameterize_entity(arc)
        rebuilt = entity_from_params(params)
        self.assertIsInstance(rebuilt, VArc)
        self.assertAlmostEqual(rebuilt.xCenter, 0.2)
        self.assertAlmostEqual(rebuilt.yCenter, -0.1)
        self.assertAlmostEqual(rebuilt.radius, 0.4)
        for got, want in zip(parameterize_entity(rebuilt), params):
            self.assertAlmostEqual(got, want, places=6)

    def test_collinear_arc_is_rejected(self):
        self.assertIsNone(entity_from_params([0.0, 0.0, 0.1, 0.1, 0.2, 0.2]))

    def test_zero_length_line_gets_default_direction(self):
        line = entity_from_params([1.0, 1.0, 1.0, 1.0])
        self.assertEqual((line.dirX, line.dirY), (1.0, 0.0))
        self.assertEqual((line.startParam, line.endParam), (0.0, 0.0))

    def test_unknown_param_count_raises(self):
        with self.assertRaises(ValueError):
            entity_from_params([1.0, 2.0, 3.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
