"""Tests for drawings.sgraphs2_entity_render."""

import math
import unittest

from harnesscad.domain.drawings.sgraphs2_entity_render import (
    TAU,
    arc_endpoints,
    arc_midpoint,
    arc_point,
    bounding_box,
    circle_center,
    line_endpoints,
    line_point,
    normalize_scene,
    render_entity,
    render_sketch,
    sample_entity,
)
from harnesscad.io.formats.sgraphs2_onshape_json import Arc, Circle, EntityType, Line, Point, Sketch


class TestLine(unittest.TestCase):
    def setUp(self):
        # Unit x-direction anchored at (0, 2), spanning t in [-1, 3].
        self.line = Line("l0", False, pntX=0.0, pntY=2.0, dirX=1.0, dirY=0.0,
                         startParam=-1.0, endParam=3.0)

    def test_line_point(self):
        self.assertEqual(line_point(self.line, 0.0), (0.0, 2.0))
        self.assertEqual(line_point(self.line, 2.0), (2.0, 2.0))

    def test_line_endpoints(self):
        start, end = line_endpoints(self.line)
        self.assertEqual(start, (-1.0, 2.0))
        self.assertEqual(end, (3.0, 2.0))

    def test_diagonal_direction(self):
        s = math.sqrt(0.5)
        line = Line("l1", False, pntX=1.0, pntY=1.0, dirX=s, dirY=s,
                    startParam=0.0, endParam=math.sqrt(2.0))
        start, end = line_endpoints(line)
        self.assertAlmostEqual(start[0], 1.0)
        self.assertAlmostEqual(end[0], 2.0)
        self.assertAlmostEqual(end[1], 2.0)

    def test_line_samples_to_two_points(self):
        self.assertEqual(sample_entity(self.line, segments=16), [(-1.0, 2.0), (3.0, 2.0)])


class TestArc(unittest.TestCase):
    def test_ccw_arc_quarter(self):
        arc = Arc("a0", False, xCenter=0.0, yCenter=0.0, xDir=1.0, yDir=0.0,
                  radius=2.0, clockwise=False, startParam=0.0, endParam=math.pi / 2)
        start, end = arc_endpoints(arc)
        self.assertAlmostEqual(start[0], 2.0)
        self.assertAlmostEqual(start[1], 0.0)
        self.assertAlmostEqual(end[0], 0.0)
        self.assertAlmostEqual(end[1], 2.0)

    def test_clockwise_negates_the_parameter(self):
        kwargs = dict(xCenter=0.0, yCenter=0.0, xDir=1.0, yDir=0.0, radius=1.0,
                      startParam=0.0, endParam=math.pi / 2)
        ccw = Arc("a1", False, clockwise=False, **kwargs)
        cw = Arc("a2", False, clockwise=True, **kwargs)
        # Same base angle: both start at (1, 0).
        self.assertAlmostEqual(arc_endpoints(ccw)[0][0], 1.0)
        self.assertAlmostEqual(arc_endpoints(cw)[0][0], 1.0)
        # Mirrored end: (0, 1) vs (0, -1).
        self.assertAlmostEqual(arc_endpoints(ccw)[1][1], 1.0)
        self.assertAlmostEqual(arc_endpoints(cw)[1][1], -1.0)

    def test_reference_direction_rotates_the_arc(self):
        arc = Arc("a3", False, xCenter=0.0, yCenter=0.0, xDir=0.0, yDir=1.0,
                  radius=1.0, clockwise=False, startParam=0.0, endParam=1.0)
        start = arc_point(arc, 0.0)
        # Base angle is pi/2, so parameter 0 sits at (0, 1).
        self.assertAlmostEqual(start[0], 0.0)
        self.assertAlmostEqual(start[1], 1.0)

    def test_midpoint_of_quarter_arc(self):
        arc = Arc("a4", False, radius=1.0, startParam=0.0, endParam=math.pi / 2)
        mid = arc_midpoint(arc)
        self.assertAlmostEqual(mid[0], math.cos(math.pi / 4))
        self.assertAlmostEqual(mid[1], math.sin(math.pi / 4))

    def test_midpoint_wraps_backwards_interval_forwards(self):
        # start wraps to 3pi/2 and end to pi/2, so the end is lifted a full turn
        # to 5pi/2; the mean is 2pi, i.e. the midpoint lands at angle 0 -- on the
        # right half that is actually drawn, not on its complement at angle pi.
        arc = Arc("a5", False, radius=1.0, startParam=-math.pi / 2, endParam=math.pi / 2)
        mid = arc_midpoint(arc)
        self.assertAlmostEqual(mid[0], 1.0)
        self.assertAlmostEqual(mid[1], 0.0, places=9)
        # The naive (unwrapped) mean would be 0 as well here; check a case where
        # wrapping actually changes the answer: [3pi/2, pi/2] wraps the same way.
        arc2 = Arc("a5b", False, radius=1.0, startParam=3 * math.pi / 2,
                   endParam=math.pi / 2)
        self.assertAlmostEqual(arc_midpoint(arc2)[0], 1.0)
        self.assertAlmostEqual(arc_midpoint(arc2)[1], 0.0, places=9)

    def test_arc_sampling_endpoints_and_count(self):
        arc = Arc("a6", False, radius=1.0, startParam=0.0, endParam=math.pi)
        points = sample_entity(arc, segments=8)
        self.assertEqual(len(points), 9)
        self.assertAlmostEqual(points[0][0], 1.0)
        self.assertAlmostEqual(points[-1][0], -1.0)
        # Every sample is on the circle of radius 1.
        for x, y in points:
            self.assertAlmostEqual(math.hypot(x, y), 1.0)

    def test_arc_sampling_is_deterministic(self):
        arc = Arc("a7", False, radius=3.0, startParam=0.2, endParam=2.0)
        self.assertEqual(sample_entity(arc, 5), sample_entity(arc, 5))


class TestCircle(unittest.TestCase):
    def setUp(self):
        self.circle = Circle("c0", False, xCenter=1.0, yCenter=-1.0, xDir=1.0,
                             yDir=0.0, radius=2.0)

    def test_center(self):
        self.assertEqual(circle_center(self.circle), (1.0, -1.0))

    def test_sampling_is_closed_and_on_radius(self):
        points = sample_entity(self.circle, segments=12)
        self.assertEqual(len(points), 13)
        self.assertEqual(points[0], points[-1])
        for x, y in points:
            self.assertAlmostEqual(math.hypot(x - 1.0, y + 1.0), 2.0)

    def test_full_turn_covered(self):
        points = sample_entity(self.circle, segments=4)
        # Quarter-turn steps starting from the reference direction.
        self.assertAlmostEqual(points[1][0], 1.0)
        self.assertAlmostEqual(points[1][1], 1.0)
        self.assertAlmostEqual(TAU, 2 * math.pi)


class TestRender(unittest.TestCase):
    def test_render_entity_flags(self):
        rendered = render_entity(Circle("c1", True, radius=1.0), segments=8)
        self.assertEqual(rendered.entity_id, "c1")
        self.assertIs(rendered.kind, EntityType.Circle)
        self.assertTrue(rendered.closed)
        self.assertTrue(rendered.construction)

    def test_arc_is_not_closed(self):
        rendered = render_entity(Arc("a8", False, radius=1.0, startParam=0.0, endParam=1.0))
        self.assertFalse(rendered.closed)

    def test_point_renders_to_single_sample(self):
        rendered = render_entity(Point("p0", False, x=3.0, y=4.0))
        self.assertEqual(rendered.polyline, ((3.0, 4.0),))

    def test_render_sketch_order_and_skipping(self):
        sketch = Sketch(
            entities={
                "p0": Point("p0", False, x=0.0, y=0.0),
                "l0": Line("l0", False, endParam=1.0),
                "c0": Circle("c0", False, radius=1.0),
            }
        )
        rendered = render_sketch(sketch, segments=4)
        self.assertEqual([r.entity_id for r in rendered], ["p0", "l0", "c0"])

    def test_bad_segments_rejected(self):
        with self.assertRaises(ValueError):
            sample_entity(Circle("c2", False, radius=1.0), segments=0)


class TestScene(unittest.TestCase):
    def test_bounding_box(self):
        rendered = [
            render_entity(Point("p0", False, x=-1.0, y=0.0)),
            render_entity(Point("p1", False, x=2.0, y=5.0)),
        ]
        self.assertEqual(bounding_box(rendered), (-1.0, 0.0, 2.0, 5.0))

    def test_bounding_box_empty_raises(self):
        with self.assertRaises(ValueError):
            bounding_box([])

    def test_normalize_fits_unit_box(self):
        rendered = render_sketch(
            Sketch(entities={"c0": Circle("c0", False, xCenter=10.0, yCenter=10.0, radius=5.0)}),
            segments=32,
        )
        norm = normalize_scene(rendered)
        min_x, min_y, max_x, max_y = bounding_box(norm)
        self.assertAlmostEqual(min_x, 0.0)
        self.assertAlmostEqual(min_y, 0.0)
        self.assertAlmostEqual(max_x, 1.0)
        self.assertAlmostEqual(max_y, 1.0)

    def test_normalize_preserves_aspect_and_centres(self):
        # Wide, flat content: x spans 4, y spans 2 -> y occupies the middle half.
        rendered = [
            render_entity(Point("p0", False, x=0.0, y=0.0)),
            render_entity(Point("p1", False, x=4.0, y=2.0)),
        ]
        norm = normalize_scene(rendered)
        min_x, min_y, max_x, max_y = bounding_box(norm)
        self.assertAlmostEqual(max_x - min_x, 1.0)
        self.assertAlmostEqual(max_y - min_y, 0.5)
        self.assertAlmostEqual(min_y, 0.25)
        self.assertAlmostEqual((min_y + max_y) / 2, 0.5)

    def test_normalize_is_similarity_invariant(self):
        def scene(scale, shift):
            return render_sketch(
                Sketch(
                    entities={
                        "l0": Line("l0", False, pntX=shift, pntY=shift, dirX=1.0,
                                   dirY=0.0, startParam=0.0, endParam=scale),
                        "c0": Circle("c0", False, xCenter=shift, yCenter=shift,
                                     radius=scale),
                    }
                ),
                segments=8,
            )

        a = normalize_scene(scene(1.0, 0.0))
        b = normalize_scene(scene(7.0, -30.0))
        for ra, rb in zip(a, b):
            for (ax, ay), (bx, by) in zip(ra.polyline, rb.polyline):
                self.assertAlmostEqual(ax, bx)
                self.assertAlmostEqual(ay, by)

    def test_normalize_degenerate_scene_maps_to_centre(self):
        norm = normalize_scene([render_entity(Point("p0", False, x=9.0, y=9.0))])
        self.assertEqual(norm[0].polyline, ((0.5, 0.5),))

    def test_normalize_margin(self):
        rendered = [
            render_entity(Point("p0", False, x=0.0, y=0.0)),
            render_entity(Point("p1", False, x=1.0, y=1.0)),
        ]
        norm = normalize_scene(rendered, margin=0.1)
        min_x, min_y, max_x, max_y = bounding_box(norm)
        self.assertAlmostEqual(min_x, 0.1)
        self.assertAlmostEqual(max_x, 0.9)

    def test_normalize_bad_margin(self):
        with self.assertRaises(ValueError):
            normalize_scene([], margin=0.5)

    def test_normalize_empty_scene(self):
        self.assertEqual(normalize_scene([]), [])

    def test_normalize_preserves_metadata(self):
        rendered = [render_entity(Circle("c0", True, radius=2.0), segments=6)]
        norm = normalize_scene(rendered)
        self.assertEqual(norm[0].entity_id, "c0")
        self.assertTrue(norm[0].construction)
        self.assertTrue(norm[0].closed)


if __name__ == "__main__":
    unittest.main()
