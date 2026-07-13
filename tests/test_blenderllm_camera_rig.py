import math
import unittest

from harnesscad.domain.geometry.blenderllm_camera_rig import (
    BoundingBox,
    FRAME_SCALE,
    bounding_box,
    camera_positions,
    camera_positions_from_obj,
    framing_radius,
    parse_obj_vertices,
)


class ParseObjTests(unittest.TestCase):
    def test_only_geometric_vertices(self):
        text = (
            "# comment\n"
            "v 1.0 2.0 3.0\n"
            "vt 0.5 0.5\n"
            "vn 0 0 1\n"
            "v -1 -2 -3\n"
            "f 1 2 3\n"
        )
        self.assertEqual(
            parse_obj_vertices(text),
            [(1.0, 2.0, 3.0), (-1.0, -2.0, -3.0)],
        )

    def test_extra_coordinates_discarded(self):
        self.assertEqual(parse_obj_vertices("v 1 2 3 1.0"), [(1.0, 2.0, 3.0)])


class BoundingBoxTests(unittest.TestCase):
    def test_center_extents_delta(self):
        box = bounding_box([(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)])
        self.assertEqual(box.center, (1.0, 2.0, 3.0))
        self.assertEqual(box.extents, (2.0, 4.0, 6.0))
        self.assertEqual(box.delta_max, 6.0)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            bounding_box([])


class CameraRigTests(unittest.TestCase):
    def test_eight_distinct_positions(self):
        box = bounding_box([(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])
        positions = camera_positions(box)
        self.assertEqual(len(positions), 8)
        self.assertEqual(len(set(positions)), 8)

    def test_matches_reference_formula(self):
        # Unit cube centred at origin: delta_max = 2.
        box = BoundingBox((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        horizontal = 2.0 * FRAME_SCALE / math.sqrt(2)
        vertical = 2.0 * FRAME_SCALE
        positions = set(camera_positions(box))
        expected = set()
        for i in (-1, 1):
            for j in (-1, 1):
                for k in (-1, 1):
                    expected.add((i * horizontal, j * horizontal, k * vertical))
        for got, exp in zip(sorted(positions), sorted(expected)):
            for a, b in zip(got, exp):
                self.assertAlmostEqual(a, b)

    def test_axis_swap_z_is_vertical(self):
        # Offset the box centre so the y/z swap is observable.
        box = BoundingBox((0.0, 10.0, 100.0), (2.0, 12.0, 102.0))
        positions = camera_positions(box)
        # vertical slot (index 2) is derived from the object's y centre (11).
        for p in positions:
            self.assertAlmostEqual(abs(p[2] - 11.0), 2.0 * FRAME_SCALE)
            # depth-ish slot (index 1) derived from object's z centre (101).
            self.assertAlmostEqual(abs(p[1] - 101.0), 2.0 * FRAME_SCALE / math.sqrt(2))

    def test_scale_invariance_of_geometry(self):
        small = camera_positions(BoundingBox((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))
        big = camera_positions(BoundingBox((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0)))
        # Big box has 10x the delta, so every camera offset scales by 10.
        for s, b in zip(small, big):
            for a, c in zip(s, b):
                self.assertAlmostEqual(c, a * 10.0)

    def test_from_obj_roundtrip(self):
        text = "v -1 -1 -1\nv 1 1 1\n"
        self.assertEqual(
            camera_positions_from_obj(text),
            camera_positions(bounding_box([(-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)])),
        )

    def test_framing_radius_positive(self):
        box = BoundingBox((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        r = framing_radius(box)
        # Distance to a corner camera matches the vector norm.
        self.assertAlmostEqual(r, math.dist((0.0, 0.0, 0.0), camera_positions(box)[0]))


if __name__ == "__main__":
    unittest.main()
