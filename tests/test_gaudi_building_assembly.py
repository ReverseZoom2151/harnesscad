import math
import unittest

from generation.gaudi_building_assembly import (
    Building,
    BuildingAssemblyError,
    PlacedPlate,
    assemble_building,
    catmull_rom_closed,
    center_xy,
    plate_outline,
    rotate_z,
)


def _square(name="p", thickness=1.0, position=None):
    plate = {
        "name": name,
        "category": "vertex",
        "thickness": thickness,
        "vertices": [(0, 0), (2, 0), (2, 2), (0, 2)],
    }
    if position is not None:
        plate["position"] = position
    return plate


def _parametric(name="wave"):
    return {
        "name": name,
        "category": "parametric",
        "thickness": 1.5,
        "formula": {"x": "4*cos(t)", "y": "4*sin(t)"},
        "range": (0, 2 * math.pi),
        "steps": 60,
    }


def _mixed(name="blob"):
    return {
        "name": name,
        "category": "mixed",
        "thickness": 0.5,
        "vertices": [(0, 0), (3, 0), (3, 3), (0, 3)],
    }


class OutlineTests(unittest.TestCase):
    def test_vertex_outline(self):
        out = plate_outline(_square())
        self.assertEqual(out, [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])

    def test_parametric_outline_count(self):
        out = plate_outline(_parametric())
        self.assertEqual(len(out), 60)

    def test_mixed_outline_smooth(self):
        out = plate_outline(_mixed(), bezier_samples=8)
        self.assertEqual(len(out), 32)  # 4 segments * 8 samples

    def test_degenerate_rejected(self):
        collinear = {
            "name": "line",
            "category": "vertex",
            "thickness": 1.0,
            "vertices": [(0, 0), (1, 1), (2, 2)],
        }
        with self.assertRaises(BuildingAssemblyError):
            plate_outline(collinear)


class CatmullRomTests(unittest.TestCase):
    def test_passes_through_control_points(self):
        control = [(0, 0), (4, 0), (4, 4), (0, 4)]
        pts = catmull_rom_closed(control, samples_per_segment=4)
        # segment starts (u=0) coincide with the control points
        self.assertAlmostEqual(pts[0][0], 0.0)
        self.assertAlmostEqual(pts[0][1], 0.0)
        self.assertAlmostEqual(pts[4][0], 4.0)
        self.assertAlmostEqual(pts[4][1], 0.0)

    def test_deterministic(self):
        c = [(0, 0), (1, 0), (1, 1)]
        self.assertEqual(catmull_rom_closed(c, 5), catmull_rom_closed(c, 5))


class TransformTests(unittest.TestCase):
    def test_center_xy(self):
        out = center_xy([(0, 0), (2, 0), (2, 2), (0, 2)])
        self.assertEqual(out, [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])

    def test_center_leaves_z_untouched_by_being_2d(self):
        out = center_xy([(10, 10), (12, 10), (12, 12), (10, 12)])
        # centre was (11,11)
        self.assertEqual(out[0], (-1.0, -1.0))

    def test_rotate_z_90(self):
        out = rotate_z([(1, 0)], 90.0)
        self.assertAlmostEqual(out[0][0], 0.0, places=9)
        self.assertAlmostEqual(out[0][1], 1.0, places=9)

    def test_rotate_z_zero_identity(self):
        pts = [(1.0, 2.0), (3.0, 4.0)]
        out = rotate_z(pts, 0.0)
        for a, b in zip(out, pts):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])


class AssembleTests(unittest.TestCase):
    def test_auto_stack_z(self):
        b = assemble_building([_square("a", 1.0), _square("b", 2.0), _square("c", 0.5)])
        self.assertEqual([(p.z_bottom, p.z_top) for p in b.plates],
                         [(0.0, 1.0), (1.0, 3.0), (3.0, 3.5)])
        self.assertAlmostEqual(b.height, 3.5)

    def test_no_auto_stack(self):
        b = assemble_building(
            [_square("a", 1.0), _square("b", 2.0)], auto_stack=False
        )
        self.assertEqual([(p.z_bottom, p.z_top) for p in b.plates],
                         [(0.0, 1.0), (0.0, 2.0)])

    def test_position_z_offsets_within_stack(self):
        b = assemble_building([_square("a", 1.0, position={"z": 5.0})])
        self.assertEqual((b.plates[0].z_bottom, b.plates[0].z_top), (5.0, 6.0))

    def test_centering_applied(self):
        b = assemble_building([_square("a", 1.0)])
        ring = b.plates[0].bottom_ring
        # square centred: corners at +-1
        self.assertAlmostEqual(ring[0][0], -1.0)
        self.assertAlmostEqual(ring[0][1], -1.0)
        self.assertAlmostEqual(ring[0][2], 0.0)

    def test_position_xy_offset(self):
        b = assemble_building([_square("a", 1.0, position={"x": 10.0, "y": 3.0})])
        ring = b.plates[0].bottom_ring
        self.assertAlmostEqual(ring[0][0], 9.0)  # -1 + 10
        self.assertAlmostEqual(ring[0][1], 2.0)  # -1 + 3

    def test_ring_lengths_match(self):
        b = assemble_building([_parametric("w")])
        p = b.plates[0]
        self.assertEqual(len(p.bottom_ring), len(p.top_ring))
        self.assertEqual(len(p.bottom_ring), 60)

    def test_bbox(self):
        b = assemble_building([_square("a", 1.0), _square("b", 1.0)])
        (minx, miny, minz), (maxx, maxy, maxz) = b.bbox()
        self.assertAlmostEqual(minx, -1.0)
        self.assertAlmostEqual(maxx, 1.0)
        self.assertAlmostEqual(minz, 0.0)
        self.assertAlmostEqual(maxz, 2.0)

    def test_empty_building_rejected(self):
        with self.assertRaises(BuildingAssemblyError):
            assemble_building([])

    def test_invalid_plate_rejected(self):
        with self.assertRaises(BuildingAssemblyError):
            assemble_building([{"name": "x", "category": "vertex", "thickness": -1,
                                "vertices": [(0, 0), (1, 0), (1, 1)]}])

    def test_deterministic_assembly(self):
        p = [_parametric("w"), _mixed("m")]
        a = assemble_building(p)
        c = assemble_building(p)
        self.assertEqual([q.bottom_ring for q in a.plates],
                         [q.bottom_ring for q in c.plates])

    def test_empty_building_height_and_bbox(self):
        b = Building()
        self.assertEqual(b.height, 0.0)
        self.assertEqual(b.bbox(), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
