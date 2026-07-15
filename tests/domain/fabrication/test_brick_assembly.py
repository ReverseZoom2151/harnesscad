import unittest

from harnesscad.domain.fabrication.brick_assembly import (
    Brick,
    BrickStructure,
    parse_text,
    validate,
)


class BrickTests(unittest.TestCase):
    def test_voxels_and_footprint(self):
        b = Brick(h=2, w=1, x=0, y=0, z=0)
        self.assertEqual(set(b.voxels()), {(0, 0, 0), (1, 0, 0)})
        self.assertEqual(set(b.footprint()), {(0, 0), (1, 0)})
        self.assertEqual(b.area, 2)

    def test_text_round_trip(self):
        b = Brick(h=2, w=4, x=1, y=3, z=5)
        self.assertEqual(Brick.from_text(b.to_text()), b)

    def test_from_text_bad(self):
        with self.assertRaises(ValueError):
            Brick.from_text("not a brick")

    def test_bad_dimensions(self):
        with self.assertRaises(ValueError):
            Brick(h=0, w=1, x=0, y=0, z=0)

    def test_overlaps_xy(self):
        a = Brick(h=2, w=2, x=0, y=0, z=0)
        b = Brick(h=2, w=2, x=1, y=1, z=1)
        c = Brick(h=2, w=2, x=5, y=5, z=1)
        self.assertTrue(a.overlaps_xy(b))
        self.assertFalse(a.overlaps_xy(c))


class BrickStructureTests(unittest.TestCase):
    def test_valid_stack_is_buildable(self):
        s = BrickStructure([
            Brick(h=2, w=2, x=0, y=0, z=0),
            Brick(h=2, w=2, x=0, y=0, z=1),
        ])
        rep = validate(s)
        self.assertTrue(rep.buildable)
        self.assertEqual(rep.reasons, ())

    def test_out_of_bounds(self):
        s = BrickStructure([Brick(h=2, w=2, x=19, y=0, z=0)], world_dim=20)
        self.assertTrue(s.has_out_of_bounds_bricks())
        self.assertTrue(validate(s).out_of_bounds)

    def test_collision(self):
        s = BrickStructure([
            Brick(h=2, w=2, x=0, y=0, z=0),
            Brick(h=2, w=2, x=1, y=1, z=0),
        ])
        self.assertTrue(s.has_collisions())
        self.assertTrue(validate(s).collisions)

    def test_floating_brick(self):
        s = BrickStructure([
            Brick(h=1, w=1, x=0, y=0, z=0),
            Brick(h=1, w=1, x=5, y=5, z=3),  # floats far above ground
        ])
        self.assertTrue(s.has_floating_bricks())
        self.assertTrue(validate(s).floating)

    def test_ground_brick_not_floating(self):
        s = BrickStructure([Brick(h=1, w=1, x=0, y=0, z=0)])
        self.assertFalse(s.has_floating_bricks())

    def test_disconnected_but_supported(self):
        # Two separate ground-resting towers are each connected (both touch ground).
        s = BrickStructure([
            Brick(h=1, w=1, x=0, y=0, z=0),
            Brick(h=1, w=1, x=0, y=0, z=1),
            Brick(h=1, w=1, x=10, y=10, z=0),
        ])
        self.assertTrue(s.is_connected())
        self.assertEqual(s.disconnected_bricks(), [])

    def test_connectivity_chain(self):
        s = BrickStructure([
            Brick(h=2, w=2, x=0, y=0, z=0),
            Brick(h=2, w=2, x=1, y=1, z=1),  # overlaps first, one level up
            Brick(h=2, w=2, x=2, y=2, z=2),  # overlaps second
        ])
        self.assertTrue(s.is_connected())

    def test_parse_text_structure(self):
        s = parse_text("2x2 (0,0,0)\n2x2 (0,0,1)\n")
        self.assertEqual(len(s), 2)
        self.assertTrue(validate(s).buildable)

    def test_to_text_round_trip(self):
        s = BrickStructure([Brick(h=2, w=2, x=0, y=0, z=0), Brick(h=1, w=3, x=0, y=0, z=1)])
        s2 = parse_text(s.to_text())
        self.assertEqual(s.bricks, s2.bricks)

    def test_empty_is_connected(self):
        self.assertTrue(BrickStructure([]).is_connected())


if __name__ == "__main__":
    unittest.main()
