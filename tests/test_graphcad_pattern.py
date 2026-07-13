import math
import unittest

from harnesscad.domain.geometry.graphcad_pattern import (
    GridPattern,
    Instance,
    PolarPattern,
    expand_grid,
    expand_pattern,
    expand_polar,
    parse_pattern,
)


class ParseTests(unittest.TestCase):
    def test_parse_grid(self):
        pattern = parse_pattern(
            "pattern=grid(rows:2, cols:3, x_spacing:0.1, y_spacing:0.2, "
            "start_offset:(0.05, -0.05))"
        )
        self.assertIsInstance(pattern, GridPattern)
        self.assertEqual((pattern.rows, pattern.cols), (2, 3))
        self.assertEqual(pattern.x_spacing, 0.1)
        self.assertEqual(pattern.y_spacing, 0.2)
        self.assertEqual(pattern.start_offset, (0.05, -0.05))
        self.assertEqual(pattern.count, 6)

    def test_grid_defaults(self):
        pattern = parse_pattern("pattern=grid(rows:1, cols:2)")
        self.assertEqual(pattern.start_offset, (0.0, 0.0))
        self.assertEqual(pattern.x_spacing, 0.0)

    def test_parse_polar_with_degree_signs(self):
        pattern = parse_pattern(
            "pattern=polar(count:6, radius:0.08, start_angle:30deg, angle_step:60deg)"
        )
        self.assertIsInstance(pattern, PolarPattern)
        self.assertEqual(pattern.count, 6)
        self.assertEqual(pattern.start_angle, 30.0)
        self.assertEqual(pattern.step(), 60.0)

    def test_polar_step_defaults_to_even_distribution(self):
        self.assertEqual(parse_pattern("pattern=polar(count:4, radius:1)").step(), 90.0)

    def test_missing_fields_rejected(self):
        with self.assertRaises(ValueError):
            parse_pattern("pattern=grid(cols:2)")
        with self.assertRaises(ValueError):
            parse_pattern("pattern=polar(radius:1)")

    def test_non_pattern_text(self):
        with self.assertRaises(ValueError):
            parse_pattern("offset(0,0,0)")

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            GridPattern(rows=0, cols=2, x_spacing=1.0, y_spacing=1.0)
        with self.assertRaises(ValueError):
            PolarPattern(count=0, radius=1.0)
        with self.assertRaises(ValueError):
            PolarPattern(count=3, radius=-1.0)


class ExpandGridTests(unittest.TestCase):
    def test_row_major_ids_and_offsets(self):
        pattern = GridPattern(rows=2, cols=2, x_spacing=1.0, y_spacing=2.0,
                              start_offset=(0.5, 0.5))
        instances = expand_grid("tile", pattern)
        self.assertEqual(
            [item.instance_id for item in instances],
            ["tile_0_0", "tile_0_1", "tile_1_0", "tile_1_1"],
        )
        self.assertEqual(instances[0].offset, (0.5, 0.5, 0.0))
        self.assertEqual(instances[1].offset, (1.5, 0.5, 0.0))
        self.assertEqual(instances[2].offset, (0.5, 2.5, 0.0))
        self.assertEqual(instances[3].offset, (1.5, 2.5, 0.0))

    def test_indices(self):
        instances = expand_grid("t", GridPattern(rows=1, cols=3, x_spacing=1.0, y_spacing=0.0))
        self.assertEqual([item.index for item in instances], [(0, 0), (0, 1), (0, 2)])

    def test_count_matches_expansion(self):
        pattern = GridPattern(rows=3, cols=4, x_spacing=1.0, y_spacing=1.0)
        self.assertEqual(len(expand_grid("t", pattern)), pattern.count)


class ExpandPolarTests(unittest.TestCase):
    def test_ids_and_positions(self):
        instances = expand_polar("leg", PolarPattern(count=4, radius=2.0))
        self.assertEqual(
            [item.instance_id for item in instances],
            ["leg_0", "leg_1", "leg_2", "leg_3"],
        )
        self.assertAlmostEqual(instances[0].offset[0], 2.0)
        self.assertAlmostEqual(instances[0].offset[1], 0.0)
        self.assertAlmostEqual(instances[1].offset[0], 0.0, places=9)
        self.assertAlmostEqual(instances[1].offset[1], 2.0)
        self.assertAlmostEqual(instances[2].offset[0], -2.0)

    def test_start_angle_and_step(self):
        instances = expand_polar(
            "b", PolarPattern(count=3, radius=1.0, start_angle=90.0, angle_step=120.0)
        )
        self.assertAlmostEqual(instances[0].offset[1], 1.0)
        expected = math.radians(210.0)
        self.assertAlmostEqual(instances[1].offset[0], math.cos(expected))
        self.assertAlmostEqual(instances[1].offset[1], math.sin(expected))

    def test_all_on_circle(self):
        instances = expand_polar("p", PolarPattern(count=7, radius=1.5, start_angle=13.0))
        for item in instances:
            radius = math.hypot(item.offset[0], item.offset[1])
            self.assertAlmostEqual(radius, 1.5)
            self.assertEqual(item.offset[2], 0.0)

    def test_zero_radius_stacks_at_anchor(self):
        instances = expand_polar("p", PolarPattern(count=2, radius=0.0))
        self.assertEqual(instances[0].offset, (0.0, 0.0, 0.0))


class DispatchTests(unittest.TestCase):
    def test_expand_pattern_dispatches(self):
        grid = expand_pattern("g", GridPattern(rows=1, cols=1, x_spacing=0.0, y_spacing=0.0))
        polar = expand_pattern("p", PolarPattern(count=1, radius=1.0))
        self.assertIsInstance(grid[0], Instance)
        self.assertEqual(polar[0].instance_id, "p_0")

    def test_unsupported_type(self):
        with self.assertRaises(TypeError):
            expand_pattern("x", object())

    def test_end_to_end_from_clause(self):
        instances = expand_pattern("hole", parse_pattern("pattern=polar(count:2, radius:1)"))
        self.assertEqual([item.instance_id for item in instances], ["hole_0", "hole_1"])
        self.assertAlmostEqual(instances[1].offset[0], -1.0)


if __name__ == "__main__":
    unittest.main()
