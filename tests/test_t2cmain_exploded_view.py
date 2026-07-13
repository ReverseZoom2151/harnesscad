import math
import unittest

from harnesscad.domain.geometry.assembly.exploded_view import (
    Part,
    bounds_center,
    bounds_radius,
    bounds_size,
    ease_progress,
    exploded_bounds,
    group_key,
    merge_bounds,
    normalize_settings,
    shift_bounds,
    solve_exploded_view,
    translation_at_progress,
)


def _slab(part_id, z0, z1, half=1.0):
    return Part(part_id=part_id, bounds=((-half, -half, z0), (half, half, z1)))


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        settings = normalize_settings()
        self.assertEqual(settings.axis, "z")
        self.assertEqual(settings.direction, "positive")
        self.assertEqual(settings.spacing, 1.45)
        self.assertEqual(settings.depth, 1)

    def test_negative_axis_prefix_sets_direction(self):
        self.assertEqual(normalize_settings(axis="-y").direction, "negative")
        self.assertEqual(normalize_settings(axis="-y").axis, "y")

    def test_spacing_and_depth_are_clamped(self):
        self.assertEqual(normalize_settings(spacing=99.0).spacing, 4.0)
        self.assertEqual(normalize_settings(spacing=0.0).spacing, 0.25)
        self.assertEqual(normalize_settings(depth=99).depth, 4)
        self.assertEqual(normalize_settings(depth=0).depth, 1)

    def test_unknown_axis_falls_back_to_z(self):
        self.assertEqual(normalize_settings(axis="w").axis, "z")

    def test_radial_axis_is_preserved(self):
        self.assertEqual(normalize_settings(axis="radial").axis, "radial")


class BoundsHelperTest(unittest.TestCase):
    def test_merge_and_measure(self):
        merged = merge_bounds([((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)), ((-1.0, 0.0, 1.0), (0.0, 0.0, 5.0))])
        self.assertEqual(merged, ((-1.0, 0.0, 0.0), (1.0, 2.0, 5.0)))
        self.assertEqual(bounds_size(merged, 0), 2.0)
        self.assertEqual(bounds_center(merged), (0.0, 1.0, 2.5))
        self.assertIsNone(merge_bounds([]))

    def test_radius_is_half_the_diagonal(self):
        radius = bounds_radius(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
        self.assertAlmostEqual(radius, 1.0)

    def test_shift_bounds(self):
        shifted = shift_bounds(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (0.0, 0.0, 2.0))
        self.assertEqual(shifted, ((0.0, 0.0, 2.0), (1.0, 1.0, 3.0)))


class GroupKeyTest(unittest.TestCase):
    def test_depth_one_keeps_top_segment(self):
        self.assertEqual(group_key("o1.2.3", depth=1, common_prefix=()), "1")

    def test_depth_two(self):
        self.assertEqual(group_key("o1.2.3", depth=2, common_prefix=()), "1.2")

    def test_common_prefix_shifts_the_window(self):
        self.assertEqual(group_key("o1.2.3", depth=1, common_prefix=("1",)), "1.2")

    def test_never_exceeds_available_segments(self):
        self.assertEqual(group_key("o7", depth=4, common_prefix=()), "7")

    def test_non_occurrence_id_groups_by_itself(self):
        self.assertEqual(group_key("solid_body", depth=1), "solid_body")


class AxisExplosionTest(unittest.TestCase):
    def setUp(self):
        self.parts = [
            _slab("o1", 0.0, 1.0),
            _slab("o2", 1.0, 2.0),
            _slab("o3", 2.0, 3.0),
        ]

    def test_three_stacked_slabs_separate_monotonically(self):
        states = solve_exploded_view(self.parts)
        distances = [state.distance for state in states]
        self.assertEqual(len(states), 3)
        self.assertEqual(distances[0], 0.0)
        self.assertLess(distances[0], distances[1])
        self.assertLess(distances[1], distances[2])

    def test_base_stays_grounded(self):
        states = solve_exploded_view(self.parts)
        self.assertEqual(states[0].translation, (0.0, 0.0, 0.0))

    def test_ungrounded_base_still_starts_at_zero_offset(self):
        states = solve_exploded_view(self.parts, keep_base_grounded=False)
        # The first layer has no predecessor, so its target min is its own min.
        self.assertEqual(states[0].distance, 0.0)
        self.assertGreater(states[1].distance, 0.0)

    def test_exploded_layers_never_overlap(self):
        states = solve_exploded_view(self.parts)
        by_id = {part.part_id: part for part in self.parts}
        spans = [
            shift_bounds(by_id[state.part_id].bounds, state.translation)
            for state in sorted(states, key=lambda s: s.layer_index)
        ]
        for lower, upper in zip(spans, spans[1:]):
            self.assertGreater(upper[0][2], lower[1][2])

    def test_negative_direction_flips_the_translation(self):
        states = solve_exploded_view(self.parts, axis="-z")
        self.assertLess(states[2].translation[2], 0.0)

    def test_x_axis_translates_in_x_only(self):
        parts = [
            Part("o1", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
            Part("o2", ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))),
        ]
        states = solve_exploded_view(parts, axis="x")
        self.assertGreater(states[1].translation[0], 0.0)
        self.assertEqual(states[1].translation[1], 0.0)
        self.assertEqual(states[1].translation[2], 0.0)

    def test_larger_spacing_pushes_further(self):
        tight = solve_exploded_view(self.parts, spacing=0.5)
        loose = solve_exploded_view(self.parts, spacing=4.0)
        self.assertGreater(loose[2].distance, tight[2].distance)

    def test_coplanar_groups_merge_into_one_layer(self):
        parts = [
            _slab("o1", 0.0, 1.0),
            _slab("o2", 0.0, 1.0),
            _slab("o3", 1.0, 2.0),
        ]
        states = solve_exploded_view(parts, merge_coplanar=True)
        layers = {state.part_id: state.layer_index for state in states}
        self.assertEqual(layers["o1"], layers["o2"])
        self.assertNotEqual(layers["o1"], layers["o3"])

    def test_merge_coplanar_off_keeps_layers_distinct(self):
        parts = [
            _slab("o1", 0.0, 1.0),
            _slab("o2", 0.0, 1.0),
            _slab("o3", 1.0, 2.0),
        ]
        states = solve_exploded_view(parts, merge_coplanar=False)
        layers = {state.part_id: state.layer_index for state in states}
        self.assertNotEqual(layers["o1"], layers["o2"])

    def test_parts_in_the_same_group_move_together(self):
        parts = [
            _slab("o1.1", 0.0, 1.0),
            _slab("o1.2", 1.0, 2.0),
            _slab("o2.1", 3.0, 4.0),
        ]
        states = {state.part_id: state for state in solve_exploded_view(parts, depth=1)}
        self.assertEqual(states["o1.1"].group_key, states["o1.2"].group_key)
        self.assertEqual(states["o1.1"].translation, states["o1.2"].translation)
        self.assertNotEqual(states["o2.1"].group_key, states["o1.1"].group_key)

    def test_depth_two_splits_subassembly(self):
        parts = [
            _slab("o1.1", 0.0, 1.0),
            _slab("o1.2", 1.0, 2.0),
            _slab("o2.1", 3.0, 4.0),
        ]
        states = {state.part_id: state for state in solve_exploded_view(parts, depth=2)}
        self.assertNotEqual(states["o1.1"].group_key, states["o1.2"].group_key)

    def test_is_deterministic(self):
        self.assertEqual(solve_exploded_view(self.parts), solve_exploded_view(self.parts))


class DegenerateTest(unittest.TestCase):
    def test_single_part_does_not_explode(self):
        self.assertEqual(solve_exploded_view([_slab("o1", 0.0, 1.0)]), ())

    def test_single_group_does_not_explode(self):
        # Two bodies of the same occurrence collapse to one group: nothing to move.
        parts = [_slab("o1", 0.0, 1.0), _slab("o1", 1.0, 2.0)]
        self.assertEqual(solve_exploded_view(parts, depth=1), ())

    def test_shared_root_still_separates_siblings(self):
        # The common-prefix rule deliberately shifts the grouping window past a
        # root that every part shares, so "o1.1" and "o1.2" remain two groups.
        parts = [_slab("o1.1", 0.0, 1.0), _slab("o1.2", 1.0, 2.0)]
        keys = {state.group_key for state in solve_exploded_view(parts, depth=1)}
        self.assertEqual(keys, {"1.1", "1.2"})

    def test_model_placeholder_is_ignored(self):
        parts = [_slab("__model__", 0.0, 3.0), _slab("o1", 0.0, 1.0)]
        self.assertEqual(solve_exploded_view(parts), ())

    def test_empty_input(self):
        self.assertEqual(solve_exploded_view([]), ())


class RadialExplosionTest(unittest.TestCase):
    def test_groups_fly_away_from_the_centroid(self):
        parts = [
            Part("o1", ((-2.0, -1.0, 0.0), (-1.0, 1.0, 1.0))),
            Part("o2", ((1.0, -1.0, 0.0), (2.0, 1.0, 1.0))),
        ]
        states = {s.part_id: s for s in solve_exploded_view(parts, axis="radial")}
        self.assertLess(states["o1"].translation[0], 0.0)
        self.assertGreater(states["o2"].translation[0], 0.0)

    def test_negative_direction_implodes(self):
        parts = [
            Part("o1", ((-2.0, -1.0, 0.0), (-1.0, 1.0, 1.0))),
            Part("o2", ((1.0, -1.0, 0.0), (2.0, 1.0, 1.0))),
        ]
        states = {
            s.part_id: s
            for s in solve_exploded_view(parts, axis="radial", direction="negative")
        }
        self.assertGreater(states["o1"].translation[0], 0.0)

    def test_concentric_groups_get_distinct_spiral_directions(self):
        # Both groups share the model centroid, so both need fallback directions.
        parts = [
            Part("o1", ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))),
            Part("o2", ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))),
        ]
        states = solve_exploded_view(
            parts, axis="radial", keep_base_grounded=False
        )
        self.assertEqual(len(states), 2)
        first, second = states[0].direction, states[1].direction
        self.assertNotEqual(first, second)
        for direction in (first, second):
            self.assertAlmostEqual(
                math.sqrt(sum(c ** 2 for c in direction)), 1.0, places=9
            )

    def test_keep_base_grounded_lifts_nothing_below_the_floor(self):
        parts = [
            Part("o1", ((-2.0, -1.0, 0.0), (-1.0, 1.0, 1.0))),
            Part("o2", ((1.0, -1.0, 0.0), (2.0, 1.0, 1.0))),
            Part("o3", ((-0.5, -0.5, -2.0), (0.5, 0.5, -1.0))),
        ]
        states = solve_exploded_view(parts, axis="radial", keep_base_grounded=True)
        result = exploded_bounds(parts, states)
        self.assertGreaterEqual(result[0][2], -2.0 - 1e-6)


class ProgressTest(unittest.TestCase):
    def test_translation_scales_with_progress(self):
        state = solve_exploded_view(
            [_slab("o1", 0.0, 1.0), _slab("o2", 1.0, 2.0)]
        )[1]
        half = translation_at_progress(state, 0.5)
        self.assertAlmostEqual(half[2], state.translation[2] / 2.0)
        self.assertEqual(translation_at_progress(state, 0.0), (0.0, 0.0, 0.0))
        self.assertEqual(translation_at_progress(state, 1.0), state.translation)

    def test_progress_is_clamped(self):
        state = solve_exploded_view(
            [_slab("o1", 0.0, 1.0), _slab("o2", 1.0, 2.0)]
        )[1]
        self.assertEqual(translation_at_progress(state, 5.0), state.translation)
        self.assertEqual(translation_at_progress(state, -5.0), (0.0, 0.0, 0.0))

    def test_ease_is_monotonic_and_bounded(self):
        self.assertEqual(ease_progress(0.0), 0.0)
        self.assertEqual(ease_progress(1.0), 1.0)
        self.assertEqual(ease_progress(2.0), 1.0)
        self.assertGreater(ease_progress(0.5), 0.5)

    def test_exploded_bounds_grow_with_progress(self):
        parts = [_slab("o1", 0.0, 1.0), _slab("o2", 1.0, 2.0)]
        states = solve_exploded_view(parts)
        closed = exploded_bounds(parts, states, progress=0.0)
        open_ = exploded_bounds(parts, states, progress=1.0)
        self.assertGreater(bounds_size(open_, 2), bounds_size(closed, 2))


if __name__ == "__main__":
    unittest.main()
