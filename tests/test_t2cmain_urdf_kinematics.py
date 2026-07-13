import math
import unittest

from harnesscad.domain.geometry.t2cmain_urdf_kinematics import (
    IDENTITY_TRANSFORM,
    Joint,
    JointMimic,
    RobotModel,
    UrdfKinematicsError,
    axis_angle_transform,
    build_default_joint_values,
    clamp_joint_value_deg,
    invert_rigid_transform,
    link_origin_in_frame,
    merge_bounds,
    multiply_transforms,
    normalize_vector,
    pose_transform_from_xyz_rpy,
    posed_joint_local_transform,
    resolve_joint_value,
    root_point_in_frame,
    rotation_transform_from_rpy,
    solve_link_world_transforms,
    transform_bounds,
    transform_point,
    translation_transform,
)


def _almost(test, actual, expected, places=9):
    test.assertEqual(len(actual), len(expected))
    for got, want in zip(actual, expected):
        test.assertAlmostEqual(got, want, places=places)


class TransformMathTest(unittest.TestCase):
    def test_identity_is_multiplicative_unit(self):
        arbitrary = pose_transform_from_xyz_rpy((1.0, 2.0, 3.0, 0.3, -0.2, 1.1))
        _almost(self, multiply_transforms(IDENTITY_TRANSFORM, arbitrary), arbitrary)
        _almost(self, multiply_transforms(arbitrary, IDENTITY_TRANSFORM), arbitrary)

    def test_rejects_wrong_sized_transform(self):
        with self.assertRaises(UrdfKinematicsError):
            multiply_transforms((1.0, 0.0), IDENTITY_TRANSFORM)
        with self.assertRaises(UrdfKinematicsError):
            transform_point((1.0, 0.0), (0.0, 0.0, 0.0))

    def test_translation_moves_point(self):
        moved = transform_point(translation_transform(1.0, -2.0, 4.0), (1.0, 1.0, 1.0))
        _almost(self, moved, (2.0, -1.0, 5.0))

    def test_rpy_yaw_rotates_x_axis_to_y(self):
        rotation = rotation_transform_from_rpy(0.0, 0.0, math.pi / 2)
        _almost(self, transform_point(rotation, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))

    def test_pose_applies_rotation_before_translation(self):
        pose = pose_transform_from_xyz_rpy((10.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2))
        _almost(self, transform_point(pose, (1.0, 0.0, 0.0)), (10.0, 1.0, 0.0))

    def test_axis_angle_about_z(self):
        rotation = axis_angle_transform((0.0, 0.0, 5.0), math.pi / 2)
        _almost(self, transform_point(rotation, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))

    def test_normalize_vector_degenerate_returns_fallback(self):
        self.assertEqual(normalize_vector((0.0, 0.0, 0.0)), (0.0, 0.0, 1.0))
        _almost(self, normalize_vector((0.0, 3.0, 4.0)), (0.0, 0.6, 0.8))

    def test_inverse_round_trips(self):
        pose = pose_transform_from_xyz_rpy((3.0, -1.0, 2.0, 0.4, 0.1, -0.7))
        back = multiply_transforms(invert_rigid_transform(pose), pose)
        _almost(self, back, IDENTITY_TRANSFORM)


class BoundsTest(unittest.TestCase):
    def test_transform_bounds_of_rotated_box(self):
        bounds = ((-1.0, -1.0, 0.0), (1.0, 1.0, 2.0))
        rotated = transform_bounds(bounds, rotation_transform_from_rpy(0.0, 0.0, math.pi / 4))
        half = math.sqrt(2.0)
        self.assertAlmostEqual(rotated[1][0], half, places=9)
        self.assertAlmostEqual(rotated[0][2], 0.0, places=9)
        self.assertAlmostEqual(rotated[1][2], 2.0, places=9)

    def test_merge_bounds(self):
        merged = merge_bounds(
            [((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), None, ((-2.0, 0.5, 0.0), (0.5, 3.0, 0.5))]
        )
        self.assertEqual(merged, ((-2.0, 0.0, 0.0), (1.0, 3.0, 1.0)))

    def test_merge_bounds_empty(self):
        self.assertEqual(merge_bounds([]), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))


def _two_link_model():
    shoulder = Joint(
        name="shoulder",
        type="revolute",
        parent_link="base",
        child_link="arm",
        origin_transform=pose_transform_from_xyz_rpy((0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
        axis=(0.0, 0.0, 1.0),
        min_value_deg=-90.0,
        max_value_deg=90.0,
    )
    slide = Joint(
        name="slide",
        type="prismatic",
        parent_link="arm",
        child_link="tool",
        origin_transform=pose_transform_from_xyz_rpy((1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        axis=(1.0, 0.0, 0.0),
        min_value_deg=0.0,
        max_value_deg=0.5,
    )
    return RobotModel(
        root_link="base",
        joints=(shoulder, slide),
        link_names=("base", "arm", "tool"),
    )


class JointValueTest(unittest.TestCase):
    def test_clamp_respects_limits(self):
        model = _two_link_model()
        shoulder = model.joints_by_name["shoulder"]
        self.assertEqual(clamp_joint_value_deg(shoulder, 200.0), 90.0)
        self.assertEqual(clamp_joint_value_deg(shoulder, -200.0), -90.0)
        self.assertEqual(clamp_joint_value_deg(shoulder, 45.0), 45.0)

    def test_continuous_joint_is_unbounded(self):
        joint = Joint("spin", "continuous", "a", "b", min_value_deg=-180.0, max_value_deg=180.0)
        self.assertEqual(clamp_joint_value_deg(joint, 720.0), 720.0)

    def test_fixed_joint_ignores_value(self):
        joint = Joint("weld", "fixed", "a", "b")
        self.assertEqual(clamp_joint_value_deg(joint, 33.0), 0.0)

    def test_default_values_exclude_fixed_and_mimic(self):
        model = RobotModel(
            root_link="a",
            joints=(
                Joint("weld", "fixed", "a", "b"),
                Joint("drive", "revolute", "b", "c", min_value_deg=-10.0, max_value_deg=10.0),
                Joint(
                    "follow",
                    "revolute",
                    "c",
                    "d",
                    min_value_deg=-10.0,
                    max_value_deg=10.0,
                    mimic=JointMimic("drive"),
                ),
            ),
        )
        self.assertEqual(build_default_joint_values(model), {"drive": 0.0})

    def test_mimic_applies_multiplier_and_offset_in_native_units(self):
        model = RobotModel(
            root_link="a",
            joints=(
                Joint("drive", "revolute", "a", "b", min_value_deg=-180.0, max_value_deg=180.0),
                Joint(
                    "follow",
                    "revolute",
                    "b",
                    "c",
                    min_value_deg=-180.0,
                    max_value_deg=180.0,
                    mimic=JointMimic("drive", multiplier=-2.0, offset=0.0),
                ),
            ),
        )
        follow = model.joints_by_name["follow"]
        self.assertAlmostEqual(resolve_joint_value(model, follow, {"drive": 30.0}), -60.0)

    def test_mimic_result_is_clamped(self):
        model = RobotModel(
            root_link="a",
            joints=(
                Joint("drive", "revolute", "a", "b", min_value_deg=-180.0, max_value_deg=180.0),
                Joint(
                    "follow",
                    "revolute",
                    "b",
                    "c",
                    min_value_deg=-15.0,
                    max_value_deg=15.0,
                    mimic=JointMimic("drive", multiplier=3.0),
                ),
            ),
        )
        follow = model.joints_by_name["follow"]
        self.assertAlmostEqual(resolve_joint_value(model, follow, {"drive": 90.0}), 15.0)

    def test_mimic_cycle_falls_back_to_default(self):
        model = RobotModel(
            root_link="a",
            joints=(
                Joint(
                    "left",
                    "revolute",
                    "a",
                    "b",
                    min_value_deg=-90.0,
                    max_value_deg=90.0,
                    default_value_deg=7.0,
                    mimic=JointMimic("right"),
                ),
                Joint(
                    "right",
                    "revolute",
                    "b",
                    "c",
                    min_value_deg=-90.0,
                    max_value_deg=90.0,
                    default_value_deg=5.0,
                    mimic=JointMimic("left"),
                ),
            ),
        )
        # left -> right -> left: the re-entrant visit of "left" short-circuits to
        # left's own default (7.0), which then propagates back out through the
        # identity mimic of "right" instead of recursing forever.
        self.assertAlmostEqual(
            resolve_joint_value(model, model.joints_by_name["left"], {}), 7.0
        )
        self.assertAlmostEqual(
            resolve_joint_value(model, model.joints_by_name["right"], {}), 5.0
        )


class ForwardKinematicsTest(unittest.TestCase):
    def test_zero_pose_places_links_by_origins(self):
        transforms = solve_link_world_transforms(_two_link_model(), {})
        self.assertEqual(set(transforms), {"base", "arm", "tool"})
        _almost(self, transform_point(transforms["arm"], (0.0, 0.0, 0.0)), (0.0, 0.0, 1.0))
        _almost(self, transform_point(transforms["tool"], (0.0, 0.0, 0.0)), (1.0, 0.0, 1.0))

    def test_revolute_rotation_propagates_to_child(self):
        transforms = solve_link_world_transforms(_two_link_model(), {"shoulder": 90.0})
        _almost(self, transform_point(transforms["tool"], (0.0, 0.0, 0.0)), (0.0, 1.0, 1.0))

    def test_prismatic_translation_is_clamped_along_axis(self):
        transforms = solve_link_world_transforms(_two_link_model(), {"slide": 10.0})
        _almost(self, transform_point(transforms["tool"], (0.0, 0.0, 0.0)), (1.5, 0.0, 1.0))

    def test_posed_local_transform_of_fixed_joint_is_its_origin(self):
        joint = Joint(
            "weld",
            "fixed",
            "a",
            "b",
            origin_transform=translation_transform(0.0, 2.0, 0.0),
        )
        _almost(self, posed_joint_local_transform(joint, 45.0), translation_transform(0.0, 2.0, 0.0))

    def test_link_origin_in_frame(self):
        model = _two_link_model()
        local = link_origin_in_frame(model, {"shoulder": 90.0}, "tool", "arm")
        _almost(self, local, (1.0, 0.0, 0.0))
        self.assertIsNone(link_origin_in_frame(model, {}, "tool", "missing"))
        self.assertIsNone(link_origin_in_frame(model, {}, "", "arm"))

    def test_root_point_in_frame(self):
        model = _two_link_model()
        local = root_point_in_frame(model, {}, (0.0, 0.0, 1.0), "arm")
        _almost(self, local, (0.0, 0.0, 0.0))
        self.assertIsNone(root_point_in_frame(model, {}, (0.0, 0.0, 0.0), "nope"))

    def test_empty_root_yields_no_transforms(self):
        self.assertEqual(solve_link_world_transforms(RobotModel(root_link="")), {})

    def test_solver_is_deterministic(self):
        model = _two_link_model()
        first = solve_link_world_transforms(model, {"shoulder": 33.0, "slide": 0.25})
        second = solve_link_world_transforms(model, {"shoulder": 33.0, "slide": 0.25})
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
