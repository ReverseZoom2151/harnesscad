import math
import unittest

from harnesscad.domain.geometry.kinematics.forward_kinematics import (
    solve_link_world_transforms,
    transform_point,
)
from harnesscad.domain.spec.urdf import (
    UrdfParseError,
    parse_rgba_color,
    parse_urdf,
    validate_tree,
)

ARM_URDF = """
<robot name="arm">
  <material name="steel"><color rgba="0.2 0.4 0.6 1"/></material>
  <link name="base">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
      <material name="steel"/>
    </visual>
  </link>
  <link name="upper">
    <visual>
      <geometry><cylinder radius="0.03" length="0.4"/></geometry>
      <material name="inline"><color rgba="1 0 0 1"/></material>
    </visual>
  </link>
  <link name="tool">
    <visual>
      <geometry><mesh filename="meshes/tool.stl" scale="0.001 0.001 0.001"/></geometry>
    </visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5707963267948966" upper="1.5707963267948966" effort="10" velocity="1"/>
  </joint>
  <joint name="grip" type="prismatic">
    <parent link="upper"/>
    <child link="tool"/>
    <origin xyz="0.4 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.05" effort="10" velocity="1"/>
  </joint>
</robot>
"""


class ParseUrdfTest(unittest.TestCase):
    def setUp(self):
        self.document = parse_urdf(ARM_URDF)

    def test_robot_name_and_root(self):
        self.assertEqual(self.document.name, "arm")
        self.assertEqual(self.document.root_link, "base")
        self.assertEqual(self.document.model.link_names, ("base", "upper", "tool"))

    def test_revolute_limits_are_converted_to_degrees(self):
        shoulder = self.document.model.joints_by_name["shoulder"]
        self.assertAlmostEqual(shoulder.min_value_deg, -90.0, places=9)
        self.assertAlmostEqual(shoulder.max_value_deg, 90.0, places=9)

    def test_prismatic_limits_stay_in_native_units(self):
        grip = self.document.model.joints_by_name["grip"]
        self.assertEqual((grip.min_value_deg, grip.max_value_deg), (0.0, 0.05))

    def test_named_material_resolves_by_reference(self):
        base = self.document.links[0]
        self.assertEqual(base.visuals[0].color, "#336699")

    def test_inline_material_color_wins(self):
        upper = self.document.links[1]
        self.assertEqual(upper.visuals[0].color, "#ff0000")

    def test_primitives_are_typed(self):
        self.assertEqual(self.document.links[0].visuals[0].primitive.type, "box")
        cylinder = self.document.links[1].visuals[0].primitive
        self.assertEqual(cylinder.type, "cylinder")
        self.assertEqual((cylinder.radius, cylinder.length), (0.03, 0.4))

    def test_mesh_visual_records_filename_and_scale(self):
        tool = self.document.links[2].visuals[0]
        self.assertIsNone(tool.primitive)
        self.assertEqual(tool.mesh_filename, "meshes/tool.stl")
        self.assertEqual(tool.mesh_scale, (0.001, 0.001, 0.001))

    def test_parsed_model_is_directly_solvable(self):
        transforms = solve_link_world_transforms(self.document.model, {"shoulder": 90.0})
        point = transform_point(transforms["tool"], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(point[0], 0.0, places=9)
        self.assertAlmostEqual(point[1], 0.4, places=9)
        self.assertAlmostEqual(point[2], 0.1, places=9)

    def test_visual_origin_is_composed(self):
        origin = self.document.links[0].visuals[0].origin_transform
        self.assertAlmostEqual(origin[11], 0.05, places=9)

    def test_parsing_is_deterministic(self):
        self.assertEqual(parse_urdf(ARM_URDF).model.joints, self.document.model.joints)


class RgbaTest(unittest.TestCase):
    def test_rounds_to_hex(self):
        self.assertEqual(parse_rgba_color("1 1 1 1", "ctx"), "#ffffff")
        self.assertEqual(parse_rgba_color("0 0 0 1", "ctx"), "#000000")

    def test_out_of_range_rejected(self):
        with self.assertRaises(UrdfParseError):
            parse_rgba_color("1.5 0 0 1", "ctx")

    def test_wrong_arity_rejected(self):
        with self.assertRaises(UrdfParseError):
            parse_rgba_color("1 0 0", "ctx")


class ValidationTest(unittest.TestCase):
    def _urdf(self, body, name="r"):
        return f'<robot name="{name}">{body}</robot>'

    def test_not_xml(self):
        with self.assertRaises(UrdfParseError):
            parse_urdf("<robot>")

    def test_wrong_root_element(self):
        with self.assertRaises(UrdfParseError):
            parse_urdf("<model name='x'/>")

    def test_no_links(self):
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf(""))

    def test_duplicate_link_name(self):
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf('<link name="a"/><link name="a"/>'))

    def test_unsupported_joint_type(self):
        body = (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="floating"><parent link="a"/><child link="b"/></joint>'
        )
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf(body))

    def test_revolute_requires_limit(self):
        body = (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
            '<axis xyz="0 0 1"/></joint>'
        )
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf(body))

    def test_joint_referencing_missing_link(self):
        body = (
            '<link name="a"/>'
            '<joint name="j" type="fixed"><parent link="a"/><child link="ghost"/></joint>'
        )
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf(body))

    def test_link_with_two_parents(self):
        body = (
            '<link name="a"/><link name="b"/><link name="c"/>'
            '<joint name="j1" type="fixed"><parent link="a"/><child link="c"/></joint>'
            '<joint name="j2" type="fixed"><parent link="b"/><child link="c"/></joint>'
        )
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf(body))

    def test_forest_is_rejected(self):
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf('<link name="a"/><link name="b"/>'))

    def test_negative_box_size_rejected(self):
        body = (
            '<link name="a"><visual><geometry><box size="1 -1 1"/></geometry></visual></link>'
        )
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf(body))

    def test_zero_radius_cylinder_rejected(self):
        body = (
            '<link name="a"><visual><geometry>'
            '<cylinder radius="0" length="1"/></geometry></visual></link>'
        )
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf(body))

    def test_mimic_referencing_missing_joint(self):
        body = (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1"/>'
            '<mimic joint="nope"/></joint>'
        )
        with self.assertRaises(UrdfParseError):
            parse_urdf(self._urdf(body))

    def test_mimic_defaults_are_parsed(self):
        body = (
            '<link name="a"/><link name="b"/><link name="c"/>'
            '<joint name="drive" type="revolute"><parent link="a"/><child link="b"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>'
            '<joint name="follow" type="revolute"><parent link="b"/><child link="c"/>'
            '<axis xyz="0 0 1"/><limit lower="-1" upper="1"/>'
            '<mimic joint="drive" multiplier="-1" offset="0.5"/></joint>'
        )
        document = parse_urdf(self._urdf(body))
        mimic = document.model.joints_by_name["follow"].mimic
        self.assertEqual((mimic.joint, mimic.multiplier, mimic.offset), ("drive", -1.0, 0.5))

    def test_continuous_joint_gets_display_range(self):
        body = (
            '<link name="a"/><link name="b"/>'
            '<joint name="spin" type="continuous"><parent link="a"/><child link="b"/>'
            '<axis xyz="0 1 0"/></joint>'
        )
        joint = parse_urdf(self._urdf(body)).model.joints_by_name["spin"]
        self.assertEqual((joint.min_value_deg, joint.max_value_deg), (-180.0, 180.0))
        self.assertEqual(joint.axis, (0.0, 1.0, 0.0))

    def test_namespaced_document_is_tolerated(self):
        text = (
            '<robot xmlns:xacro="http://ros.org/xacro" name="ns">'
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint>'
            "</robot>"
        )
        self.assertEqual(parse_urdf(text).root_link, "a")

    def test_validate_tree_detects_cycle(self):
        from harnesscad.domain.geometry.kinematics.forward_kinematics import Joint

        joints = (
            Joint("j1", "fixed", "a", "b"),
            Joint("j2", "fixed", "b", "a"),
        )
        with self.assertRaises(UrdfParseError):
            validate_tree(["a", "b"], joints)


if __name__ == "__main__":
    unittest.main()
