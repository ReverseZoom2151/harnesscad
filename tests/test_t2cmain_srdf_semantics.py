import math
import unittest

from spec.t2cmain_srdf_semantics import (
    SrdfParseError,
    adjacent_collision_pairs,
    chain_joint_names,
    classify_collision_reason,
    group_joint_names,
    group_link_names,
    links_are_adjacent,
    missing_adjacent_disables,
    parse_srdf,
)
from spec.t2cmain_urdf_parser import parse_urdf

ROBOT_URDF = """
<robot name="panda">
  <link name="base"/>
  <link name="shoulder"/>
  <link name="elbow"/>
  <link name="wrist"/>
  <link name="hand"/>
  <link name="finger"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="shoulder"/>
    <axis xyz="0 0 1"/><limit lower="-1.57" upper="1.57"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="shoulder"/><child link="elbow"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="elbow"/><child link="wrist"/>
    <axis xyz="0 1 0"/><limit lower="-2" upper="2"/>
  </joint>
  <joint name="weld" type="fixed">
    <parent link="wrist"/><child link="hand"/>
  </joint>
  <joint name="grip" type="prismatic">
    <parent link="hand"/><child link="finger"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.04"/>
  </joint>
</robot>
"""

ROBOT_SRDF = """
<robot name="panda">
  <group name="arm">
    <chain base_link="base" tip_link="hand"/>
  </group>
  <group name="gripper">
    <joint name="grip"/>
  </group>
  <group name="whole">
    <group name="arm"/>
    <group name="gripper"/>
  </group>
  <end_effector name="eef" parent_link="hand" group="gripper" parent_group="arm"/>
  <group_state name="home" group="arm">
    <joint name="j1" value="0"/>
    <joint name="j2" value="0.5"/>
  </group_state>
  <disable_collisions link1="base" link2="shoulder" reason="Adjacent"/>
  <disable_collisions link1="base" link2="elbow" reason="Never"/>
  <disable_collisions link1="hand" link2="wrist" reason="assumed contact"/>
</robot>
"""


class ReasonClassificationTest(unittest.TestCase):
    def test_sources(self):
        self.assertEqual(classify_collision_reason("Adjacent"), "adjacent")
        self.assertEqual(classify_collision_reason("Never"), "sampled")
        self.assertEqual(classify_collision_reason("Default"), "sampled")
        self.assertEqual(classify_collision_reason("setup assistant"), "setup_assistant")
        self.assertEqual(classify_collision_reason("assumed contact"), "assumed")
        self.assertEqual(classify_collision_reason("because I said so"), "manual")


class ChainAndClosureTest(unittest.TestCase):
    def setUp(self):
        self.urdf = parse_urdf(ROBOT_URDF)
        self.srdf = parse_srdf(ROBOT_SRDF, self.urdf)
        self.groups = {g.name: g for g in self.srdf.planning_groups}

    def test_chain_walk_is_ordered(self):
        self.assertEqual(chain_joint_names(self.urdf, "base", "wrist"), ("j1", "j2", "j3"))

    def test_chain_to_self_is_empty(self):
        self.assertEqual(chain_joint_names(self.urdf, "base", "base"), ())

    def test_chain_to_unreachable_link_is_empty(self):
        self.assertEqual(chain_joint_names(self.urdf, "wrist", "base"), ())

    def test_chain_group_expands_to_plannable_joints(self):
        names = group_joint_names(self.groups["arm"], self.urdf, self.groups)
        self.assertEqual(names, ("j1", "j2", "j3"))

    def test_subgroup_closure_unions_children(self):
        names = group_joint_names(self.groups["whole"], self.urdf, self.groups)
        self.assertEqual(names, ("j1", "j2", "j3", "grip"))

    def test_fixed_joint_excluded_from_chain_closure(self):
        # "weld" lies on the raw base -> hand joint path, but is fixed and so is
        # never a plannable degree of freedom of the group.
        self.assertEqual(
            chain_joint_names(self.urdf, "base", "hand"), ("j1", "j2", "j3", "weld")
        )
        self.assertNotIn(
            "weld", group_joint_names(self.groups["arm"], self.urdf, self.groups)
        )

    def test_group_link_names_from_chain(self):
        links = group_link_names(self.groups["arm"], self.urdf, self.groups)
        self.assertEqual(links, {"shoulder", "elbow", "wrist", "hand"})

    def test_group_link_names_from_joint(self):
        links = group_link_names(self.groups["gripper"], self.urdf, self.groups)
        self.assertEqual(links, {"finger"})

    def test_links_are_adjacent(self):
        self.assertTrue(links_are_adjacent(self.urdf, "hand", {"finger"}))
        self.assertFalse(links_are_adjacent(self.urdf, "base", {"finger"}))


class ParseSrdfTest(unittest.TestCase):
    def setUp(self):
        self.urdf = parse_urdf(ROBOT_URDF)
        self.srdf = parse_srdf(ROBOT_SRDF, self.urdf)

    def test_robot_name(self):
        self.assertEqual(self.srdf.robot_name, "panda")

    def test_end_effector_link_is_group_tip(self):
        eef = self.srdf.end_effectors[0]
        self.assertEqual((eef.group, eef.parent_group), ("gripper", "arm"))
        # gripper declares no explicit links or chains, so the effector link falls
        # back to the parent_link.
        self.assertEqual(eef.link, "hand")

    def test_group_state_values_are_radians(self):
        state = self.srdf.group_states[0]
        self.assertEqual(state.group, "arm")
        self.assertAlmostEqual(state.joint_values_rad["j2"], 0.5)

    def test_disabled_pairs_classified(self):
        sources = [pair.source for pair in self.srdf.disabled_collision_pairs]
        self.assertEqual(sources, ["adjacent", "sampled", "assumed"])

    def test_robot_name_must_match_urdf(self):
        with self.assertRaises(SrdfParseError):
            parse_srdf(ROBOT_SRDF.replace('name="panda"', 'name="other"', 1), self.urdf)

    def test_group_state_outside_limits_rejected(self):
        text = ROBOT_SRDF.replace('<joint name="j2" value="0.5"/>', '<joint name="j2" value="9"/>')
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_group_state_below_limit_rejected(self):
        text = ROBOT_SRDF.replace('<joint name="j2" value="0.5"/>', '<joint name="j2" value="-9"/>')
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_group_state_within_radian_limit_accepted(self):
        # j1 limit is +/-1.57 rad, i.e. +/-89.95 deg after URDF parsing; 1.5 rad is fine.
        text = ROBOT_SRDF.replace('<joint name="j1" value="0"/>', '<joint name="j1" value="1.5"/>')
        state = parse_srdf(text, self.urdf).group_states[0]
        self.assertAlmostEqual(state.joint_values_rad["j1"], 1.5)

    def test_group_state_cannot_set_joint_outside_its_group(self):
        text = ROBOT_SRDF.replace(
            '<joint name="j2" value="0.5"/>', '<joint name="grip" value="0.01"/>'
        )
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_group_state_cannot_set_fixed_joint(self):
        text = ROBOT_SRDF.replace(
            '<joint name="j2" value="0.5"/>', '<joint name="weld" value="0"/>'
        )
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_duplicate_group_rejected(self):
        text = ROBOT_SRDF.replace(
            '<group name="gripper">', '<group name="arm"><joint name="grip"/></group><group name="gripper">', 1
        )
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_missing_subgroup_rejected(self):
        text = ROBOT_SRDF.replace('<group name="gripper"/>', '<group name="ghost"/>')
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_group_referencing_missing_joint(self):
        text = ROBOT_SRDF.replace('<joint name="grip"/>', '<joint name="ghost"/>')
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_duplicate_disabled_pair_rejected_regardless_of_order(self):
        text = ROBOT_SRDF.replace(
            '<disable_collisions link1="base" link2="elbow" reason="Never"/>',
            '<disable_collisions link1="shoulder" link2="base" reason="Never"/>',
        )
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_self_pair_rejected(self):
        text = ROBOT_SRDF.replace(
            '<disable_collisions link1="base" link2="elbow" reason="Never"/>',
            '<disable_collisions link1="base" link2="base" reason="Never"/>',
        )
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_reasonless_pair_rejected(self):
        text = ROBOT_SRDF.replace(
            '<disable_collisions link1="base" link2="elbow" reason="Never"/>',
            '<disable_collisions link1="base" link2="elbow" reason=""/>',
        )
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_end_effector_group_overlapping_parent_group_rejected(self):
        text = ROBOT_SRDF.replace(
            'group="gripper" parent_group="arm"', 'group="arm" parent_group="arm"'
        )
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_end_effector_parent_link_must_be_in_parent_group(self):
        text = ROBOT_SRDF.replace('parent_link="hand"', 'parent_link="base"')
        with self.assertRaises(SrdfParseError):
            parse_srdf(text, self.urdf)

    def test_no_groups_rejected(self):
        with self.assertRaises(SrdfParseError):
            parse_srdf('<robot name="panda"/>', self.urdf)

    def test_bad_xml_rejected(self):
        with self.assertRaises(SrdfParseError):
            parse_srdf("<robot", self.urdf)


class CollisionMatrixTest(unittest.TestCase):
    def setUp(self):
        self.urdf = parse_urdf(ROBOT_URDF)
        self.srdf = parse_srdf(ROBOT_SRDF, self.urdf)

    def test_adjacent_pairs_from_urdf(self):
        self.assertEqual(
            adjacent_collision_pairs(self.urdf),
            (
                ("base", "shoulder"),
                ("elbow", "shoulder"),
                ("elbow", "wrist"),
                ("finger", "hand"),
                ("hand", "wrist"),
            ),
        )

    def test_missing_adjacent_disables(self):
        self.assertEqual(
            missing_adjacent_disables(self.srdf, self.urdf),
            (("elbow", "shoulder"), ("elbow", "wrist"), ("finger", "hand")),
        )


if __name__ == "__main__":
    unittest.main()
