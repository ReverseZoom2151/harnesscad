"""Tests for mechanism mobility and kinematic-tree validity (AADvark/ArtiCAD/ASSEMCAD)."""

import unittest

from harnesscad.domain.geometry.assembly import mobility as mob


class KutzbachTest(unittest.TestCase):
    def test_scissors_one_dof(self):
        # Two blades + one revolute pivot: M = 1 (the "scissors test").
        self.assertEqual(mob.kutzbach_mobility(2, ["revolute"]), 1)

    def test_planar_four_bar_linkage(self):
        # 4 links, 4 revolute joints, planar: M = 3*(4-1) - 4*(3-1) = 9 - 8 = 1.
        self.assertEqual(mob.kutzbach_mobility(4, ["revolute"] * 4, planar=True), 1)

    def test_rigid_assembly_zero_dof(self):
        self.assertEqual(mob.kutzbach_mobility(2, ["fixed"]), 0)

    def test_spatial_free_joint(self):
        self.assertEqual(mob.kutzbach_mobility(2, ["free"]), 6)

    def test_mate_type_accepted(self):
        # coaxial mate leaves 2 freedom; spatial 2-link => 6 - (6-2) = 2.
        self.assertEqual(mob.kutzbach_mobility(2, ["coaxial"]), 2)

    def test_bad_link_count(self):
        with self.assertRaises(ValueError):
            mob.kutzbach_mobility(0, [])


class TreeDofTest(unittest.TestCase):
    def test_sum_of_joint_freedoms(self):
        self.assertEqual(mob.tree_dof(["revolute", "fixed", "ball"]), 1 + 0 + 3)


class ValidateTreeTest(unittest.TestCase):
    def test_valid_tree(self):
        parts = ["base", "arm", "hand"]
        joints = [("base", "arm", "revolute"), ("arm", "hand", "revolute")]
        rep = mob.validate_kinematic_tree(parts, joints, root="base")
        self.assertTrue(rep.valid)
        self.assertEqual(rep.total_dof, 2)
        self.assertEqual(rep.root, "base")
        self.assertEqual(set(rep.reachable), set(parts))

    def test_wrong_joint_count(self):
        rep = mob.validate_kinematic_tree(["a", "b", "c"], [("a", "b", "fixed")])
        self.assertFalse(rep.valid)
        self.assertTrue(any("needs" in e for e in rep.errors))

    def test_two_parents_detected(self):
        parts = ["a", "b", "c"]
        joints = [("a", "c", "fixed"), ("b", "c", "fixed")]
        rep = mob.validate_kinematic_tree(parts, joints)
        self.assertFalse(rep.valid)
        self.assertTrue(any("more than one parent" in e for e in rep.errors))

    def test_disconnected_part(self):
        parts = ["a", "b", "c"]
        joints = [("a", "b", "fixed"), ("a", "b", "revolute")]  # c unreachable, dup edge
        rep = mob.validate_kinematic_tree(parts, joints, root="a")
        self.assertFalse(rep.valid)

    def test_unknown_part_reference(self):
        rep = mob.validate_kinematic_tree(["a", "b"], [("a", "ghost", "fixed")])
        self.assertFalse(rep.valid)


class ConnectivityTest(unittest.TestCase):
    def test_connected(self):
        ok, reach = mob.assembly_connectivity(["a", "b", "c"],
                                               [("a", "b"), ("b", "c")], root="a")
        self.assertTrue(ok)
        self.assertEqual(set(reach), {"a", "b", "c"})

    def test_disconnected(self):
        ok, _ = mob.assembly_connectivity(["a", "b", "c"], [("a", "b")])
        self.assertFalse(ok)


class ClashTest(unittest.TestCase):
    def test_clear_when_no_overlap(self):
        self.assertEqual(mob.classify_clash("a", "b", 0.0, []), "clear")

    def test_expected_for_contact_mate(self):
        mates = [("a", "b", "press_fit")]
        self.assertEqual(mob.classify_clash("a", "b", 1.5, mates), "expected")

    def test_clash_for_unexplained_overlap(self):
        mates = [("a", "b", "face_to_face")]
        self.assertEqual(mob.classify_clash("a", "b", 1.5, mates), "clash")


if __name__ == "__main__":
    unittest.main()
