import unittest

from numeric.cq_assembly_dof import (
    BINARY_KINDS,
    DOF_REMOVED,
    UNARY_KINDS,
    AssemblyConstraint,
    AssemblyDOF,
    ConstraintError,
)


class TestConstraintAlgebra(unittest.TestCase):
    def test_kind_partition(self):
        self.assertEqual(UNARY_KINDS & BINARY_KINDS, frozenset())
        self.assertEqual(set(DOF_REMOVED), UNARY_KINDS | BINARY_KINDS)

    def test_plane_is_axis_plus_point(self):
        self.assertEqual(
            DOF_REMOVED["Plane"], DOF_REMOVED["Axis"] + DOF_REMOVED["Point"]
        )

    def test_constraint_props(self):
        c = AssemblyConstraint("Point", ("a", "b"))
        self.assertEqual(c.arity, 2)
        self.assertEqual(c.dof_removed, 3)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.a = AssemblyDOF()
        self.a.add_part("base").add_part("arm")

    def test_unknown_kind(self):
        with self.assertRaises(ConstraintError):
            self.a.constrain("Glue", ("base", "arm"))

    def test_wrong_arity(self):
        with self.assertRaises(ConstraintError):
            self.a.constrain("Point", ("base",))  # binary needs 2
        with self.assertRaises(ConstraintError):
            self.a.constrain("Fixed", ("base", "arm"))  # unary needs 1

    def test_unknown_part(self):
        with self.assertRaises(ConstraintError):
            self.a.constrain("Point", ("base", "ghost"))

    def test_duplicate_part(self):
        with self.assertRaises(ConstraintError):
            self.a.add_part("base")


class TestDOFAnalysis(unittest.TestCase):
    def test_single_free_part(self):
        r = AssemblyDOF().add_part("p").analyze()
        self.assertEqual(r.total_dof, 6)
        self.assertEqual(r.mobility, 6)
        self.assertEqual(r.status, "under")

    def test_fixed_part_well(self):
        a = AssemblyDOF().add_part("p")
        a.constrain("Fixed", ("p",))
        r = a.analyze()
        self.assertEqual(r.removed, 6)
        self.assertEqual(r.mobility, 0)
        self.assertEqual(r.status, "well")
        self.assertTrue(r.grounded)

    def test_two_parts_underconstrained(self):
        a = AssemblyDOF().add_part("base").add_part("arm")
        a.constrain("Fixed", ("base",))       # 6
        a.constrain("Point", ("base", "arm"))  # 3
        r = a.analyze()
        # 12 - 9 = 3 remaining
        self.assertEqual(r.mobility, 3)
        self.assertEqual(r.status, "under")

    def test_two_parts_well(self):
        a = AssemblyDOF().add_part("base").add_part("arm")
        a.constrain("Fixed", ("base",))              # 6
        a.constrain("Plane", ("base", "arm"))        # 5
        a.constrain("PointInPlane", ("base", "arm"))  # 1
        # 6 + 5 + 1 = 12 -> mobility 0
        r = a.analyze()
        self.assertEqual(r.removed, 12)
        self.assertEqual(r.status, "well")

    def test_overconstrained(self):
        a = AssemblyDOF().add_part("base").add_part("arm")
        a.constrain("Fixed", ("base",))         # 6
        a.constrain("Plane", ("base", "arm"))   # 5
        a.constrain("Plane", ("arm", "base"))   # duplicate signature -> redundant
        a.constrain("Point", ("base", "arm"))   # 3 -> 6+5+3 = 14 > 12
        r = a.analyze()
        self.assertEqual(r.status, "over")
        self.assertTrue(r.mobility < 0 or r.redundant)

    def test_redundant_duplicate(self):
        a = AssemblyDOF().add_part("base").add_part("arm")
        a.constrain("Fixed", ("base",))
        a.constrain("Point", ("base", "arm"))
        a.constrain("Point", ("arm", "base"))  # same signature (set of parts)
        r = a.analyze()
        self.assertEqual(len(r.redundant), 1)

    def test_no_anchor_note(self):
        a = AssemblyDOF().add_part("x").add_part("y")
        # remove exactly 12 via four distinct binary mates, no Fixed anchor
        a.constrain("Plane", ("x", "y"))        # 5
        a.constrain("Point", ("x", "y"))        # 3
        a.constrain("Axis", ("x", "y"))         # 2
        a.constrain("PointOnLine", ("x", "y"))  # 2  -> 12 total
        r = a.analyze()
        self.assertEqual(r.status, "well")
        self.assertFalse(r.grounded)
        self.assertTrue(any("rigid-body" in n for n in r.notes))

    def test_local_overconstraint_note(self):
        a = AssemblyDOF().add_part("solo")
        a.constrain("Fixed", ("solo",))       # 6
        a.constrain("FixedPoint", ("solo",))  # +3 -> 9 on one part
        r = a.analyze()
        self.assertTrue(any("over-constrained" in n for n in r.notes))


if __name__ == "__main__":
    unittest.main()
