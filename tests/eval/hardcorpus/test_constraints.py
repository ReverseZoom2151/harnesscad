"""Constraint briefs: every shipped constraint is checkable, and IoU is not.

The reference solution of every constraint brief must build, pass the gate, and
satisfy every constraint (the positive control). A second, DIFFERENT satisfying
answer must ALSO satisfy -- and score a low IoU against the reference -- which is the
proof that no shape metric can grade a constraint set. And a part that violates a
constraint must be caught, or the constraint is decoration.
"""

from __future__ import annotations

import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, Hole, NewSketch
from harnesscad.eval.hardcorpus import constraints as con
from harnesscad.eval.hardcorpus import occt
from harnesscad.io import gate


class TestConstraints(unittest.TestCase):

    def test_every_reference_satisfies_and_passes_the_gate(self):
        for b in con.BRIEFS:
            r = con.grade(b, b.reference)
            self.assertTrue(r.built, "%s reference did not build" % b.id)
            self.assertTrue(r.satisfied,
                            "%s reference does not satisfy its own constraints: %s"
                            % (b.id, [(x.name, x.satisfied) for x in r.results]))
            eng = occt.build(b.reference).engine
            self.assertTrue(gate.check(eng, source=eng).ok,
                            "%s reference fails the gate" % b.id)

    def test_a_second_answer_satisfies_but_has_low_iou(self):
        # The point of the whole module: many shapes satisfy, so IoU is undefined
        # as a grader. Two satisfying answers disagree on shape.
        for b in con.BRIEFS:
            r = con.grade(b, b.alt_reference)
            self.assertTrue(r.satisfied,
                            "%s alt does not satisfy: %s"
                            % (b.id, [(x.name, x.satisfied) for x in r.results]))
            a = occt.build(b.reference).shape
            c = occt.build(b.alt_reference).shape
            iou = occt.boolean_iou(a, c)
            self.assertLess(iou, 0.8,
                            "%s: two satisfying answers score IoU %.3f -- if that "
                            "were near 1, IoU could grade this brief" % (b.id, iou))

    def test_a_violating_part_is_caught(self):
        b = con.BRIEFS[0]                              # bracket: fits 50x50x20
        # A part that overflows the envelope must fail the envelope constraint.
        too_big = [NewSketch("XY"), AddRectangle("sk1", 0, 0, 80, 80),
                   Extrude("sk1", 6),
                   Hole("sk1", 25, 25, con.CLEARANCE_ISO273["M8"], None, True,
                        "simple")]
        r = con.grade(b, too_big)
        self.assertFalse(r.satisfied)
        env = next(x for x in r.results if x.name == "envelope")
        self.assertFalse(env.satisfied)

    def test_a_part_with_no_bolt_hole_fails_bolt_constraint(self):
        b = con.BRIEFS[0]
        no_hole = [NewSketch("XY"), AddRectangle("sk1", 0, 0, 50, 50),
                   Extrude("sk1", 6)]
        r = con.grade(b, no_hole)
        bolt = next(x for x in r.results if x.name == "bolt_bore")
        self.assertFalse(bolt.satisfied)

    def test_bending_uses_an_exact_section_modulus(self):
        # A too-thin part must exceed the allowable stress under the load case.
        b = con.BRIEFS[0]
        thin = [NewSketch("XY"), AddRectangle("sk1", 0, 0, 50, 50),
                Extrude("sk1", 1.2),
                Hole("sk1", 25, 25, con.CLEARANCE_ISO273["M8"], None, True,
                     "simple")]
        r = con.grade(b, thin)
        bend = next(x for x in r.results if x.name == "bending")
        self.assertFalse(bend.satisfied,
                         "a 1.2 mm arm must yield under 200 N at 45 mm")

    def test_dropped_constraints_are_named_with_reasons(self):
        self.assertIn("min_wall", con.DROPPED_CONSTRAINTS)
        for name, why in con.DROPPED_CONSTRAINTS.items():
            self.assertTrue(why.strip(), "%s dropped with no reason" % name)

    def test_a_brief_with_no_constraint_is_refused(self):
        with self.assertRaises(ValueError):
            con.ConstraintBrief(id="empty", text="", proc="", envelope=(1, 1, 1),
                                material="AL6061", constraints=(),
                                reference=(NewSketch("XY"),))


if __name__ == "__main__":
    unittest.main()
