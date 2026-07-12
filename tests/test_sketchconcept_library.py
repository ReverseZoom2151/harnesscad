import unittest

from library.sketchconcept_library import ConceptCycleError, ConceptLibrary
from reconstruction.sketchconcept_template import (
    Concept,
    Const,
    ConstraintSpec,
    Member,
    Slot,
    SubInstance,
    input_ref,
    instantiate,
    sub_out_ref,
)


def corner(name="corner"):
    """Two lines meeting perpendicularly; exports the second line."""
    return Concept(
        name=name,
        slots=("x", "y", "w", "h"),
        members=(
            Member.make("a", "line", {"x1": Slot("x"), "y1": Slot("y"),
                                      "x2": Slot("w"), "y2": Slot("y")}),
            Member.make("b", "line", {"x1": Slot("w"), "y1": Slot("y"),
                                      "x2": Slot("w"), "y2": Slot("h")}),
        ),
        constraints=(ConstraintSpec("perpendicular", ("a", "b")),),
        out_refs=("a", "b"),
        defaults=(("y", 0.0),),
    )


def hole(name="hole"):
    """A circle constrained concentric with one external input."""
    return Concept(
        name=name,
        slots=("cx", "cy", "r"),
        members=(Member.make("c", "circle", {"x": Slot("cx"), "y": Slot("cy"), "r": Slot("r")}),),
        constraints=(ConstraintSpec("concentric", ("c", input_ref(0))),),
        in_arity=1,
        out_refs=("c",),
    )


class TestAddAndDedup(unittest.TestCase):
    def test_add_and_get(self):
        lib = ConceptLibrary()
        self.assertEqual(lib.add(corner()), "corner")
        self.assertEqual(len(lib), 1)
        self.assertIn("corner", lib)
        self.assertEqual(lib.get("corner").name, "corner")

    def test_dedup_returns_existing(self):
        lib = ConceptLibrary()
        lib.add(corner("c1"))
        same = corner("c2")
        self.assertEqual(lib.add(same), "c1")
        self.assertEqual(len(lib), 1)
        self.assertEqual(lib.aliases(), {"c2": "c1"})
        self.assertEqual(lib.resolve("c2"), "c1")
        self.assertEqual(lib.get("c2").name, "c1")

    def test_dedup_disabled(self):
        lib = ConceptLibrary()
        lib.add(corner("c1"))
        self.assertEqual(lib.add(corner("c2"), dedup=False), "c2")
        self.assertEqual(len(lib), 2)

    def test_distinct_structures_both_kept(self):
        lib = ConceptLibrary()
        lib.add(corner())
        lib.add(hole())
        self.assertEqual(sorted(lib.names()), ["corner", "hole"])

    def test_duplicate_name_rejected(self):
        lib = ConceptLibrary()
        lib.add(corner())
        with self.assertRaises(ValueError):
            lib.add(hole("corner"))

    def test_invalid_concept_rejected(self):
        lib = ConceptLibrary()
        bad = Concept(name="bad", members=(Member.make("m", "point",
                                                       {"x": Slot("nope"), "y": Const(0)}),))
        with self.assertRaises(ValueError):
            lib.add(bad)

    def test_missing_subconcept_rejected(self):
        lib = ConceptLibrary()
        with self.assertRaises(KeyError):
            lib.add(Concept(name="h", subs=(SubInstance.make("s", "absent"),)))


class TestHierarchy(unittest.TestCase):
    def build(self):
        lib = ConceptLibrary()
        lib.add(corner())
        lib.add(hole())
        # bracket: a corner, plus a hole whose circle is concentric with the
        # corner's second line; the hole's radius is a bracket-level slot.
        bracket = Concept(
            name="bracket",
            slots=("bx", "by", "bw", "bh", "hr"),
            members=(Member.make("p", "point", {"x": Slot("bx"), "y": Slot("by")}),),
            subs=(
                SubInstance.make("k", "corner", {"x": Slot("bx"), "y": Slot("by"),
                                                 "w": Slot("bw"), "h": Slot("bh")}),
                SubInstance.make("g", "hole",
                                 {"cx": Slot("bx"), "cy": Slot("by"), "r": Slot("hr")},
                                 inputs=[sub_out_ref("k", 1)]),
            ),
            constraints=(ConstraintSpec("coincident", ("p", sub_out_ref("k", 0))),),
            out_refs=(sub_out_ref("g", 0),),
        )
        lib.add(bracket)
        return lib

    def test_depth_and_children(self):
        lib = self.build()
        self.assertEqual(lib.depth("corner"), 1)
        self.assertEqual(lib.depth("bracket"), 2)
        self.assertEqual(sorted(lib.children("bracket")), ["corner", "hole"])

    def test_topological_order(self):
        lib = self.build()
        order = lib.topological_order()
        self.assertLess(order.index("corner"), order.index("bracket"))
        self.assertLess(order.index("hole"), order.index("bracket"))

    def test_flatten_members_and_wiring(self):
        lib = self.build()
        flat = lib.flatten("bracket")
        self.assertTrue(flat.is_flat)
        self.assertEqual(flat.member_ids(), ("p", "k/a", "k/b", "g/c"))
        self.assertEqual(flat.validate(), [])
        # the hole's input reference was wired to the corner's exported line k/b
        self.assertIn(ConstraintSpec("concentric", ("g/c", "k/b")), flat.constraints)
        self.assertIn(ConstraintSpec("perpendicular", ("k/a", "k/b")), flat.constraints)
        self.assertIn(ConstraintSpec("coincident", ("p", "k/a")), flat.constraints)
        self.assertEqual(flat.out_refs, ("g/c",))
        # sub-concept default (corner.y) was overridden by the parent binding
        self.assertEqual(flat.member("k/a").param_map()["y1"], Slot("by"))

    def test_flatten_instantiates(self):
        lib = self.build()
        inst = instantiate(lib.flatten("bracket"),
                           {"bx": 0, "by": 0, "bw": 4, "bh": 3, "hr": 1}, prefix="i")
        ids = [p.pid for p in inst.primitives]
        self.assertEqual(ids, ["i/p", "i/k/a", "i/k/b", "i/g/c"])
        self.assertEqual(inst.as_sketch().validate(), [])
        self.assertEqual(inst.outputs, ("i/g/c",))
        circle = inst.primitives[3].param_map()
        self.assertEqual(circle, {"x": 0.0, "y": 0.0, "r": 1.0})

    def test_sub_default_used_when_unbound(self):
        lib = ConceptLibrary()
        lib.add(corner())
        top = Concept(name="t", slots=("a", "b", "c"),
                      subs=(SubInstance.make("k", "corner",
                                             {"x": Slot("a"), "w": Slot("b"), "h": Slot("c")}),),
                      out_refs=(sub_out_ref("k", 0),))
        lib.add(top)
        flat = lib.flatten("t")
        self.assertEqual(flat.member("k/a").param_map()["y1"], Const(0.0))

    def test_unbound_sub_slot_raises(self):
        lib = ConceptLibrary()
        lib.add(hole())
        top = Concept(name="t", slots=("r",),
                      subs=(SubInstance.make("k", "hole", {"r": Slot("r")},
                                             inputs=[input_ref(0)]),),
                      in_arity=1)
        with self.assertRaises(KeyError):
            lib.add(top)

    def test_wrong_sub_input_arity(self):
        lib = ConceptLibrary()
        lib.add(hole())
        top = Concept(name="t", slots=("r",),
                      subs=(SubInstance.make("k", "hole",
                                             {"cx": Const(0), "cy": Const(0), "r": Slot("r")}),))
        with self.assertRaises(ValueError):
            lib.add(top)

    def test_sub_input_cycle_rejected(self):
        lib = ConceptLibrary()
        lib.add(hole())
        top = Concept(
            name="t", slots=("r",),
            subs=(
                SubInstance.make("a", "hole",
                                 {"cx": Const(0), "cy": Const(0), "r": Slot("r")},
                                 inputs=[sub_out_ref("b", 0)]),
                SubInstance.make("b", "hole",
                                 {"cx": Const(0), "cy": Const(0), "r": Slot("r")},
                                 inputs=[sub_out_ref("a", 0)]),
            ),
        )
        with self.assertRaises(ConceptCycleError):
            lib.add(top)

    def test_three_level_hierarchy(self):
        lib = self.build()
        top = Concept(
            name="plate",
            slots=("s",),
            subs=(SubInstance.make("b1", "bracket",
                                   {"bx": Const(0), "by": Const(0), "bw": Const(2),
                                    "bh": Const(2), "hr": Slot("s")}),),
            out_refs=(sub_out_ref("b1", 0),),
        )
        lib.add(top)
        self.assertEqual(lib.depth("plate"), 3)
        flat = lib.flatten("plate")
        self.assertEqual(flat.member_ids(), ("b1/p", "b1/k/a", "b1/k/b", "b1/g/c"))
        self.assertEqual(flat.validate(), [])

    def test_dedup_across_hierarchy_levels(self):
        """A hand-written flat copy of a hierarchical concept dedups against it."""
        lib = self.build()
        flat = lib.flatten("bracket")
        clone = Concept(name="bracket_flat", slots=flat.slots, members=flat.members,
                        constraints=flat.constraints, in_arity=flat.in_arity,
                        out_refs=flat.out_refs, defaults=flat.defaults)
        self.assertEqual(lib.add(clone), "bracket")


class TestUsage(unittest.TestCase):
    def test_usage_counts(self):
        lib = ConceptLibrary()
        lib.add(corner())
        lib.add(hole())
        lib.record_use("corner")
        lib.record_use("corner", 2)
        self.assertEqual(lib.usage(), {"corner": 3, "hole": 0})
        self.assertEqual(lib.unused(), ("hole",))
        self.assertEqual(lib.most_used(), (("corner", 3), ("hole", 0)))

    def test_usage_through_alias(self):
        lib = ConceptLibrary()
        lib.add(corner("c1"))
        lib.add(corner("c2"))
        lib.record_use("c2")
        self.assertEqual(lib.usage(), {"c1": 1})


if __name__ == "__main__":
    unittest.main()
