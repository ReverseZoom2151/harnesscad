import unittest

from harnesscad.domain.reconstruction.sketch.concept_template import (
    Concept,
    ConceptInstance,
    Const,
    Constraint,
    ConstraintSpec,
    Member,
    Primitive,
    Sketch,
    Slot,
    SubInstance,
    canonical_signature,
    input_ref,
    instantiate,
    parse_ref,
    realise,
    resolve_param,
    sub_out_ref,
)


def rect_concept(name="rect"):
    """A 2-line L-shape concept with a perpendicular constraint and one input."""
    return Concept(
        name=name,
        slots=("x0", "y0", "w", "h"),
        members=(
            Member.make("a", "line", {"x1": Slot("x0"), "y1": Slot("y0"),
                                      "x2": Slot("w"), "y2": Slot("y0")}),
            Member.make("b", "line", {"x1": Slot("w"), "y1": Slot("y0"),
                                      "x2": Slot("w"), "y2": Slot("h")}),
        ),
        constraints=(
            ConstraintSpec("coincident", ("a", "b")),
            ConstraintSpec("perpendicular", ("a", "b")),
            ConstraintSpec("tangent", ("a", input_ref(0))),
        ),
        in_arity=1,
        out_refs=("b",),
        defaults=(("y0", 0.0),),
    )


class TestSketchData(unittest.TestCase):
    def test_primitive_params_sorted_and_typed(self):
        p = Primitive.make("p1", "circle", {"r": 2, "x": 1, "y": 0})
        self.assertEqual(p.params, (("r", 2.0), ("x", 1.0), ("y", 0.0)))
        self.assertEqual(p.param_map()["r"], 2.0)

    def test_unknown_type_rejected(self):
        with self.assertRaises(ValueError):
            Primitive.make("p", "spline", {})

    def test_sketch_validate(self):
        s = Sketch((Primitive.make("p", "point", {"x": 0, "y": 0}),),
                   (Constraint("coincident", ("p", "q")),))
        self.assertTrue(any("unknown primitive q" in e for e in s.validate()))
        good = Sketch((Primitive.make("p", "point", {"x": 0, "y": 0}),))
        self.assertEqual(good.validate(), [])


class TestRefs(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_ref("in:2"), ("input", 2))
        self.assertEqual(parse_ref(sub_out_ref("s", 1)), ("sub", ("s", 1)))
        self.assertEqual(parse_ref("m1"), ("member", "m1"))

    def test_resolve_param(self):
        self.assertEqual(resolve_param(Const(3), {}), 3.0)
        self.assertEqual(resolve_param(Slot("a"), {"a": 5}), 5.0)
        with self.assertRaises(KeyError):
            resolve_param(Slot("a"), {})


class TestValidation(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(rect_concept().validate(), [])

    def test_unknown_slot(self):
        c = Concept(name="c", slots=(),
                    members=(Member.make("a", "point", {"x": Slot("q"), "y": Const(0)}),))
        self.assertTrue(any("unknown slot q" in e for e in c.validate()))

    def test_missing_param(self):
        c = Concept(name="c", members=(Member("a", "point", (("x", Const(0)),)),))
        self.assertTrue(any("expected params" in e for e in c.validate()))

    def test_bad_input_index(self):
        c = Concept(name="c", in_arity=1,
                    members=(Member.make("a", "point", {"x": Const(0), "y": Const(0)}),),
                    constraints=(ConstraintSpec("coincident", ("a", input_ref(3))),))
        self.assertTrue(any("out of range" in e for e in c.validate()))

    def test_free_slots(self):
        self.assertEqual(rect_concept().free_slots(), ("x0", "w", "h"))


class TestInstantiate(unittest.TestCase):
    def test_bind_and_prefix(self):
        inst = instantiate(rect_concept(), {"x0": 0, "w": 4, "h": 3},
                           inputs=["ext"], prefix="i0")
        self.assertEqual([p.pid for p in inst.primitives], ["i0/a", "i0/b"])
        self.assertEqual(inst.primitives[0].param_map(),
                         {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 0.0})
        self.assertEqual(inst.outputs, ("i0/b",))
        # the external input reference is left as the caller's id
        self.assertIn(Constraint("tangent", ("i0/a", "ext")), inst.constraints)
        self.assertEqual(inst.as_sketch().validate(),
                         ["constraint 2 references unknown primitive ext"])

    def test_default_used(self):
        inst = instantiate(rect_concept(), {"x0": 1, "w": 2, "h": 5}, inputs=["e"])
        self.assertEqual(inst.primitives[0].param_map()["y1"], 0.0)

    def test_missing_binding(self):
        with self.assertRaises(KeyError):
            instantiate(rect_concept(), {"x0": 0}, inputs=["e"])

    def test_unknown_binding(self):
        with self.assertRaises(KeyError):
            instantiate(rect_concept(), {"x0": 0, "w": 1, "h": 1, "zz": 9}, inputs=["e"])

    def test_wrong_input_arity(self):
        with self.assertRaises(ValueError):
            instantiate(rect_concept(), {"x0": 0, "w": 1, "h": 1}, inputs=[])

    def test_hierarchical_rejected(self):
        c = Concept(name="h", subs=(SubInstance.make("s", "rect"),))
        with self.assertRaises(ValueError):
            instantiate(c)

    def test_realise_roundtrip(self):
        ci = ConceptInstance.make("rect", "u1", {"x0": 0, "w": 1, "h": 2}, ["ext"])
        inst = realise(rect_concept(), ci)
        self.assertEqual(inst.outputs, ("u1/b",))
        with self.assertRaises(ValueError):
            realise(rect_concept("other"), ci)

    def test_deterministic(self):
        a = instantiate(rect_concept(), {"x0": 0, "w": 1, "h": 2}, inputs=["e"], prefix="p")
        b = instantiate(rect_concept(), {"x0": 0, "w": 1, "h": 2}, inputs=["e"], prefix="p")
        self.assertEqual(a, b)


class TestCanonicalSignature(unittest.TestCase):
    def test_invariant_to_member_naming_and_order(self):
        c1 = rect_concept()
        c2 = Concept(
            name="other",
            slots=("p", "q", "r", "s"),
            members=(
                Member.make("zz", "line", {"x1": Slot("q"), "y1": Slot("p"),
                                           "x2": Slot("q"), "y2": Slot("r")}),
                Member.make("aa", "line", {"x1": Slot("s"), "y1": Slot("p"),
                                           "x2": Slot("q"), "y2": Slot("p")}),
            ),
            constraints=(
                ConstraintSpec("perpendicular", ("aa", "zz")),
                ConstraintSpec("tangent", ("aa", input_ref(0))),
                ConstraintSpec("coincident", ("aa", "zz")),
            ),
            in_arity=1,
            out_refs=("zz",),
        )
        self.assertEqual(canonical_signature(c1), canonical_signature(c2))

    def test_distinguishes_structure(self):
        c1 = rect_concept()
        c2 = Concept(
            name="rect2",
            slots=("x0", "y0", "w", "h"),
            members=c1.members,
            constraints=(ConstraintSpec("coincident", ("a", "b")),),
            in_arity=1,
            out_refs=("b",),
        )
        self.assertNotEqual(canonical_signature(c1), canonical_signature(c2))

    def test_constants_significant(self):
        c1 = Concept(name="a", slots=("x",),
                     members=(Member.make("m", "point", {"x": Slot("x"), "y": Const(0)}),),
                     out_refs=("m",))
        c2 = Concept(name="b", slots=("x",),
                     members=(Member.make("m", "point", {"x": Slot("x"), "y": Const(1)}),),
                     out_refs=("m",))
        self.assertNotEqual(canonical_signature(c1), canonical_signature(c2))

    def test_shared_slot_pattern_significant(self):
        shared = Concept(name="s", slots=("t",),
                         members=(Member.make("m", "point", {"x": Slot("t"), "y": Slot("t")}),))
        distinct = Concept(name="d", slots=("t", "u"),
                           members=(Member.make("m", "point", {"x": Slot("t"), "y": Slot("u")}),))
        self.assertNotEqual(canonical_signature(shared), canonical_signature(distinct))

    def test_flat_required(self):
        with self.assertRaises(ValueError):
            canonical_signature(Concept(name="h", subs=(SubInstance.make("s", "x"),)))


if __name__ == "__main__":
    unittest.main()
