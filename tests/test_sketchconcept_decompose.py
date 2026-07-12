import unittest

from library.sketchconcept_library import ConceptLibrary
from reconstruction.sketchconcept_decompose import (
    decompose,
    find_matches,
    is_exact,
    reconstruct,
)
from reconstruction.sketchconcept_template import (
    Concept,
    Const,
    Constraint,
    ConstraintSpec,
    Member,
    Primitive,
    Sketch,
    Slot,
    input_ref,
)


def hslot_pair():
    """Concept: two circles of equal radius, linked by an 'equal' constraint."""
    return Concept(
        name="pair",
        slots=("x1", "y1", "x2", "y2", "r"),
        members=(
            Member.make("c1", "circle", {"x": Slot("x1"), "y": Slot("y1"), "r": Slot("r")}),
            Member.make("c2", "circle", {"x": Slot("x2"), "y": Slot("y2"), "r": Slot("r")}),
        ),
        constraints=(ConstraintSpec("equal", ("c1", "c2")),),
        out_refs=("c1", "c2"),
    )


def unit_circle_concept():
    """Concept with a constant radius of 1."""
    return Concept(
        name="unit",
        slots=("x", "y"),
        members=(Member.make("c", "circle", {"x": Slot("x"), "y": Slot("y"), "r": Const(1.0)}),),
        out_refs=("c",),
    )


def pinned_circle():
    """Circle concentric with an external primitive (one input reference)."""
    return Concept(
        name="pinned",
        slots=("x", "y", "r"),
        members=(Member.make("c", "circle", {"x": Slot("x"), "y": Slot("y"), "r": Slot("r")}),),
        constraints=(ConstraintSpec("concentric", ("c", input_ref(0))),),
        in_arity=1,
        out_refs=("c",),
    )


def sketch_two_equal_circles():
    return Sketch(
        primitives=(
            Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 2}),
            Primitive.make("b", "circle", {"x": 5, "y": 0, "r": 2}),
        ),
        constraints=(Constraint("equal", ("a", "b")),),
    )


class TestFindMatches(unittest.TestCase):
    def test_basic_match_with_shared_slot(self):
        ms = find_matches(hslot_pair(), sketch_two_equal_circles())
        # two embeddings (c1->a,c2->b and the swap), both consistent
        self.assertEqual(len(ms), 2)
        m = ms[0]
        self.assertEqual(m.member_map, (("c1", "a"), ("c2", "b")))
        self.assertEqual(m.binding_map()["r"], 2.0)
        self.assertEqual(m.covered_constraints, (0,))

    def test_shared_slot_rejects_unequal_radii(self):
        sk = Sketch(
            primitives=(
                Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 2}),
                Primitive.make("b", "circle", {"x": 5, "y": 0, "r": 3}),
            ),
            constraints=(Constraint("equal", ("a", "b")),),
        )
        self.assertEqual(find_matches(hslot_pair(), sk), [])

    def test_missing_constraint_rejects(self):
        sk = Sketch(primitives=sketch_two_equal_circles().primitives, constraints=())
        self.assertEqual(find_matches(hslot_pair(), sk), [])

    def test_constant_parameter_filter(self):
        sk = Sketch(primitives=(
            Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 1}),
            Primitive.make("b", "circle", {"x": 3, "y": 0, "r": 4}),
        ))
        ms = find_matches(unit_circle_concept(), sk)
        self.assertEqual([m.member_map for m in ms], [(("c", "a"),)])

    def test_type_filter(self):
        sk = Sketch(primitives=(Primitive.make("p", "point", {"x": 0, "y": 0}),))
        self.assertEqual(find_matches(unit_circle_concept(), sk), [])

    def test_input_reference_not_owned(self):
        sk = Sketch(
            primitives=(
                Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 2}),
                Primitive.make("b", "point", {"x": 0, "y": 0}),
            ),
            constraints=(Constraint("concentric", ("a", "b")),),
        )
        ms = find_matches(pinned_circle(), sk)
        self.assertEqual(len(ms), 1)
        self.assertEqual(ms[0].member_map, (("c", "a"),))
        self.assertEqual(ms[0].inputs, ("b",))
        self.assertNotIn("b", ms[0].owned())

    def test_unordered_refs_by_default(self):
        sk = Sketch(
            primitives=sketch_two_equal_circles().primitives,
            constraints=(Constraint("equal", ("b", "a")),),
        )
        self.assertEqual(len(find_matches(hslot_pair(), sk)), 2)
        self.assertEqual(len(find_matches(hslot_pair(), sk, ordered_refs=True)), 1)

    def test_limit(self):
        ms = find_matches(hslot_pair(), sketch_two_equal_circles(), limit=1)
        self.assertEqual(len(ms), 1)

    def test_deterministic(self):
        a = find_matches(hslot_pair(), sketch_two_equal_circles())
        b = find_matches(hslot_pair(), sketch_two_equal_circles())
        self.assertEqual(a, b)

    def test_hierarchical_rejected(self):
        from reconstruction.sketchconcept_template import SubInstance
        with self.assertRaises(ValueError):
            find_matches(Concept(name="h", subs=(SubInstance.make("s", "pair"),)),
                         sketch_two_equal_circles())


class TestDecompose(unittest.TestCase):
    def library(self):
        lib = ConceptLibrary()
        lib.add(hslot_pair())
        lib.add(unit_circle_concept())
        lib.add(pinned_circle())
        return lib

    def test_full_cover(self):
        lib = self.library()
        d = decompose(sketch_two_equal_circles(), lib)
        self.assertEqual(len(d.placements), 1)
        self.assertEqual(d.placements[0].instance.concept, "pair")
        self.assertEqual(d.primitive_coverage(), 1.0)
        self.assertEqual(d.constraint_coverage(), 1.0)
        self.assertEqual(d.residual_primitives, ())
        self.assertEqual(d.concept_counts(), {"pair": 1})

    def test_residual(self):
        sk = Sketch(
            primitives=(
                Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 2}),
                Primitive.make("b", "circle", {"x": 5, "y": 0, "r": 2}),
                Primitive.make("z", "line", {"x1": 0, "y1": 0, "x2": 1, "y2": 1}),
            ),
            constraints=(Constraint("equal", ("a", "b")),
                         Constraint("horizontal", ("z",))),
        )
        d = decompose(sk, self.library())
        self.assertEqual(d.residual_primitives, ("z",))
        self.assertEqual(d.residual_constraints, (1,))
        self.assertAlmostEqual(d.primitive_coverage(), 2 / 3)
        self.assertAlmostEqual(d.constraint_coverage(), 0.5)

    def test_non_overlapping_instances(self):
        sk = Sketch(
            primitives=(
                Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 2}),
                Primitive.make("b", "circle", {"x": 5, "y": 0, "r": 2}),
                Primitive.make("c", "circle", {"x": 9, "y": 0, "r": 3}),
                Primitive.make("d", "circle", {"x": 12, "y": 0, "r": 3}),
            ),
            constraints=(Constraint("equal", ("a", "b")), Constraint("equal", ("c", "d"))),
        )
        d = decompose(sk, self.library())
        self.assertEqual(d.concept_counts(), {"pair": 2})
        covered = sorted(d.covered_primitives())
        self.assertEqual(covered, ["a", "b", "c", "d"])
        self.assertEqual(len(set(covered)), 4)

    def test_prefers_larger_concept(self):
        # 'unit' also matches unit circles, but the 2-member 'pair' wins on coverage
        sk = Sketch(
            primitives=(
                Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 1}),
                Primitive.make("b", "circle", {"x": 5, "y": 0, "r": 1}),
            ),
            constraints=(Constraint("equal", ("a", "b")),),
        )
        d = decompose(sk, self.library())
        self.assertEqual(d.concept_counts(), {"pair": 1})

    def test_empty_library(self):
        d = decompose(sketch_two_equal_circles(), ConceptLibrary())
        self.assertEqual(d.placements, ())
        self.assertEqual(d.primitive_coverage(), 0.0)

    def test_deterministic(self):
        lib = self.library()
        sk = sketch_two_equal_circles()
        self.assertEqual(decompose(sk, lib), decompose(sk, lib))


class TestReconstruct(unittest.TestCase):
    def test_roundtrip_exact(self):
        lib = ConceptLibrary()
        lib.add(hslot_pair())
        sk = sketch_two_equal_circles()
        d = decompose(sk, lib)
        rebuilt = reconstruct(d, lib)
        self.assertEqual([p.pid for p in rebuilt.primitives], ["a", "b"])
        self.assertEqual(rebuilt.primitives[1].param_map(), {"x": 5.0, "y": 0.0, "r": 2.0})
        self.assertEqual(rebuilt.validate(), [])
        self.assertTrue(is_exact(d, lib))

    def test_roundtrip_with_residual(self):
        lib = ConceptLibrary()
        lib.add(hslot_pair())
        sk = Sketch(
            primitives=(
                Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 2}),
                Primitive.make("b", "circle", {"x": 5, "y": 0, "r": 2}),
                Primitive.make("z", "point", {"x": 7, "y": 7}),
            ),
            constraints=(Constraint("equal", ("a", "b")), Constraint("fixed", ("z",))),
        )
        d = decompose(sk, lib)
        self.assertTrue(is_exact(d, lib))
        self.assertEqual([p.pid for p in reconstruct(d, lib).primitives], ["a", "b", "z"])

    def test_inputs_preserved_in_reconstruction(self):
        lib = ConceptLibrary()
        lib.add(pinned_circle())
        sk = Sketch(
            primitives=(
                Primitive.make("a", "circle", {"x": 0, "y": 0, "r": 2}),
                Primitive.make("b", "point", {"x": 0, "y": 0}),
            ),
            constraints=(Constraint("concentric", ("a", "b")),),
        )
        d = decompose(sk, lib)
        self.assertEqual(d.placements[0].instance.inputs, ("b",))
        self.assertTrue(is_exact(d, lib))

    def test_not_exact_when_uncovered_primitive_dropped(self):
        lib = ConceptLibrary()
        lib.add(hslot_pair())
        sk = sketch_two_equal_circles()
        d = decompose(sk, lib)
        broken = type(d)(sk, d.placements, ("ghost",), d.residual_constraints)
        with self.assertRaises(KeyError):
            reconstruct(broken, lib)


if __name__ == "__main__":
    unittest.main()
