import unittest

from harnesscad.eval.bench.sketch.sketchconcept_metrics import (
    concept_cost,
    coverage,
    decomposition_cost,
    evaluate_corpus,
    library_compactness,
    library_cost,
    sketch_cost,
)
from harnesscad.domain.library.sketchconcept_induction import induce_library
from harnesscad.domain.library.sketchconcept_library import ConceptLibrary
from harnesscad.domain.reconstruction.sketch.sketchconcept_decompose import decompose
from harnesscad.domain.reconstruction.sketch.sketchconcept_template import (
    Concept,
    Const,
    Constraint,
    ConstraintSpec,
    Member,
    Primitive,
    Sketch,
    Slot,
    SubInstance,
    sub_out_ref,
)


def pair_concept(name="pair"):
    return Concept(
        name=name,
        slots=("x1", "y1", "x2", "y2", "r"),
        members=(
            Member.make("c1", "circle", {"x": Slot("x1"), "y": Slot("y1"), "r": Slot("r")}),
            Member.make("c2", "circle", {"x": Slot("x2"), "y": Slot("y2"), "r": Slot("r")}),
        ),
        constraints=(ConstraintSpec("equal", ("c1", "c2")),),
        out_refs=("c1", "c2"),
    )


def sk(prefix, r, dx):
    return Sketch(
        primitives=(
            Primitive.make(prefix + "1", "circle", {"x": dx, "y": 0, "r": r}),
            Primitive.make(prefix + "2", "circle", {"x": dx + 5, "y": 0, "r": r}),
        ),
        constraints=(Constraint("equal", (prefix + "1", prefix + "2")),),
    )


def corpus():
    return [sk("a", 2.0, 0.0), sk("b", 4.0, 10.0), sk("c", 1.0, 30.0)]


class TestCosts(unittest.TestCase):
    def test_sketch_cost(self):
        # 2 circles (1+3 each) + 1 binary constraint (1+2)
        self.assertEqual(sketch_cost(sk("a", 2.0, 0.0)), 11)

    def test_empty_sketch_cost(self):
        self.assertEqual(sketch_cost(Sketch(())), 0)

    def test_concept_cost(self):
        c = pair_concept()
        # header 1 + 5 slots + 0 inputs + 2 outs = 8; members 8; constraint 3
        self.assertEqual(concept_cost(c), 19)

    def test_library_cost(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        self.assertEqual(library_cost(lib), concept_cost(pair_concept()))

    def test_decomposition_cost(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        d = decompose(sk("a", 2.0, 0.0), lib)
        # one call: 1 + 5 bindings + 0 inputs
        self.assertEqual(decomposition_cost(d), 6)

    def test_residual_costed(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        s = Sketch(
            primitives=sk("a", 2.0, 0.0).primitives
            + (Primitive.make("z", "point", {"x": 1, "y": 1}),),
            constraints=sk("a", 2.0, 0.0).constraints + (Constraint("fixed", ("z",)),),
        )
        d = decompose(s, lib)
        # call (6) + residual point (1+2) + residual unary constraint (1+1)
        self.assertEqual(decomposition_cost(d), 6 + 3 + 2)


class TestLibraryCompactness(unittest.TestCase):
    def test_empty_library(self):
        st = library_compactness(ConceptLibrary())
        self.assertEqual(st.n_concepts, 0)
        self.assertEqual(st.max_depth, 0)
        self.assertEqual(st.cost, 0)

    def test_flat_library(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        st = library_compactness(lib)
        self.assertEqual(st.n_concepts, 1)
        self.assertEqual(st.total_members, 2)
        self.assertEqual(st.avg_members, 2.0)
        self.assertEqual(st.max_depth, 1)
        self.assertEqual(st.total_constraints, 1)
        self.assertEqual(st.hierarchy_saving, 0)

    def test_aliases_counted(self):
        lib = ConceptLibrary()
        lib.add(pair_concept("p1"))
        lib.add(pair_concept("p2"))
        st = library_compactness(lib)
        self.assertEqual(st.n_concepts, 1)
        self.assertEqual(st.n_aliases, 1)

    def test_hierarchy_saves_description_length(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        quad = Concept(
            name="quad",
            slots=("r", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"),
            subs=(
                SubInstance.make("u", "pair", {"r": Slot("r"), "x1": Slot("x1"),
                                               "y1": Slot("y1"), "x2": Slot("x2"),
                                               "y2": Slot("y2")}),
                SubInstance.make("v", "pair", {"r": Slot("r"), "x1": Slot("x3"),
                                               "y1": Slot("y3"), "x2": Slot("x4"),
                                               "y2": Slot("y4")}),
            ),
            out_refs=(sub_out_ref("u", 0), sub_out_ref("v", 0)),
        )
        lib.add(quad)
        st = library_compactness(lib)
        self.assertEqual(st.n_concepts, 2)
        self.assertEqual(st.max_depth, 2)
        self.assertEqual(st.total_members, 2 + 4)
        self.assertGreater(st.hierarchy_saving, 0)
        self.assertGreater(st.flat_cost, st.cost)


class TestCoverage(unittest.TestCase):
    def test_full_coverage_and_compression(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        d = decompose(sk("a", 2.0, 0.0), lib)
        st = coverage(d, lib)
        self.assertEqual(st.primitive_coverage, 1.0)
        self.assertEqual(st.constraint_coverage, 1.0)
        self.assertEqual(st.n_instances, 1)
        self.assertEqual((st.raw_cost, st.encoded_cost), (11, 6))
        self.assertAlmostEqual(st.compression_ratio, 11 / 6)
        self.assertTrue(st.lossless)

    def test_no_library_no_coverage(self):
        lib = ConceptLibrary()
        d = decompose(sk("a", 2.0, 0.0), lib)
        st = coverage(d, lib)
        self.assertEqual(st.primitive_coverage, 0.0)
        self.assertEqual(st.compression_ratio, 1.0)
        self.assertTrue(st.lossless)


class TestCorpusEvaluation(unittest.TestCase):
    def test_evaluate_with_handwritten_library(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        st = evaluate_corpus(corpus(), lib)
        self.assertEqual(st.n_sketches, 3)
        self.assertEqual(st.n_instances, 3)
        self.assertEqual(st.primitive_coverage, 1.0)
        self.assertEqual(st.constraint_coverage, 1.0)
        self.assertTrue(st.lossless)
        self.assertEqual(st.usage, (("pair", 3),))
        self.assertEqual(st.unused_concepts, ())
        self.assertEqual(st.reuse_rate, 1.0)
        # 3 * 11 raw vs library (19) + 3 calls (6 each)
        self.assertEqual(st.raw_cost, 33)
        self.assertEqual(st.encoded_cost, 19 + 18)
        self.assertAlmostEqual(st.compression_ratio, 33 / 37)

    def test_compression_beats_one_when_corpus_is_large(self):
        big = [sk("s%d" % i, 2.0 + i, 10.0 * i) for i in range(20)]
        lib = ConceptLibrary()
        lib.add(pair_concept())
        st = evaluate_corpus(big, lib)
        self.assertGreater(st.compression_ratio, 1.0)
        self.assertTrue(st.lossless)

    def test_induced_library_compresses(self):
        big = [sk("s%d" % i, 2.0 + i, 10.0 * i) for i in range(20)]
        lib = induce_library(big, max_concepts=2)
        st = evaluate_corpus(big, lib)
        self.assertEqual(st.primitive_coverage, 1.0)
        self.assertTrue(st.lossless)
        self.assertGreater(st.compression_ratio, 1.0)
        self.assertGreater(st.library.n_concepts, 0)

    def test_unused_concept_reported(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        lib.add(Concept(name="lonely", slots=("x", "y"),
                        members=(Member.make("p", "point",
                                             {"x": Slot("x"), "y": Slot("y")}),),
                        out_refs=("p",)))
        st = evaluate_corpus(corpus(), lib)
        self.assertEqual(st.unused_concepts, ("lonely",))
        self.assertEqual(st.reuse_rate, 0.5)

    def test_mismatched_decompositions_rejected(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        with self.assertRaises(ValueError):
            evaluate_corpus(corpus(), lib, decompositions=[decompose(corpus()[0], lib)])

    def test_deterministic(self):
        lib = ConceptLibrary()
        lib.add(pair_concept())
        self.assertEqual(evaluate_corpus(corpus(), lib), evaluate_corpus(corpus(), lib))


if __name__ == "__main__":
    unittest.main()
