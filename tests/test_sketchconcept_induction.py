import unittest

from library.sketchconcept_induction import (
    ConceptCandidate,
    abstract_region,
    adjacency,
    build_library,
    compression_gain,
    connected_subsets,
    induce_concepts,
    induce_library,
    instance_cost,
    region_cost,
)
from reconstruction.sketchconcept_decompose import decompose, is_exact
from reconstruction.sketchconcept_template import (
    Constraint,
    Primitive,
    Sketch,
    Slot,
    canonical_signature,
)


def equal_circles(prefix, r=2.0, dx=0.0):
    """Two circles of equal radius joined by an 'equal' constraint."""
    return (
        Primitive.make(prefix + "1", "circle", {"x": dx, "y": 0, "r": r}),
        Primitive.make(prefix + "2", "circle", {"x": dx + 5, "y": 0, "r": r}),
    ), (Constraint("equal", (prefix + "1", prefix + "2")),)


def sketch_equal_circles(prefix="a", r=2.0, dx=0.0):
    prims, cons = equal_circles(prefix, r, dx)
    return Sketch(prims, cons)


def corpus():
    # three sketches, each containing the same 'two equal circles' structure at
    # different positions and radii, plus a stray line in the last one.
    s3_prims, s3_cons = equal_circles("c", r=1.5, dx=20)
    s3 = Sketch(s3_prims + (Primitive.make("z", "line", {"x1": 0, "y1": 0, "x2": 9, "y2": 9}),),
                s3_cons)
    return [sketch_equal_circles("a", 2.0, 0.0),
            sketch_equal_circles("b", 4.0, 10.0),
            s3]


class TestEnumeration(unittest.TestCase):
    def test_adjacency(self):
        adj = adjacency(sketch_equal_circles())
        self.assertEqual(adj, {"a1": {"a2"}, "a2": {"a1"}})

    def test_connected_subsets(self):
        subs = connected_subsets(sketch_equal_circles(), min_size=2, max_size=3)
        self.assertEqual(subs, [("a1", "a2")])

    def test_isolated_primitive_not_grouped(self):
        sk = corpus()[2]
        subs = connected_subsets(sk, min_size=2, max_size=3)
        self.assertEqual(subs, [("c1", "c2")])

    def test_size_range(self):
        sk = Sketch(
            primitives=(
                Primitive.make("p", "point", {"x": 0, "y": 0}),
                Primitive.make("q", "point", {"x": 1, "y": 0}),
                Primitive.make("r", "point", {"x": 2, "y": 0}),
            ),
            constraints=(Constraint("coincident", ("p", "q")),
                         Constraint("coincident", ("q", "r"))),
        )
        s2 = connected_subsets(sk, min_size=2, max_size=2)
        self.assertEqual(s2, [("p", "q"), ("q", "r")])
        s3 = connected_subsets(sk, min_size=3, max_size=3)
        self.assertEqual(s3, [("p", "q", "r")])

    def test_bad_range(self):
        with self.assertRaises(ValueError):
            connected_subsets(sketch_equal_circles(), min_size=3, max_size=2)


class TestAbstraction(unittest.TestCase):
    def test_all_params_free(self):
        sk = sketch_equal_circles()
        c = abstract_region(sk, ("a1", "a2"), share_equal_params=False)
        self.assertEqual(len(c.slots), 6)
        self.assertEqual(c.member_ids(), ("m0", "m1"))
        self.assertEqual(c.out_refs, ("m0", "m1"))
        self.assertEqual(c.validate(), [])

    def test_equal_params_shared(self):
        sk = sketch_equal_circles()
        c = abstract_region(sk, ("a1", "a2"), share_equal_params=True)
        # both radii (2.0) share one slot; y=0 of both circles shares one slot
        self.assertEqual(c.member("m0").param_map()["r"], c.member("m1").param_map()["r"])
        self.assertEqual(c.member("m0").param_map()["y"], c.member("m1").param_map()["y"])
        # x and y of the same circle do NOT collapse even though both are 0
        self.assertNotEqual(c.member("m0").param_map()["x"], c.member("m0").param_map()["y"])

    def test_induced_constraints_kept(self):
        c = abstract_region(sketch_equal_circles(), ("a1", "a2"))
        self.assertEqual([(k.ctype, k.refs) for k in c.constraints], [("equal", ("m0", "m1"))])

    def test_signature_matches_across_sketches(self):
        c1 = abstract_region(sketch_equal_circles("a", 2.0, 0.0), ("a1", "a2"))
        c2 = abstract_region(sketch_equal_circles("b", 4.0, 9.0), ("b1", "b2"))
        self.assertEqual(canonical_signature(c1), canonical_signature(c2))


class TestScoring(unittest.TestCase):
    def test_costs(self):
        c = abstract_region(sketch_equal_circles(), ("a1", "a2"))
        # 2 circles: (1 + 3 params) each = 8; 1 binary constraint: 1 + 2 = 3
        self.assertEqual(region_cost(c), 11)
        self.assertEqual(instance_cost(c), 1 + len(c.slots))
        self.assertGreater(compression_gain(c, 10), compression_gain(c, 2))

    def test_gain_negative_when_rare(self):
        c = abstract_region(sketch_equal_circles(), ("a1", "a2"))
        self.assertLess(compression_gain(c, 1), compression_gain(c, 5))


class TestInduce(unittest.TestCase):
    def test_mines_the_recurring_structure(self):
        cands = induce_concepts(corpus(), min_size=2, max_size=3, min_occurrences=2)
        self.assertTrue(cands)
        top = cands[0]
        self.assertEqual(top.occurrences, 3)
        self.assertEqual(top.sketches, 3)
        self.assertEqual(top.size(), 2)
        self.assertGreater(top.gain, 0)
        self.assertIsInstance(top, ConceptCandidate)

    def test_min_occurrences_filters(self):
        self.assertEqual(induce_concepts(corpus(), min_occurrences=4), [])

    def test_deterministic(self):
        self.assertEqual(induce_concepts(corpus()), induce_concepts(corpus()))

    def test_invalid_sketch(self):
        bad = Sketch((Primitive.make("p", "point", {"x": 0, "y": 0}),),
                     (Constraint("coincident", ("p", "q")),))
        with self.assertRaises(ValueError):
            induce_concepts([bad])


class TestBuildLibrary(unittest.TestCase):
    def test_library_is_deduplicated_and_named(self):
        lib = induce_library(corpus(), max_concepts=3)
        self.assertGreaterEqual(len(lib), 1)
        self.assertEqual(lib.names()[0], "c0")
        # every stored concept has a distinct signature
        sigs = {lib.signature(n) for n in lib.names()}
        self.assertEqual(len(sigs), len(lib))

    def test_max_concepts_respected(self):
        lib = induce_library(corpus(), max_concepts=1)
        self.assertEqual(len(lib), 1)

    def test_min_gain_stops_admission(self):
        cands = induce_concepts(corpus())
        lib = build_library(cands, min_gain=10 ** 9)
        self.assertEqual(len(lib), 0)

    def test_induced_library_explains_the_corpus(self):
        lib = induce_library(corpus(), max_concepts=2)
        d = decompose(corpus()[0], lib)
        self.assertEqual(d.primitive_coverage(), 1.0)
        self.assertEqual(d.constraint_coverage(), 1.0)
        self.assertTrue(is_exact(d, lib))

    def test_induced_concept_generalises_to_other_radii(self):
        # mined from the corpus, the concept must also explain sketch 2 (r = 4)
        lib = induce_library(corpus(), max_concepts=2)
        d = decompose(corpus()[1], lib)
        self.assertEqual(d.primitive_coverage(), 1.0)
        self.assertTrue(is_exact(d, lib))

    def test_residual_stays_uncovered(self):
        lib = induce_library(corpus(), max_concepts=2)
        d = decompose(corpus()[2], lib)
        self.assertEqual(d.residual_primitives, ("z",))
        self.assertTrue(is_exact(d, lib))


if __name__ == "__main__":
    unittest.main()
