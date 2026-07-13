import unittest

from harnesscad.domain.reconstruction.sequences.histcad_sequence import (
    Line, Circle, SketchPlane, Sketch, Constraint, Extrusion, Feature,
    ModelingSequence,
)
from harnesscad.eval.bench.data.histcad_history_quality import (
    sequence_length_stats, constraint_overhead, constraint_distribution,
    total_variation, flattening_ratio, hierarchy_free, REFERENCE_DISTRIBUTION,
)


def _square(dx=0.0, dy=0.0, s=1.0):
    return [
        Line(dx, dy, dx + s, dy), Line(dx + s, dy, dx + s, dy + s),
        Line(dx + s, dy + s, dx, dy + s), Line(dx, dy + s, dx, dy),
    ]


def _seq(constraints=()):
    sk = Sketch(SketchPlane(), tuple(_square()), tuple(constraints))
    return ModelingSequence((Feature(sk, Extrusion(0, 0, 1, 5.0), "create"),))


class TestStats(unittest.TestCase):
    def test_length_stats(self):
        seqs = [_seq(), _seq(), _seq()]
        st = sequence_length_stats(seqs)
        self.assertEqual(st["count"], 3)
        self.assertGreater(st["mean"], 0)
        self.assertEqual(st["mean"], st["median"])  # identical seqs

    def test_empty(self):
        self.assertEqual(sequence_length_stats([])["count"], 0)

    def test_percentile(self):
        # build sequences of varying constraint counts to vary length
        seqs = [_seq([Constraint("horizontal", (0,))] * n) for n in range(1, 11)]
        st = sequence_length_stats(seqs)
        self.assertGreaterEqual(st["p95"], st["median"])


class TestOverhead(unittest.TestCase):
    def test_positive_overhead(self):
        seqs = [_seq([Constraint("horizontal", (0,)), Constraint("vertical", (1,))])]
        self.assertGreater(constraint_overhead(seqs), 0.0)

    def test_no_constraints_zero(self):
        self.assertEqual(constraint_overhead([_seq()]), 0.0)


class TestDistribution(unittest.TestCase):
    def test_distribution_sums_to_one(self):
        seqs = [_seq([Constraint("horizontal", (0,)), Constraint("parallel", (0, 1))])]
        d = constraint_distribution(seqs)
        self.assertAlmostEqual(sum(d.values()), 1.0)
        self.assertAlmostEqual(d["horizontal"], 0.5)

    def test_tv_zero_for_reference(self):
        self.assertAlmostEqual(total_variation(REFERENCE_DISTRIBUTION), 0.0, places=6)

    def test_tv_bounded(self):
        d = {"coincident": 1.0}
        tv = total_variation(d)
        self.assertGreaterEqual(tv, 0.0)
        self.assertLessEqual(tv, 1.0)


class TestFlattening(unittest.TestCase):
    def test_flattening_removes_shared(self):
        left = [Line(0, 0, 1, 0), Line(1, 0, 1, 1), Line(1, 1, 0, 1), Line(0, 1, 0, 0)]
        right = [Line(1, 0, 2, 0), Line(2, 0, 2, 1), Line(2, 1, 1, 1), Line(1, 1, 1, 0)]
        ratio = flattening_ratio([[left], [right]])
        self.assertGreater(ratio, 0.0)  # shared edge removed

    def test_no_shared_zero(self):
        self.assertEqual(flattening_ratio([[_square()]]), 0.0)


class TestHierarchyFree(unittest.TestCase):
    def test_flat_is_one(self):
        self.assertEqual(hierarchy_free([_seq()]), 1.0)

    def test_nested_lowers(self):
        outer = _square(0, 0, 10)
        inner = _square(3, 3, 2)
        sk = Sketch(SketchPlane(), tuple(outer + inner))
        seq = ModelingSequence((Feature(sk, Extrusion(0, 0, 1, 1.0), "create"),))
        self.assertEqual(hierarchy_free([seq]), 0.0)

    def test_empty_is_one(self):
        self.assertEqual(hierarchy_free([]), 1.0)


if __name__ == "__main__":
    unittest.main()
