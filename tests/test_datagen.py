"""Tests for the synthetic-data bootstrap generator (datagen/).

Exercises the whole data engine against the dependency-free StubBackend: seeded
determinism, the solver-in-the-loop keep-only-ok filter + reported yield, JSONL
round-trip, and the verifiers-as-cheap-labor decomposition.
"""

import os
import tempfile
import unittest

from backends.stub import StubBackend
from cisp.ops import NewSketch, AddRectangle, Constrain, Extrude
from datagen import (
    ParametricSampler, Sample,
    gen_plate, gen_bracket, gen_plate_with_holes, DEFAULT_GENERATORS,
    generate_dataset, generate_dataset_report,
    to_jsonl, read_jsonl, verifiers_as_labor,
)
from loop import HarnessSession


def _bad_generator(rng):
    """A candidate that over-constrains its sketch (dof 4 - 5 = -1) -> the
    plural verifier rejects it -> it must be filtered by the solver-in-the-loop."""
    ops = [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0),
        Constrain(kind="distance", a="e1", value=10.0),
        Constrain(kind="distance", a="e1", value=10.0),
        Constrain(kind="distance", a="e1", value=10.0),
        Constrain(kind="distance", a="e1", value=10.0),
        Constrain(kind="distance", a="e1", value=10.0),
        Extrude(sketch="sk1", distance=2.0),
    ]
    return "a deliberately over-constrained plate", ops, {"generator": "bad"}


class TestSamplerDeterminism(unittest.TestCase):
    def test_same_seed_same_draws(self):
        a = ParametricSampler(7)
        b = ParametricSampler(7)
        self.assertEqual([a.dim(1, 100) for _ in range(5)],
                         [b.dim(1, 100) for _ in range(5)])

    def test_generators_deterministic(self):
        for gen in DEFAULT_GENERATORS:
            _, ops1, p1 = gen(ParametricSampler(3))
            _, ops2, p2 = gen(ParametricSampler(3))
            self.assertEqual([o.to_dict() for o in ops1],
                             [o.to_dict() for o in ops2])
            self.assertEqual(p1, p2)


class TestDatasetDeterminism(unittest.TestCase):
    def test_same_seed_same_dataset(self):
        d1 = generate_dataset(12, seed=42, backend_factory=StubBackend)
        d2 = generate_dataset(12, seed=42, backend_factory=StubBackend)
        self.assertTrue(d1)  # non-empty
        self.assertEqual([s.to_dict() for s in d1], [s.to_dict() for s in d2])

    def test_different_seed_differs(self):
        d1 = generate_dataset(9, seed=1, backend_factory=StubBackend)
        d2 = generate_dataset(9, seed=2, backend_factory=StubBackend)
        self.assertNotEqual([s.brief for s in d1], [s.brief for s in d2])


class TestSolverInTheLoop(unittest.TestCase):
    def test_yield_reported_and_all_default_build(self):
        report = generate_dataset_report(15, seed=5, backend_factory=StubBackend)
        self.assertEqual(report.total, 15)
        self.assertEqual(report.kept, len(report.samples))
        self.assertGreaterEqual(report.yield_rate, 0.0)
        self.assertLessEqual(report.yield_rate, 1.0)
        # The default generators are constructed to build cleanly on the stub.
        self.assertEqual(report.kept, 15)
        self.assertEqual(report.yield_rate, 1.0)

    def test_filter_drops_non_building_samples(self):
        # Half the mix is the deliberately-broken generator -> yield must drop and
        # only the good ones survive.
        gens = [gen_plate, _bad_generator]
        report = generate_dataset_report(
            10, seed=0, backend_factory=StubBackend, generators=gens)
        self.assertEqual(report.total, 10)
        self.assertLess(report.kept, report.total)
        self.assertLess(report.yield_rate, 1.0)
        for s in report.samples:
            self.assertNotEqual(s.generator, "bad")

    def test_kept_samples_reapply_ok_on_fresh_session(self):
        samples = generate_dataset(12, seed=99, backend_factory=StubBackend)
        self.assertTrue(samples)
        for s in samples:
            session = HarnessSession(StubBackend())
            result = session.apply_ops(s.reference_ops())
            self.assertTrue(result.ok, f"{s.generator} failed to rebuild")
            # Deterministic replay: the digest tagged at generation matches.
            self.assertEqual(result.digest, s.digest)


class TestJsonlRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        samples = generate_dataset(8, seed=11, backend_factory=StubBackend)
        self.assertTrue(samples)
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            to_jsonl(path, samples)
            loaded = read_jsonl(path)
            self.assertEqual([s.to_dict() for s in samples],
                             [s.to_dict() for s in loaded])
        finally:
            os.remove(path)


class TestVerifiersAsLabor(unittest.TestCase):
    def _one(self, gen):
        brief, ops, params = gen(ParametricSampler(4))
        session = HarnessSession(StubBackend())
        result = session.apply_ops(ops)
        self.assertTrue(result.ok)
        return Sample(brief=brief, generator=params["generator"], params=params,
                      ops=[o.to_dict() for o in ops], digest=result.digest,
                      summary=session.summary())

    def test_binary_questions(self):
        for gen in (gen_plate, gen_bracket, gen_plate_with_holes):
            sample = self._one(gen)
            qs = verifiers_as_labor(sample)
            self.assertTrue(qs)  # at least one question
            for item in qs:
                self.assertIn("question", item)
                self.assertIn("answer", item)
                self.assertIsInstance(item["question"], str)
                self.assertIsInstance(item["answer"], bool)  # strictly binary

    def test_hole_questions_present_for_holes(self):
        sample = self._one(gen_plate_with_holes)
        text = " ".join(q["question"] for q in verifiers_as_labor(sample))
        self.assertIn("hole", text.lower())


if __name__ == "__main__":
    unittest.main()
