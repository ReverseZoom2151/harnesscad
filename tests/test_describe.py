"""Tests for design-to-text grounding (describe.py).

Runs on the dependency-free StubBackend with the heuristic (no-network) path, so
narration and Q&A are exercised deterministically and grounded in query('summary').
The Hole op is imported defensively. An offline fake LLM checks the injected-LLM
phrasing seam without any network.
"""

import unittest

from backends.stub import StubBackend
from state.opdag import OpDAG
from cisp.ops import NewSketch, AddRectangle, Extrude, Fillet

try:
    from cisp.ops import Hole
    HAVE_HOLE = True
except Exception:  # noqa: BLE001
    Hole = None
    HAVE_HOLE = False

from describe import describe_part, answer_query


def _plate(backend, n_holes=0, fillet=False):
    assert backend.apply(NewSketch(plane="XY")).ok
    assert backend.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=60.0, h=40.0)).ok
    assert backend.apply(Extrude(sketch="sk1", distance=8.0)).ok
    if HAVE_HOLE:
        for i in range(n_holes):
            assert backend.apply(Hole(face_or_sketch="", x=5.0 + i, y=5.0,
                                      diameter=5.0, through=True)).ok
    if fillet:
        assert backend.apply(Fillet(edges=(), radius=2.0)).ok
    return backend


def _dag(backend_ops_builder):
    """Build an OpDAG that mirrors what _plate applies, for the opdag path."""
    dag = OpDAG()
    for op in backend_ops_builder:
        dag.append(op)
    return dag


class _FakeLLM:
    """Offline LLM stub: echoes the rephrase prompt's ground-truth line back."""

    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, response_schema=None, **opts):
        self.calls += 1

        class _R:
            text = "Rephrased: " + messages[-1].content.splitlines()[-1]
        return _R()

    def stream(self, messages, tools=None, response_schema=None, **opts):
        yield ""


class TestDescribe(unittest.TestCase):
    def test_describe_returns_string_with_real_counts(self):
        b = _plate(StubBackend(), n_holes=2, fillet=True)
        text = describe_part(b)
        self.assertIsInstance(text, str)
        self.assertTrue(text)
        summary = b.query("summary")
        # narration exposes the real feature count from query('summary')
        self.assertIn(str(summary["feature_count"]), text)
        if HAVE_HOLE:
            self.assertIn("hole", text.lower())

    def test_describe_never_invents_bbox_without_metrics(self):
        # StubBackend has no metrics -> no fabricated dimensions/volume.
        b = _plate(StubBackend(), n_holes=1)
        text = describe_part(b)
        self.assertNotIn(" mm", text)  # no dimension leaked
        self.assertNotIn("cm3", text)  # no volume leaked

    def test_describe_deterministic(self):
        b1 = _plate(StubBackend(), n_holes=2, fillet=True)
        b2 = _plate(StubBackend(), n_holes=2, fillet=True)
        self.assertEqual(describe_part(b1), describe_part(b2))

    def test_describe_with_opdag_path(self):
        ops = [
            NewSketch(plane="XY"),
            AddRectangle(sketch="sk1", x=0.0, y=0.0, w=60.0, h=40.0),
            Extrude(sketch="sk1", distance=8.0),
        ]
        b = StubBackend()
        for op in ops:
            b.apply(op)
        text = describe_part(b, opdag=_dag(ops))
        self.assertIsInstance(text, str)
        self.assertIn("1 feature", text.lower())

    def test_injected_llm_is_used_offline(self):
        b = _plate(StubBackend(), n_holes=1)
        llm = _FakeLLM()
        text = describe_part(b, llm=llm)
        self.assertEqual(llm.calls, 1)
        self.assertTrue(text.startswith("Rephrased:"))


class TestAnswerQuery(unittest.TestCase):
    def test_how_many_features_is_correct(self):
        b = _plate(StubBackend(), n_holes=2, fillet=True)
        ans = answer_query("how many features?", b)
        n = b.query("summary")["feature_count"]
        self.assertIn(str(n), ans)

    @unittest.skipUnless(HAVE_HOLE, "Hole op not available")
    def test_how_many_holes_is_correct(self):
        b = _plate(StubBackend(), n_holes=3)
        ans = answer_query("How many holes are there?", b)
        self.assertIn("3", ans)

    def test_how_many_sketches(self):
        b = _plate(StubBackend())
        ans = answer_query("how many sketches?", b)
        self.assertIn("1", ans)

    def test_bounding_box_unavailable_on_stub(self):
        b = _plate(StubBackend())
        ans = answer_query("what is the bounding box?", b)
        self.assertIn("unavailable", ans.lower())

    def test_volume_unavailable_on_stub(self):
        b = _plate(StubBackend())
        ans = answer_query("what is the volume?", b)
        self.assertIn("unavailable", ans.lower())

    def test_unknown_question_gives_guidance(self):
        b = _plate(StubBackend())
        ans = answer_query("what colour is it?", b)
        self.assertIn("counts", ans.lower())

    def test_answer_query_deterministic(self):
        b = _plate(StubBackend(), n_holes=2)
        self.assertEqual(answer_query("how many features?", b),
                         answer_query("how many features?", b))


if __name__ == "__main__":
    unittest.main()
