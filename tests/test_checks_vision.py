"""Tests for checks_vision.py — the advisory VLM-as-judge verifier.

Covered: score parsing (plain + G-Eval + prose-wrapped + garbage),
swap-augmentation averaging, A-vs-B comparison with position-bias defence,
headless INFO-skip (stub backend), and the advisory-only guarantee (never
ERROR). A MockVisionLLM returns canned JSON; the real render path is exercised
only when cadquery is installed.
"""

import unittest

from harnesscad.eval.verifiers.vlm_judge import (
    VLMJudgeCheck, GEvalScore, JudgeVerdict,
    parse_judge_json, build_judge_messages, DEFAULT_RUBRIC,
)
from harnesscad.agents.llm.base import CompletionResult
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()


class MockVisionLLM:
    """LLM-protocol stub: returns queued canned JSON strings, records calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def _next(self) -> CompletionResult:
        item = self._responses.pop(0) if self._responses else '{"score": 0.0}'
        return item if isinstance(item, CompletionResult) else CompletionResult(text=str(item))

    def complete(self, messages, tools=None, response_schema=None, **opts) -> CompletionResult:
        self.calls.append({"messages": list(messages), "opts": opts})
        return self._next()

    def stream(self, messages, tools=None, response_schema=None, **opts):
        yield self._next().text


def _stub_with_solid() -> StubBackend:
    b = StubBackend()
    b.apply(NewSketch(plane="XY"))
    b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
    b.apply(Extrude(sketch="sk1", distance=5.0))
    return b


def _cq_plate():
    from harnesscad.io.backends.cadquery import CadQueryBackend
    b = CadQueryBackend()
    b.apply(NewSketch(plane="XY"))
    b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
    b.apply(Extrude(sketch="sk1", distance=5.0))
    return b


class TestScoreParsing(unittest.TestCase):
    def test_plain_score(self):
        v = parse_judge_json('{"score": 0.8, "rationale": "clean"}')
        self.assertAlmostEqual(v.score, 0.8)
        self.assertEqual(v.rationale, "clean")

    def test_score_clamped_and_normalised(self):
        self.assertAlmostEqual(parse_judge_json('{"score": 8}').score, 0.8)
        self.assertAlmostEqual(parse_judge_json('{"score": 80}').score, 0.8)
        self.assertEqual(parse_judge_json('{"score": -5}').score, 0.0)

    def test_prose_wrapped_json(self):
        v = parse_judge_json('Sure! Here: {"score": 0.5} hope that helps')
        self.assertAlmostEqual(v.score, 0.5)

    def test_garbage_is_zero_not_crash(self):
        self.assertEqual(parse_judge_json("not json at all").score, 0.0)
        self.assertEqual(parse_judge_json("").score, 0.0)

    def test_geval_distribution(self):
        # buckets 1..5, mass on 4 and 5 -> normalised ~0.85
        v = parse_judge_json('{"scores": {"1":0, "2":0, "3":0, "4":0.4, "5":0.6}}')
        self.assertAlmostEqual(v.score, ((4 * 0.4 + 5 * 0.6) - 1) / 4.0, places=6)


class TestGEvalScore(unittest.TestCase):
    def test_probability_weighted_expectation(self):
        g = GEvalScore.from_distribution({1: 0.5, 5: 0.5}, scale=(1.0, 5.0))
        self.assertAlmostEqual(g.value, 0.5)  # mean 3.0 on 1..5 -> 0.5

    def test_empty_distribution_is_zero(self):
        self.assertEqual(GEvalScore({}).value, 0.0)


class TestHeadlessSkip(unittest.TestCase):
    def test_stub_backend_info_skips(self):
        llm = MockVisionLLM(['{"score": 0.9}'])
        report = VLMJudgeCheck(llm, brief="a plate").check(_stub_with_solid())
        self.assertTrue(report.ok)  # advisory: no ERROR
        self.assertEqual(len(report.diagnostics), 1)
        d = report.diagnostics[0]
        self.assertIs(d.severity, Severity.INFO)
        self.assertEqual(d.code, "vlm-judge-skip")
        # The judge LLM must not even be called when nothing renders.
        self.assertEqual(llm.calls, [])


class TestSwapAugmentation(unittest.TestCase):
    def test_two_passes_averaged(self):
        # Judge is stubbed at the render level: feed a fake RenderResult.
        from harnesscad.io.surfaces.render import RenderResult
        result = RenderResult(images={"iso": b"<svg/>", "front": b"<svg/>"},
                             fmt="svg", note=None)
        llm = MockVisionLLM(['{"score": 0.2}', '{"score": 0.8}'])
        check = VLMJudgeCheck(llm, brief="x", swap_augment=True)
        verdict = check._judge(result)
        self.assertAlmostEqual(verdict.score, 0.5)  # (0.2 + 0.8)/2
        self.assertEqual(len(llm.calls), 2)  # both orderings judged

    def test_swap_disabled_single_pass(self):
        from harnesscad.io.surfaces.render import RenderResult
        result = RenderResult(images={"iso": b"<svg/>", "front": b"<svg/>"},
                             fmt="svg", note=None)
        llm = MockVisionLLM(['{"score": 0.2}', '{"score": 0.8}'])
        check = VLMJudgeCheck(llm, brief="x", swap_augment=False)
        verdict = check._judge(result)
        self.assertAlmostEqual(verdict.score, 0.2)
        self.assertEqual(len(llm.calls), 1)


class TestComparePositionBias(unittest.TestCase):
    @unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
    def test_compare_averages_both_orderings(self):
        # A always scores 0.9, B always 0.3, regardless of ordering.
        responses = []
        for _ in range(8):
            responses.append('{"score": 0.9}')  # A
            responses.append('{"score": 0.3}')  # B
        llm = MockVisionLLM(responses * 2)
        check = VLMJudgeCheck(llm, brief="a plate", swap_augment=False)
        out = check.compare(_cq_plate(), _cq_plate())
        self.assertFalse(out["skipped"])
        self.assertIn(out["winner"], ("a", "b", "tie"))

    def test_compare_skips_when_headless(self):
        llm = MockVisionLLM(['{"score": 0.9}'])
        out = VLMJudgeCheck(llm, brief="x").compare(_stub_with_solid(),
                                                    _stub_with_solid())
        self.assertTrue(out["skipped"])
        self.assertIsNone(out["winner"])


class TestPromptSafety(unittest.TestCase):
    def test_prompt_contains_brief_and_rubric_not_answer(self):
        msgs = build_judge_messages(
            brief="a 20x10x5 plate", rubric=DEFAULT_RUBRIC,
            view_order=["iso", "front"],
            data_uris={"iso": "data:image/svg+xml;base64,AAAA",
                       "front": "data:image/svg+xml;base64,BBBB"})
        joined = " ".join(m.content for m in msgs)
        self.assertIn("a 20x10x5 plate", joined)
        self.assertIn("STRICT JSON", joined)
        self.assertIn("data:image/svg+xml", joined)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestRealRenderJudge(unittest.TestCase):
    def test_advisory_score_from_real_render(self):
        llm = MockVisionLLM(['{"score": 0.9, "rationale": "reads as a plate"}',
                             '{"score": 0.7}'])
        report = VLMJudgeCheck(llm, brief="a rectangular plate").check(_cq_plate())
        self.assertTrue(report.ok)  # never ERROR
        d = report.diagnostics[0]
        self.assertEqual(d.code, "vlm-judge")
        self.assertIs(d.severity, Severity.INFO)  # (0.9+0.7)/2 = 0.8 >= 0.5
        self.assertIn("0.8", d.message)

    def test_low_score_is_warning_not_error(self):
        llm = MockVisionLLM(['{"score": 0.1}', '{"score": 0.1}'])
        report = VLMJudgeCheck(llm, brief="a sphere").check(_cq_plate())
        self.assertTrue(report.ok)  # advisory-only
        self.assertIs(report.diagnostics[0].severity, Severity.WARNING)


if __name__ == "__main__":
    unittest.main()
