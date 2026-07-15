"""The grounding-LoRA data contract and its 16GB verdict are pure python and must
be tested with no torch and no VLM: a coordinate parser that silently returns (0,0)
on a non-answer, or an answer format that does not round-trip, would launder misses
into hits exactly as a mis-masked text example would. The predictor and trainer
touch a GPU and SKIP cleanly when the training stack is absent."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import train
from harnesscad.agents.selftrain.train import grounding as GR
from harnesscad.eval.grounding import cadspot as C


def _viewport_target(x: int, y: int, r: int = 12) -> C.Target:
    return C.Target(region="viewport", instruction="the bore's front wall",
                    image="images/s1_iso.png", width=1920, height=1080,
                    bbox=(x - r, y - r, x + r, y + r), entity="Face6",
                    sample="s1", view="isometric")


class TestAnswerRoundTrip(unittest.TestCase):

    def test_format_then_parse_is_identity(self):
        for x, y in [(0, 0), (960, 540), (1919, 1079)]:
            xy = GR.parse_answer(GR.format_answer(x, y))
            self.assertIsNotNone(xy)
            self.assertEqual((int(xy[0]), int(xy[1])), (x, y))

    def test_parse_tolerates_bare_pair_and_equals(self):
        self.assertEqual(GR.parse_answer("(300, 210)"), (300.0, 210.0))
        self.assertEqual(GR.parse_answer("x=42, y=7"), (42.0, 7.0))

    def test_parse_returns_none_on_nonanswer(self):
        # The load-bearing case: a non-answer must be None, NEVER a silent (0, 0)
        # that would score as a click in the top-left corner.
        self.assertIsNone(GR.parse_answer("I cannot see the element."))
        self.assertIsNone(GR.parse_answer(""))
        self.assertIsNone(GR.parse_answer(None))


class TestBuildExamples(unittest.TestCase):

    def test_example_shape_and_supervised_pixel(self):
        tgt = _viewport_target(400, 300)
        ex = GR.build_examples([tgt])
        self.assertEqual(len(ex), 1)
        e = ex[0]
        # Supervised answer is the target centre in absolute pixels.
        self.assertEqual((e["x"], e["y"]), (400, 300))
        roles = [m["role"] for m in e["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])
        # The assistant turn must be the parseable coordinate, nothing else.
        self.assertEqual(GR.parse_answer(e["messages"][-1]["content"]), (400.0, 300.0))
        # The user turn carries the image and the instruction.
        content = e["messages"][1]["content"]
        self.assertTrue(any(c.get("type") == "image" for c in content))

    def test_root_is_joined_onto_image(self):
        ex = GR.build_examples([_viewport_target(10, 10)], root="corpusdir")
        self.assertTrue(ex[0]["image"].replace("\\", "/").startswith("corpusdir/"))

    def test_stats_count_regions(self):
        ex = GR.build_examples([_viewport_target(1, 1), _viewport_target(2, 2)])
        st = GR.dataset_stats(ex)
        self.assertEqual(st.records, 2)
        self.assertEqual(st.viewport, 2)


class TestVerdict(unittest.TestCase):

    def test_verdict_says_not_run_and_names_the_upgrade(self):
        v = GR.VRAM_VERDICT
        self.assertEqual(v["vram_gb"], 16)
        self.assertFalse(v["run_now"])
        self.assertIn("24GB", v["recommended_for_production"])
        self.assertEqual(v["corpus_size"], 938)
        # The 7B training verdict is a clear "does NOT fit".
        self.assertIn("does NOT fit", v["7b_qlora_train"])

    def test_baselines_are_the_benchmark_floor(self):
        self.assertEqual(GR.VIEWPORT_BASELINES,
                         {"random": 0.034, "center": 0.085, "oracle": 1.000})


@unittest.skipUnless(train.MISSING, "training stack present; skip the absence path")
class TestPredictorAbsentStack(unittest.TestCase):

    def test_predictor_requires_the_stack(self):
        with self.assertRaises(RuntimeError):
            GR.LoRAGroundingPredictor()


if __name__ == "__main__":
    unittest.main()
