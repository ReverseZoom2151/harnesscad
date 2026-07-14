"""Offline proof that the showcase loop is the harness's loop.

Nothing here touches ollama, litellm or the network: the model is a scripted
`LLM` that hands back a canned response per turn. What is exercised for real is
everything downstream of the model -- the planner, the CISP op parser, the
transactional session, the verifier fleet at verify_level="full", the kernel
preflight gate, the renderer and the PNG validator.

The load-bearing test is `test_bad_fillet_is_blocked_and_the_retry_succeeds`:
a model asks for a 25 mm fillet on a 30 mm cube, the harness answers with the
typed `preflight-RADIUS_TOO_LARGE` diagnostic, that diagnostic reaches the
model's next prompt verbatim, and the corrected op stream verifies. That is the
block-and-correct loop, end to end, with no human in it.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

from harnesscad.agents.llm.base import CompletionResult, Message
from harnesscad.eval.showcase import report as report_mod
from harnesscad.eval.showcase.briefs import (
    BRIEFS, Brief, brief_by_id, brief_ids, grade_geometry,
)
from harnesscad.eval.showcase.image import (
    MIN_SILHOUETTE, PngError, load_png, png_stats, validate_png,
)
from harnesscad.eval.showcase.loop import (
    apply_ops, blocking_diagnostics, run_brief,
)
from harnesscad.eval.showcase.models import MODELS, model_slug
from harnesscad.eval.showcase.runner import render_record


# --- the scripted model ----------------------------------------------------
class ScriptedLLM:
    """An `llm.base.LLM` that replays canned responses and records its prompts.

    `prompts` keeps every user message it was sent, so a test can assert that a
    diagnostic actually reached the model rather than merely being logged.
    """

    def __init__(self, responses: List[Any]) -> None:
        self.responses = list(responses)
        self.prompts: List[str] = []
        self.calls = 0

    def complete(self, messages: List[Message], tools=None, response_schema=None,
                 **opts: Any) -> CompletionResult:
        self.prompts.append("\n".join(m.content for m in messages if m.role == "user"))
        self.calls += 1
        if not self.responses:
            return CompletionResult(text="")
        body = self.responses.pop(0)
        text = body if isinstance(body, str) else json.dumps(body)
        return CompletionResult(text=text)

    def stream(self, messages, tools=None, response_schema=None, **opts):  # pragma: no cover
        yield self.complete(messages).text


# --- op streams the scripted model emits -----------------------------------
def cube_ops(size: float = 30.0, fillet: Optional[float] = None,
             hole: Optional[float] = None) -> List[dict]:
    ops: List[dict] = [
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": size, "h": size},
        {"op": "constrain", "kind": "distance", "a": "e1", "value": size},
        {"op": "extrude", "sketch": "sk1", "distance": size},
    ]
    if hole is not None:
        ops.append({"op": "hole", "face_or_sketch": "f1", "x": size / 2,
                    "y": size / 2, "diameter": hole, "through": True})
    if fillet is not None:
        ops.append({"op": "fillet", "edges": ["f1"], "radius": fillet})
    return ops


PLATE_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 60, "h": 40},
    {"op": "constrain", "kind": "distance", "a": "e1", "value": 60},
    {"op": "extrude", "sketch": "sk1", "distance": 6},
]

# References a sketch entity that was never created: the classic model error.
BAD_REF_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 60, "h": 40},
    {"op": "constrain", "kind": "distance", "a": "e7", "value": 60},
    {"op": "extrude", "sketch": "sk1", "distance": 6},
]

PLATE = brief_by_id("plate")
FILLETED = brief_by_id("filleted_cube")


class TestBriefs(unittest.TestCase):
    def test_brief_set_loads(self):
        self.assertGreaterEqual(len(BRIEFS), 10)
        ids = brief_ids()
        self.assertEqual(len(ids), len(set(ids)), "brief ids must be unique")
        for b in BRIEFS:
            self.assertIsInstance(b, Brief)
            self.assertTrue(b.text.strip())
            self.assertGreater(b.volume_mm3, 0.0, b.id)
            self.assertIn(b.tier, (1, 2, 3, 4, 5))
            self.assertTrue(b.rationale)

    def test_briefs_span_the_difficulty_range(self):
        tiers = {b.tier for b in BRIEFS}
        self.assertTrue({1, 2, 3, 4, 5}.issubset(tiers) or len(tiers) >= 4)

    def test_brief_by_id_rejects_unknown(self):
        with self.assertRaises(KeyError):
            brief_by_id("no-such-brief")

    def test_grading_is_a_measurement_not_an_opinion(self):
        # The right volume with the right features: on brief.
        good = grade_geometry(PLATE, PLATE.volume_mm3, ["new_sketch", "extrude"])
        self.assertTrue(good["on_brief"])
        # A solid that verifies but is the wrong size: NOT on brief.
        wrong_size = grade_geometry(PLATE, PLATE.volume_mm3 * 1.9,
                                    ["new_sketch", "extrude"])
        self.assertFalse(wrong_size["on_brief"])
        self.assertFalse(wrong_size["volume_ok"])
        # A bracket-shaped brief built with no holes in it: NOT on brief, and the
        # reason names the missing feature.
        bracket = brief_by_id("bracket")
        holeless = grade_geometry(bracket, bracket.volume_mm3, ["extrude"])
        self.assertFalse(holeless["on_brief"])
        self.assertFalse(holeless["features_ok"])
        self.assertIn("hole|boolean", holeless["missing_features"])


class TestLoop(unittest.TestCase):
    def test_a_good_op_stream_solves_on_the_first_attempt(self):
        llm = ScriptedLLM([PLATE_OPS])
        rec = run_brief(PLATE, llm, model="scripted", seed=7)
        self.assertTrue(rec.solved)
        self.assertEqual(rec.attempt_count, 1)
        self.assertTrue(rec.digest)
        self.assertTrue(rec.grade["on_brief"], rec.grade["reasons"])
        self.assertFalse(rec.hand_fixed)

    def test_typed_diagnostics_are_fed_back_and_the_retry_succeeds(self):
        """A bad entity reference is caught, reported, and fixed on the retry."""
        llm = ScriptedLLM([BAD_REF_OPS, PLATE_OPS])
        rec = run_brief(PLATE, llm, model="scripted", seed=7)

        first = rec.attempts[0]
        self.assertFalse(first.ok)
        self.assertIn("bad-ref", first.error_codes)
        # The diagnostic reached the MODEL, not just the log.
        self.assertEqual(llm.calls, 2)
        self.assertIn("bad-ref", llm.prompts[1])
        self.assertIn("PRIOR ATTEMPT FAILED", llm.prompts[1])
        # And the corrected stream verified.
        self.assertTrue(rec.solved)
        self.assertEqual(rec.attempt_count, 2)
        self.assertIn("bad-ref", rec.diagnostics_seen)

    def test_bad_fillet_is_blocked_but_the_unsound_rule_does_not_instruct(self):
        """RADIUS_TOO_LARGE blocks the part. It does NOT tell the model what to do.

        Blocking and instructing are different powers and they carry different
        risks. Blocking a good part costs a retry. INSTRUCTING on a good part
        destroys it: `assets/pressure/report.md` measured the 14b obeying a false
        typed diagnostic precisely and turning a correct washer into scrap.

        `preflight-RADIUS_TOO_LARGE` is HEURISTIC (verifiers.soundness): it
        compares the radius against half the smallest extent of the whole
        bounding box, but a fillet acts on an EDGE, which need not span that
        extent -- a 50x30x6 plate filleted at r=3.1 is valid and the rule
        rejects it. So the showcase still refuses to SHIP the r=25 cube (the
        finding is logged, the attempt is not accepted, the model is re-asked),
        and the model is never handed the order "Reduce the radius below 15".
        It resamples instead, and resampling is what beat the typed loop.
        """
        llm = ScriptedLLM([
            cube_ops(30.0, fillet=25.0),            # r25 on a 30 cube: impossible
            cube_ops(30.0, fillet=2.0, hole=10.0),  # the corrected part
        ])
        rec = run_brief(FILLETED, llm, model="scripted", seed=7)

        first = rec.attempts[0]
        self.assertFalse(first.accepted)
        # Still detected, still blocking, still logged for the human.
        self.assertIn("preflight-RADIUS_TOO_LARGE", first.error_codes)
        self.assertIn("preflight-RADIUS_TOO_LARGE", rec.diagnostics_seen)
        # ...and never spoken to the model.
        self.assertNotIn("RADIUS_TOO_LARGE", llm.prompts[1])
        self.assertNotIn("Reduce the radius", llm.prompts[1])

        self.assertTrue(rec.solved)
        self.assertEqual(rec.attempt_count, 2)
        self.assertEqual(rec.attempts[1].error_codes, [])
        self.assertTrue(rec.grade["on_brief"], rec.grade["reasons"])

    def test_the_preflight_gate_only_promotes_kernel_findings(self):
        """Advisory fleet chatter the op set cannot fix is never sent back."""
        _server, result = apply_ops(cube_ops(30.0))
        self.assertTrue(result["ok"])
        codes = {d["code"] for d in result["diagnostics"]}
        # The fleet does report unfixable metadata errors at verify_level=full...
        self.assertIn("missing-metadata", codes)
        # ...and the loop does not push them at the model.
        self.assertEqual(blocking_diagnostics(result), [])

    def test_unparseable_output_is_retried_as_a_typed_diagnostic(self):
        llm = ScriptedLLM(["Sure! Here is your bracket, I hope you like it.",
                           PLATE_OPS])
        rec = run_brief(PLATE, llm, model="scripted", seed=7)
        self.assertIsNotNone(rec.attempts[0].parse_error)
        self.assertIn("plan-parse-error", llm.prompts[1])
        self.assertTrue(rec.solved)
        self.assertEqual(rec.attempt_count, 2)

    def test_a_model_that_never_fixes_it_fails_and_says_why(self):
        llm = ScriptedLLM([BAD_REF_OPS, BAD_REF_OPS, BAD_REF_OPS])
        rec = run_brief(PLATE, llm, model="scripted", seed=7, max_attempts=3)
        self.assertFalse(rec.solved)
        self.assertEqual(rec.attempt_count, 3)
        self.assertIn("bad-ref", rec.failure_reason)
        self.assertIn("could not fix", rec.failure_reason)
        self.assertIsNone(rec.render)

    def test_ops_that_make_no_solid_are_not_a_part(self):
        sketch_only = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0, "y": 0, "w": 60, "h": 40},
        ]
        llm = ScriptedLLM([sketch_only, sketch_only])
        rec = run_brief(PLATE, llm, model="scripted", seed=7, max_attempts=2)
        self.assertFalse(rec.solved)
        self.assertIn("no-solid", rec.diagnostics_seen)

    def test_a_provider_failure_is_recorded_not_raised(self):
        class Broken:
            def complete(self, *a, **k):
                raise TimeoutError("ollama did not answer")

            def stream(self, *a, **k):  # pragma: no cover
                raise TimeoutError

        rec = run_brief(PLATE, Broken(), model="broken", seed=7)
        self.assertFalse(rec.solved)
        self.assertIn("provider error", rec.failure_reason)


class TestRenderValidation(unittest.TestCase):
    def test_a_real_render_is_decoded_and_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "plate.png")
            img = render_record(PLATE_OPS, png, width=320, height=200)
            self.assertTrue(img["ok"], img.get("failures"))
            self.assertEqual((img["width"], img["height"]), (320, 200))
            self.assertGreater(img["silhouette"], MIN_SILHOUETTE)
            self.assertGreater(img["variance"], 0.0)
            # The file on disk really is that image.
            decoded = load_png(png)
            self.assertEqual(decoded.width, 320)
            self.assertEqual(len(decoded.pixels),
                             320 * 200 * decoded.channels)

    def test_a_blank_render_is_rejected(self):
        from harnesscad.io.render import write_png

        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "blank.png")
            write_png(png, [200] * (64 * 64 * 3), 64, 64)
            stats = png_stats(png)
            self.assertFalse(stats.ok)
            self.assertTrue(any("blank" in f or "flat" in f for f in stats.failures),
                            stats.failures)

    def test_a_black_render_is_rejected(self):
        from harnesscad.io.render import write_png

        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "black.png")
            write_png(png, [0] * (64 * 64 * 3), 64, 64)
            self.assertFalse(png_stats(png).ok)

    def test_a_broken_image_is_never_shipped(self):
        """render_record deletes any PNG that fails validation."""
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "nope.png")
            img = render_record(
                [{"op": "new_sketch", "plane": "XY"}], png, width=64, height=64)
            self.assertFalse(img["ok"])
            self.assertFalse(os.path.exists(png))

    def test_a_non_png_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "not.png")
            with open(path, "wb") as fh:
                fh.write(b"definitely not a png")
            with self.assertRaises(PngError):
                load_png(path)
            self.assertFalse(validate_png(path)["ok"])


class TestScoreboard(unittest.TestCase):
    def _runs(self) -> List[Dict[str, Any]]:
        llm_good = ScriptedLLM([PLATE_OPS])
        good = run_brief(PLATE, llm_good, model="qwen2.5-coder:7b", seed=7)
        llm_bad = ScriptedLLM([BAD_REF_OPS, BAD_REF_OPS, BAD_REF_OPS])
        bad = run_brief(PLATE, llm_bad, model="codellama:7b", seed=7)
        return [good.to_dict(), bad.to_dict()]

    def test_scoreboard_separates_solved_from_on_brief(self):
        board = report_mod.scoreboard(self._runs())
        self.assertEqual(board["totals"]["pairs"], 2)
        self.assertEqual(board["totals"]["solved"], 1)
        self.assertEqual(board["totals"]["on_brief"], 1)
        self.assertEqual(board["totals"]["hand_fixed"], 0)
        rows = {r["model"]: r for r in board["per_model"]}
        self.assertEqual(rows["qwen2.5-coder:7b"]["unaided"], 1)
        self.assertEqual(rows["codellama:7b"]["solved"], 0)
        self.assertIn("bad-ref", rows["codellama:7b"]["diagnostics_seen"])

    def test_report_and_results_are_written(self):
        runs = self._runs()
        with tempfile.TemporaryDirectory() as tmp:
            board = report_mod.write_results(runs, tmp)
            with open(os.path.join(tmp, "results.json"), encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(len(payload["runs"]), 2)
            self.assertEqual(len(payload["briefs"]), len(BRIEFS))
            md = report_mod.render_markdown(runs, board)
            self.assertIn("Scoreboard (model)", md)
            self.assertIn("could not fix", md)

    def test_best_per_brief_prefers_the_on_brief_result(self):
        best = report_mod.best_per_brief(self._runs())
        self.assertEqual(best["plate"]["model"], "qwen2.5-coder:7b")
        self.assertIsNone(best["spur_gear"])

    def test_model_slugs_are_file_safe(self):
        for m in MODELS:
            self.assertNotIn(":", model_slug(m))


if __name__ == "__main__":
    unittest.main(verbosity=2)
