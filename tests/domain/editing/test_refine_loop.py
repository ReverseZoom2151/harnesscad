"""Tests for editing/cadreasoner_edit_loop.py (render->compare->refine loop)."""

import math
import unittest

from harnesscad.eval.bench.geometry.chamfer import symmetric_chamfer
from harnesscad.domain.editing.refine_loop import EditLoopResult, run_edit_loop


# A toy "program" is a float v; its render is the single point (v, 0, 0).
def _render(program):
    if program is None or program == "invalid":
        return None
    return [(float(program), 0.0, 0.0)]


def _target_x(target_points):
    return sum(p[0] for p in target_points) / len(target_points)


def _converging_editor(target_points, prev_render, prev_program, encoding):
    """Move the parameter halfway toward the target's x each step."""
    tx = _target_x(target_points)
    v = 0.0 if prev_program is None else float(prev_program)
    return v + (tx - v) * 0.5


class TestEditLoopConverges(unittest.TestCase):
    def _run(self, **kw):
        target = [(10.0, 0.0, 0.0)]
        return run_edit_loop(
            target, 0.0, _converging_editor, _render,
            select_metric=symmetric_chamfer, **kw)

    def test_best_so_far_improves_monotonically(self):
        res = self._run(max_steps=5)
        self.assertIsInstance(res, EditLoopResult)
        self.assertTrue(res.converged)
        # 5 halving steps from 0 toward 10: best score is the last (smallest).
        scores = [s.select_score for s in res.steps if s.valid]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLess(res.best_select_score, 1.0)
        self.assertEqual(res.best_index, len(res.steps) - 1)

    def test_invalid_rate_zero_when_all_valid(self):
        res = self._run(max_steps=4)
        self.assertEqual(res.invalid_rate, 0.0)

    def test_respects_step_budget(self):
        res = self._run(max_steps=3)
        self.assertLessEqual(len(res.steps), 3)

    def test_deterministic(self):
        a = self._run(max_steps=5).to_dict()
        b = self._run(max_steps=5).to_dict()
        self.assertEqual(a, b)


class TestEarlyStop(unittest.TestCase):
    def test_stops_when_improvement_below_threshold(self):
        target = [(10.0, 0.0, 0.0)]

        # Editor that jumps straight to the target then stalls: no further gain.
        def editor(tp, pr, prog, enc):
            return 10.0

        res = run_edit_loop(
            target, 0.0, editor, _render, select_metric=symmetric_chamfer,
            max_steps=10, min_improvement=1e-6)
        self.assertEqual(res.stopped_reason, "min-improvement")
        # t=1 lands on target (score 0); t=2 makes no progress -> stop.
        self.assertEqual(len(res.steps), 2)
        self.assertAlmostEqual(res.best_select_score, 0.0)


class TestInvalidHandling(unittest.TestCase):
    def test_invalid_render_counted_and_not_selected(self):
        target = [(10.0, 0.0, 0.0)]

        # Alternate: odd steps invalid, even steps land on the target.
        state = {"n": 0}

        def editor(tp, pr, prog, enc):
            state["n"] += 1
            return "invalid" if state["n"] % 2 == 1 else 10.0

        res = run_edit_loop(
            target, 0.0, editor, _render, select_metric=symmetric_chamfer,
            max_steps=4, min_improvement=0.0)
        self.assertGreater(res.invalid_rate, 0.0)
        # best-so-far must be a valid step
        self.assertTrue(res.steps[res.best_index].valid)

    def test_editor_exception_is_contained(self):
        target = [(10.0, 0.0, 0.0)]

        def editor(tp, pr, prog, enc):
            raise RuntimeError("boom")

        res = run_edit_loop(
            target, 0.0, editor, _render, select_metric=symmetric_chamfer,
            max_steps=2)
        self.assertEqual(res.invalid_rate, 1.0)
        self.assertFalse(res.converged)
        self.assertIn("editor", res.steps[0].error)


class TestSelectionVsReporting(unittest.TestCase):
    def test_selection_uses_scan_report_uses_clean(self):
        # Selection target (scan) shifted; report target (clean) at the true spot.
        scan = [(9.0, 0.0, 0.0)]
        clean = [(10.0, 0.0, 0.0)]

        res = run_edit_loop(
            scan, 0.0, _converging_editor, _render,
            select_metric=symmetric_chamfer,
            report_target_points=clean, report_metric=symmetric_chamfer,
            max_steps=6)
        best = res.steps[res.best_index]
        # The selected best minimizes distance to the SCAN, not the clean target.
        self.assertAlmostEqual(best.select_score,
                               abs(best.program - 9.0), places=6)
        # Reported score is measured against the CLEAN target.
        self.assertAlmostEqual(res.best_report_score,
                               abs(best.program - 10.0), places=6)
        self.assertNotAlmostEqual(res.best_select_score, res.best_report_score)


class TestNullInitPath(unittest.TestCase):
    def test_first_step_uses_null_init_encoding(self):
        target = [(3.0, 4.0, 0.0)]
        seen = {}

        def editor(tp, pr, prog, enc):
            if 1 not in seen:
                seen[1] = enc
            return 5.0

        run_edit_loop(target, None, editor, _render,
                      select_metric=symmetric_chamfer, max_steps=1)
        self.assertTrue(seen[1].t1)


if __name__ == "__main__":
    unittest.main()
