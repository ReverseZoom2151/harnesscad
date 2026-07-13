"""Tests for the GenCAD-Self-Repairing repair-loop convergence driver."""

from __future__ import annotations

import unittest

from harnesscad.domain.reconstruction.tokens.deepcad_commands import Command, command
from harnesscad.eval.reliability.repair_loop import LoopResult, repair_until_feasible
from harnesscad.eval.reliability.infeasibility_taxonomy import is_feasible


def _ext(**kw):
    params = dict(theta=0.0, phi=0.0, gamma=0.0, px=0.0, py=0.0, pz=0.0,
                  s=1.0, e1=0.5, e2=0.0, b=0.0, u=0.0)
    params.update(kw)
    return command("Ext", **params)


def _rect_loop():
    return [
        Command("SOL"),
        command("Line", x=0.0, y=0.0),
        command("Line", x=0.5, y=0.0),
        command("Line", x=0.5, y=0.5),
        command("Line", x=0.0, y=0.5),
    ]


def _circle_loop(r=0.5):
    return [Command("SOL"), command("Circle", x=0.0, y=0.0, r=r)]


class TestConvergence(unittest.TestCase):
    def test_already_feasible_zero_iterations(self):
        seq = _rect_loop() + [_ext(), Command("EOS")]
        res = repair_until_feasible(seq)
        self.assertIsInstance(res, LoopResult)
        self.assertTrue(res.feasible)
        self.assertTrue(res.converged)
        self.assertEqual(res.reason, "feasible")
        self.assertEqual(res.iterations, 0)

    def test_infeasible_converges_in_one_step(self):
        seq = [command("Line", x=9.0, y=0.0)] + _rect_loop()[1:] + [
            _ext(e1=0.0, e2=0.0), Command("EOS"), Command("SOL")]
        res = repair_until_feasible(seq)
        self.assertTrue(res.feasible)
        self.assertTrue(res.converged)
        self.assertEqual(res.reason, "feasible")
        self.assertEqual(res.iterations, 1)
        self.assertTrue(is_feasible(res.sequence))
        self.assertEqual(len(res.history), 1)
        self.assertGreater(res.history[0].findings_before, 0)
        self.assertEqual(res.history[0].findings_after, 0)

    def test_history_records_fixes(self):
        seq = _circle_loop(r=-1.0) + [_ext(u=9.0), Command("EOS")]
        res = repair_until_feasible(seq)
        self.assertTrue(res.feasible)
        self.assertTrue(res.history[0].fixes)


class TestStopConditions(unittest.TestCase):
    def test_no_progress_when_checker_never_accepts_feasible_input(self):
        # A feasible structural sequence but a checker that always rejects:
        # repair makes no change -> stall guard fires.
        seq = _rect_loop() + [_ext(), Command("EOS")]
        res = repair_until_feasible(seq, checker=lambda s: False)
        self.assertFalse(res.feasible)
        self.assertTrue(res.converged)
        self.assertEqual(res.reason, "no-progress")

    def test_max_iterations_reached(self):
        # Infeasible input, checker never accepts, single iteration budget:
        # the first repair changes+reduces so the stall guard does not fire,
        # and the loop exits on the iteration cap.
        seq = [_ext(), Command("EOS"), command("Line", x=9.0, y=0.0)]
        res = repair_until_feasible(seq, max_iterations=1,
                                    checker=lambda s: False)
        self.assertFalse(res.converged)
        self.assertEqual(res.reason, "max-iterations")
        self.assertEqual(res.iterations, 1)

    def test_invalid_max_iterations(self):
        with self.assertRaises(ValueError):
            repair_until_feasible(_rect_loop(), max_iterations=0)


class TestDeterminism(unittest.TestCase):
    def test_repeatable(self):
        seq = _circle_loop(r=-2.0) + [_ext(e1=0.0, e2=0.0), Command("EOS")]
        a = repair_until_feasible(seq)
        b = repair_until_feasible(seq)
        self.assertEqual(a.to_dict(), b.to_dict())


if __name__ == "__main__":
    unittest.main()
