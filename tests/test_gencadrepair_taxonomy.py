"""Tests for the GenCAD-Self-Repairing infeasibility taxonomy."""

from __future__ import annotations

import unittest

from harnesscad.domain.reconstruction.deepcad_command_spec import Command, command
from harnesscad.eval.reliability.gencadrepair_taxonomy import (
    COMMANDS_AFTER_EOS,
    CURVE_BEFORE_LOOP,
    DEGENERATE_EXTRUDE,
    DEGENERATE_PROFILE,
    EMPTY_LOOP,
    EXTRUDE_WITHOUT_PROFILE,
    FEASIBILITY_CODES,
    MISSING_EOS,
    NONPOSITIVE_RADIUS,
    PARAM_NOT_DISCRETE,
    PARAM_OUT_OF_RANGE,
    TRAILING_PROFILE,
    diagnose,
    is_feasible,
)


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


class TestFeasibleSequences(unittest.TestCase):
    def test_rectangle_extrude_is_feasible(self):
        seq = _rect_loop() + [_ext(), Command("EOS")]
        d = diagnose(seq)
        self.assertTrue(d.feasible, d.codes())
        self.assertTrue(is_feasible(seq))

    def test_circle_extrude_is_feasible(self):
        seq = _circle_loop() + [_ext(), Command("EOS")]
        self.assertTrue(diagnose(seq).feasible)

    def test_two_extrusions_feasible(self):
        seq = (_rect_loop() + [_ext()] + _circle_loop()
               + [_ext(b=1.0), Command("EOS")])
        self.assertTrue(diagnose(seq).feasible, diagnose(seq).codes())


class TestOrderInfeasibility(unittest.TestCase):
    def test_curve_before_loop(self):
        seq = [command("Line", x=0.0, y=0.0)] + _rect_loop()[1:] + [
            _ext(), Command("EOS")]
        self.assertIn(CURVE_BEFORE_LOOP, diagnose(seq).codes())

    def test_commands_after_eos(self):
        seq = _rect_loop() + [_ext(), Command("EOS"), Command("SOL")]
        self.assertIn(COMMANDS_AFTER_EOS, diagnose(seq).codes())

    def test_missing_eos(self):
        seq = _rect_loop() + [_ext()]
        self.assertIn(MISSING_EOS, diagnose(seq).codes())

    def test_extrude_without_profile(self):
        seq = [_ext(), Command("EOS")]
        self.assertIn(EXTRUDE_WITHOUT_PROFILE, diagnose(seq).codes())


class TestLoopProfileInfeasibility(unittest.TestCase):
    def test_empty_loop(self):
        seq = [Command("SOL"), _ext(), Command("EOS")]
        codes = diagnose(seq).codes()
        self.assertIn(EMPTY_LOOP, codes)

    def test_degenerate_single_line_profile(self):
        seq = [Command("SOL"), command("Line", x=0.5, y=0.5),
               _ext(), Command("EOS")]
        self.assertIn(DEGENERATE_PROFILE, diagnose(seq).codes())

    def test_trailing_profile_without_extrude(self):
        seq = _rect_loop() + [Command("EOS")]
        self.assertIn(TRAILING_PROFILE, diagnose(seq).codes())


class TestParamInfeasibility(unittest.TestCase):
    def test_out_of_range_coordinate(self):
        seq = [Command("SOL"), command("Line", x=5.0, y=0.0),
               command("Line", x=0.5, y=0.5), _ext(), Command("EOS")]
        self.assertIn(PARAM_OUT_OF_RANGE, diagnose(seq).codes())

    def test_nonpositive_radius(self):
        seq = [Command("SOL"), command("Circle", x=0.0, y=0.0, r=-0.2),
               _ext(), Command("EOS")]
        self.assertIn(NONPOSITIVE_RADIUS, diagnose(seq).codes())

    def test_bad_discrete_flag(self):
        seq = _circle_loop() + [_ext(u=9.0), Command("EOS")]
        self.assertIn(PARAM_NOT_DISCRETE, diagnose(seq).codes())

    def test_degenerate_extrude_zero_thickness(self):
        seq = _circle_loop() + [_ext(e1=0.0, e2=0.0), Command("EOS")]
        self.assertIn(DEGENERATE_EXTRUDE, diagnose(seq).codes())

    def test_degenerate_extrude_nonpositive_scale(self):
        seq = _circle_loop() + [_ext(s=0.0), Command("EOS")]
        self.assertIn(DEGENERATE_EXTRUDE, diagnose(seq).codes())


class TestDeterminismAndCodes(unittest.TestCase):
    def test_findings_sorted_and_stable(self):
        seq = [command("Line", x=9.0, y=0.0), Command("SOL"), _ext()]
        d1 = diagnose(seq)
        d2 = diagnose(seq)
        self.assertEqual([f.to_dict() for f in d1.findings],
                         [f.to_dict() for f in d2.findings])
        idx = [f.index for f in d1.findings]
        self.assertEqual(idx, sorted(idx))

    def test_all_codes_registered(self):
        self.assertEqual(len(set(FEASIBILITY_CODES)), len(FEASIBILITY_CODES))

    def test_empty_sequence_feasible(self):
        self.assertTrue(diagnose([]).feasible)


if __name__ == "__main__":
    unittest.main()
