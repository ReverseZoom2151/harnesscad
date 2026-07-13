"""Tests for the GenCAD-Self-Repairing sequence-level repair procedure."""

from __future__ import annotations

import unittest

from harnesscad.domain.reconstruction.deepcad_command_spec import Command, command
from harnesscad.eval.reliability.gencadrepair_sequence import repair_sequence
from harnesscad.eval.reliability.gencadrepair_taxonomy import is_feasible


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


class TestRepairMakesFeasible(unittest.TestCase):
    def _assert_repaired_feasible(self, seq):
        out = repair_sequence(seq)
        self.assertTrue(out.feasible, out.diagnosis_after.codes())
        self.assertTrue(is_feasible(out.repaired))
        return out

    def test_curve_before_loop(self):
        seq = [command("Line", x=0.0, y=0.0), command("Line", x=0.5, y=0.5),
               command("Line", x=0.0, y=0.5), _ext(), Command("EOS")]
        out = self._assert_repaired_feasible(seq)
        self.assertTrue(out.changed)
        self.assertEqual(out.repaired[0].type, "SOL")

    def test_commands_after_eos_truncated(self):
        seq = _rect_loop() + [_ext(), Command("EOS"), Command("SOL"),
                              command("Line", x=0.1, y=0.1)]
        out = self._assert_repaired_feasible(seq)
        self.assertEqual(out.repaired[-1].type, "EOS")
        self.assertEqual(sum(c.type == "EOS" for c in out.repaired), 1)

    def test_missing_eos_appended(self):
        seq = _rect_loop() + [_ext()]
        out = self._assert_repaired_feasible(seq)
        self.assertEqual(out.repaired[-1].type, "EOS")

    def test_empty_loop_dropped(self):
        seq = [Command("SOL")] + _rect_loop() + [_ext(), Command("EOS")]
        self._assert_repaired_feasible(seq)

    def test_degenerate_profile_dropped(self):
        seq = ([Command("SOL"), command("Line", x=0.4, y=0.4)]
               + _circle_loop() + [_ext(), Command("EOS")])
        self._assert_repaired_feasible(seq)

    def test_extrude_without_profile_dropped(self):
        seq = [_ext(), Command("EOS")]
        out = self._assert_repaired_feasible(seq)
        self.assertNotIn("Ext", [c.type for c in out.repaired])

    def test_trailing_profile_gets_extrude(self):
        seq = _rect_loop() + [Command("EOS")]
        out = self._assert_repaired_feasible(seq)
        self.assertIn("Ext", [c.type for c in out.repaired])

    def test_out_of_range_param_clamped(self):
        seq = [Command("SOL"), command("Line", x=9.0, y=-9.0),
               command("Line", x=0.5, y=0.5), _ext(), Command("EOS")]
        out = self._assert_repaired_feasible(seq)
        line = out.repaired[1]
        self.assertLessEqual(abs(line.get("x")), 1.0)

    def test_nonpositive_radius_fixed(self):
        seq = [Command("SOL"), command("Circle", x=0.0, y=0.0, r=-3.0),
               _ext(), Command("EOS")]
        out = self._assert_repaired_feasible(seq)
        circ = out.repaired[1]
        self.assertGreater(circ.get("r"), 0.0)

    def test_bad_flag_snapped(self):
        seq = _circle_loop() + [_ext(u=7.0), Command("EOS")]
        self._assert_repaired_feasible(seq)

    def test_degenerate_extrude_fixed(self):
        seq = _circle_loop() + [_ext(e1=0.0, e2=0.0, s=0.0), Command("EOS")]
        out = self._assert_repaired_feasible(seq)
        ext = [c for c in out.repaired if c.type == "Ext"][0]
        self.assertGreater(ext.get("s"), 0.0)
        self.assertTrue(ext.get("e1") != 0.0 or ext.get("e2") != 0.0)


class TestIdempotence(unittest.TestCase):
    def test_feasible_sequence_unchanged(self):
        seq = _rect_loop() + [_ext(), Command("EOS")]
        out = repair_sequence(seq)
        self.assertFalse(out.changed)
        self.assertEqual(out.repaired, seq)

    def test_repair_is_idempotent(self):
        seq = [command("Line", x=9.0, y=0.0)] + _rect_loop()[1:] + [
            _ext(e1=0.0, e2=0.0), Command("EOS"), Command("SOL")]
        first = repair_sequence(seq)
        second = repair_sequence(first.repaired)
        self.assertFalse(second.changed)
        self.assertEqual(second.repaired, first.repaired)

    def test_determinism(self):
        seq = _circle_loop(r=-1.0) + [_ext(u=9.0), Command("EOS")]
        a = repair_sequence(seq)
        b = repair_sequence(seq)
        self.assertEqual(a.repaired, b.repaired)
        self.assertEqual(a.fixes, b.fixes)


if __name__ == "__main__":
    unittest.main()
