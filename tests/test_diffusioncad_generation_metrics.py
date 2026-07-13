import unittest

from harnesscad.eval.bench.generative.diffusioncad_generation_metrics import (
    ARC,
    CIRCLE,
    EOS,
    EXTRUDE,
    LINE,
    SOL,
    generation_report,
    invalidity_percent,
    is_wellformed,
    novel_percent,
    unique_percent,
)


class TestUnique(unittest.TestCase):
    def test_all_unique(self):
        gen = [(1, 2), (3, 4), (5, 6)]
        self.assertEqual(unique_percent(gen), 100.0)

    def test_with_duplicates(self):
        gen = [(1,), (1,), (2,), (3,)]  # only (2,) and (3,) singletons
        self.assertEqual(unique_percent(gen), 50.0)

    def test_empty(self):
        self.assertEqual(unique_percent([]), 0.0)


class TestNovel(unittest.TestCase):
    def test_all_novel(self):
        gen = [(1,), (2,)]
        train = [(9,), (8,)]
        self.assertEqual(novel_percent(gen, train), 100.0)

    def test_none_novel(self):
        gen = [(1,), (2,)]
        train = [(1,), (2,)]
        self.assertEqual(novel_percent(gen, train), 0.0)

    def test_half(self):
        gen = [(1,), (5,)]
        train = [(1,)]
        self.assertEqual(novel_percent(gen, train), 50.0)


class TestWellformed(unittest.TestCase):
    def test_valid_line_loop(self):
        seq = [SOL, LINE, LINE, LINE, EXTRUDE, EOS]
        self.assertTrue(is_wellformed(seq))

    def test_valid_circle(self):
        seq = [SOL, CIRCLE, EXTRUDE, EOS]
        self.assertTrue(is_wellformed(seq))

    def test_valid_two_loops(self):
        seq = [SOL, LINE, LINE, SOL, CIRCLE, EXTRUDE, EOS]
        # second loop is a circle (self-closing), first closed by... actually
        # first loop stays open until circle? No: SOL opens, geometry, then a new
        # SOL is illegal while open. Expect False.
        self.assertFalse(is_wellformed(seq))

    def test_geometry_outside_loop(self):
        self.assertFalse(is_wellformed([LINE, EXTRUDE, EOS]))

    def test_no_eos(self):
        self.assertFalse(is_wellformed([SOL, LINE, EXTRUDE]))

    def test_nothing_after_eos(self):
        self.assertFalse(is_wellformed([SOL, CIRCLE, EXTRUDE, EOS, LINE]))

    def test_extrude_without_loop(self):
        self.assertFalse(is_wellformed([EXTRUDE, EOS]))

    def test_dangling_open_loop(self):
        self.assertFalse(is_wellformed([SOL, LINE, EOS]))

    def test_empty_loop_then_extrude(self):
        self.assertFalse(is_wellformed([SOL, EXTRUDE, EOS]))

    def test_unknown_token(self):
        self.assertFalse(is_wellformed([SOL, "X", EXTRUDE, EOS]))

    def test_arc_valid(self):
        self.assertTrue(is_wellformed([SOL, LINE, ARC, EXTRUDE, EOS]))


class TestInvalidityAndReport(unittest.TestCase):
    def test_invalidity_percent(self):
        seqs = [
            [SOL, CIRCLE, EXTRUDE, EOS],  # valid
            [LINE, EOS],  # invalid
        ]
        self.assertEqual(invalidity_percent(seqs), 50.0)

    def test_report(self):
        gen = [(SOL, CIRCLE, EXTRUDE, EOS), (SOL, LINE, LINE, EXTRUDE, EOS)]
        train = [(SOL, CIRCLE, EXTRUDE, EOS)]
        rep = generation_report(gen, train)
        self.assertEqual(rep["count"], 2)
        self.assertEqual(rep["unique_pct"], 100.0)
        self.assertEqual(rep["novel_pct"], 50.0)
        self.assertEqual(rep["invalidity_pct"], 0.0)

    def test_report_separate_command_view(self):
        gen = [(1, 2, 3)]
        train = []
        cmd = [[SOL, CIRCLE, EXTRUDE, EOS]]
        rep = generation_report(gen, train, command_view=cmd)
        self.assertEqual(rep["invalidity_pct"], 0.0)
        self.assertEqual(rep["novel_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
