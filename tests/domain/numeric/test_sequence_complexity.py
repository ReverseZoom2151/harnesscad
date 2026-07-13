import math
import unittest

from harnesscad.domain.numeric.sequence_complexity import (
    effective_length, command_type_entropy, loop_structure,
    parameter_richness, transition_diversity, sequence_complexity,
    is_complex, multiscale_complexity_profile,
)

# command type ids: 0=SOL, 1=Line, 2=Arc, 3=Circle, 4=Extrude, -1=EOS, -2=pad
SOL, LINE, ARC, CIRCLE, EXT, EOS, PAD = 0, 1, 2, 3, 4, -1, -2


def cmd(ctype, *params):
    return (ctype, tuple(params))


class TestEffectiveLength(unittest.TestCase):
    def test_counts_real_commands(self):
        prog = (cmd(SOL, 1.0), cmd(LINE, 1.0), cmd(EOS), cmd(LINE, 1.0))
        self.assertEqual(effective_length(prog), 2)

    def test_skips_padding(self):
        prog = (cmd(LINE, 1.0), cmd(PAD), cmd(ARC, 1.0))
        self.assertEqual(effective_length(prog), 2)


class TestEntropy(unittest.TestCase):
    def test_single_type_zero(self):
        prog = (cmd(LINE, 1.0), cmd(LINE, 1.0), cmd(LINE, 1.0))
        self.assertAlmostEqual(command_type_entropy(prog), 0.0)

    def test_two_balanced_types_one_bit(self):
        prog = (cmd(LINE, 1.0), cmd(ARC, 1.0))
        self.assertAlmostEqual(command_type_entropy(prog), 1.0)

    def test_empty_zero(self):
        self.assertEqual(command_type_entropy(()), 0.0)


class TestLoopStructure(unittest.TestCase):
    def test_two_loops(self):
        prog = (cmd(SOL), cmd(LINE, 1.0), cmd(LINE, 1.0),
                cmd(SOL), cmd(ARC, 1.0))
        info = loop_structure(prog, SOL)
        self.assertEqual(info["num_loops"], 2.0)
        self.assertEqual(info["max_loop"], 2.0)
        self.assertAlmostEqual(info["avg_loop"], 1.5)

    def test_no_loops(self):
        prog = (cmd(LINE, 1.0), cmd(ARC, 1.0))
        info = loop_structure(prog, SOL)
        self.assertEqual(info["num_loops"], 0.0)


class TestParameterRichness(unittest.TestCase):
    def test_all_used(self):
        prog = (cmd(LINE, 1.0, 2.0), cmd(ARC, 3.0, 4.0))
        self.assertAlmostEqual(parameter_richness(prog), 1.0)

    def test_half_used(self):
        prog = (cmd(LINE, 1.0, -1.0),)
        self.assertAlmostEqual(parameter_richness(prog), 0.5)

    def test_empty(self):
        self.assertEqual(parameter_richness(()), 0.0)


class TestTransitionDiversity(unittest.TestCase):
    def test_all_distinct(self):
        prog = (cmd(LINE, 1.0), cmd(ARC, 1.0), cmd(CIRCLE, 1.0))
        # bigrams (L,A),(A,C) both distinct over 2 transitions -> 1.0
        self.assertAlmostEqual(transition_diversity(prog), 1.0)

    def test_repeated(self):
        prog = (cmd(LINE, 1.0), cmd(LINE, 1.0), cmd(LINE, 1.0))
        # only bigram (L,L) over 2 transitions -> 0.5
        self.assertAlmostEqual(transition_diversity(prog), 0.5)

    def test_too_short(self):
        self.assertEqual(transition_diversity((cmd(LINE, 1.0),)), 0.0)


class TestSequenceComplexity(unittest.TestCase):
    def test_range_and_monotonic(self):
        simple = (cmd(SOL), cmd(LINE, 1.0), cmd(EXT, 1.0))
        complex_prog = tuple(
            cmd(SOL) if i % 5 == 0 else cmd((i % 4) + 1, float(i), 2.0)
            for i in range(120)
        )
        s_simple = sequence_complexity(simple, SOL)
        s_complex = sequence_complexity(complex_prog, SOL)
        self.assertGreaterEqual(s_simple, 0.0)
        self.assertLessEqual(s_complex, 1.0)
        self.assertGreater(s_complex, s_simple)

    def test_empty_zero(self):
        self.assertEqual(sequence_complexity((), SOL), 0.0)

    def test_deterministic(self):
        prog = (cmd(SOL), cmd(LINE, 1.0), cmd(ARC, 2.0), cmd(EXT, 3.0))
        self.assertEqual(sequence_complexity(prog, SOL),
                         sequence_complexity(prog, SOL))


class TestIsComplex(unittest.TestCase):
    def test_short_rejected(self):
        prog = tuple(cmd(LINE, 1.0) for _ in range(10))
        self.assertFalse(is_complex(prog, SOL))

    def test_sketch_extrude_only_rejected(self):
        # length in band but only 2 distinct types -> rejected
        prog = tuple(cmd(LINE if i % 2 else EXT, 1.0) for i in range(100))
        self.assertFalse(is_complex(prog, SOL))

    def test_accepted(self):
        prog = tuple(
            cmd([SOL, LINE, ARC, CIRCLE, EXT][i % 5], 1.0) for i in range(100)
        )
        self.assertTrue(is_complex(prog, SOL))

    def test_too_long_rejected(self):
        prog = tuple(cmd([SOL, LINE, ARC][i % 3], 1.0) for i in range(300))
        self.assertFalse(is_complex(prog, SOL))


class TestMultiscaleProfile(unittest.TestCase):
    def test_shape(self):
        prog = tuple(cmd((i % 4) + 1, float(i)) for i in range(64))
        prof = multiscale_complexity_profile(prog, SOL, levels=3, factor=2)
        self.assertEqual(len(prof), 3)
        for v in prof:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            multiscale_complexity_profile((), SOL, levels=0)
        with self.assertRaises(ValueError):
            multiscale_complexity_profile((), SOL, factor=1)


if __name__ == "__main__":
    unittest.main()
