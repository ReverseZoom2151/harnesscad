import unittest

from reconstruction.cadparser_grammar import State, allowed, is_valid, run
from reconstruction.cadparser_schema import SOS, EOS, PAD


def wrap(*tokens):
    return [SOS, *tokens, EOS]


class TestGrammar(unittest.TestCase):
    def test_simple_sketch_extrude(self):
        self.assertTrue(is_valid(wrap("L", "L", "L", "E")))

    def test_padding_after_eos(self):
        self.assertTrue(is_valid([SOS, "C", "E", EOS, PAD, PAD]))

    def test_revolution_requires_axis(self):
        # R without a preceding Ax is illegal
        self.assertFalse(is_valid(wrap("L", "L", "R")))
        # Ax then R is valid
        self.assertTrue(is_valid(wrap("Ax", "R")))

    def test_axis_must_be_followed_by_revolve(self):
        state, issues = run([SOS, "Ax", "E"])
        self.assertTrue(issues)

    def test_cut_needs_existing_solid(self):
        # first shaping op cannot be a cut
        self.assertFalse(is_valid(wrap("L", "L", "Ec")))
        # but after building a solid, a cut is fine
        self.assertTrue(is_valid(wrap("C", "E", "C", "Ec")))

    def test_edge_feature_needs_solid(self):
        self.assertFalse(is_valid(wrap("F")))
        self.assertTrue(is_valid(wrap("C", "E", "F")))

    def test_chamfer_after_solid(self):
        self.assertTrue(is_valid(wrap("C", "E", "Cf", "F")))

    def test_unterminated_flagged(self):
        state, issues = run([SOS, "C", "E"])  # no EOS
        self.assertEqual(state, State.SOLID)
        self.assertFalse(issues)  # SOLID is an accepting resting state
        state2, issues2 = run([SOS, "L"])  # open profile, no closer/eos
        self.assertTrue(issues2)

    def test_illegal_reports_index(self):
        _, issues = run([SOS, "Ec"])
        self.assertEqual(issues, ("illegal:1:Ec",))

    def test_allowed_masks(self):
        self.assertEqual(allowed(State.START), frozenset({SOS}))
        self.assertIn("R", allowed(State.AXIS))
        self.assertNotIn("E", allowed(State.AXIS))
        self.assertEqual(allowed(State.PAD), frozenset({PAD}))


if __name__ == "__main__":
    unittest.main()
