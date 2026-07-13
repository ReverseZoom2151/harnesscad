import unittest

from harnesscad.io.surfaces.modality_fusion import (
    FusedIntent,
    ModalityFuser,
    ModalityKind,
    ModalitySignal,
)


class TestModalitySignal(unittest.TestCase):
    def test_rejects_bad_confidence(self):
        with self.assertRaises(ValueError):
            ModalitySignal(ModalityKind.VOICE, "extrude", confidence=1.5)

    def test_rejects_empty_operation(self):
        with self.assertRaises(ValueError):
            ModalitySignal(ModalityKind.VOICE, "")


class TestComplementaryFusion(unittest.TestCase):
    def test_slots_merge_across_modalities(self):
        signals = [
            ModalitySignal(ModalityKind.VOICE, "extrude", {"sketch": "s1"}, 0.9, 0),
            ModalitySignal(ModalityKind.GESTURE, None, {"depth": 12.0}, 0.8, 1),
        ]
        fused = ModalityFuser().fuse(signals)
        self.assertEqual(fused.operation, "extrude")
        self.assertEqual(fused.slots, {"sketch": "s1", "depth": 12.0})
        self.assertEqual(fused.provenance["depth"], ModalityKind.GESTURE)
        self.assertEqual(fused.provenance["sketch"], ModalityKind.VOICE)
        self.assertFalse(fused.needs_clarification)

    def test_stronger_signal_wins_conflicting_slot(self):
        signals = [
            ModalitySignal(ModalityKind.GESTURE, None, {"depth": 5.0}, 0.6, 0),
            ModalitySignal(ModalityKind.KEYBOARD, None, {"depth": 9.0}, 0.9, 1),
        ]
        fused = ModalityFuser().fuse(signals)
        self.assertEqual(fused.slots["depth"], 9.0)
        self.assertEqual(fused.provenance["depth"], ModalityKind.KEYBOARD)


class TestCompetitiveFusion(unittest.TestCase):
    def test_highest_weighted_operation_wins(self):
        signals = [
            ModalitySignal(ModalityKind.GESTURE, "revolve", confidence=0.9, order=0),
            ModalitySignal(ModalityKind.KEYBOARD, "extrude", confidence=0.9, order=1),
        ]
        fused = ModalityFuser().fuse(signals)
        self.assertEqual(fused.operation, "extrude")  # keyboard weight highest

    def test_close_rivals_flag_ambiguous(self):
        signals = [
            ModalitySignal(ModalityKind.VOICE, "extrude", confidence=0.9, order=0),
            ModalitySignal(ModalityKind.SKETCH, "revolve", confidence=0.95, order=1),
        ]
        fused = ModalityFuser(conflict_margin=0.2).fuse(signals)
        self.assertTrue(fused.ambiguous)
        self.assertTrue(fused.needs_clarification)
        self.assertIn(("extrude", "revolve"), fused.conflicts)

    def test_clear_winner_not_ambiguous(self):
        signals = [
            ModalitySignal(ModalityKind.KEYBOARD, "extrude", confidence=1.0, order=0),
            ModalitySignal(ModalityKind.GESTURE, "revolve", confidence=0.3, order=1),
        ]
        fused = ModalityFuser(conflict_margin=0.1).fuse(signals)
        self.assertEqual(fused.operation, "extrude")
        self.assertFalse(fused.ambiguous)

    def test_deterministic_tie_break_by_precedence(self):
        # Equal score: keyboard should win by precedence rank.
        a = ModalitySignal(ModalityKind.GESTURE, "revolve", confidence=1.0, order=0)
        b = ModalitySignal(ModalityKind.KEYBOARD, "extrude", confidence=0.7, order=1)
        # scores: gesture 0.7, keyboard 0.7 -> tie -> keyboard precedence wins
        fused = ModalityFuser().fuse([a, b])
        self.assertEqual(fused.operation, "extrude")


class TestConfidenceAndFiltering(unittest.TestCase):
    def test_confidence_floor_discards_weak_signals(self):
        signals = [
            ModalitySignal(ModalityKind.VOICE, "extrude", confidence=0.9, order=0),
            ModalitySignal(ModalityKind.GESTURE, "revolve", confidence=0.05, order=1),
        ]
        fused = ModalityFuser(confidence_floor=0.1).fuse(signals)
        self.assertEqual(fused.operation, "extrude")
        self.assertEqual(fused.conflicts, ())
        self.assertNotIn(ModalityKind.GESTURE, fused.contributors)

    def test_empty_input_yields_empty_intent(self):
        fused = ModalityFuser().fuse([])
        self.assertIsNone(fused.operation)
        self.assertEqual(fused.slots, {})
        self.assertEqual(fused.confidence, 0.0)
        self.assertTrue(fused.needs_clarification)

    def test_fused_confidence_is_weighted_mean(self):
        signals = [
            ModalitySignal(ModalityKind.KEYBOARD, "extrude", confidence=1.0, order=0),
        ]
        fused = ModalityFuser().fuse(signals)
        # single keyboard signal: score/weight = (1.0*1.0)/1.0 = 1.0
        self.assertAlmostEqual(fused.confidence, 1.0)

    def test_determinism_across_input_order(self):
        s1 = ModalitySignal(ModalityKind.VOICE, "extrude", {"sketch": "s1"}, 0.9, 0)
        s2 = ModalitySignal(ModalityKind.GESTURE, None, {"depth": 4.0}, 0.8, 1)
        f = ModalityFuser()
        self.assertEqual(f.fuse([s1, s2]), f.fuse([s2, s1]))


if __name__ == "__main__":
    unittest.main()
