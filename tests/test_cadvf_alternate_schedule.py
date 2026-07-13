import unittest

from harnesscad.agents.generation.cadvf_alternate_schedule import (
    SL, VF, Stage, build_schedule, epochs_by_kind, stage_sequence,
    total_epochs, validate_schedule,
)


class TestBuildSchedule(unittest.TestCase):
    def test_default_shape(self):
        stages = build_schedule()
        # 1 initial SL + 5 blocks of (VF, SL) = 11 stages.
        self.assertEqual(len(stages), 11)
        self.assertEqual(stage_sequence(stages),
                         [SL, VF, SL, VF, SL, VF, SL, VF, SL, VF, SL])

    def test_starts_with_sl(self):
        self.assertEqual(build_schedule()[0].kind, SL)

    def test_zero_rounds_is_only_initial_sl(self):
        stages = build_schedule(num_rounds=0)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].kind, SL)
        self.assertEqual(stages[0].round_index, 0)

    def test_model_tags_chain(self):
        stages = build_schedule(num_rounds=2)
        for prev, nxt in zip(stages, stages[1:]):
            self.assertEqual(nxt.input_model, prev.output_model)
        self.assertEqual(stages[0].input_model, "pretrained")

    def test_vf_references_last_sl(self):
        stages = build_schedule(num_rounds=2)
        # round1 VF references f0_SL; round2 VF references f1_SL.
        vf_stages = [s for s in stages if s.kind == VF]
        self.assertEqual(vf_stages[0].reference_model, "f0_SL")
        self.assertEqual(vf_stages[1].reference_model, "f1_SL")

    def test_sl_has_no_reference(self):
        for s in build_schedule():
            if s.kind == SL:
                self.assertIsNone(s.reference_model)

    def test_epoch_defaults(self):
        stages = build_schedule()
        self.assertEqual(stages[0].epochs, 40)          # initial SL
        by = epochs_by_kind(stages)
        self.assertEqual(by[VF], 5 * 5)                 # 5 rounds x 5 epochs
        self.assertEqual(by[SL], 40 + 5 * 1)            # init + 5 x 1
        self.assertEqual(total_epochs(stages), 40 + 25 + 5)

    def test_custom_epochs(self):
        stages = build_schedule(num_rounds=1, init_sl_epochs=10,
                                vf_epochs=3, sl_epochs=2)
        self.assertEqual([s.epochs for s in stages], [10, 3, 2])

    def test_bad_args_rejected(self):
        with self.assertRaises(ValueError):
            build_schedule(num_rounds=-1)
        with self.assertRaises(ValueError):
            build_schedule(vf_epochs=0)
        with self.assertRaises(ValueError):
            build_schedule(init_sl_epochs=0)

    def test_deterministic(self):
        self.assertEqual(build_schedule(num_rounds=3), build_schedule(num_rounds=3))


class TestValidateSchedule(unittest.TestCase):
    def test_default_valid(self):
        self.assertTrue(validate_schedule(build_schedule()))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            validate_schedule([])

    def test_must_start_with_sl(self):
        bad = [Stage(kind=VF, round_index=1, epochs=5,
                     input_model="a", output_model="b", reference_model=None)]
        with self.assertRaises(ValueError):
            validate_schedule(bad)

    def test_adjacent_vf_rejected(self):
        bad = [
            Stage(SL, 0, 40, "pretrained", "f0_SL"),
            Stage(VF, 1, 5, "f0_SL", "f1_VF", reference_model="f0_SL"),
            Stage(VF, 2, 5, "f1_VF", "f2_VF", reference_model="f0_SL"),
        ]
        with self.assertRaises(ValueError):
            validate_schedule(bad)

    def test_broken_chain_rejected(self):
        bad = [
            Stage(SL, 0, 40, "pretrained", "f0_SL"),
            Stage(VF, 1, 5, "WRONG", "f1_VF", reference_model="f0_SL"),
        ]
        with self.assertRaises(ValueError):
            validate_schedule(bad)

    def test_wrong_reference_rejected(self):
        bad = [
            Stage(SL, 0, 40, "pretrained", "f0_SL"),
            Stage(VF, 1, 5, "f0_SL", "f1_VF", reference_model="pretrained"),
        ]
        with self.assertRaises(ValueError):
            validate_schedule(bad)


if __name__ == "__main__":
    unittest.main()
