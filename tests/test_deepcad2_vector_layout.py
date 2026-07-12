"""Tests for the DeepCAD 17-column dataset vector layout."""

import unittest

from reconstruction import deepcad2_vector_layout as vl


class TestConstants(unittest.TestCase):
    def test_reference_command_order(self):
        self.assertEqual(vl.ALL_COMMANDS,
                         ("Line", "Arc", "Circle", "EOS", "SOL", "Ext"))
        self.assertEqual((vl.LINE_IDX, vl.ARC_IDX, vl.CIRCLE_IDX,
                          vl.EOS_IDX, vl.SOL_IDX, vl.EXT_IDX), (0, 1, 2, 3, 4, 5))

    def test_arg_widths(self):
        self.assertEqual(vl.N_ARGS_EXT, 11)
        self.assertEqual(vl.N_ARGS, 16)
        self.assertEqual(vl.ROW_LEN, 17)
        self.assertEqual(len(vl.ARG_NAMES), 16)

    def test_sentinel_rows(self):
        self.assertEqual(vl.SOL_VEC[0], vl.SOL_IDX)
        self.assertEqual(set(vl.SOL_VEC[1:]), {-1})
        self.assertEqual(vl.EOS_VEC[0], vl.EOS_IDX)
        self.assertEqual(len(vl.EOS_VEC), 17)

    def test_mask_shape_and_counts(self):
        self.assertEqual(len(vl.CMD_ARGS_MASK), 6)
        counts = [sum(m) for m in vl.CMD_ARGS_MASK]
        self.assertEqual(counts, [2, 4, 3, 0, 0, 11])


class TestRows(unittest.TestCase):
    def test_line_row(self):
        row = vl.line_row(10, 20)
        self.assertEqual(row[0], vl.LINE_IDX)
        self.assertEqual(row[1:3], (10, 20))
        self.assertEqual(set(row[3:]), {-1})
        vl.validate_row(row)

    def test_arc_row_slots(self):
        row = vl.arc_row(4, 5, 64, 1)
        self.assertEqual(vl.used_args(row), {"x": 4, "y": 5, "alpha": 64, "f": 1})

    def test_circle_row_uses_radius_slot_not_alpha(self):
        row = vl.circle_row(128, 128, 30)
        self.assertEqual(row[3], -1)   # alpha
        self.assertEqual(row[4], -1)   # f
        self.assertEqual(row[5], 30)   # r
        self.assertEqual(vl.used_args(row), {"x": 128, "y": 128, "r": 30})

    def test_ext_row_leaves_sketch_slots_empty(self):
        row = vl.ext_row(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0)
        self.assertEqual(row[1:6], (-1, -1, -1, -1, -1))
        self.assertEqual(row[6:], (1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0))
        self.assertEqual(len(vl.used_args(row)), 11)
        vl.validate_row(row)

    def test_validate_row_rejects_stray_value(self):
        bad = list(vl.line_row(1, 2))
        bad[5] = 7  # radius on a Line
        with self.assertRaises(ValueError):
            vl.validate_row(bad)

    def test_validate_row_rejects_bad_width(self):
        with self.assertRaises(ValueError):
            vl.validate_row((0, 1, 2))


class TestLoopAssembly(unittest.TestCase):
    def test_loop_has_sol_prefix_and_eos_suffix(self):
        rows = vl.loop_vector([vl.line_row(1, 1), vl.line_row(2, 2)])
        self.assertEqual([r[0] for r in rows],
                         [vl.SOL_IDX, vl.LINE_IDX, vl.LINE_IDX, vl.EOS_IDX])

    def test_loop_padding(self):
        rows = vl.loop_vector([vl.line_row(1, 1)], max_len=6)
        self.assertEqual(len(rows), 6)
        self.assertEqual([r[0] for r in rows[2:]], [vl.EOS_IDX] * 4)

    def test_over_budget_loop_is_none(self):
        curves = [vl.line_row(i, i) for i in range(20)]
        self.assertIsNone(vl.loop_vector(curves, max_len=15))


class TestProfileAssembly(unittest.TestCase):
    def _loops(self):
        return [[vl.line_row(1, 1), vl.line_row(2, 2)], [vl.circle_row(5, 5, 2)]]

    def test_unpadded_profile_structure(self):
        rows = vl.profile_vector(self._loops(), pad=False)
        self.assertEqual([r[0] for r in rows],
                         [vl.SOL_IDX, vl.LINE_IDX, vl.LINE_IDX,
                          vl.SOL_IDX, vl.CIRCLE_IDX, vl.EOS_IDX])

    def test_padded_profile_length(self):
        rows = vl.profile_vector(self._loops(), pad=True)
        self.assertEqual(len(rows), vl.MAX_N_LOOPS * vl.MAX_N_CURVES)  # 90

    def test_too_many_loops_is_none(self):
        loops = [[vl.circle_row(i, i, 1)] for i in range(7)]
        self.assertIsNone(vl.profile_vector(loops))

    def test_too_long_loop_is_none(self):
        loops = [[vl.line_row(i, i) for i in range(20)]]
        self.assertIsNone(vl.profile_vector(loops))


class TestExtrudeAndSequence(unittest.TestCase):
    def _ext(self):
        return vl.ext_row(128, 128, 128, 128, 128, 128, 100, 200, 128, 0, 0)

    def test_ext_row_precedes_the_eos(self):
        rows = vl.extrude_vector([[vl.circle_row(4, 4, 2)]], self._ext(), pad=False)
        self.assertEqual([r[0] for r in rows],
                         [vl.SOL_IDX, vl.CIRCLE_IDX, vl.EXT_IDX, vl.EOS_IDX])

    def test_cad_vector_has_single_trailing_eos_before_padding(self):
        loops = [[vl.circle_row(4, 4, 2)]]
        vec = vl.cad_vector([(loops, self._ext()), (loops, self._ext())], pad=True)
        self.assertEqual(len(vec), vl.MAX_TOTAL_LEN)
        cmds = [r[0] for r in vec]
        self.assertEqual(cmds[:6], [vl.SOL_IDX, vl.CIRCLE_IDX, vl.EXT_IDX,
                                    vl.SOL_IDX, vl.CIRCLE_IDX, vl.EXT_IDX])
        self.assertEqual(set(cmds[6:]), {vl.EOS_IDX})
        self.assertEqual(vl.sequence_length(vec), 6)

    def test_too_many_extrudes_is_none(self):
        loops = [[vl.circle_row(4, 4, 2)]]
        self.assertIsNone(vl.cad_vector([(loops, self._ext())] * 11))

    def test_every_row_is_well_formed(self):
        loops = [[vl.line_row(1, 1), vl.arc_row(2, 2, 64, 1)], [vl.circle_row(5, 5, 2)]]
        vec = vl.cad_vector([(loops, self._ext())])
        for row in vec:
            vl.validate_row(row)
            self.assertEqual(len(row), 17)


class TestDisassembly(unittest.TestCase):
    def _vec(self):
        loops_a = [[vl.line_row(1, 1), vl.line_row(2, 2)]]
        loops_b = [[vl.circle_row(9, 9, 3)], [vl.circle_row(1, 1, 1)]]
        ext = vl.ext_row(1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0)
        return vl.cad_vector([(loops_a, ext), (loops_b, ext)])

    def test_split_extrudes(self):
        parts = vl.split_extrudes(self._vec())
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][-1][0], vl.EXT_IDX)
        self.assertEqual([r[0] for r in parts[1]],
                         [vl.SOL_IDX, vl.CIRCLE_IDX, vl.SOL_IDX,
                          vl.CIRCLE_IDX, vl.EXT_IDX])

    def test_split_loops_of_second_extrude(self):
        parts = vl.split_extrudes(self._vec())
        loops = vl.split_loops(parts[1])
        self.assertEqual(len(loops), 2)
        self.assertTrue(all(lp[0][0] == vl.SOL_IDX for lp in loops))
        self.assertEqual(loops[0][1][1:3], (9, 9))

    def test_split_loops_drops_empty_loop(self):
        vec = [vl.sol_row(), vl.sol_row(), vl.line_row(3, 3), vl.eos_row()]
        loops = vl.split_loops(vec)
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0][1][1:3], (3, 3))

    def test_trim_eos_strips_padding(self):
        vec = self._vec()
        self.assertEqual(len(vl.trim_eos(vec)), vl.sequence_length(vec))
        self.assertTrue(all(r[0] != vl.EOS_IDX for r in vl.trim_eos(vec)))


if __name__ == "__main__":
    unittest.main()
