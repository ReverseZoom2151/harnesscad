"""Tests for the curriculum ordering package.

These assert REAL orderings on the REAL pressure corpus -- a plate before an
l_bracket before a flange -- not "the function returned a list", and they pin the
determinism the rest of the repo depends on: the same collection, and a shuffled
copy of it, must produce byte-identical sequences.
"""

from __future__ import annotations

import contextlib
import io
import unittest

from harnesscad.core.cisp import ops as cisp_ops
from harnesscad.eval.curriculum import complexity, difficulty, ordering, validate


def _sk(plane="XY"):
    return {"op": "new_sketch", "plane": plane}


def _rect(w=10.0, h=10.0):
    return {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": w, "h": h}


def _circ(r=5.0):
    return {"op": "add_circle", "sketch": "sk1", "cx": 0.0, "cy": 0.0, "r": r}


def _ext(d=5.0):
    return {"op": "extrude", "sketch": "sk1", "distance": d}


def _hole(x=0.0, y=0.0, dia=4.0):
    return {"op": "hole", "face_or_sketch": "solid", "x": x, "y": y,
            "diameter": dia, "through": True}


PLATE = (_sk(), _rect(60, 40), _ext(5))
PLATE_HOLE = (_sk(), _rect(60, 40), _ext(12), _hole())
TWO_BODY = (_sk(), _rect(60, 40), _ext(5), _sk(), _rect(20, 20), _ext(20),
            {"op": "boolean", "kind": "union", "target": "s1", "tool": "s2"})


def _briefs():
    from harnesscad.eval.pressure.briefs import BRIEFS

    return BRIEFS


def _by_id(briefs):
    return {b.id: b for b in briefs}


# --------------------------------------------------------------------------- #
# the metric
# --------------------------------------------------------------------------- #
class TestComplexityFeatures(unittest.TestCase):
    def test_counts_are_what_the_docstring_says(self):
        f = complexity.features(PLATE_HOLE)
        self.assertEqual(f.op_count, 4)
        self.assertEqual(f.distinct_op_types, 4)
        self.assertEqual(f.curve_count, 4)          # a rectangle is four curves
        self.assertEqual(f.constraint_count, 0)
        self.assertEqual(f.sketch_count, 1)
        self.assertEqual(f.feature_count, 2)        # extrude + hole
        self.assertEqual(f.feature_depth, 2)
        self.assertEqual(f.max_op_tier, 2)          # hole is tier 2

    def test_curve_count_is_recads_number_of_curves(self):
        self.assertEqual(complexity.curve_count((_rect(),)), 4)
        self.assertEqual(complexity.curve_count((_circ(),)), 1)
        self.assertEqual(
            complexity.curve_count(
                ({"op": "add_polygon", "sketch": "sk1",
                  "points": (0, 0, 1, 0, 1, 1, 0, 1, 0, 2)},)),
            5,
        )
        self.assertEqual(
            complexity.curve_count(({"op": "add_point", "sketch": "sk1"},)), 0)

    def test_feature_depth_is_tree_depth_not_feature_count(self):
        """Two extrusions unioned is DEPTH 2; one extrusion with two holes is 3.

        Both streams contain three solid ops, so a naive feature COUNT scores
        them identically. The tree depth is the thing that says how many chances
        the geometry has to go wrong downstream of a correct prefix.
        """
        two_holes = PLATE + (_hole(-10, 0), _hole(10, 0))
        self.assertEqual(complexity.features(TWO_BODY).feature_count, 3)
        self.assertEqual(complexity.features(two_holes).feature_count, 3)
        self.assertEqual(complexity.feature_depth(TWO_BODY), 2)
        self.assertEqual(complexity.feature_depth(two_holes), 3)

    def test_score_is_the_declared_weighted_sum(self):
        f = complexity.features(PLATE)
        expected = (
            1.0 * f.op_count
            + 1.0 * f.distinct_op_types
            + 0.25 * f.curve_count
            + 0.5 * f.constraint_count
            + 2.0 * f.feature_depth
            + 1.0 * f.max_op_tier
        )
        self.assertAlmostEqual(f.score, expected, places=9)
        self.assertAlmostEqual(complexity.score(PLATE), expected, places=9)

    def test_score_is_monotone_in_added_work(self):
        self.assertLess(complexity.score(PLATE), complexity.score(PLATE_HOLE))
        self.assertLess(complexity.score(PLATE_HOLE), complexity.score(TWO_BODY))

    def test_metric_is_deterministic_and_side_effect_free(self):
        first = complexity.features(PLATE_HOLE)
        second = complexity.features(list(PLATE_HOLE))
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_empty_stream_scores_zero(self):
        f = complexity.features(())
        self.assertEqual(f.op_count, 0)
        self.assertEqual(f.feature_depth, 0)
        self.assertEqual(f.score, 0.0)


class TestOpTables(unittest.TestCase):
    def test_every_cisp_op_has_a_declared_tier(self):
        """A new op must not slip into the DEFAULT_TIER bucket unnoticed."""
        missing = sorted(set(cisp_ops._REGISTRY) - set(complexity.OP_TIER))
        self.assertEqual(
            missing, [],
            "these CISP ops have no OP_TIER entry and would silently score as "
            "DEFAULT_TIER: %r" % (missing,))

    def test_no_tier_entry_names_a_nonexistent_op(self):
        stray = sorted(set(complexity.OP_TIER) - set(cisp_ops._REGISTRY))
        self.assertEqual(stray, [], "OP_TIER names ops that do not exist: %r" % (stray,))

    def test_solid_op_classes_are_disjoint(self):
        self.assertFalse(complexity.GENERATOR_OPS & complexity.MODIFIER_OPS)
        self.assertFalse(complexity.GENERATOR_OPS & complexity.COMBINER_OPS)
        self.assertFalse(complexity.COMBINER_OPS & complexity.MODIFIER_OPS)

    def test_op_tag_reads_all_three_op_shapes(self):
        self.assertEqual(complexity.op_tag({"op": "extrude"}), "extrude")
        self.assertEqual(complexity.op_tag(cisp_ops.Extrude()), "extrude")
        self.assertEqual(complexity.op_tag("extrude"), "extrude")

    def test_typed_ops_score_the_same_as_their_dicts(self):
        typed = (cisp_ops.NewSketch(), cisp_ops.AddRectangle(sketch="sk1", w=60, h=40),
                 cisp_ops.Extrude(sketch="sk1", distance=5.0))
        self.assertEqual(complexity.features(typed), complexity.features(PLATE))


# --------------------------------------------------------------------------- #
# ordering
# --------------------------------------------------------------------------- #
class TestStructuralLevel(unittest.TestCase):
    def test_levels_follow_recads_hierarchy(self):
        self.assertEqual(ordering.structural_level((_rect(),)), "L")
        self.assertEqual(ordering.structural_level((_sk(), _rect())), "F")
        self.assertEqual(ordering.structural_level((_sk(), _rect(), _circ())), "S")
        self.assertEqual(ordering.structural_level(PLATE), "SE")
        self.assertEqual(ordering.structural_level(TWO_BODY), "MSE")

    def test_level_ranks_are_the_papers_order(self):
        ranks = [ordering.level_rank(x) for x in
                 ((_rect(),), (_sk(), _rect()), (_sk(), _rect(), _circ()),
                  PLATE, TWO_BODY)]
        self.assertEqual(ranks, [0, 1, 2, 3, 4])

    def test_pressure_two_extrude_briefs_are_mse(self):
        by_id = _by_id(_briefs())
        for bid in ("l_bracket", "step_block", "slotted_block"):
            self.assertEqual(ordering.structural_level(by_id[bid]), "MSE", bid)
        self.assertEqual(ordering.structural_level(by_id["plate_60x40x5"]), "SE")


class TestFlatOrdering(unittest.TestCase):
    def setUp(self):
        self.briefs = list(_briefs())
        self.order = [b.id for b in ordering.order_tasks(self.briefs)]

    def test_a_plate_comes_before_an_l_bracket_before_a_flange(self):
        pos = {bid: i for i, bid in enumerate(self.order)}
        self.assertLess(pos["plate_60x40x5"], pos["l_bracket"])
        self.assertLess(pos["l_bracket"], pos["flange_square"])
        self.assertLess(pos["plate_square_25"], pos["plate_hole_centre"])
        self.assertLess(pos["plate_hole_centre"], pos["plate_hole_four"])
        self.assertLess(pos["disc_d30_h8"], pos["disc_bore"])

    def test_the_five_tier_one_plates_open_the_curriculum(self):
        self.assertEqual(
            set(self.order[:5]),
            {"disc_d30_h8", "bar_100x10x10", "plate_60x40x5",
             "plate_square_25", "plate_thin_80x50x2"},
        )

    def test_the_two_most_complex_briefs_close_it(self):
        self.assertEqual(set(self.order[-2:]), {"flange_round", "flange_square"})

    def test_order_is_deterministic_across_repeated_calls(self):
        again = [b.id for b in ordering.order_tasks(list(_briefs()))]
        self.assertEqual(self.order, again)
        third = [b.id for b in ordering.order_tasks(list(_briefs()))]
        self.assertEqual(self.order, third)

    def test_order_is_independent_of_input_order(self):
        """A reversed collection must produce the identical curriculum.

        This is the property a stable sort alone does NOT give: ties would keep
        their incoming order. The trailing task-id tie-break is what makes the
        key a total order.
        """
        reversed_in = list(reversed(self.briefs))
        self.assertEqual(
            [b.id for b in ordering.order_tasks(reversed_in)], self.order)
        rotated = self.briefs[7:] + self.briefs[:7]
        self.assertEqual([b.id for b in ordering.order_tasks(rotated)], self.order)

    def test_score_is_non_decreasing_along_the_curriculum(self):
        scores = [complexity.task_score(b)
                  for b in ordering.order_tasks(self.briefs)]
        self.assertEqual(scores, sorted(scores))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ordering.order_tasks(self.briefs, mode="by_vibes")


class TestHierarchicalOrdering(unittest.TestCase):
    def setUp(self):
        self.briefs = list(_briefs())

    def test_stages_walk_the_hierarchy_low_to_high(self):
        levels = [level for level, _members in ordering.stages(self.briefs)]
        self.assertEqual(levels, sorted(levels, key=ordering.PRIMITIVE_ORDER.index))
        self.assertEqual(levels, ["SE", "MSE"])

    def test_every_brief_lands_in_exactly_one_stage(self):
        seen = [b.id for _lvl, members in ordering.stages(self.briefs)
                for b in members]
        self.assertEqual(sorted(seen), sorted(b.id for b in self.briefs))

    def test_within_a_stage_curve_count_is_non_decreasing(self):
        """ReCAD's within-level rule: order by the number of curves."""
        for _level, members in ordering.stages(self.briefs):
            curves = [complexity.task_features(b).curve_count for b in members]
            self.assertEqual(curves, sorted(curves))

    def test_mse_briefs_come_after_every_se_brief(self):
        order = [b.id for b in ordering.order_tasks(self.briefs, mode="hierarchical")]
        by_id = _by_id(self.briefs)
        last_se = max(i for i, bid in enumerate(order)
                      if ordering.structural_level(by_id[bid]) == "SE")
        first_mse = min(i for i, bid in enumerate(order)
                        if ordering.structural_level(by_id[bid]) == "MSE")
        self.assertLess(last_se, first_mse)

    def test_hierarchical_order_is_deterministic(self):
        a = [b.id for b in ordering.order_tasks(self.briefs, mode="hierarchical")]
        b = [x.id for x in ordering.order_tasks(list(reversed(self.briefs)),
                                                mode="hierarchical")]
        self.assertEqual(a, b)

    def test_batches_partition_the_curriculum_easiest_first(self):
        chunks = ordering.batches(self.briefs, 5)
        self.assertEqual(sum(len(c) for c in chunks), len(self.briefs))
        self.assertEqual([len(c) for c in chunks], [5, 5, 5, 5, 5, 3])
        flat = [b.id for c in chunks for b in c]
        self.assertEqual(flat, [b.id for b in ordering.order_tasks(self.briefs)])
        with self.assertRaises(ValueError):
            ordering.batches(self.briefs, 0)

    def test_order_table_is_a_reportable_row_per_task(self):
        rows = ordering.order_table(self.briefs)
        self.assertEqual(len(rows), len(self.briefs))
        self.assertEqual([r["position"] for r in rows], list(range(len(rows))))
        self.assertEqual(rows[0]["id"], "disc_d30_h8")
        self.assertIn("level", rows[0])


class TestTaskAdapters(unittest.TestCase):
    def test_raw_op_streams_dicts_and_objects_all_work(self):
        self.assertEqual(complexity.task_ops(PLATE), PLATE)
        self.assertEqual(complexity.task_ops({"id": "x", "ops": list(PLATE)}), PLATE)

        class _Task(object):
            id = "y"
            op_stream = PLATE

        self.assertEqual(complexity.task_ops(_Task()), PLATE)
        self.assertEqual(complexity.task_id(_Task()), "y")

    def test_a_pressure_brief_is_read_through_its_reference(self):
        by_id = _by_id(_briefs())
        brief = by_id["plate_60x40x5"]
        self.assertEqual(complexity.task_ops(brief), brief.reference)
        self.assertEqual(complexity.task_id(brief), "plate_60x40x5")

    def test_a_task_with_no_ops_sorts_first_rather_than_raising(self):
        empty = {"id": "empty"}
        seq = ordering.order_tasks([empty, {"id": "p", "ops": list(PLATE)}])
        self.assertEqual(complexity.task_id(seq[0]), "empty")


# --------------------------------------------------------------------------- #
# empirical difficulty
# --------------------------------------------------------------------------- #
class TestEmpiricalDifficulty(unittest.TestCase):
    def test_difficulty_is_one_minus_the_best_result(self):
        self.assertEqual(difficulty.empirical_difficulty([0.1, 0.9, 0.4]), 0.1)
        self.assertEqual(difficulty.empirical_difficulty([0.0, 0.0]), 1.0)
        self.assertIsNone(difficulty.empirical_difficulty([]))

    def test_pass_fail_verdicts_are_accepted_as_rewards(self):
        self.assertEqual(difficulty.as_reward(True), 1.0)
        self.assertEqual(difficulty.as_reward(False), 0.0)
        self.assertEqual(difficulty.empirical_difficulty([False, True, False]), 0.0)

    def test_rewards_are_clamped_into_the_unit_interval(self):
        self.assertEqual(difficulty.as_reward(1.7), 1.0)
        self.assertEqual(difficulty.as_reward(-3.0), 0.0)

    def test_max_not_mean_decides(self):
        """A task solved once is a task the policy can solve."""
        once = difficulty.TaskOutcome("t", (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(once.difficulty, 0.0)
        self.assertAlmostEqual(once.mean_reward, 0.25)
        self.assertFalse(once.is_hard())


class TestDifficultyLedger(unittest.TestCase):
    def setUp(self):
        self.ledger = difficulty.DifficultyLedger()
        self.ledger.extend("easy", [0.9, 1.0, 0.95])
        self.ledger.extend("medium", [0.5, 0.79, 0.2])
        self.ledger.extend("hopeless", [0.0, 0.1, 0.05])

    def test_hard_tasks_is_recads_max_reward_below_h_rule(self):
        self.assertEqual(self.ledger.hard_tasks(0.8), ("hopeless", "medium"))
        self.assertEqual(self.ledger.hard_tasks(0.05), ())
        # The rule is a STRICT inequality, so a task solved perfectly once is
        # never hard, not even at threshold 1.0.
        self.assertEqual(self.ledger.hard_tasks(1.0), ("hopeless", "medium"))
        self.assertEqual(self.ledger.hard_tasks(0.96), ("hopeless", "medium"))

    def test_default_threshold_is_the_papers_zero_point_eight(self):
        self.assertEqual(difficulty.HARD_THRESHOLD, 0.8)
        self.assertEqual(self.ledger.hard_tasks(), self.ledger.hard_tasks(0.8))

    def test_hard_tasks_are_ordered_hardest_first(self):
        self.assertEqual(self.ledger.hard_tasks(0.8)[0], "hopeless")

    def test_an_unobserved_task_is_unmeasured_not_hard(self):
        self.assertIsNone(self.ledger.difficulty("never_run"))
        self.assertFalse(self.ledger.outcome("never_run").is_hard())
        self.assertNotIn("never_run", self.ledger.hard_tasks(0.8))

    def test_ledger_is_deterministic_regardless_of_record_order(self):
        other = difficulty.DifficultyLedger()
        other.extend("hopeless", [0.05, 0.1, 0.0])
        other.extend("easy", [1.0, 0.95, 0.9])
        other.extend("medium", [0.2, 0.79, 0.5])
        self.assertEqual(other.hard_tasks(0.8), self.ledger.hard_tasks(0.8))
        self.assertEqual([o.task_id for o in other.outcomes()],
                         [o.task_id for o in self.ledger.outcomes()])

    def test_pass_rate_counts_samples_that_cleared_the_bar(self):
        self.assertAlmostEqual(self.ledger.outcome("easy").pass_rate(0.8), 1.0)
        self.assertAlmostEqual(self.ledger.outcome("medium").pass_rate(0.8), 0.0)

    def test_seeded_from_a_mapping(self):
        seeded = difficulty.DifficultyLedger({"a": [0.1], "b": [1.0]})
        self.assertEqual(seeded.hard_tasks(0.8), ("a",))
        self.assertEqual(len(seeded), 2)


class TestMeasuredReordering(unittest.TestCase):
    def setUp(self):
        self.briefs = list(_briefs())

    def test_empty_ledger_reduces_to_the_structural_order(self):
        empty = difficulty.DifficultyLedger()
        self.assertEqual(
            [b.id for b in difficulty.order_by_measured(self.briefs, empty)],
            [b.id for b in ordering.order_tasks(self.briefs)],
        )

    def test_measured_traps_are_promoted_past_their_structural_twins(self):
        """The traps are the whole point of the empirical channel.

        ``trap_shell_too_thick`` is four ops -- structurally the same brief as
        ``shell_box_3mm``, which is why the static metric cannot separate them.
        Feed in outcomes and the trap moves to the end of the curriculum.
        """
        ledger = difficulty.DifficultyLedger()
        for b in self.briefs:
            ledger.extend(b.id, [0.0, 0.1] if b.trap else [0.95, 1.0])
        order = [b.id for b in difficulty.order_by_measured(self.briefs, ledger)]
        traps = {b.id for b in self.briefs if b.trap}
        self.assertEqual(set(order[-len(traps):]), traps)
        self.assertEqual(ledger.hard_tasks(0.8), tuple(sorted(traps)))

    def test_measured_reordering_is_deterministic(self):
        ledger = difficulty.DifficultyLedger()
        for b in self.briefs:
            ledger.extend(b.id, [0.0, 0.1] if b.trap else [0.95, 1.0])
        a = [b.id for b in difficulty.order_by_measured(self.briefs, ledger)]
        b = [x.id for x in difficulty.order_by_measured(
            list(reversed(self.briefs)), ledger)]
        self.assertEqual(a, b)

    def test_partially_observed_corpora_stay_totally_ordered(self):
        ledger = difficulty.DifficultyLedger({"trap_hole_oversize": [0.0]})
        order = [b.id for b in difficulty.order_by_measured(self.briefs, ledger)]
        self.assertEqual(len(order), len(self.briefs))
        self.assertEqual(order[-1], "trap_hole_oversize")


# --------------------------------------------------------------------------- #
# agreement with the hand-assigned difficulty column
# --------------------------------------------------------------------------- #
class TestAgreementWithHandLabels(unittest.TestCase):
    """The sanity check, and the finding it produced.

    On the 23 non-trap briefs the computed complexity reproduces the author's
    difficulty column closely. Over all 28 it does not, and the gap is not a bug
    in the metric: a trap brief is labelled 4 because its stated dimensions are
    infeasible, and no count over its op stream can see that. These tests pin
    both halves so a future reweighting cannot quietly "fix" the second one by
    fitting the label.
    """

    def setUp(self):
        self.report = validate.report()

    def test_non_trap_agreement_is_strong(self):
        rho = self.report["ablations"]["full"]["non_trap"]["spearman"]
        self.assertGreater(rho, 0.9, "non-trap Spearman fell to %.4f" % rho)

    def test_the_traps_are_what_breaks_the_overall_correlation(self):
        full = self.report["ablations"]["full"]
        self.assertLess(full["all"]["spearman"], 0.75)
        self.assertGreater(full["non_trap"]["spearman"], full["all"]["spearman"] + 0.2)

    def test_traps_score_identically_to_their_non_trap_twins(self):
        by_id = {r["id"]: r for r in self.report["rows"]}
        self.assertEqual(by_id["trap_shell_too_thick"]["score"],
                         by_id["shell_box_3mm"]["score"])
        self.assertEqual(by_id["trap_fillet_too_big"]["score"],
                         by_id["fillet_plate_3mm"]["score"])
        self.assertEqual(by_id["trap_hole_oversize"]["score"],
                         by_id["plate_hole_centre"]["score"])
        # ...and yet the hand labels differ by two whole tiers.
        self.assertEqual(by_id["trap_shell_too_thick"]["labelled_difficulty"], 4)
        self.assertEqual(by_id["shell_box_3mm"]["labelled_difficulty"], 2)

    def test_every_ablation_is_reported(self):
        self.assertEqual(sorted(self.report["ablations"]), sorted(validate.ABLATIONS))
        for entry in self.report["ablations"].values():
            for half in ("all", "non_trap"):
                self.assertIn("kendall_tau_b", entry[half])
                self.assertIn("discordant_pairs", entry[half])

    def test_report_is_deterministic(self):
        self.assertEqual(validate.report(), self.report)

    def test_render_produces_a_table(self):
        text = validate.render(self.report)
        self.assertIn("plate_60x40x5", text)
        self.assertIn("op_count_only", text)
        self.assertTrue(all(ord(ch) < 128 for ch in text))

    def test_main_runs(self):
        # Captured: the CLI prints the whole table, and the suite is not the
        # place for it.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = validate.main([])
        self.assertEqual(code, 0)
        self.assertIn("op_count_only", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
