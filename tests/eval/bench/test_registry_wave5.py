"""Scoring tests for the fifth adapter wave of the bench registry.

Two groups land in this wave:

*   **Process / stability** -- ReCAD's Invalidity Ratio (arXiv:2512.06328) with
    CAD-RL's Executability (arXiv:2508.10118) as its complement, and the linked
    SUC / Pass@1 / AVG Re triple (arXiv:2605.19748, Table 1).
*   **Set-level generative** -- RLCAD's COV / MMD-CD / JSD (arXiv:2503.18549,
    Tables 1/4/5/6) plus the Diffusion-CAD set ratios that were sitting in the
    unadapted ledger.

Same contract as waves three and four: every metric is materialised, leaves
``unadapted()``, and is scored on a hand-built pred/gold pair whose expected
number is worked out by hand from the published definition. This wave also pins
the rival families it creates or extends -- there are now three different things
in this tree called an "invalidity" rate and two different things called
"coverage", and no suite may pool them.
"""

from __future__ import annotations

import unittest

from harnesscad.eval.bench import registry as bench


#: Every metric the fifth wave added, by name.
WAVE_FIVE = (
    "process.build_invalidity_ratio",
    "process.correction_budget",
    "generative.rlcad_set_metrics",
    "generative.brep_set_coverage_mmd",
    "generative.sequence_set_ratios",
)

#: The families this wave introduces or extends.
WAVE_FIVE_RIVALS = {
    "validity_rate": ("sequence.invalidity_ratio", "sequence.code_validity",
                      "process.build_invalidity_ratio"),
    "set_coverage": ("generative.rlcad_set_metrics",
                     "generative.brep_set_coverage_mmd"),
}


def _score(name, pred, gold):
    return bench.metric(name).score(pred, gold)


def P(x, y=0.0, z=0.0):
    """A one-point cloud; symmetric Chamfer between two of these is |a - b|."""
    return [(float(x), float(y), float(z))]


# -- fixtures ---------------------------------------------------------------

#: 8 generated outputs, 3 of which did not build -> IR 37.5 / Executability 62.5.
GENERATED_BUILDS = [
    {"id": "a", "built": True}, {"id": "b", "built": False, "error": "SyntaxError"},
    {"id": "c", "built": True}, {"id": "d", "built": False, "error": "SyntaxError"},
    {"id": "e", "built": True}, {"id": "f", "built": False, "error": "KernelError"},
    {"id": "g", "built": True}, {"id": "h", "built": True},
]
#: The reference corpus builds by construction -> reference IR 0.0.
REFERENCE_BUILDS = [{"id": "r%d" % i, "built": True} for i in range(4)]

_OK = {"executed": True, "valid": True}
_BROKEN = {"executed": True, "valid": False}
_CRASH = {"executed": False, "valid": False}

#: SUC 3/4, Pass@1 1/4, AVG Re (0 + 2 + 1) / 3 = 1.0.
RUN_TRACES = {
    "traces": [
        {"task_id": "t0", "attempts": [_OK]},
        {"task_id": "t1", "attempts": [_CRASH, _BROKEN, _OK]},
        {"task_id": "t2", "attempts": [_BROKEN, _OK]},
        {"task_id": "t3", "attempts": [_CRASH, _CRASH, _BROKEN]},
    ],
    "budget": None,
}
#: The same traces under a one-retry budget: SUC 0.5, Pass@1 0.25, AVG Re 0.5.
BASELINE_TRACES = dict(RUN_TRACES, budget=1)

#: reference {0, 10} vs generated {1, 2}: the shape at 10 is never covered.
GEN_CLUMP = [P(1.0), P(2.0)]
REF_SPLIT = [P(0.0), P(10.0)]

SOL, L, C, E, EOS = "<SOL>", "L", "C", "E", "<EOS>"
#: 4 generated sequences: one repeated twice, one seen in training, one ill-formed.
GEN_SEQS = [[SOL, L, L, L, E, EOS], [SOL, L, L, L, E, EOS],
            [SOL, C, E, EOS], [L, E, EOS]]
TRAIN_SEQS = [[SOL, C, E, EOS]]


class DiscoveryTest(unittest.TestCase):
    def test_every_new_metric_is_discovered_and_bound(self):
        known = {m.name: m for m in bench.metrics()}
        for name in WAVE_FIVE:
            self.assertIn(name, known, f"{name} was not discovered")
            self.assertNotIn(known[name].dotted, bench.unadapted(),
                             f"{name}'s module is still listed unadapted")
            self.assertIn(known[name].kind, bench.KINDS)
            for key in known[name].inputs:
                self.assertIn(key, bench.INPUT_KINDS, f"{name} needs unknown {key}")

    def test_process_is_a_first_class_kind(self):
        names = {m.name for m in bench.metrics(kind="process")}
        self.assertEqual(names, {"process.build_invalidity_ratio",
                                 "process.correction_budget"})
        self.assertIn("process", bench.kinds())

    def test_the_ledger_is_complete_in_both_directions(self):
        unadapted = set(bench.unadapted())
        stated = set(bench.reasons())
        self.assertTrue(unadapted)
        self.assertEqual(sorted(unadapted - stated), [],
                         "unadapted modules with no recorded reason")
        self.assertEqual(sorted(stated - unadapted), [],
                         "reasons naming modules that are in fact adapted")

    def test_the_two_bound_set_modules_left_the_ledger(self):
        for dotted in ("harnesscad.eval.bench.generative.brep_set_metrics",
                       "harnesscad.eval.bench.generative.sequence_set_ratios"):
            self.assertNotIn(dotted, bench.unadapted())

    def test_the_refusals_this_wave_kept_are_still_recorded(self):
        """Ambiguous or callable-dependent set metrics stay unadapted, with a why."""
        why = bench.reasons()
        for dotted in ("harnesscad.eval.bench.generative.prompt_similarity",
                       "harnesscad.eval.bench.generative.render_distribution",
                       "harnesscad.eval.bench.generative.text2cad_complexity",
                       "harnesscad.eval.bench.geometry.cd_tolerance_recall"):
            self.assertIn(dotted, bench.unadapted())
            self.assertTrue(why[dotted].strip())


class BuildInvalidityScoringTest(unittest.TestCase):
    """ReCAD IR / CAD-RL Executability, in percent, over the generated set."""

    def test_ir_and_executability_are_complements_on_one_population(self):
        value = _score("process.build_invalidity_ratio",
                       {"build_outcomes": GENERATED_BUILDS},
                       {"build_outcomes": REFERENCE_BUILDS})
        self.assertEqual(value["n_outputs"], 8)
        self.assertEqual(value["n_invalid"], 3)
        self.assertEqual(value["ir_percent"], 37.5)
        self.assertEqual(value["executability_percent"], 62.5)
        self.assertEqual(value["ir_ratio"], 0.375)

    def test_the_reference_population_is_reported_alongside(self):
        value = _score("process.build_invalidity_ratio",
                       {"build_outcomes": GENERATED_BUILDS},
                       {"build_outcomes": REFERENCE_BUILDS})
        self.assertEqual(value["reference_ir_percent"], 0.0)

    def test_a_flawless_run_scores_zero_not_one(self):
        value = _score("process.build_invalidity_ratio",
                       {"build_outcomes": REFERENCE_BUILDS},
                       {"build_outcomes": REFERENCE_BUILDS})
        self.assertEqual(value["ir_percent"], 0.0)
        self.assertEqual(value["executability_percent"], 100.0)

    def test_this_is_not_the_structural_invalidity_ratio(self):
        """Same name, different definition, different scale, different module."""
        build = bench.metric("process.build_invalidity_ratio")
        structural = bench.metric("sequence.invalidity_ratio")
        self.assertNotEqual(build.dotted, structural.dotted)
        self.assertNotEqual(build.inputs, structural.inputs)
        family = dict(bench.RIVAL_FAMILIES)["validity_rate"]
        self.assertIn(build.name, family)
        self.assertIn(structural.name, family)


class CorrectionBudgetScoringTest(unittest.TestCase):
    """SUC / Pass@1 / AVG Re, always reported as one linked triple."""

    def _value(self):
        return _score("process.correction_budget",
                      {"correction_traces": RUN_TRACES},
                      {"correction_traces": BASELINE_TRACES})

    def test_the_triple_is_scored_by_hand(self):
        value = self._value()
        self.assertEqual(value["n_tasks"], 4)
        self.assertEqual(value["suc"], 0.75)
        self.assertEqual(value["pass_at_1"], 0.25)
        self.assertEqual(value["avg_retries"], 1.0)

    def test_the_reference_run_is_scored_under_its_own_budget(self):
        value = self._value()
        self.assertEqual(value["reference_suc"], 0.5)
        self.assertEqual(value["reference_pass_at_1"], 0.25)
        self.assertEqual(value["reference_avg_retries"], 0.5)

    def test_more_retries_with_more_success_is_not_called_a_regression(self):
        value = self._value()
        self.assertEqual(value["suc_delta"], 0.25)
        self.assertEqual(value["avg_retries_delta"], 0.5)
        self.assertEqual(value["verdict"], "more_success_more_retries")
        self.assertTrue(value["note"].strip())

    def test_the_three_numbers_travel_together_through_the_runner(self):
        sample = {"id": "run-1",
                  "pred": {"correction_traces": RUN_TRACES},
                  "gold": {"correction_traces": BASELINE_TRACES}}
        report = bench.run_suite("process_stability", [sample])
        aggregates = report.aggregates()["process.correction_budget"]
        for key in ("suc", "pass_at_1", "avg_retries"):
            self.assertIn(key, aggregates)
        self.assertNotIn("verdict", aggregates)  # a string, not a number


class RlcadSetScoringTest(unittest.TestCase):
    """RLCAD COV / MMD-CD / JSD over a generated and a reference SET."""

    def test_a_perfect_generator_scores_cov_one_mmd_zero_jsd_zero(self):
        shapes = [P(0.1), P(-0.2), P(0.5)]
        value = _score("generative.rlcad_set_metrics",
                       {"point_sets": shapes}, {"point_sets": shapes})
        self.assertEqual(value["cov"], 1.0)
        self.assertEqual(value["mmd_cd"], 0.0)
        self.assertAlmostEqual(value["jsd"], 0.0, places=12)

    def test_cov_and_mmd_on_the_hand_worked_split(self):
        value = _score("generative.rlcad_set_metrics",
                       {"point_sets": GEN_CLUMP}, {"point_sets": REF_SPLIT})
        # both generated shapes are nearest to the reference at 0 -> 1 of 2.
        self.assertEqual(value["cov"], 0.5)
        # (min(1, 2) + min(9, 8)) / 2 = (1 + 8) / 2.
        self.assertAlmostEqual(value["mmd_cd"], 4.5, places=12)


class CoverageRivalTest(unittest.TestCase):
    """Two coverage directions, two different numbers, one input."""

    def test_the_two_coverage_protocols_disagree_by_design(self):
        rlcad = _score("generative.rlcad_set_metrics",
                       {"point_sets": GEN_CLUMP}, {"point_sets": REF_SPLIT})
        brep = _score("generative.brep_set_coverage_mmd",
                      {"point_sets": GEN_CLUMP}, {"point_sets": REF_SPLIT})
        self.assertEqual(rlcad["cov"], 0.5)
        self.assertEqual(brep["coverage_reference_nearest"], 1.0)
        self.assertNotEqual(rlcad["cov"], brep["coverage_reference_nearest"])

    def test_but_their_mmd_terms_agree_because_that_direction_is_shared(self):
        rlcad = _score("generative.rlcad_set_metrics",
                       {"point_sets": GEN_CLUMP}, {"point_sets": REF_SPLIT})
        brep = _score("generative.brep_set_coverage_mmd",
                      {"point_sets": GEN_CLUMP}, {"point_sets": REF_SPLIT})
        self.assertAlmostEqual(rlcad["mmd_cd"], brep["mmd_cd"], places=12)

    def test_the_rival_coverage_is_never_reported_as_plain_cov(self):
        brep = _score("generative.brep_set_coverage_mmd",
                      {"point_sets": GEN_CLUMP}, {"point_sets": REF_SPLIT})
        self.assertNotIn("cov", brep)
        self.assertIn("coverage_reference_nearest", brep)


class SetRatioScoringTest(unittest.TestCase):
    """Diffusion-CAD singleton-unique / novel / grammar-invalidity percentages."""

    def test_diffusion_cad_percentages_are_scored_by_hand(self):
        value = _score("generative.sequence_set_ratios",
                       {"token_sequences": GEN_SEQS},
                       {"token_sequences": TRAIN_SEQS})
        self.assertEqual(value["count"], 4.0)
        # only 2 of the 4 sequences occur exactly once.
        self.assertEqual(value["unique_pct"], 50.0)
        # 3 of the 4 are absent from the training set.
        self.assertEqual(value["novel_pct"], 75.0)
        # [L, E, EOS] opens geometry outside a loop -> 1 of 4 ill-formed.
        self.assertEqual(value["invalidity_pct"], 25.0)

    def test_the_rival_unique_definition_is_left_unbound_not_renamed(self):
        """brep_set_metrics.ratios also says "unique" and means something else.

        Its unique is |distinct signatures| / |generated| = 3/4 = 0.75, not the
        singleton rate 0.50 above, and its ``valid`` term defaults to a vacuous
        1.0 because the predicate is injected. It is therefore not bound, and no
        metric in the registry adapts it.
        """
        from harnesscad.eval.bench.generative import brep_set_metrics as brep
        theirs = brep.ratios(GEN_SEQS, TRAIN_SEQS, signature=tuple)
        self.assertEqual(theirs["unique"], 0.75)
        self.assertEqual(theirs["valid"], 1.0)   # vacuous: predicate defaulted
        mine = _score("generative.sequence_set_ratios",
                      {"token_sequences": GEN_SEQS},
                      {"token_sequences": TRAIN_SEQS})
        self.assertNotEqual(theirs["unique"], mine["unique_pct"] / 100.0)
        self.assertNotIn("generative.brep_set_ratios",
                         {m.name for m in bench.metrics()})


class RivalEnforcementTest(unittest.TestCase):
    def test_every_new_family_is_declared(self):
        families = bench.rivals()
        for family, members in WAVE_FIVE_RIVALS.items():
            self.assertIn(family, families)
            self.assertEqual(set(families[family]), set(members))

    def test_a_suite_definition_blending_new_rivals_is_refused(self):
        for family, members in WAVE_FIVE_RIVALS.items():
            self.assertTrue(bench._rival_conflicts(list(members)),
                            f"{family} is not enforced")

    def test_no_suite_selects_two_members_of_any_new_family(self):
        for name in bench.suites():
            chosen = set(bench.suite(name).metric_names)
            for family, members in WAVE_FIVE_RIVALS.items():
                self.assertLessEqual(len(chosen.intersection(members)), 1,
                                     f"suite {name!r} blends {family!r}")


class SuiteRunTest(unittest.TestCase):
    def test_the_new_suites_run_clean(self):
        sample = {
            "id": "s1",
            "pred": {"build_outcomes": GENERATED_BUILDS,
                     "correction_traces": RUN_TRACES,
                     "point_sets": GEN_CLUMP,
                     "token_sequences": GEN_SEQS,
                     "points": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]},
            "gold": {"build_outcomes": REFERENCE_BUILDS,
                     "correction_traces": BASELINE_TRACES,
                     "point_sets": REF_SPLIT,
                     "token_sequences": TRAIN_SEQS,
                     "points": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]},
        }
        for name in ("process_stability", "rlcad"):
            report = bench.run_suite(name, [sample])
            self.assertFalse(report.errors(), [r.error for r in report.errors()])
            self.assertTrue(report.aggregates(), name)

    def test_the_rlcad_suite_selects_the_rlcad_coverage_direction(self):
        chosen = bench.suite("rlcad").metric_names
        self.assertIn("generative.rlcad_set_metrics", chosen)
        self.assertNotIn("generative.brep_set_coverage_mmd", chosen)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
