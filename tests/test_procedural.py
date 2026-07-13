import unittest

from harnesscad.agents.exploration.technique_trials import (
    Placement,
    PlacementRules,
    ProceduralTechnique,
    TechniqueRegistry,
    derive_child_seeds,
    replay,
    run_trials,
    solution_space_coverage,
    validate_placements,
)


class TestTechniqueRegistry(unittest.TestCase):
    def test_weighted_selection_and_stable_tie_break(self):
        registry = TechniqueRegistry([
            ProceduralTechnique(
                "casting",
                {"precision": .5, "repeatability": .8, "cost": .9, "manufacturability": .9},
                frozenset({"metal"}),
            ),
            ProceduralTechnique(
                "machining",
                {"precision": 1, "repeatability": .9, "cost": .3, "manufacturability": .7},
                frozenset({"metal"}),
            ),
        ])
        ranked = registry.select({"precision": 3, "cost": 1}, required_tags={"metal"})
        self.assertEqual(ranked[0][0].name, "machining")

    def test_minimum_filter_and_validation(self):
        registry = TechniqueRegistry([
            ProceduralTechnique("a", {"precision": .4}),
            ProceduralTechnique("b", {"precision": .8}),
        ])
        self.assertEqual(
            [item.name for item, _ in registry.select(
                {"precision": 1}, minimums={"precision": .7}
            )],
            ["b"],
        )
        with self.assertRaises(ValueError):
            ProceduralTechnique("bad", {"magic": .5})


class TestTrials(unittest.TestCase):
    def test_seeds_are_stable_and_distinct(self):
        first = derive_child_seeds(42, 20)
        self.assertEqual(first, derive_child_seeds(42, 20))
        self.assertEqual(len(set(first)), 20)

    def test_winner_replays_exactly(self):
        def generate(seed):
            return {"seed": seed, "value": seed % 101}

        run = run_trials(
            generate, lambda result: result["value"], master_seed=8, attempts=12
        )
        self.assertIsNotNone(run.winning_seed)
        replayed = replay(generate, lambda result: result["value"], run.winning_seed)
        self.assertEqual(replayed.result, run.winning_result)
        self.assertEqual(replayed.score, run.winning_score)

    def test_failures_continue_and_are_diagnostic(self):
        seeds = derive_child_seeds(3, 3)

        def generate(seed):
            if seed == seeds[0]:
                raise RuntimeError("kernel rejected shape")
            return seed

        run = run_trials(generate, lambda value: value % 7, master_seed=3, attempts=3)
        self.assertEqual(run.attempts[0].status, "failed")
        self.assertIn("kernel rejected shape", run.attempts[0].diagnostic)
        self.assertIsNotNone(run.winning_seed)

    def test_injected_clock_marks_timeout(self):
        ticks = iter([0.0, 2.0, 2.0, 2.25])
        run = run_trials(
            lambda seed: seed,
            lambda value: 1,
            master_seed=1,
            attempts=2,
            timeout=1.0,
            clock=lambda: next(ticks),
        )
        self.assertEqual([attempt.status for attempt in run.attempts], ["timeout", "ok"])
        self.assertIn("exceeded", run.attempts[0].diagnostic)


class TestPlacement(unittest.TestCase):
    def test_valid_layout(self):
        placements = [
            Placement("motor", (0, 0), "drive"),
            Placement("gear", (2, 0), "drive"),
        ]
        rules = PlacementRules(
            adjacency=(("motor", "gear", 3),),
            cluster_radius=3,
            obstacles=(((10, 10), 2),),
            clearance=.5,
        )
        self.assertEqual(validate_placements(placements, rules), ())

    def test_reports_each_constraint_family(self):
        placements = [
            Placement("a", (0, 0), "g"),
            Placement("b", (5, 0), "g"),
        ]
        diagnostics = validate_placements(
            placements,
            PlacementRules(
                adjacency=(("a", "b", 2),),
                cluster_radius=3,
                obstacles=(((0, 0), .5),),
                clearance=.1,
            ),
        )
        self.assertEqual(len(diagnostics), 3)
        self.assertIn("adjacency", diagnostics[0])
        self.assertIn("cluster", diagnostics[1])
        self.assertIn("obstacle", diagnostics[2])


class TestCoverage(unittest.TestCase):
    def test_declared_space_coverage_and_diversity(self):
        report = solution_space_coverage(
            {"material": ["steel", "aluminum"], "holes": [2, 4]},
            [
                {"material": "steel", "holes": 2},
                {"material": "aluminum", "holes": 4},
                {"material": "steel", "holes": 2},
            ],
        )
        self.assertEqual(report.dimension_coverage, {"holes": 1.0, "material": 1.0})
        self.assertEqual(report.configuration_coverage, .5)
        self.assertEqual(report.unique_configurations, 2)
        self.assertEqual(report.diversity, 1.0)

    def test_rejects_undeclared_values(self):
        with self.assertRaises(ValueError):
            solution_space_coverage(
                {"material": ["steel"]},
                [{"material": "wood"}],
            )


if __name__ == "__main__":
    unittest.main()
