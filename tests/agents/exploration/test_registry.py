"""The search surface: discovery, a real search that improves, rivals, failures."""

import unittest

from harnesscad.agents.exploration import registry as X
from harnesscad.core.cli import DEMO_OPS
from harnesscad.io.surfaces.server import CISPServer


def session(**overrides):
    ops = [dict(op) for op in DEMO_OPS]
    if overrides:
        ops[1].update(overrides)
    server = CISPServer(backend="stub")
    result = server.applyOps(ops)
    assert result["ok"], result
    return server.session


#: Every strategy that searches a session (block_decomp targets a polygon).
SESSION_STRATEGIES = tuple(n for n in X.strategies()
                           if X.strategy(n).target == "session")


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_five_real_exploration_modules(self):
        from harnesscad import registry as capability_registry

        indexed = {e.dotted for e in capability_registry.find(package="exploration")}
        bound = set()
        for name in X.strategies():
            bound.update(X.strategy(name).modules)
        self.assertGreater(len(bound), 5)
        for dotted in bound:
            self.assertIn(dotted, indexed)      # nothing invented

    def test_every_strategy_is_described_and_targeted(self):
        self.assertGreaterEqual(len(X.strategies()), 8)
        for name in X.strategies():
            s = X.strategy(name)
            self.assertTrue(s.description)
            self.assertIn(s.target, ("session", "polygon"))

    def test_unadapted_modules_are_reported_not_hidden(self):
        # These need a generative model (text-to-3D / image / LEGO bricks) that this
        # repo does not carry, so they stay honestly orphaned.
        for dotted in ("harnesscad.agents.exploration.image_prompt_sweep",
                       "harnesscad.agents.exploration.prompt_encoding",
                       "harnesscad.agents.exploration.variant_consensus"):
            self.assertIn(dotted, X.unadapted())

    def test_unknown_strategy_raises(self):
        with self.assertRaises(X.UnknownStrategy):
            X.strategy("no-such-search")


class TestDesignSpace(unittest.TestCase):
    def test_the_space_is_the_shape_bearing_parameters_of_the_real_model(self):
        dims = X.space(session())
        names = {(d.index, d.param) for d in dims}
        self.assertIn((1, "w"), names)
        self.assertIn((1, "h"), names)
        self.assertIn((6, "distance"), names)
        # A constraint's value is not searchable: the objective cannot see it.
        self.assertNotIn((2, "value"), names)
        for d in dims:
            self.assertLessEqual(d.low, d.value)
            self.assertLessEqual(d.value, d.high)

    def test_a_design_vector_realises_into_a_real_applicable_op_stream(self):
        dims = X.space(session())
        ops = X.realise(session(), dims, [40.0, 25.0, 9.0])
        server = CISPServer(backend="stub")
        result = server.applyOps([op.to_dict() for op in ops])
        self.assertTrue(result["ok"], result)
        self.assertTrue(server.query("summary")["result"]["solid_present"])

    def test_a_model_with_no_shape_parameters_is_refused(self):
        empty = CISPServer(backend="stub").session
        with self.assertRaises(X.Unsupported):
            X.space(empty)


class TestSearchImproves(unittest.TestCase):
    """A search must actually move the objective -- that is the whole point."""

    def _objective(self):
        return X.shape_objective(session(w=40.0, h=25.0))

    def test_every_session_strategy_improves_the_objective(self):
        for name in SESSION_STRATEGIES:
            with self.subTest(strategy=name):
                result = X.search(name, session(), self._objective(), seed=3)
                self.assertNotIn("error", result.detail, result.detail.get("error"))
                self.assertGreater(result.evaluations, 1)
                self.assertTrue(result.improved,
                                "%s did not improve: %.3f -> %.3f"
                                % (name, result.start_score, result.best.score))
                self.assertLess(result.best.score, result.start_score)

    def test_the_best_so_far_history_is_monotone_non_increasing(self):
        result = X.search("greedy_refine", session(), self._objective(), seed=3)
        self.assertGreater(len(result.history), 2)
        for earlier, later in zip(result.history, result.history[1:]):
            self.assertLessEqual(later, earlier)

    def test_a_seeded_search_is_reproducible(self):
        a = X.search("evolution", session(), self._objective(), seed=11)
        b = X.search("evolution", session(), self._objective(), seed=11)
        self.assertEqual(a.best.vector, b.best.vector)
        self.assertEqual(a.best.score, b.best.score)
        self.assertEqual(a.evaluations, b.evaluations)

    def test_different_seeds_explore_differently(self):
        a = X.search("evolution_strategy", session(), self._objective(), seed=1)
        b = X.search("evolution_strategy", session(), self._objective(), seed=2)
        self.assertNotEqual(a.best.vector, b.best.vector)

    def test_the_winning_design_is_a_model_the_harness_can_build(self):
        result = X.search("technique_trials", session(), self._objective(), seed=5)
        server = CISPServer(backend="stub")
        applied = server.applyOps([op.to_dict() for op in result.best.ops])
        self.assertTrue(applied["ok"], applied)
        self.assertTrue(result.detail["replay_matches"])   # the winner replays exactly


class TestRivals(unittest.TestCase):
    def test_the_rival_families_are_exposed_by_name(self):
        families = X.rivals()
        self.assertEqual(families["evolutionary"],
                         ("evolution", "evolution_strategy"))
        self.assertEqual(families["sampling"],
                         ("designspace_sampler", "constrained_designspace",
                          "latin_hypercube"))
        self.assertEqual(families["local"],
                         ("greedy_refine", "guided_contact_search"))
        for names in families.values():
            for name in names:
                self.assertIn(name, X.strategies())

    def test_rivals_are_run_by_name_and_never_averaged(self):
        objective = X.shape_objective(session(w=40.0, h=25.0))
        scores = {}
        for name in X.rivals()["evolutionary"]:
            scores[name] = X.search(name, session(), objective, seed=3).best.score
        self.assertEqual(len(scores), 2)
        # The GA over CAD programs and the numeric ES genuinely disagree.
        self.assertNotEqual(round(scores["evolution"], 6),
                            round(scores["evolution_strategy"], 6))
        # And the surface offers no way to merge them.
        self.assertFalse(hasattr(X, "run_all"))
        self.assertFalse(hasattr(X, "ensemble"))

    def test_the_three_samplers_are_three_answers_not_one(self):
        objective = X.shape_objective(session(w=40.0, h=25.0))
        results = {name: X.search(name, session(), objective, seed=4)
                   for name in X.rivals()["sampling"]}
        for name, result in results.items():
            self.assertTrue(result.improved, name)
        self.assertGreater(
            len({round(r.best.score, 6) for r in results.values()}), 1)


class TestFailureIsCaptured(unittest.TestCase):
    def test_a_raising_strategy_component_is_captured_not_fatal(self):
        def explodes(_ops):
            raise RuntimeError("the objective blew up")

        result = X.search("designspace_sampler", session(), explodes, seed=0)
        self.assertIn("error", result.detail)
        self.assertIn("RuntimeError", result.detail["error"])
        # The surface still works afterwards.
        good = X.search("designspace_sampler", session(),
                        X.shape_objective(session(w=40.0)), seed=0)
        self.assertTrue(good.improved)

    def test_a_session_search_without_an_objective_is_refused(self):
        with self.assertRaises(X.SearchError):
            X.search("evolution", session())


class TestBlockDecomposition(unittest.TestCase):
    def test_the_greedy_mdp_rollout_decomposes_an_l_shape_into_quads(self):
        from harnesscad.domain.geometry.mesh.block_domain import Shape

        domain = Shape.from_rectangles([(0, 0, 4, 2), (0, 2, 2, 4)])
        result = X.search("block_decomp", domain, seed=0)
        detail = result.detail
        self.assertNotIn("error", detail)
        self.assertTrue(detail["terminal"])
        self.assertGreaterEqual(detail["blocks"], 2)
        self.assertGreater(detail["reward"], 0.0)
        self.assertEqual(detail["observations"], 6)     # the L has six corners


if __name__ == "__main__":
    unittest.main()
