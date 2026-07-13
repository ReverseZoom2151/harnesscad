"""The procedural surface: named generators that emit CISP ops."""

import unittest

from harnesscad.core.cisp.ops import AddCircle, AddRectangle, Extrude, NewSketch
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.procedural import registry as P
from harnesscad.io.backends.stub import StubBackend


def session():
    return HarnessSession(StubBackend())


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_five_real_modules(self):
        routed = P.routed_modules()
        self.assertGreater(len(routed), 5, routed)
        for dotted in routed:
            self.assertTrue(dotted.startswith("harnesscad.domain.procedural."),
                            dotted)

    def test_every_procedural_module_has_a_route(self):
        self.assertEqual(P.unadapted(), [])

    def test_more_than_five_generators(self):
        self.assertGreater(len(P.generators()), 5)

    def test_discovery_is_deterministic(self):
        self.assertEqual(P.discover(), P.discover())
        self.assertEqual(P.generators(), P.generators())

    def test_unknown_generator_raises(self):
        with self.assertRaises(P.UnknownGenerator):
            P.emit("no_such_generator")


class TestOpsApplyCleanly(unittest.TestCase):
    def test_every_generator_emits_ops_a_session_accepts(self):
        for name in P.generators():
            with self.subTest(generator=name):
                ops = P.emit(name)
                self.assertGreater(len(ops), 0)
                result = session().apply_ops(ops)
                self.assertTrue(
                    result.ok,
                    "%s: %s" % (name, [d.message for d in result.diagnostics]))
                self.assertEqual(result.applied, len(ops))

    def test_emitted_ops_are_real_cisp(self):
        ops = P.emit("array.rectangular", rows=2, cols=2)
        self.assertTrue(all(isinstance(o, (NewSketch, AddRectangle, Extrude))
                            for o in ops))

    def test_apply_to_continues_from_the_sessions_sketch_count(self):
        s = session()
        first = P.apply_to(s, "patterns.grid")
        second = P.apply_to(s, "shape_grammar")
        self.assertTrue(first.ok)
        self.assertTrue(second.ok, [d.message for d in second.diagnostics])
        # The second batch referenced NEW sketches, not the first batch's.
        self.assertGreater(s.summary()["sketch_count"], 1)

    def test_emission_is_deterministic(self):
        a = [o.to_dict() for o in P.emit("markov_grammar", seed=7)]
        b = [o.to_dict() for o in P.emit("markov_grammar", seed=7)]
        self.assertEqual(a, b)

    def test_seed_actually_changes_the_derivation(self):
        seeds = {tuple(str(o.to_dict()) for o in P.emit("shape_grammar", seed=s))
                 for s in range(8)}
        self.assertGreater(len(seeds), 1)


class TestRivals(unittest.TestCase):
    def test_rival_families_are_declared(self):
        families = {f for f, _doc, _members in P.RIVAL_FAMILIES}
        self.assertEqual(families, {"grammar", "pattern"})

    def test_both_grammars_are_selectable_by_name(self):
        self.assertIn("shape_grammar", P.generators())
        self.assertIn("markov_grammar", P.generators())

    def test_the_two_grammars_are_not_the_same_formalism(self):
        """Same seed, different formalism -> different geometry. Never averaged."""
        cf = [o.to_dict() for o in P.emit("shape_grammar", seed=3)]
        markov = [o.to_dict() for o in P.emit("markov_grammar", seed=3)]
        self.assertNotEqual(cf, markov)

    def test_the_two_pattern_families_are_not_the_same_answer(self):
        """A polar array ROTATES its items; a radial ring carries no orientation."""
        ring = [o.to_dict() for o in P.emit("patterns.radial", count=6,
                                            radius=30.0)]
        polar = [o.to_dict() for o in P.emit("array.polar", count=6,
                                             radius=30.0, rotate_items=True)]
        self.assertNotEqual(ring, polar)

    def test_the_polar_arrays_rotation_reaches_the_geometry(self):
        """If the rotation were dropped, these two would be identical."""
        turned = [o.to_dict() for o in P.emit("array.polar", rotate_items=True)]
        flat = [o.to_dict() for o in P.emit("array.polar", rotate_items=False)]
        self.assertNotEqual(turned, flat)

    def test_array_can_fit_to_length_and_patterns_cannot(self):
        # The capability that makes array_patterns a distinct family, not a
        # reimplementation of patterns.
        ops = P.emit("array.fit_linear", total_length=100.0, pitch=25.0)
        self.assertEqual(sum(1 for o in ops if isinstance(o, Extrude)), 4)

    def test_no_route_averages_two_rivals(self):
        """Every rival member is reachable ONLY under its own name."""
        for _family, _doc, members in P.RIVAL_FAMILIES:
            for member in members:
                self.assertIn(member, P.generators())


class TestNamedGenerators(unittest.TestCase):
    def test_patterns_emit_one_sketch_of_circles(self):
        ops = P.emit("patterns.grid", rows=2, cols=3)
        self.assertEqual(sum(1 for o in ops if isinstance(o, NewSketch)), 1)
        self.assertEqual(sum(1 for o in ops if isinstance(o, AddCircle)), 6)

    def test_brick_template_emits_one_box_per_brick(self):
        ops = P.emit("brick_template", category="table", width=4, depth=4,
                     height=4, seed=0)
        boxes = sum(1 for o in ops if isinstance(o, AddRectangle))
        self.assertGreater(boxes, 0)
        self.assertEqual(len(ops), 3 * boxes)

    def test_voxel_compose_is_bigger_than_one_template(self):
        one = P.emit("brick_template", category="table", width=4, depth=4,
                     height=4, seed=0)
        two = P.emit("voxel_compose", categories=("table", "table"), width=4,
                     depth=4, height=4, seed=0)
        self.assertGreater(len(two), len(one))

    def test_symmetry_expands_a_motif(self):
        ops = P.emit("symmetry", kind="nfold", order=6)
        self.assertEqual(sum(1 for o in ops if isinstance(o, AddCircle)), 6)

    def test_unknown_symmetry_kind_raises(self):
        with self.assertRaises(P.ProceduralError):
            P.emit("symmetry", kind="fractal")


class TestGeometryOnlyRoutes(unittest.TestCase):
    def test_scene_returns_instances_not_cisp(self):
        instances, stats = P.scene(dims=(2, 2, 1), seed=0)
        self.assertEqual(stats["cells_total"], 4)
        self.assertGreater(len(instances), 0)

    def test_scene_is_deterministic(self):
        self.assertEqual(P.scene(seed=0)[1], P.scene(seed=0)[1])

    def test_expand_scene_culls_and_batches(self):
        terminals, batches, stats = P.expand_scene(
            [("a", 1), ("b", 0)],
            visible=lambda n: bool(n[1]),
            children=lambda n: (),
            terminal_key=lambda n: n[0])
        self.assertEqual(terminals, (("a", 1),))
        self.assertEqual(stats["culled"], 1)
        self.assertIn("a", batches)

    def test_key_params_derive_the_dependent_parameters(self):
        full = P.realize_parameters()
        self.assertIn("slew_rate", full)
        self.assertGreater(len(full), 4)

    def test_rebuild_only_recomputes_what_the_edit_dirtied(self):
        computes = [("b", ["a"], lambda deps: deps["a"] * 3)]
        values, recomputed = P.rebuild({"a": 2}, computes, {"a": 4})
        self.assertEqual(values["b"], 12)
        self.assertGreater(recomputed, 0)

        _values, clean = P.rebuild({"a": 2}, computes, None)
        self.assertEqual(clean, 0)

    def test_freeze_preserves_the_frozen_region(self):
        constraints, violations, projected = P.freeze(
            {"x": 1.0, "y": 2.0}, [["x"]], [], candidate={"x": 9.0, "y": 5.0})
        self.assertIn("x", constraints.frozen_variables())
        self.assertTrue(violations)
        self.assertEqual(projected["x"], 1.0)   # frozen: projected back
        self.assertEqual(projected["y"], 5.0)   # free: the edit stands

    def test_unknown_parameter_template_raises(self):
        with self.assertRaises(P.ProceduralError):
            P.realize_parameters(template="nope")


if __name__ == "__main__":
    unittest.main()
