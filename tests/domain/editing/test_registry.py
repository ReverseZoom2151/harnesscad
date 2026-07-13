"""The edit surface: discovery, a real edit on a real session, diffs, rival loops."""

import unittest

from harnesscad.core.cisp.ops import SetParam
from harnesscad.core.cli import DEMO_OPS
from harnesscad.domain.editing import registry as R
from harnesscad.io.surfaces.server import CISPServer


def session(**overrides):
    """The demo plate (20 x 10 x 5), optionally with the rectangle overridden."""
    ops = [dict(op) for op in DEMO_OPS]
    if overrides:
        ops[1].update(overrides)
    server = CISPServer(backend="stub")
    result = server.applyOps(ops)
    assert result["ok"], result
    return server.session


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_five_real_editing_modules(self):
        mods = R.modules()
        self.assertGreater(len(mods), 5)
        for dotted in mods:
            self.assertTrue(dotted.startswith("harnesscad.domain.editing."))
        # Every bound module is really in the capability index (nothing invented).
        from harnesscad import registry as capability_registry

        indexed = {e.dotted for e in capability_registry.find(package="editing")}
        for dotted in mods:
            self.assertIn(dotted, indexed)

    def test_every_edit_kind_and_strategy_declares_a_known_target(self):
        self.assertGreater(len(R.edits()), 5)
        self.assertGreater(len(R.strategies()), 4)
        for name in R.edits():
            self.assertIn(R.edit_kind(name).target, R.TARGETS)
        for name in R.strategies():
            self.assertTrue(R.strategy(name).description)

    def test_unadapted_modules_are_reported_not_hidden(self):
        # VoxHammer's mask/trajectory need a 3D diffusion model; they stay orphaned.
        self.assertIn("harnesscad.domain.editing.voxel_mask", R.unadapted())
        self.assertIn("harnesscad.domain.editing.inversion_trajectory", R.unadapted())

    def test_unknown_names_raise(self):
        with self.assertRaises(R.UnknownEdit):
            R.edit_kind("no-such-edit")
        with self.assertRaises(R.UnknownStrategy):
            R.strategy("no-such-strategy")


class TestParametricEdit(unittest.TestCase):
    def test_a_real_edit_applies_to_a_session_and_the_diff_is_correct(self):
        s = session()
        before = R.snapshot(s)
        self.assertEqual(before.shape.extents, (20.0, 10.0, 5.0))

        result = R.apply_edit(s, R.Edit("param", target=1, param="w", value=40.0))

        self.assertTrue(result.ok)
        d = result.diff
        self.assertTrue(d.changed)
        self.assertEqual(d.changed_params, ((1, "w", 20.0, 40.0),))
        self.assertEqual(d.added_ops, ())
        self.assertEqual(d.removed_ops, ())
        self.assertNotEqual(d.digest_before, d.digest_after)
        # The model really changed: the session rebuilt at the new width.
        after = R.snapshot(s)
        self.assertEqual(after.shape.extents, (40.0, 10.0, 5.0))
        self.assertEqual(after.shape.volume, 2000.0)
        self.assertAlmostEqual(d.shape_delta[0], 20.0)
        # The op stream itself carries the edit (SetParam folded into the log).
        self.assertEqual(R.ops_of(s)[1].w, 40.0)

    def test_the_edit_reached_the_backend_via_setparam(self):
        s = session()
        R.apply_edit(s, R.Edit("param", target=6, param="distance", value=9.0))
        self.assertTrue(any(isinstance(op, SetParam) for op in s.opdag.ops()))
        self.assertEqual(R.shape_of(s).extents[2], 9.0)

    def test_a_blocked_edit_is_a_result_not_a_crash_and_leaves_the_model_alone(self):
        s = session()
        digest = s.digest()
        result = R.apply_edit(s, R.Edit("param", target=1, param="w", value=-5.0))
        self.assertFalse(result.ok)
        self.assertTrue(result.diagnostics)
        self.assertEqual(s.digest(), digest)          # block-and-correct
        self.assertEqual(R.shape_of(s).extents, (20.0, 10.0, 5.0))

    def test_an_out_of_range_target_is_reported(self):
        result = R.apply_edit(session(), R.Edit("param", target=99, param="w",
                                                value=1.0))
        self.assertFalse(result.ok)
        self.assertIn("bad-ref", result.diagnostics[0])

    def test_parameters_are_addressable(self):
        refs = {(p.index, p.param): p.value for p in R.parameters(session())}
        self.assertEqual(refs[(1, "w")], 20.0)
        self.assertEqual(refs[(6, "distance")], 5.0)

    def test_diff_of_two_op_streams_is_pure(self):
        d = R.diff(session(), session(w=40.0))
        self.assertEqual(d.changed_params, ((1, "w", 20.0, 40.0),))
        self.assertTrue(d.changed)
        self.assertFalse(R.diff(session(), session()).changed)


class TestEditLoops(unittest.TestCase):
    """The three rival loops each edit the model TOWARD a target -- differently."""

    def test_plan_verify_improves_toward_the_target(self):
        out = R.run_strategy("plan_verify", session(), session(w=40.0), seed=1)
        self.assertTrue(out.ok, out.error)
        self.assertLess(out.value["distance"], out.value["start_distance"])
        self.assertGreater(out.value["rounds"], 0)

    def test_refine_improves_toward_the_target(self):
        out = R.run_strategy("refine", session(), session(w=40.0), seed=1)
        self.assertTrue(out.ok, out.error)
        self.assertLess(out.value["distance"], out.value["start_distance"])

    def test_geometry_beam_improves_toward_the_target(self):
        out = R.run_strategy("geometry_beam", session(), session(w=40.0), seed=1)
        self.assertTrue(out.ok, out.error)
        self.assertLess(out.value["distance"], out.value["start_distance"])
        self.assertGreater(out.value["renders"], 0)

    def test_the_loops_are_reproducible_for_a_seed(self):
        a = R.run_strategy("geometry_beam", session(), session(w=40.0), seed=7)
        b = R.run_strategy("geometry_beam", session(), session(w=40.0), seed=7)
        self.assertEqual(a.value["distance"], b.value["distance"])
        self.assertEqual(a.value["ops"], b.value["ops"])

    def test_the_edited_ops_still_apply_to_a_real_session(self):
        out = R.run_strategy("refine", session(), session(w=40.0), seed=1)
        server = CISPServer(backend="stub")
        result = server.applyOps([op.to_dict() for op in out.value["ops"]])
        self.assertTrue(result["ok"], result)
        self.assertTrue(server.query("summary")["result"]["solid_present"])


class TestRivals(unittest.TestCase):
    def test_the_rival_families_are_exposed_by_name(self):
        families = R.rivals()
        self.assertEqual(families["edit_search"],
                         ("plan_verify", "refine", "geometry_beam"))
        self.assertEqual(families["push_pull_integration"],
                         R.PUSH_PULL_INTEGRATIONS)
        for name in families["edit_search"]:
            self.assertIn(name, R.strategies())

    def test_rivals_are_never_averaged(self):
        """Each rival reports its OWN number; the surface offers no blended one."""
        results = {name: R.run_strategy(name, session(), session(w=40.0), seed=1)
                   for name in R.rivals()["edit_search"]}
        distances = {n: r.value["distance"] for n, r in results.items()}
        self.assertEqual(len(distances), 3)
        # They genuinely disagree (different algorithms, different answers) ...
        self.assertGreater(len(set(round(d, 6) for d in distances.values())), 1)
        # ... and there is no API that merges them.
        self.assertFalse(hasattr(R, "run_all"))
        self.assertFalse(hasattr(R, "average"))

    def test_a_push_pull_edit_must_name_its_integration_strategy(self):
        from harnesscad.domain.editing import hybrid_consistency as hc
        from harnesscad.domain.editing import hybrid_model as hm

        model = hc.HybridModel(brep=hm.DirectBRep(), tree=hm.FeatureTree())
        with self.assertRaises(R.RivalBlend):
            R.apply_edit(model, R.Edit("push_pull", param="top", value=2.0))


class TestDirectManipulation(unittest.TestCase):
    """A direct face drag, reconciled into a parametric tree (Zou 2025)."""

    def _model(self):
        from harnesscad.domain.editing import hybrid_consistency as hc
        from harnesscad.domain.editing import hybrid_model as hm

        tree = hm.FeatureTree(features=[
            hm.ParametricFeature(fid="f1", ftype="extrude",
                                 params={"height": 5.0, "width": 20.0})])
        brep = hm.DirectBRep()
        brep.add_face(hm.Face(name="top", nx=0.0, ny=0.0, nz=1.0, offset=5.0,
                              origin="f1"))
        brep.add_face(hm.Face(name="bottom", nx=0.0, ny=0.0, nz=-1.0, offset=0.0,
                              origin="f1"))
        return hc.HybridModel(brep=brep, tree=tree), hm

    def test_operation_translation_turns_the_drag_into_a_parameter_edit(self):
        from harnesscad.domain.editing import operation_translation as ot

        model, _hm = self._model()
        result = R.apply_edit(model, R.Edit(
            "push_pull", param="top", value=3.0,
            payload={"integration": "operation_translation",
                     "links": [ot.FaceParamLink(face_name="top", fid="f1",
                                                param="height", gain=1.0)]}))
        self.assertTrue(result.ok, result.diagnostics)
        self.assertEqual(result.detail["integration"], "operation_translation")
        self.assertEqual(result.after.tree.parameter("f1", "height"), 8.0)

    def test_pseudo_feature_is_a_different_answer_to_the_same_drag(self):
        model, _hm = self._model()
        result = R.apply_edit(model, R.Edit(
            "push_pull", param="top", value=3.0,
            payload={"integration": "pseudo_feature"}))
        self.assertTrue(result.ok, result.diagnostics)
        # The parametric height is UNTOUCHED: the edit became an appended feature.
        self.assertEqual(result.after.tree.parameter("f1", "height"), 5.0)
        self.assertGreater(len(result.after.tree.features), 1)

    def test_synchronous_partition_drops_the_feature_out_of_the_history(self):
        model, _hm = self._model()
        result = R.apply_edit(model, R.Edit(
            "push_pull", param="top", value=3.0,
            payload={"integration": "synchronous_partition", "fid": "f1"}))
        self.assertTrue(result.ok, result.diagnostics)
        self.assertIn("f1", result.detail["direct_edit"])
        self.assertGreater(result.detail["parametric_loss"], 0)


class TestOtherTargets(unittest.TestCase):
    def test_layout_edits_align_the_placed_instances(self):
        server = CISPServer(backend="stub")
        ops = [dict(op) for op in DEMO_OPS] + [
            {"op": "add_instance", "part": "solid", "x": 0.0, "y": 0.0, "z": 0.0},
            {"op": "add_instance", "part": "solid", "x": 8.0, "y": 3.0, "z": 0.0},
        ]
        self.assertTrue(server.applyOps(ops)["ok"])
        result = R.apply_edit(server.session,
                              R.Edit("align", payload={"mode": "left", "size": 2.0}))
        self.assertTrue(result.ok, result.diagnostics)
        xs = [op.x for op in R.ops_of(server.session) if op.OP == "add_instance"]
        self.assertEqual(len(set(xs)), 1)          # they really are aligned now

    def test_a_layout_edit_on_a_model_with_no_instances_is_refused_not_faked(self):
        with self.assertRaises(R.Unsupported):
            R.apply_edit(session(), R.Edit("align", payload={"mode": "left"}))

    def test_the_token_mask_locates_exactly_what_changed(self):
        result = R.apply_edit(["a", "b", "c"],
                              R.Edit("mask", payload={"edited": ["a", "z", "c"],
                                                      "replacements": [["z"]]}))
        self.assertTrue(result.ok)
        self.assertEqual(result.detail["coarse_mask"], ["a", "<mask>", "c"])
        self.assertEqual(result.detail["infilled"], ["a", "z", "c"])

    def test_a_text_edit_preserves_the_line_structure(self):
        result = R.apply_edit("a = 1\nb = 2", R.Edit("text",
                                                     payload={"action": "indent"}))
        self.assertTrue(result.ok)
        self.assertEqual(result.after, "    a = 1\n    b = 2")


class TestFailureIsCaptured(unittest.TestCase):
    def test_a_raising_strategy_is_captured_not_fatal(self):
        # No target solid -> the refine loop genuinely cannot run. It must report,
        # not explode.
        empty = CISPServer(backend="stub").session
        out = R.run_strategy("refine", session(), empty)
        self.assertFalse(out.ok)
        self.assertIn("Unsupported", out.error)
        # ... and the surface is still usable afterwards.
        self.assertTrue(R.run_strategy("refine", session(), session(w=40.0),
                                       seed=1).ok)

    def test_a_strategy_whose_component_raises_is_reported_by_name(self):
        out = R.run_strategy("consistency", object())
        self.assertFalse(out.ok)
        self.assertEqual(out.name, "consistency")
        self.assertTrue(out.error)


class TestHistory(unittest.TestCase):
    def test_revisions_are_recorded_and_rollback_works(self):
        out = R.run_strategy("history", session(),
                             [R.Edit("param", 1, "w", 40.0),
                              R.Edit("param", 6, "distance", 8.0)])
        self.assertTrue(out.ok, out.error)
        value = out.value
        self.assertEqual(value["revisions"], 2)
        self.assertEqual(R.shape_of(value["current"]).extents, (40.0, 10.0, 8.0))
        self.assertNotEqual(value["digests"][0], value["digests"][1])
        rolled = value["session"].rollback(1)
        self.assertEqual(R.shape_of(rolled).extents, (40.0, 10.0, 5.0))


if __name__ == "__main__":
    unittest.main()
