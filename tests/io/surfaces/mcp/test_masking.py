"""Valid-action masking by preflight (mcp/masking.py) + its wiring into CADGymEnv.

Three things are under test:

  * the mask is a REAL mask -- it answers per geometric state and per parameter
    (a fillet radius the stock cannot carry is absent; the same op with a legal
    radius is present);
  * it AGREES with RLCAD's trial-execution masking (arXiv:2503.18549 Alg.1) on a
    corpus of states -- exactly, for every structural tier, and with the numeric
    tier's only disagreements recorded here in full;
  * it is CHEAPER: zero kernel round trips against one per candidate.
"""

from __future__ import annotations

import unittest

from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.stub import StubBackend
from harnesscad.io.surfaces.mcp import masking
from harnesscad.io.surfaces.mcp.gym import CADGymEnv

try:  # the real B-rep kernel, when this box has it
    import cadquery as _cq  # noqa: F401
    HAVE_CADQUERY = True
except Exception:  # noqa: BLE001 - pragma: no cover
    HAVE_CADQUERY = False


#: 10 x 10 x 2 plate: the smallest extent is 2 mm, so a 1 mm fillet radius is
#: exactly the degenerate limit (2r == limit) and 0.3 mm is comfortably legal.
PLATE = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 2.0},
]

#: States the mask is asked about. Each is a prefix of ops applied to a session.
CORPUS = {
    "empty": [],
    "sketch_only": [{"op": "new_sketch", "plane": "XY"}],
    "sketch_with_profile": PLATE[:2],
    "plate": PLATE,
    "thick_block": [
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 20.0},
        {"op": "extrude", "sketch": "sk1", "distance": 10.0},
        {"op": "fillet", "edges": [], "radius": 1.0},
    ],
    "two_solids": PLATE + [
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_circle", "sketch": "sk2", "cx": 0.0, "cy": 0.0, "r": 3.0},
        {"op": "extrude", "sketch": "sk2", "distance": 4.0},
    ],
    "primitive_box": [
        {"op": "primitive", "shape": "box", "dx": 10.0, "dy": 10.0, "dz": 10.0},
    ],
}


def _session(ops, backend=None):
    session = HarnessSession(backend if backend is not None else StubBackend())
    if ops:
        result = session.apply_ops([parse_op(o) for o in ops])
        assert result.ok, [d.to_dict() for d in result.diagnostics]
    return session


class TestMaskIsARealMask(unittest.TestCase):
    def test_empty_state_masks_out_every_solid_consuming_op(self):
        valid = masking.valid_actions(_session([]))
        self.assertIn("new_sketch", valid)
        for name in ("fillet", "chamfer", "shell", "draft", "mirror",
                     "linear_pattern", "circular_pattern", "extrude", "boolean"):
            self.assertNotIn(name, valid, "%s cannot run on an empty model" % name)

    def test_fillet_radius_decides_membership(self):
        session = _session(PLATE)
        mask = masking.action_mask(session, {
            "fillet_illegal": {"op": "fillet", "radius": 5.0},
            "fillet_degenerate": {"op": "fillet", "radius": 1.0},
            "fillet_legal": {"op": "fillet", "radius": 0.3},
        })
        self.assertFalse(mask["fillet_illegal"])
        self.assertFalse(mask["fillet_degenerate"])   # 2r == smallest extent
        self.assertTrue(mask["fillet_legal"])

    def test_rejected_fillet_carries_the_preflight_code(self):
        verdicts = masking.mask_verdicts(_session(PLATE), {
            "fillet": {"op": "fillet", "radius": 5.0}})
        self.assertEqual(verdicts["fillet"].code, "preflight-RADIUS_TOO_LARGE")

    def test_shell_thickness_decides_membership(self):
        mask = masking.action_mask(_session(PLATE), {
            "shell_illegal": {"op": "shell", "thickness": 1.5},
            "shell_legal": {"op": "shell", "thickness": 0.4},
        })
        self.assertFalse(mask["shell_illegal"])
        self.assertTrue(mask["shell_legal"])

    def test_thin_wall_is_manufacturability_not_validity(self):
        """A 0.4 mm wall is thin to MAKE but perfectly buildable: masking it out
        would delete a legal action, so MASK_RULES switches min_wall off."""
        self.assertEqual(masking.MASK_RULES.min_wall, 0.0)
        verdict = masking.mask_verdicts(_session(PLATE), {
            "shell": {"op": "shell", "thickness": 0.4}})["shell"]
        self.assertTrue(verdict.valid, verdict.reason)

    def test_dangling_reference_is_masked_out(self):
        mask = masking.action_mask(_session(PLATE), {
            "extrude_bad": {"op": "extrude", "sketch": "nope", "distance": 1.0},
            "extrude_ok": {"op": "extrude", "sketch": "sk1", "distance": 1.0},
        })
        self.assertFalse(mask["extrude_bad"])
        self.assertTrue(mask["extrude_ok"])

    def test_malformed_selector_is_masked_out(self):
        verdicts = masking.mask_verdicts(_session(PLATE), {
            "hole_bad": {"op": "hole", "face_or_sketch": "top", "x": 5.0, "y": 5.0,
                         "diameter": 3.0},
            "hole_ok": {"op": "hole", "face_or_sketch": ">Z", "x": 5.0, "y": 5.0,
                        "diameter": 3.0},
        })
        self.assertFalse(verdicts["hole_bad"].valid)
        self.assertEqual(verdicts["hole_bad"].code, "preflight-bad-selector")
        self.assertTrue(verdicts["hole_ok"].valid, verdicts["hole_ok"].reason)

    def test_boolean_needs_two_solids(self):
        self.assertNotIn("boolean", masking.valid_actions(_session(PLATE)))
        self.assertIn("boolean", masking.valid_actions(_session(CORPUS["two_solids"])))

    def test_set_param_target_is_checked_symbolically(self):
        session = _session(PLATE)
        mask = masking.action_mask(session, {
            "ok": {"op": "set_param", "target": 2, "param": "distance", "value": 3.0},
            "bad_index": {"op": "set_param", "target": 99, "param": "distance",
                          "value": 3.0},
            "bad_param": {"op": "set_param", "target": 2, "param": "nope",
                          "value": 3.0},
        })
        self.assertTrue(mask["ok"])
        self.assertFalse(mask["bad_index"])
        self.assertFalse(mask["bad_param"])

    def test_masking_never_mutates_the_session(self):
        session = _session(PLATE)
        before = session.digest()
        masking.valid_actions(session)
        self.assertEqual(session.digest(), before)

    def test_numeric_tier_is_silent_without_stock(self):
        """No knowable stock -> the numeric rule must not fire (a false
        rejection silently shrinks the action space)."""
        session = _session(CORPUS["primitive_box"])
        view = masking.state_view(session)
        self.assertIsNone(view.stock)
        verdict = masking.op_verdict(view, parse_op({"op": "fillet", "radius": 99.0}))
        self.assertTrue(verdict.valid)


class TestGymWiring(unittest.TestCase):
    def test_action_space_contract_is_unchanged(self):
        env = CADGymEnv()
        names = env.action_space()
        self.assertEqual(names, [t.name for t in env.catalog.op_tools()])

    def test_masked_action_space_is_a_strict_subset_at_reset(self):
        env = CADGymEnv()
        full = set(env.action_space())
        masked = set(env.action_space(masked=True))
        self.assertTrue(masked < full)
        self.assertIn("new_sketch", masked)
        self.assertNotIn("fillet", masked)

    def test_valid_action_space_grows_once_a_solid_exists(self):
        env = CADGymEnv()
        before = set(env.valid_action_space())
        env.step(PLATE)
        after = set(env.valid_action_space())
        self.assertNotIn("chamfer", before)
        self.assertIn("chamfer", after)

    def test_observation_carries_the_mask(self):
        env = CADGymEnv()
        obs = env.reset()
        self.assertIn("action_mask", obs)
        self.assertIn("valid_actions", obs)
        self.assertEqual(set(obs["action_mask"]), set(env.action_space()))
        for value in obs["action_mask"].values():
            self.assertIsInstance(value, bool)
        self.assertEqual(obs["valid_actions"],
                         sorted(n for n, ok in obs["action_mask"].items() if ok))
        obs2, _reward, _done, _info = env.step({"op": "new_sketch", "plane": "XY"})
        self.assertIn("action_mask", obs2)

    def test_mask_can_be_switched_off(self):
        obs = CADGymEnv(mask_actions=False).reset()
        self.assertNotIn("action_mask", obs)
        self.assertNotIn("valid_actions", obs)

    def test_mask_leaks_no_ground_truth(self):
        obs = CADGymEnv().reset()
        for banned in ("target", "answer", "ground_truth", "solution", "expected"):
            self.assertNotIn(banned, obs)

    def test_env_level_verdicts_explain_a_rejection(self):
        env = CADGymEnv()
        env.step(PLATE)
        verdict = env.action_mask_verdicts({"f": {"op": "fillet", "radius": 5.0}})["f"]
        self.assertFalse(verdict.valid)
        self.assertIn("RADIUS_TOO_LARGE", verdict.code)


class TestAgreementWithTrialExecution(unittest.TestCase):
    """Preflight masking vs RLCAD's trial-execution masking, over the corpus."""

    def _compare(self, numeric):
        report = masking.MaskComparison()
        for ops in CORPUS.values():
            masking.compare_masks(_session(ops), numeric=numeric, report=report)
        return report

    def test_structural_masking_agrees_exactly(self):
        report = self._compare(numeric=False)
        self.assertGreater(report.total, 200)
        self.assertEqual(report.over_permissive, [])
        self.assertEqual(report.over_restrictive, [])
        self.assertEqual(report.agreement, 1.0)

    def test_numeric_tier_disagreements_are_exactly_the_known_two(self):
        """THE FINDING, recorded rather than hidden.

        With the semantic StubBackend as "the kernel", the numeric tier refuses
        two actions the stub would have accepted: a 1 mm fillet and a 1 mm shell
        on a 2 mm plate. The stub models no geometry, so it accepts both; a real
        B-rep kernel does not (see TestAgreementWithTheRealKernel, where OCCT
        refuses exactly these). The disagreement is therefore against the stub,
        not against a kernel -- but it is over-RESTRICTIVE in direction, so it is
        pinned here: any growth in this set shrinks the agent's action space.
        """
        report = self._compare(numeric=True)
        self.assertEqual(report.over_permissive, [])
        self.assertEqual([name for name, _why in report.over_restrictive],
                         ["fillet", "shell"])
        self.assertGreater(report.agreement, 0.98)

    def test_preflight_costs_no_kernel_round_trips(self):
        report = self._compare(numeric=True)
        self.assertEqual(report.kernel_calls_preflight, 0)
        self.assertEqual(report.kernel_calls_trial, report.total)
        self.assertLess(report.seconds_preflight, report.seconds_trial)


@unittest.skipUnless(HAVE_CADQUERY, "cadquery/OCCT not installed")
class TestAgreementWithTheRealKernel(unittest.TestCase):
    """The comparison that matters: preflight vs the OCCT kernel itself."""

    CANDIDATES = [
        {"op": "fillet", "radius": 1.0},
        {"op": "fillet", "radius": 0.3},
        {"op": "shell", "thickness": 1.0},
        {"op": "shell", "thickness": 0.4},
        {"op": "chamfer", "distance": 0.3},
        {"op": "hole", "face_or_sketch": "top", "x": 5.0, "y": 5.0, "diameter": 3.0},
        {"op": "hole", "face_or_sketch": ">Z", "x": 5.0, "y": 5.0, "diameter": 3.0},
        {"op": "hole", "face_or_sketch": ">Z", "x": 5.0, "y": 5.0, "diameter": 30.0},
        {"op": "extrude", "sketch": "nope", "distance": 1.0},
        {"op": "new_sketch", "plane": "XY"},
        {"op": "mirror", "plane": "XZ"},
        {"op": "draft", "angle": 5.0, "neutral_plane": "XY"},
    ]

    def test_preflight_matches_occt_and_is_far_cheaper(self):
        from harnesscad.io.backends.cadquery import CadQueryBackend

        session = _session(PLATE, backend=CadQueryBackend())
        proposals = {"c%d" % i: c for i, c in enumerate(self.CANDIDATES)}
        report = masking.compare_masks(session, proposals)
        self.assertEqual(report.over_restrictive, [])
        self.assertEqual(report.over_permissive, [])
        self.assertEqual(report.agreement, 1.0)
        # The whole point: no kernel round trips, orders of magnitude cheaper.
        self.assertEqual(report.kernel_calls_preflight, 0)
        self.assertLess(report.seconds_preflight * 10.0, report.seconds_trial)
        print("\n[masking] OCCT agreement %d/%d; preflight %.4fs vs trial %.4fs "
              "(%.0fx cheaper)" % (report.agree, report.total,
                                   report.seconds_preflight, report.seconds_trial,
                                   report.seconds_trial / report.seconds_preflight))


if __name__ == "__main__":
    unittest.main()
