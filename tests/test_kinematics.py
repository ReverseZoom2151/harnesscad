"""Tests for the mechanism kinematics / motion-behaviour validator.

Covers the behavioural counterpart to the static assembly DOF graph:

  * Kutzbach-Gruebler mobility on a textbook four-bar linkage (M = 1);
  * a ratchet MotionSpec (one-way revolute) flagging a joint that still allows
    reverse rotation -> ``forbidden-motion`` ERROR;
  * a matching mechanism (the joint modelled one-way) passing cleanly;
  * a mobility-mismatch on a locked mechanism that is meant to move;
  * graceful INFO-skip on a dependency-free StubBackend / single-part stub;
  * MotionSpec round-trips through from_dict/to_dict.

Deterministic; stdlib + the harness only.
"""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.eval.verifiers.assembly import AssemblyModel, Mate
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.eval.quality.physics.kinematics import (
    ROT_NEG, ROT_POS, TRANS_POS,
    JointIntent, MotionSpec, MechanismGraph, KinematicsCheck,
    kinematics_diagnostics, with_kinematics,
    joint_freedom, joint_removed_dof, joint_directions, normalize_direction,
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def _by_severity(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


def _four_bar() -> AssemblyModel:
    """A closed four-bar loop: frame + 3 bars, four revolute pins."""
    return AssemblyModel(
        parts=["frame", "crank", "coupler", "rocker"],
        mates=[
            Mate(kind="revolute", a="frame", b="crank", name="pin_a"),
            Mate(kind="revolute", a="crank", b="coupler", name="pin_b"),
            Mate(kind="revolute", a="coupler", b="rocker", name="pin_c"),
            Mate(kind="revolute", a="rocker", b="frame", name="pin_d"),
        ],
        grounded=["frame"],
    )


class TestJointTable(unittest.TestCase):
    def test_freedom_and_removed(self):
        self.assertEqual(joint_removed_dof("revolute"), 5)
        self.assertEqual(joint_freedom("revolute"), 1)
        self.assertEqual(joint_freedom("cylindrical"), 2)
        self.assertEqual(joint_freedom("planar"), 3)
        self.assertEqual(joint_freedom("rigid"), 0)
        # one-way variants are revolute-/slider-like on the DOF count.
        self.assertEqual(joint_freedom("ratchet"), 1)
        self.assertIsNone(joint_freedom("nonsense"))

    def test_permitted_directions(self):
        self.assertEqual(joint_directions("revolute"), {ROT_POS, ROT_NEG})
        self.assertEqual(joint_directions("ratchet"), {ROT_POS})
        self.assertEqual(joint_directions("rigid"), frozenset())

    def test_direction_aliases(self):
        self.assertEqual(normalize_direction("reverse"), ROT_NEG)
        self.assertEqual(normalize_direction("forward"), ROT_POS)
        self.assertEqual(normalize_direction("extend"), TRANS_POS)


class TestMobility(unittest.TestCase):
    def test_four_bar_is_mobility_one(self):
        # Textbook planar Kutzbach: M = 3(4-1) - 2*4 = 1.
        graph = MechanismGraph(_four_bar(), planar=True)
        self.assertEqual(graph.n_links(), 4)
        self.assertEqual(graph.constraints_removed(), 8)
        self.assertEqual(graph.mobility(), 1)

    def test_four_bar_dof_summary(self):
        summary = MechanismGraph(_four_bar(), planar=True).dof_summary()
        self.assertEqual(summary["mobility"], 1)
        self.assertEqual(summary["n_links"], 4)
        self.assertEqual(summary["n_joints"], 4)
        self.assertTrue(summary["planar"])
        self.assertEqual(len(summary["joints"]), 4)

    def test_spatial_mobility_default(self):
        # Two links, one revolute, spatial: M = 6(2-1) - (6-1) = 1.
        model = AssemblyModel(
            parts=["a", "b"], mates=[Mate(kind="revolute", a="a", b="b")])
        self.assertEqual(MechanismGraph(model).mobility(), 1)

    def test_locked_chain_has_zero_mobility(self):
        # Two links welded rigid, planar: M = 3(2-1) - 3 = 0.
        model = AssemblyModel(
            parts=["a", "b"], mates=[Mate(kind="rigid", a="a", b="b")])
        self.assertEqual(MechanismGraph(model, planar=True).mobility(), 0)


class TestMobilityIntent(unittest.TestCase):
    def test_matching_mobility_no_warning(self):
        spec = MotionSpec(name="four-bar", expected_mobility=1, planar=True)
        report = KinematicsCheck().check_mechanism(_four_bar(), spec)
        self.assertIn("mobility", _codes(report))
        self.assertNotIn("mobility-mismatch", _codes(report))
        self.assertNotIn("mechanism-locked", _codes(report))
        self.assertTrue(report.ok)

    def test_locked_mechanism_that_should_move_errors(self):
        # A single rigid weld where the intent expects one DOF of motion.
        model = AssemblyModel(
            parts=["a", "b"], mates=[Mate(kind="rigid", a="a", b="b")])
        spec = MotionSpec(name="hinge", expected_mobility=1, planar=True)
        report = KinematicsCheck().check_mechanism(model, spec)
        self.assertIn("mechanism-locked", _codes(report))
        self.assertFalse(report.ok)

    def test_free_mechanism_that_should_be_fixed_warns(self):
        model = AssemblyModel(
            parts=["a", "b"], mates=[Mate(kind="revolute", a="a", b="b")])
        spec = MotionSpec(name="bracket", expected_mobility=0, planar=True)
        report = KinematicsCheck().check_mechanism(model, spec)
        self.assertIn("mobility-mismatch", _codes(report))
        # advisory only -> report stays ok.
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)


class TestForbiddenMotion(unittest.TestCase):
    def _ratchet_spec(self) -> MotionSpec:
        # The pawl joint must be one-way: reverse rotation is forbidden.
        return MotionSpec(
            name="ratchet",
            joints={"pawl": JointIntent(forbidden={ROT_NEG})},
        )

    def test_reverse_allowing_joint_is_flagged(self):
        # A plain (bidirectional) revolute where the intent wants one-way.
        model = AssemblyModel(
            parts=["wheel", "pawl_arm"],
            mates=[Mate(kind="revolute", a="wheel", b="pawl_arm", name="pawl")],
        )
        report = KinematicsCheck().check_mechanism(model, self._ratchet_spec())
        self.assertIn("forbidden-motion", _codes(report))
        self.assertFalse(report.ok)

    def test_one_way_joint_passes(self):
        # Same mechanism, but the joint is modelled as a one-way ratchet.
        model = AssemblyModel(
            parts=["wheel", "pawl_arm"],
            mates=[Mate(kind="ratchet", a="wheel", b="pawl_arm", name="pawl")],
        )
        report = KinematicsCheck().check_mechanism(model, self._ratchet_spec())
        self.assertNotIn("forbidden-motion", _codes(report))
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_permitted_whitelist_flags_extra_direction(self):
        # A slider whose intent only permits extension; retraction is extra.
        model = AssemblyModel(
            parts=["body", "rod"],
            mates=[Mate(kind="slider", a="body", b="rod", name="actuator")],
        )
        spec = MotionSpec(joints={
            "actuator": JointIntent(permitted={TRANS_POS})})
        report = KinematicsCheck().check_mechanism(model, spec)
        self.assertIn("forbidden-motion", _codes(report))

    def test_unknown_joint_ref_warns(self):
        model = AssemblyModel(
            parts=["a", "b"],
            mates=[Mate(kind="revolute", a="a", b="b", name="real")])
        spec = MotionSpec(joints={"ghost": JointIntent(forbidden={ROT_NEG})})
        report = KinematicsCheck().check_mechanism(model, spec)
        self.assertIn("unknown-joint-ref", _codes(report))

    def test_travel_limit_is_advisory_info(self):
        model = AssemblyModel(
            parts=["body", "rod"],
            mates=[Mate(kind="slider", a="body", b="rod", name="actuator")])
        spec = MotionSpec(joints={
            "actuator": JointIntent(min_travel=0.0, max_travel=50.0)})
        report = KinematicsCheck().check_mechanism(model, spec)
        self.assertIn("motion-limit", _codes(report))
        self.assertEqual(_by_severity(report, Severity.ERROR), [])


class TestUnknownJoint(unittest.TestCase):
    def test_unknown_kind_warns_and_is_excluded(self):
        model = AssemblyModel(
            parts=["a", "b"],
            mates=[Mate(kind="wobble", a="a", b="b")])
        report = KinematicsCheck().check_mechanism(model, None)
        self.assertIn("unknown-joint", _codes(report))
        # unknown joint contributes no constraint -> full mobility, no ERROR.
        self.assertEqual(_by_severity(report, Severity.ERROR), [])


class TestBackendIntegration(unittest.TestCase):
    def test_stub_info_skips_cleanly(self):
        report = KinematicsCheck().check(StubBackend(), None)
        self.assertIn("kinematics-skipped", _codes(report))
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_single_part_stub_is_trivial(self):
        class _AssemblyBackend:
            def query(self, q):
                return {"parts": ["solo"], "mates": []} if q == "assembly" else {}

        report = KinematicsCheck().check(_AssemblyBackend(), None)
        self.assertIn("kinematics-trivial", _codes(report))
        self.assertTrue(report.ok)

    def test_backend_payload_with_spec(self):
        payload = {
            "parts": ["frame", "crank", "coupler", "rocker"],
            "mates": [
                {"kind": "revolute", "a": "frame", "b": "crank", "name": "pa"},
                {"kind": "revolute", "a": "crank", "b": "coupler", "name": "pb"},
                {"kind": "revolute", "a": "coupler", "b": "rocker", "name": "pc"},
                {"kind": "revolute", "a": "rocker", "b": "frame", "name": "pd"},
            ],
        }

        class _AssemblyBackend:
            def query(self, q):
                return payload if q == "assembly" else {}

        spec = MotionSpec(expected_mobility=1, planar=True)
        report = KinematicsCheck(spec).check(_AssemblyBackend(), None)
        self.assertIn("mobility", _codes(report))
        self.assertNotIn("mechanism-locked", _codes(report))
        self.assertTrue(report.ok)

    def test_never_crashes_on_broken_backend(self):
        class _Boom:
            def query(self, q):
                raise RuntimeError("no query support")

        report = KinematicsCheck().check(_Boom(), None)
        self.assertIn("kinematics-skipped", _codes(report))
        self.assertTrue(report.ok)


class TestSpecRoundTrip(unittest.TestCase):
    def test_motion_spec_round_trips(self):
        spec = MotionSpec(
            name="ratchet",
            expected_mobility=1,
            planar=True,
            joints={
                "pawl": JointIntent(forbidden={ROT_NEG}),
                "actuator": JointIntent(
                    permitted={TRANS_POS}, min_travel=0.0, max_travel=50.0),
            },
        )
        d = spec.to_dict()
        back = MotionSpec.from_dict(d)
        self.assertEqual(back.name, "ratchet")
        self.assertEqual(back.expected_mobility, 1)
        self.assertTrue(back.planar)
        self.assertEqual(back.joints["pawl"].forbidden, {ROT_NEG})
        self.assertEqual(back.joints["actuator"].permitted, {TRANS_POS})
        self.assertEqual(back.joints["actuator"].max_travel, 50.0)
        # to_dict is stable across a round trip.
        self.assertEqual(back.to_dict(), d)

    def test_one_way_convenience_flag(self):
        spec = MotionSpec.from_dict({
            "name": "ratchet",
            "joints": {"pawl": {"one_way": True}},
        })
        self.assertIn(ROT_NEG, spec.joints["pawl"].forbidden)

    def test_from_dict_accepts_aliases(self):
        spec = MotionSpec.from_dict({
            "joints": {"pawl": {"forbidden": ["reverse"]}},
        })
        self.assertEqual(spec.joints["pawl"].forbidden, {ROT_NEG})


class TestWithKinematics(unittest.TestCase):
    def test_appends_kinematics_check(self):
        base = ["x", "y"]
        result = with_kinematics(base, MotionSpec(name="m"))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[-1].name, "kinematics")
        self.assertIsNotNone(result[-1].motion_spec)
        self.assertEqual(base, ["x", "y"])  # original untouched

    def test_kinematics_diagnostics_backend_free(self):
        diags = kinematics_diagnostics(_four_bar(),
                                       MotionSpec(expected_mobility=1, planar=True))
        self.assertTrue(any(d.code == "mobility" for d in diags))


if __name__ == "__main__":
    unittest.main()
