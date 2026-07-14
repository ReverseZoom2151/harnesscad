"""Tests for the verifier registry / dispatcher and its wiring into the loop."""

import unittest

from harnesscad.core.cisp.ops import (
    AddRectangle, Constrain, Extrude, Fillet, NewSketch, Shell,
)
from harnesscad.core.loop import HarnessSession
from harnesscad.eval.verifiers import registry as vr
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.stub import StubBackend


def _plate_ops():
    """A constrained 20x10 plate, extruded 5mm."""
    return [
        NewSketch(),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0),
        Constrain(kind="distance", a="e1", value=20.0),
        Constrain(kind="distance", a="e1", value=10.0),
        Constrain(kind="distance", a="e1", value=20.0),
        Constrain(kind="distance", a="e1", value=10.0),
        Extrude(sketch="sk1", distance=5.0),
    ]


def _session(**kwargs) -> HarnessSession:
    return HarnessSession(StubBackend(), **kwargs)


class _BrokenVerifier:
    """A verifier that always explodes. The dispatcher must absorb it."""

    name = "broken"
    tier = vr.LINT

    def applies_to(self, state):
        return True

    def check(self, state):
        raise RuntimeError("boom")


class _BrokenApplies:
    """Even `applies_to` may explode; that must not abort the run either."""

    name = "broken-applies"
    tier = vr.LINT

    def applies_to(self, state):
        raise ValueError("bad gate")

    def check(self, state):  # pragma: no cover - never reached
        return []


class _AlwaysFires:
    name = "always-fires"
    tier = vr.LINT

    def applies_to(self, state):
        return True

    def check(self, state):
        return [Diagnostic(Severity.INFO, "sentinel", "sentinel diagnostic", "test")]


class TestDiscovery(unittest.TestCase):
    def test_discovers_more_than_ten_verifiers(self):
        fleet = vr.discover()
        self.assertGreater(len(fleet), 10, f"only found {len(fleet)}")

    def test_discovery_is_deterministic(self):
        a = [v.name for v in vr.discover(refresh=True)]
        b = [v.name for v in vr.discover(refresh=True)]
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a, key=lambda n: a.index(n)))  # stable order

    def test_fleet_covers_native_and_adapted_verifiers(self):
        fleet = vr.discover()
        names = {v.name for v in fleet}
        # protocol-native classes found through harnesscad.registry
        self.assertIn("dfm", names)
        self.assertIn("precheck", names)
        self.assertIn("interference", names)
        # function-style modules reached through the adapters
        self.assertIn("kernel-preflight", names)
        self.assertIn("tolerance-stack", names)
        self.assertIn("standability", names)
        self.assertTrue(any(isinstance(v, vr.NativeVerifier) for v in fleet))
        self.assertTrue(any(isinstance(v, vr.FunctionVerifier) for v in fleet))

    def test_every_verifier_has_a_known_tier(self):
        for v in vr.discover():
            self.assertIn(v.tier, vr.TIERS, v.name)

    def test_discovery_uses_the_capability_registry(self):
        from harnesscad import registry as capability_registry

        indexed = {e.dotted for e in capability_registry.find(package="verifiers")}
        for v in vr.discover():
            if v.dotted:
                self.assertIn(v.dotted, indexed, v.name)


class _MeasuredBackend(StubBackend):
    """A stub that also answers `measure` with a bbox we dictate (extents)."""

    def __init__(self, bbox):
        super().__init__()
        self._bbox = list(bbox)

    def query(self, what):
        if what == "measure":
            return {"volume": 1.0, "bbox": list(self._bbox)}
        return super().query(what)


class TestShellEnvelope(unittest.TestCase):
    """`shell-envelope`: a shell must not GROW the part (bbox_after <= before)."""

    OPS = [
        NewSketch(),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=60.0, h=40.0),
        Extrude(sketch="sk1", distance=20.0),
        Shell(faces=(), thickness=3.0),
    ]

    def _run(self, backend, ops):
        session = HarnessSession(backend)
        session.apply_ops(ops)
        state = vr.model_state(backend, session.opdag)
        return vr.run_all(state, only=["shell-envelope"])

    def test_shell_that_grows_the_part_is_flagged(self):
        # The dilated bbox the two-sided F-rep shell used to produce.
        diags = self._run(_MeasuredBackend((63.0, 43.0, 23.0)), self.OPS)
        codes = [d.code for d in diags]
        self.assertIn("shell-grew-part", codes)
        grew = [d for d in diags if d.code == "shell-grew-part"]
        self.assertEqual(len(grew), 3)                       # X, Y and Z
        for d in grew:
            self.assertIs(d.severity, Severity.ERROR)

    def test_a_shell_that_removes_material_is_silent(self):
        diags = self._run(_MeasuredBackend((60.0, 40.0, 20.0)), self.OPS)
        self.assertEqual([d.code for d in diags], [])

    def test_the_check_abstains_without_a_shell(self):
        backend = _MeasuredBackend((999.0, 999.0, 999.0))
        session = HarnessSession(backend)
        session.apply_ops(_plate_ops())
        state = vr.model_state(backend, session.opdag)
        v = next(x for x in vr.discover() if x.name == "shell-envelope")
        self.assertFalse(v.applies_to(state))


class TestRunAll(unittest.TestCase):
    def test_run_all_returns_diagnostics(self):
        session = _session()
        session.apply_ops(_plate_ops())
        diags = vr.run_all(vr.model_state(session.backend, session.opdag))
        self.assertTrue(diags)
        for d in diags:
            self.assertIsInstance(d, Diagnostic)

    def test_run_all_is_deterministic(self):
        session = _session()
        session.apply_ops(_plate_ops())
        state = vr.model_state(session.backend, session.opdag)
        first = [(d.code, d.message, d.where) for d in vr.run_all(state)]
        second = [(d.code, d.message, d.where)
                  for d in vr.run_all(vr.model_state(session.backend, session.opdag))]
        self.assertEqual(first, second)

    def test_a_broken_verifier_never_aborts_the_run(self):
        session = _session()
        session.apply_ops(_plate_ops())
        state = vr.model_state(session.backend, session.opdag)
        fleet = vr.discover() + [_BrokenVerifier(), _BrokenApplies(), _AlwaysFires()]
        diags = vr.run_all(state, verifiers=fleet)      # must not raise
        codes = [d.code for d in diags]
        self.assertIn("verifier-error", codes)
        self.assertIn("sentinel", codes)                # the run continued
        broken = [d for d in diags if d.code == "verifier-error"]
        self.assertEqual({d.where for d in broken}, {"broken", "broken-applies"})
        for d in broken:
            self.assertIs(d.severity, Severity.WARNING)  # can never roll an op back

    def test_verifiers_are_skippable(self):
        session = _session()
        session.apply_ops(_plate_ops())
        state = vr.model_state(session.backend, session.opdag)
        full = vr.run_all(state)
        without_dfm = vr.run_all(state, skip=["dfm"])
        self.assertLess(len(without_dfm), len(full))

        only_precheck = vr.run_all(state, only=["precheck"])
        self.assertLessEqual(len(only_precheck), len(full))

        # tier gating
        domain_only = vr.run_all(state, tiers=(vr.DOMAIN,))
        self.assertLessEqual(len(domain_only), len(full))

    def test_domain_verifiers_skip_when_their_data_is_absent(self):
        session = _session()
        session.apply_ops(_plate_ops())
        state = vr.model_state(session.backend, session.opdag)
        for name in ("tolerance-stack", "brick-validity", "rim-feasibility",
                     "dimension-qa", "validity-gate"):
            v = next(x for x in vr.discover() if x.name == name)
            self.assertFalse(v.applies_to(state), name)

    def test_run_report_wraps_in_verify_report(self):
        session = _session()
        session.apply_ops(_plate_ops())
        report = vr.run_report(vr.model_state(session.backend, session.opdag),
                               only=["kernel-preflight"])
        self.assertTrue(report.ok)  # no ERROR from an advisory verifier


class TestEnvelope(unittest.TestCase):
    def test_envelope_is_derived_from_the_op_stream(self):
        session = _session()
        session.apply_ops(_plate_ops())
        state = vr.model_state(session.backend, session.opdag)
        self.assertEqual(state.envelope(), (0.0, 0.0, 0.0, 20.0, 10.0, 5.0))

    def test_no_envelope_before_a_solid_exists(self):
        session = _session()
        session.apply_ops([NewSketch(), AddRectangle(sketch="sk1", w=20.0, h=10.0)])
        state = vr.model_state(session.backend, session.opdag)
        self.assertIsNone(state.envelope())


class TestSessionWiring(unittest.TestCase):
    def test_core_level_is_the_default_and_runs_no_fleet(self):
        session = _session()
        self.assertEqual(session.verify_level, "core")
        self.assertEqual(session.run_fleet(), [])
        res = session.apply_ops(_plate_ops())
        self.assertTrue(res.ok)
        codes = [d.code for d in res.diagnostics]
        self.assertNotIn("preflight-RADIUS_TOO_LARGE", codes)
        self.assertNotIn("infeasible-plan", codes)

    def test_existing_dof_diagnostics_still_appear(self):
        """Regression: the under-constrained sketch diagnostic must survive.

        It is now an INFO note rather than a WARNING -- it fires on every op
        stream that emits no constrain ops, i.e. on every correct part, so it
        cannot be evidence that anything is wrong -- but it must still be
        REPORTED, with the same code and the same location.
        """
        for level in ("core", "full"):
            session = _session(verify_level=level)
            res = session.apply_ops([
                NewSketch(),
                AddRectangle(sketch="sk1", w=20.0, h=10.0),
                Extrude(sketch="sk1", distance=5.0),
            ])
            self.assertTrue(res.ok, level)
            under = [d for d in res.diagnostics if d.code == "under-constrained"]
            self.assertTrue(under, f"under-constrained warning lost at level {level}")
            self.assertIs(under[0].severity, Severity.INFO)
            self.assertEqual(under[0].where, "sk1")

    def test_full_level_invokes_the_fleet(self):
        session = _session(verify_level="full")
        res = session.apply_ops(_plate_ops())
        self.assertTrue(res.ok)
        codes = {d.code for d in res.diagnostics}
        # a verifier-sourced diagnostic no core verifier could ever produce
        self.assertIn("simulation-skipped", codes)
        self.assertIn("dfm-not-yet-measurable", codes)
        # and the core ones are still there
        self.assertIn("under-constrained", codes)

    def test_fleet_surfaces_a_diagnostic_the_core_loop_missed(self):
        """A fillet bigger than the part, and a shell thicker than the wall.

        The core verifiers pass this model (a solid exists, the sketch solves);
        the fleet catches both as kernel-preflight failures.
        """
        ops = _plate_ops() + [
            Fillet(edges=("f1",), radius=8.0),
            Shell(faces=("f1",), thickness=9.0),
        ]
        core = _session()
        core_res = core.apply_ops(list(ops))
        self.assertTrue(core_res.ok)
        self.assertEqual([d.code for d in core_res.diagnostics
                          if d.code.startswith("preflight-")], [])

        full = _session(verify_level="full")
        full_res = full.apply_ops(list(ops))
        self.assertTrue(full_res.ok)  # advisory: the fleet does not block by default
        codes = {d.code for d in full_res.diagnostics}
        self.assertIn("preflight-RADIUS_TOO_LARGE", codes)
        self.assertIn("preflight-THICKNESS_TOO_LARGE", codes)
        self.assertIn("infeasible-plan", codes)  # precheck, on the op plan

    def test_fleet_runs_once_per_batch_not_once_per_op(self):
        session = _session(verify_level="full")
        res = session.apply_ops(_plate_ops())
        n = len([d for d in res.diagnostics if d.code == "simulation-skipped"])
        self.assertEqual(n, 1)

    def test_blocking_fleet_rolls_the_offending_op_back(self):
        session = _session(verify_level="full", fleet_blocking=True,
                           fleet_only=["precheck"])
        res = session.apply_ops(_plate_ops() + [Shell(faces=("f1",), thickness=9.0)])
        self.assertFalse(res.ok)
        self.assertEqual(res.applied, len(_plate_ops()))
        self.assertEqual(res.rejected["op"], "shell")
        self.assertIn("infeasible-plan", {d.code for d in res.diagnostics})
        # the last-good state survived the rollback
        self.assertTrue(session.summary()["solid_present"])

    def test_bad_verify_level_is_rejected(self):
        with self.assertRaises(ValueError):
            _session(verify_level="everything")

    def test_session_is_deterministic_at_full_level(self):
        a = _session(verify_level="full").apply_ops(_plate_ops())
        b = _session(verify_level="full").apply_ops(_plate_ops())
        self.assertEqual(a.digest, b.digest)
        self.assertEqual([d.to_dict() for d in a.diagnostics],
                         [d.to_dict() for d in b.diagnostics])


class TestServerWiring(unittest.TestCase):
    def test_verify_method_runs_the_fleet_but_ok_stays_core(self):
        from harnesscad.io.surfaces.server import CISPServer

        server = CISPServer()
        server.applyOps([op.to_dict() for op in _plate_ops()])
        res = server.verify()
        self.assertTrue(res["ok"])          # core verifiers still decide ok
        self.assertTrue(res["fleet"])       # the fleet ran
        codes = {d["code"] for d in res["fleet"]}
        self.assertIn("simulation-skipped", codes)


class TestConformance(unittest.TestCase):
    def test_certificate_covers_the_fleet(self):
        session = _session()
        session.apply_ops(_plate_ops())
        report = vr.conformance(session.backend, session.opdag)
        self.assertIn("fleet", report.measurements)
        self.assertGreater(report.measurements["fleet"]["verifiers"], 10)
        self.assertIn(report.verdict, ("pass", "fail"))


if __name__ == "__main__":
    unittest.main()
