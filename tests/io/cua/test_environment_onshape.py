"""Onshape as the honest bridge: the GUI is the actuator, REST is the oracle.

These tests SKIP cleanly when ONSHAPE_ACCESS_KEY / ONSHAPE_SECRET_KEY are absent -
never hang, never prompt, never fail. Everything that does NOT need a live account
(signing shape, credential redaction, SI->mm conversion, the op->GUI mapping, the
honest capability declaration, the numeric read-back defence) is tested here with
no network and no browser, because those are the load-bearing invariants.

The one thing these tests deliberately do NOT do is fabricate a REST response and
call it a measured result. A live differential run against the scripted backend is
gated behind real credentials; without them it is skipped and said to be skipped.
"""

import os
import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch, _REGISTRY
from harnesscad.core.environment import Capabilities
from harnesscad.io.cua import environment_onshape as E


HAVE_CREDS = bool(os.environ.get(E.ACCESS_ENV)) and bool(os.environ.get(E.SECRET_ENV))
SKIP_MSG = ("no Onshape credentials: set %s and %s (environment variables only - "
            "never entered here) to run the live oracle tests"
            % (E.ACCESS_ENV, E.SECRET_ENV))


class TestCredentialSafety(unittest.TestCase):
    """The hard constraint: a secret is never printed and never persisted."""

    def test_repr_redacts_the_secret(self):
        creds = E.OnshapeCredentials(access="AKAK", secret="super-secret-value")
        text = repr(creds) + str(creds)
        self.assertNotIn("super-secret-value", text)
        self.assertIn("redacted", text)

    def test_present_is_false_without_both(self):
        self.assertFalse(E.OnshapeCredentials(access="", secret="x").present)
        self.assertFalse(E.OnshapeCredentials(access="x", secret="").present)
        self.assertTrue(E.OnshapeCredentials(access="a", secret="b").present)

    def test_secret_is_not_an_ordinary_attribute(self):
        """__slots__ keeps the secret off __dict__, so a naive vars() dump can't
        leak it into a trace."""
        creds = E.OnshapeCredentials(access="a", secret="b")
        self.assertFalse(hasattr(creds, "__dict__"))


class TestSignature(unittest.TestCase):
    """The HMAC-SHA256 request signature is deterministic and shaped per the docs."""

    def test_authorization_header_shape(self):
        creds = E.OnshapeCredentials(access="MYACCESS", secret="MYSECRET")
        auth, access = creds._sign(
            "GET", "/api/v9/documents", "", "abcdef`0123456789xyz",
            "Mon, 11 Apr 2016 20:08:56 GMT", "application/json")
        self.assertTrue(auth.startswith("On MYACCESS:HmacSHA256:"))
        self.assertEqual(access, "MYACCESS")

    def test_signature_is_deterministic_for_fixed_inputs(self):
        creds = E.OnshapeCredentials(access="A", secret="S")
        args = ("GET", "/api/v9/x", "q=1", "nonce0000000000000000000",
                "Mon, 11 Apr 2016 20:08:56 GMT", "application/json")
        self.assertEqual(creds._sign(*args), creds._sign(*args))

    def test_different_secret_changes_signature(self):
        args = ("GET", "/api/v9/x", "", "nonce0000000000000000000",
                "Mon, 11 Apr 2016 20:08:56 GMT", "application/json")
        a = E.OnshapeCredentials(access="A", secret="S1")._sign(*args)[0]
        b = E.OnshapeCredentials(access="A", secret="S2")._sign(*args)[0]
        self.assertNotEqual(a, b)


class TestUnitConversion(unittest.TestCase):
    """Onshape reports SI; the oracle converts to mm at the boundary so the
    differential compare against the mm-authored scripted backend is meaningful."""

    def test_mass_properties_si_to_mm(self):
        # a 10x10x10 mm cube: 1000 mm^3 = 1e-6 m^3; 600 mm^2 = 6e-4 m^2.
        oracle = E.OnshapeOracle(E.OnshapeCredentials(access="a", secret="b"))
        oracle._request = lambda *a, **k: {  # type: ignore[method-assign]
            "bodies": {"-all-": {
                "volume": [1.0e-6, 1.0e-6, 1.0e-6],
                "area": [6.0e-4, 6.0e-4, 6.0e-4],
                "centroid": [0.005, 0.005, 0.005],
                "mass": [0.0, 0.0, 0.0]}}}
        mp = oracle.mass_properties(E.DocumentRef("d", "w", "e"))
        self.assertAlmostEqual(mp.volume_mm3, 1000.0, places=6)
        self.assertAlmostEqual(mp.surface_area_mm2, 600.0, places=6)
        self.assertAlmostEqual(mp.centroid_mm[0], 5.0, places=6)

    def test_bounding_box_si_to_mm(self):
        oracle = E.OnshapeOracle(E.OnshapeCredentials(access="a", secret="b"))
        oracle._request = lambda *a, **k: {  # type: ignore[method-assign]
            "lowX": 0.0, "lowY": 0.0, "lowZ": 0.0,
            "highX": 0.01, "highY": 0.02, "highZ": 0.03}
        bb = oracle.bounding_box(E.DocumentRef("d", "w", "e"))
        self.assertEqual(bb.size_mm, (10.0, 20.0, 30.0))

    def test_first_handles_scalar_list_and_none(self):
        self.assertEqual(E._first([2.5, 1.0, 4.0]), 2.5)
        self.assertEqual(E._first(7.0), 7.0)
        self.assertEqual(E._first(None), 0.0)
        self.assertEqual(E._first([]), 0.0)


class TestOpToGuiMapping(unittest.TestCase):
    """The op -> Onshape-GUI table is DATA and honest about what it cannot bind."""

    def test_every_op_is_either_bound_or_explicitly_refused(self):
        # Asserted as a SET, not per-op subTests: a subTest failure is reported
        # one at a time by some runners, which makes a 12-op hole look like a
        # 1-op hole. The whole gap is named at once or not at all.
        declared = set(E.RECIPES) | set(E.REQUIRES_VIEWPORT)
        self.assertEqual(
            sorted(set(_REGISTRY) - declared), [],
            "ops neither bound nor refused: implement-or-refuse means a CISP op "
            "the GUI cannot drive must say so in REQUIRES_VIEWPORT")

    def test_the_capability_surface_declares_every_op(self):
        """The refusal must reach the CAPABILITIES a caller actually reads."""
        caps = E.OnshapeGuiEnvironment.CAPABILITIES
        declared = set(caps.supported_ops) | set(caps.unsupported_ops)
        self.assertEqual(sorted(set(_REGISTRY) - declared), [])
        self.assertIn("add_arc", caps.unsupported_ops)

    def test_bound_and_refused_are_disjoint(self):
        self.assertEqual(set(E.RECIPES) & set(E.REQUIRES_VIEWPORT), set())

    def test_value_for_pulls_op_attributes(self):
        self.assertEqual(E.value_for(Extrude(distance=37.5), "distance"), 37.5)
        self.assertEqual(E.value_for(AddRectangle(w=20.0, h=10.0), "w"), 20.0)
        with self.assertRaises(KeyError):
            E.value_for(NewSketch(), "distance")

    def test_supported_subset_matches_recipes(self):
        self.assertEqual(set(E.OnshapeGuiEnvironment.CAPABILITIES.supported_ops),
                         set(E.RECIPES))


class TestNumericReadback(unittest.TestCase):
    """37.5 vs 375 is a hard failure - the same defence as the FreeCAD path."""

    def test_matching_value_passes(self):
        self.assertTrue(E._numeric_matches(37.5, "37.5 mm"))
        self.assertTrue(E._numeric_matches(37.5, "37.5"))

    def test_swallowed_separator_fails(self):
        self.assertFalse(E._numeric_matches(37.5, "375 mm"))
        self.assertFalse(E._numeric_matches(37.5, "37.6 mm"))

    def test_unreadable_field_fails(self):
        self.assertFalse(E._numeric_matches(37.5, ""))
        self.assertFalse(E._numeric_matches(37.5, "mm"))


class TestCapabilitiesHonesty(unittest.TestCase):
    """The declaration differs from FreeCAD's in exactly the honest way."""

    def test_synchronous_read_is_true_unlike_freecad(self):
        caps = E.OnshapeGuiEnvironment.CAPABILITIES
        self.assertIsInstance(caps, Capabilities)
        # THE load-bearing difference: Onshape has a real synchronous structured
        # read (REST); FreeCAD's GUI does not.
        self.assertTrue(caps.synchronous_read)

    def test_the_three_gui_impossibilities_are_still_false(self):
        caps = E.OnshapeGuiEnvironment.CAPABILITIES
        self.assertFalse(caps.content_digest)
        self.assertFalse(caps.nonmutating_reject)
        self.assertFalse(caps.deterministic_replay)
        self.assertTrue(caps.resolve_before_act)

    def test_differs_from_freecad_declaration(self):
        try:
            from harnesscad.io.cua.environment_freecad import FreeCADGuiEnvironment
        except Exception:  # noqa: BLE001 - freecad deps may be absent; that's fine
            self.skipTest("freecad environment not importable here")
        self.assertFalse(FreeCADGuiEnvironment.CAPABILITIES.synchronous_read)
        self.assertTrue(E.OnshapeGuiEnvironment.CAPABILITIES.synchronous_read)


class TestReachability(unittest.TestCase):
    """available() never raises and reports precisely what is missing."""

    def test_available_reports_missing_pieces_without_raising(self):
        class NoBrowser:
            def available(self):
                return False, "no browser here"
        ok, why = E.available(actuator=NoBrowser())
        self.assertFalse(ok)
        self.assertIn("actuator", why)
        if not HAVE_CREDS:
            self.assertIn(E.ACCESS_ENV, why)

    def test_state_digest_raises_capability_error(self):
        # No network needed: the method refuses structurally.
        from harnesscad.core.environment import CapabilityError

        class FakeAct:
            def available(self):
                return True, ""
        # Construct without running by bypassing __init__ availability gate.
        env = E.OnshapeGuiEnvironment.__new__(E.OnshapeGuiEnvironment)
        with self.assertRaises(CapabilityError):
            env.state_digest()


@unittest.skipUnless(HAVE_CREDS, SKIP_MSG)
class TestLiveOracle(unittest.TestCase):
    """The real thing: needs credentials (and, for the full run, a browser).

    Only the credential-only oracle lifecycle is exercised here; the full
    GUI-drive-then-oracle-verify differential belongs to an integration run that
    also needs an authenticated browser session, and is skipped when the actuator
    is absent rather than faked.
    """

    def test_scratch_document_roundtrip(self):
        oracle = E.OnshapeOracle()
        ref = oracle.create_scratch_document("harnesscad-selftest-%d" % os.getpid())
        try:
            self.assertTrue(ref.did and ref.wid)
            elements = oracle.list_elements(ref)
            self.assertIsInstance(elements, list)
        finally:
            oracle.delete_document(ref.did)


if __name__ == "__main__":
    unittest.main()
