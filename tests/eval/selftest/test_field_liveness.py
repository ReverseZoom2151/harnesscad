"""The field-liveness oracle must WORK -- and must not need today's bugs to work.

Same hard constraint as ``test_selftest.py``: a test that asserted "the oracle
finds the dead ``Hole.kind``" would PIN THE BUG IN PLACE. The day the FreeCAD
backend learns to counterbore, the test proving the ORACLE works goes red, and
the cheapest way to make it green again is to break the counterbore. A test must
never make a repair look like a regression.

So detection is proved against a DELIBERATELY CRIPPLED backend defined in this
file -- an engine that throws away ``Fillet.radius`` on the way in. That
corruption is ours, and it will still be here after every real bug in the repo is
fixed. Everything asserted about the LIVE engines stays true either way:

  * the schema is fully covered (no unmapped field) -- the anti-rot latch;
  * every fixture actually builds on at least one engine (a fixture that is
    refused everywhere silently tests nothing);
  * the oracle is deterministic (the same stream twice = the same signature);
  * the allow-list is short, well-formed, and names only real schema fields;
  * the report is well-formed.

The CENSUS itself -- how many fields are dead across the fleet TODAY -- is
deliberately NOT an assertion at the default level. It is a SCOREBOARD, it is
expected to be non-zero while the backend fixes land, and a scoreboard that fails
the build every time it is read is a scoreboard nobody reads. Flip it on with

    HARNESSCAD_FIELD_LIVENESS_STRICT=1     # zero dead fields, or fail
    HARNESSCAD_SELFTEST_FULL=1             # the whole six-engine matrix

RUNTIME. frep is a full SDF sample-and-march per stream (~1 s), and freecad /
openscad / blender each fork a process, so the full 83-field x 6-engine matrix is
~1000 process-or-kernel invocations and has no business in a unit-test suite. The
default here is the stub (the whole matrix, ~0.2 s -- it proves every FIXTURE is
well-formed) plus ONE op on frep. The rest is opt-in and SKIPS LOUDLY, with the
reason, never silently.
"""

from __future__ import annotations

import os
import unittest

from harnesscad.core.cisp.ops import Fillet, Op, _REGISTRY
from harnesscad.eval.selftest import field_liveness as fl
from harnesscad.io.backends.stub import StubBackend

FULL = os.environ.get("HARNESSCAD_SELFTEST_FULL") == "1"
STRICT = os.environ.get("HARNESSCAD_FIELD_LIVENESS_STRICT") == "1"


class DeadRadiusBackend(StubBackend):
    """A backend with a field-dropping bug WE put there.

    It clamps every ``Fillet.radius`` to a constant on the way in -- exactly the
    shape of the real bug (accept a typed op, silently ignore part of it, return
    a valid result, emit no diagnostic). The stub records ``edges`` but not
    ``radius``, so on the stub ``fillet.radius`` is dead either way; to make the
    corruption OBSERVABLE, this backend exposes the radius through ``summary``,
    then throws it away. A correct engine would report the radius it was given.
    """

    def apply(self, op: Op):
        if isinstance(op, Fillet):
            self._seen_radius = 1.0          # THE BUG: op.radius is discarded
            op = Fillet(edges=op.edges, radius=1.0)
        return super().apply(op)

    def query(self, q: str) -> dict:
        out = super().query(q)
        if q == "summary":
            out = dict(out)
            out["fillet_radius"] = getattr(self, "_seen_radius", None)
        return out


class HonestRadiusBackend(DeadRadiusBackend):
    """The same backend with the bug repaired -- the control.

    Without this, "the oracle says DEAD" proves nothing: an oracle that said DEAD
    to everything would pass. This one records the radius it was actually handed,
    and must come back LIVE.
    """

    def apply(self, op: Op):
        if isinstance(op, Fillet):
            self._seen_radius = float(op.radius)   # ... and it is USED
            return StubBackend.apply(self, op)
        return StubBackend.apply(self, op)


def _factory(cls):
    def make(name: str):
        return cls() if name == "victim" else None
    return make


class TestSchemaCoverage(unittest.TestCase):
    """The oracle is DERIVED from ops.py. It must stay derived."""

    def test_every_field_is_enumerated_from_the_dataclasses(self):
        pairs = fl.op_fields()
        self.assertEqual(len(pairs), len(set(pairs)), "duplicate (op, field)")
        # Every op in the registry appears, and every field of it.
        self.assertEqual({t for t, _ in pairs}, set(_REGISTRY))
        # A spot check that this is real reflection and not a copy of a list: the
        # eleven fields of Hole are the eleven fields of Hole.
        hole = [f for t, f in pairs if t == "hole"]
        self.assertEqual(len(hole), 11)
        self.assertIn("cbore_depth", hole)
        self.assertIn("csk_angle", hole)

    def test_no_unmapped_field(self):
        """THE ANTI-ROT LATCH.

        A field with neither a fixture nor an inert justification is a field this
        oracle cannot see -- and a newly added, never-wired field is precisely the
        one this oracle exists to catch. Adding a field to ops.py and not to
        field_liveness.py is a test failure, here, by design.
        """
        missing = fl.unmapped()
        self.assertEqual(missing, [],
                         "these ops.py fields have no liveness fixture and no "
                         "inert justification: %s" % missing)

    def test_every_case_names_a_real_op(self):
        for tag, case in fl.CASES.items():
            self.assertIn(tag, _REGISTRY, "CASES has an op ops.py does not: %s" % tag)
            self.assertIsNotNone(case.op)
            self.assertEqual(case.op.OP, tag)
            names = {f for _, f in fl.op_fields() if _ == tag}
            for fname in case.variants:
                self.assertIn(fname, names,
                              "%s has a fixture for a field it does not have: %s"
                              % (tag, fname))

    def test_a_variant_actually_changes_the_op(self):
        """A fixture whose 'alternate' value equals the base value tests nothing."""
        for tag, case in fl.CASES.items():
            for fname, (alt, base) in case.variants.items():
                base_op = case.stream(base)[len(case.prelude)]
                self.assertNotEqual(
                    getattr(base_op, fname), alt,
                    "%s.%s: the variant value equals the base value, so the two "
                    "streams are identical and the check is vacuous" % (tag, fname))


class TestAllowList(unittest.TestCase):
    """The allow-list is the one place a real bug could hide. Police it."""

    def test_allow_list_is_short(self):
        self.assertLessEqual(
            len(fl.INERT_FIELDS), 6,
            "the inert allow-list has grown. Every entry is a field this oracle "
            "agrees not to check; it is where a dropped field goes to hide.")

    def test_every_entry_is_a_real_field_and_carries_a_reason(self):
        real = set(fl.op_fields())
        for key, why in fl.INERT_FIELDS.items():
            self.assertIn(key, real,
                          "the allow-list exempts a field ops.py does not have: %s"
                          % (key,))
            self.assertTrue(why and len(why) > 30,
                            "an allow-list entry with no justification is not an "
                            "exception, it is a bug: %s" % (key,))

    def test_an_allow_listed_field_is_reported_na_not_live(self):
        cell = fl.check_field("stub", "add_point", "x")
        self.assertEqual(cell.verdict, fl.NA)


class TestDetection(unittest.TestCase):
    """Does the oracle actually catch a dropped field? Prove it on OUR bug."""

    def test_a_dropped_field_is_reported_dead(self):
        cell = fl.check_field("victim", "fillet", "radius",
                              factory=_factory(DeadRadiusBackend))
        self.assertEqual(cell.verdict, fl.DEAD,
                         "the oracle did not notice a backend that throws "
                         "Fillet.radius away")

    def test_the_same_field_read_honestly_is_reported_live(self):
        cell = fl.check_field("victim", "fillet", "radius",
                              factory=_factory(HonestRadiusBackend))
        self.assertEqual(cell.verdict, fl.LIVE,
                         "the oracle calls a field dead even when the backend "
                         "demonstrably reads it -- it would call everything dead")

    def test_an_absent_backend_skips_and_does_not_raise(self):
        cell = fl.check_field("no-such-engine", "fillet", "radius")
        self.assertEqual(cell.verdict, fl.SKIP)
        self.assertTrue(cell.detail)

    def test_a_backend_that_hangs_is_a_timeout_not_a_hang(self):
        class Wedged(StubBackend):
            def reset(self):
                super().reset()

            def query(self, q):
                import time
                time.sleep(30)          # never returns inside the ceiling
                return {}

        cell = fl.check_field("victim", "fillet", "radius",
                              factory=_factory(Wedged), timeout_s=0.5)
        self.assertEqual(cell.verdict, fl.ERR)
        self.assertIn("timeout", cell.detail)


class TestDeterminism(unittest.TestCase):

    def test_the_same_stream_twice_gives_the_same_signature(self):
        """If it did not, every field would score LIVE and the oracle would be a
        random number generator that always says 'fine'."""
        case = fl.CASES["fillet"]
        sigs = []
        for _ in range(2):
            backend = StubBackend()
            out = fl._run_stream(backend, case.stream())
            self.assertTrue(out.ok)
            sigs.append(out.sig)
        self.assertEqual(sigs[0], sigs[1])

    def test_the_signature_is_not_the_op_log(self):
        """state_digest() hashes the op log, so ANY field change moves it and
        every field would be trivially LIVE. The signature must be the state the
        backend BUILT, not the instructions it was handed."""
        backend = StubBackend()
        case = fl.CASES["fillet"]
        fl._run_stream(backend, case.stream())
        sig = fl.signature(backend)
        self.assertNotIn("oplog", sig)
        self.assertNotEqual(sig, backend.state_digest())


class TestStubMatrix(unittest.TestCase):
    """The whole 83-field matrix on the stub: fast, and it proves the FIXTURES.

    The stub builds no geometry, so most cells are legitimately DEAD there and
    nothing is asserted about them. What IS asserted is that every fixture is
    accepted -- a fixture the backend refuses is a fixture that tests nothing, and
    it would go unnoticed forever behind an N/A.
    """

    @classmethod
    def setUpClass(cls):
        cls.report = fl.run(backends=("stub",))

    def test_report_is_well_formed(self):
        d = self.report.to_dict()
        self.assertEqual(d["oracle"], "field_liveness")
        self.assertEqual(len(self.report.cells), len(fl.op_fields()))
        self.assertIn("inert_allow_list", d)
        self.assertTrue(fl.format_text(self.report))

    def test_no_fixture_is_refused_by_the_stub(self):
        """The stub accepts every op in the schema and rejects nothing a valid
        stream can throw at it. So a REJ cell here (BOTH streams refused) is a
        broken FIXTURE of ours -- a check that exercises nothing -- and would
        otherwise sit in the matrix looking like a finding forever.

        ERR is NOT a broken fixture: several fixtures deliberately point a
        reference field at a different target so that the alternate stream is
        REFUSED. A typed refusal is proof the field was read. That is the point.
        """
        bad = [(c.op, c.field, c.detail) for c in self.report.cells
               if c.verdict == fl.REJ]
        self.assertEqual(bad, [], "broken liveness fixtures (the stub refused BOTH "
                                  "streams, so nothing was tested): %s" % bad)

    def test_no_fixture_crashed_and_none_is_missing(self):
        for c in self.report.cells:
            self.assertNotIn("no fixture", c.detail,
                             "%s.%s has no fixture" % (c.op, c.field))
            self.assertNotIn("timeout", c.detail)
            self.assertNotIn("Traceback", c.detail)

    def test_the_stubs_dead_cells_are_not_counted_as_bugs(self):
        self.assertEqual(self.report.dead, [],
                         "the stub is non-geometric; its dead cells must not enter "
                         "the bug census")
        self.assertTrue(self.report.dead_nongeometric)


class TestFrepSubset(unittest.TestCase):
    """One real geometry engine, one op -- the cheap end of the real matrix.

    frep is in-process (no fork) and always available, so this always runs. It is
    restricted to ``fillet`` (2 fields = 4 SDF builds, a few seconds); the full
    sweep is behind HARNESSCAD_SELFTEST_FULL.
    """

    def test_fillet_fields_are_checkable_on_frep(self):
        report = fl.run(backends=("frep",), ops=("fillet",))
        cells = {c.field: c.verdict for c in report.cells}
        self.assertEqual(set(cells), {"edges", "radius"})
        for fname, verdict in cells.items():
            self.assertIn(verdict, (fl.LIVE, fl.DEAD, fl.ERR, fl.NA, fl.REJ),
                          "unknown verdict for fillet.%s" % fname)
        # frep is a real engine: a fillet radius it is handed must reach the field.
        # (radius is not on anybody's dead list and never was -- this is a live
        # assertion that stays true, not a pin holding a bug in place.)
        self.assertEqual(cells["radius"], fl.LIVE)


@unittest.skipUnless(FULL, "the full six-engine field-liveness matrix forks "
                           "freecadcmd/blender/openscad ~1000 times and takes "
                           "minutes. Set HARNESSCAD_SELFTEST_FULL=1 to run it "
                           "(or: harnesscad selftest --field-liveness).")
class TestFullMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.report = fl.run()

    def test_matrix_is_complete(self):
        n = len(fl.op_fields()) * len(fl.BACKENDS)
        self.assertEqual(len(self.report.cells), n)

    @unittest.skipUnless(STRICT, "the census is a SCOREBOARD, not a gate: dead "
                                 "fields are expected while the backend fixes "
                                 "land. Set HARNESSCAD_FIELD_LIVENESS_STRICT=1 to "
                                 "make a dead field fail the build -- that is the "
                                 "switch to flip the day the fleet is clean.")
    def test_no_dead_fields(self):
        dead = ["%s %s.%s" % (c.backend, c.op, c.field) for c in self.report.dead]
        self.assertEqual(dead, [], "these backends ignore fields they declare: %s"
                                   % dead)


if __name__ == "__main__":
    unittest.main()
