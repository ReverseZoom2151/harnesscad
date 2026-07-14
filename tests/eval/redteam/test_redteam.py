"""The red team's own invariants -- above all, that it cannot inflate its score.

An adversarial auditor that overclaims is worth less than none: it spends the one
thing it has, which is the reader's willingness to believe a finding. So the tests
that matter here are the ones that prove the auditor is HONEST, not the ones that
prove it is loud:

  * every attack it generates is a part that is legal BY CONSTRUCTION (it never
    submits a broken part and then celebrates when the fleet catches it);
  * an attack it cannot PROVE good is dropped, never promoted;
  * the certification never runs at ``verify_level="full"`` -- asking the fleet
    whether a part is good, in order to decide whether the fleet is wrong about
    the part, is a circle;
  * the search is seeded and deterministic, so a finding can be replayed.

FAST BY DESIGN. Certifying one attack builds a solid on a grid-marching engine and
runs the output gate. The suite certifies a handful; the full sweep is opt-in.
"""

from __future__ import annotations

import os
import unittest

from harnesscad.eval.redteam import attacks, oracle
from harnesscad.eval.redteam import run as redteam_run

FULL = os.environ.get("HARNESSCAD_REDTEAM_FULL") == "1"


class TestAttackGeneration(unittest.TestCase):

    def test_deterministic(self):
        a = [x.name for x in attacks.generate()]
        b = [x.name for x in attacks.generate()]
        self.assertEqual(a, b)
        self.assertGreater(len(a), 20)

    def test_a_different_seed_gives_different_attacks(self):
        a = {x.name for x in attacks.generate(seed=1)}
        b = {x.name for x in attacks.generate(seed=2)}
        self.assertNotEqual(a, b)

    def test_every_attack_states_why_the_part_is_fine(self):
        """'Verifier X rejected op stream Y' is not a bug report. It becomes one
        only when somebody says why Y was a good part."""
        for a in attacks.generate():
            self.assertTrue(a.why_fine.strip(), a.name)
            self.assertGreater(len(a.why_fine), 40, a.name)
            self.assertGreater(a.volume, 0.0, a.name)

    def test_every_attack_is_legal_by_construction(self):
        """The boundary families must sit strictly INSIDE the rules' limits. An
        attack that was actually degenerate would make the fleet right and the red
        team a liar."""
        from harnesscad.core.cisp.ops import Chamfer, Fillet, Shell

        for a in attacks.generate():
            thin = min(a.bbox)
            for op in a.ops:
                if isinstance(op, Fillet):
                    self.assertLess(2 * op.radius, thin,
                                    "%s: 2r >= the smallest extent -- this attack "
                                    "is a genuinely degenerate part" % a.name)
                if isinstance(op, Chamfer):
                    self.assertLess(2 * op.distance, thin, a.name)
                if isinstance(op, Shell):
                    self.assertLess(2 * op.thickness, thin, a.name)
                    self.assertGreaterEqual(
                        op.thickness, 0.5,
                        "%s: below the declared minimum manufacturable wall"
                        % a.name)

    def test_it_walks_right_up_to_the_boundary(self):
        """The whole point. RADIUS_TOO_LARGE fired at r=3.1 on a 6 mm plate and
        stayed silent at r=3.0. An off-by-one is invisible to a corpus of round
        numbers and obvious one EPS from the limit."""
        from harnesscad.core.cisp.ops import Fillet, Shell

        near = 0
        for a in attacks.generate():
            thin = min(a.bbox)
            for op in a.ops:
                r = (op.radius if isinstance(op, Fillet)
                     else op.thickness if isinstance(op, Shell) else None)
                if r is None:
                    continue
                if abs(2 * r - thin) <= 2 * attacks.EPS + 1e-9:
                    near += 1
        self.assertGreater(near, 3, "no attack sits within one EPS of a rule's "
                                    "threshold; the search is not adversarial")

    def test_it_covers_the_families_no_brief_covers(self):
        fams = {a.family for a in attacks.generate()}
        for required in ("fillet_boundary", "shell_boundary", "shell_min_wall",
                         "hole_edge", "hole_thin_plate", "uncovered"):
            self.assertIn(required, fams)


class TestOracleCannotBeGamed(unittest.TestCase):

    def test_certification_never_consults_the_fleet(self):
        """verify_level='full' would run the verifiers. Using the thing under test
        to decide whether the thing under test is wrong is a circle, and it is a
        one-line change to introduce."""
        import inspect
        src = inspect.getsource(oracle)
        self.assertNotIn('verify_level="full"', src)
        self.assertNotIn("verify_level='full'", src)
        self.assertIn('verify_level="core"', src)

    def test_a_part_that_does_not_build_is_not_certified(self):
        """It is dropped, not promoted. A red team that inflates its own hit count
        is worth less than no red team."""
        from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch

        bogus = attacks.Attack(
            name="claims_a_volume_it_does_not_have", family="uncovered",
            ops=(NewSketch("XY"), AddRectangle("sk1", 0, 0, 40, 40),
                 Extrude("sk1", 10.0)),
            why_fine="a lie: the closed form below is ten times the real volume",
            volume=160000.0,          # the truth is 16000
            bbox=(40.0, 40.0, 10.0), min_feature=10.0, genus=0)
        cert = oracle.certify(bogus)
        self.assertFalse(cert.certified)
        self.assertIn("closed form", cert.reason)

    def test_a_real_part_is_certified(self):
        from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch

        good = attacks.Attack(
            name="plate", family="uncovered",
            ops=(NewSketch("XY"), AddRectangle("sk1", 0, 0, 60, 40),
                 Extrude("sk1", 10.0)),
            why_fine="a 60x40x10 plate. Volume 24000 mm3 by arithmetic.",
            volume=24000.0, bbox=(60.0, 40.0, 10.0), min_feature=10.0, genus=0)
        cert = oracle.certify(good)
        self.assertTrue(cert.certified, cert.reason)
        self.assertTrue(cert.gate_ok)


class TestRedTeamRun(unittest.TestCase):

    def test_a_small_run_reports_the_triple(self):
        """(verifier, op_stream, why_the_part_is_actually_fine). Whatever it finds
        or does not find, the SHAPE of a finding is the deliverable."""
        r = redteam_run.run(limit=3)
        self.assertGreater(r.attacks, 0)
        for fp in r.false_positives + r.false_alarms:
            self.assertTrue(fp.verifier)
            self.assertTrue(fp.ops)
            self.assertTrue(fp.why_the_part_is_fine)
            self.assertTrue(fp.proof)
        # It must never claim a false positive on a part it did not certify.
        certified_names = set(r.rejected_parts) | set(r.warned_parts)
        uncertified_names = {u["attack"] for u in r.uncertified}
        self.assertEqual(certified_names & uncertified_names, set())

    def test_the_report_is_serialisable(self):
        r = redteam_run.run(limit=2)
        d = r.to_dict()
        self.assertIn("false_positive_rate", d)
        self.assertIn("false_positives", d)

    @unittest.skipUnless(FULL, "the full adversarial sweep certifies ~40 parts on "
                               "a grid-marching engine and runs the whole fleet "
                               "over each; it takes minutes. Set "
                               "HARNESSCAD_REDTEAM_FULL=1 to run it.")
    def test_full_sweep(self):
        r = redteam_run.run()
        # This test does NOT assert zero false positives. The red team's job is to
        # REPORT them; a concurrent agent owns eval/verifiers/. Asserting zero here
        # would create exactly the incentive that rigs the result -- the next
        # person to see it red would fix the verifier to make it green.
        self.assertGreater(r.certified, 10)
        print(redteam_run.format_text(r))


if __name__ == "__main__":
    unittest.main()
