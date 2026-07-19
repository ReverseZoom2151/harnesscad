"""The adversarial safety corpus is only worth anything if it is RUN against
the checker it exists to test. These tests are that guarantee, in the suite.

They assert the three properties the corpus was built to prove:

* every ``attack`` snippet is REJECTED by ``check_cad_code`` -- and with the
  specific violation codes it is defined to raise;
* every ``benign`` snippet is ACCEPTED -- the checker does not over-refuse;
* the documented ``gap`` snippets are still UNCAUGHT -- recorded honestly, and
  this test is where a newly-closed gap is discovered.

No geometry kernel is imported: the checker is pure AST.
"""

from __future__ import annotations

import unittest

from harnesscad.domain.programs.validate.code_safety import check_cad_code
from harnesscad.eval.corpus.fixtures import adversarial_code as adv


class TestCorpusWiredIntoHub(unittest.TestCase):
    def test_loader_is_reachable_through_the_package_hub(self):
        from harnesscad.eval.corpus import fixtures
        self.assertIn("adversarial_code", fixtures.LOADERS)
        self.assertIs(fixtures.loader("adversarial_code"), adv)

    def test_manifest_is_authored_not_resources_backed(self):
        m = adv.manifest()
        self.assertEqual(m.license, "REIMPLEMENTED")
        self.assertFalse(m.verify_vendored(), "vendored sha mismatch")
        for e in m.entries:
            self.assertTrue(e.vendored, e.name)
            self.assertIsNone(e.resource, e.name)

    def test_corpus_has_bad_and_good_cases(self):
        self.assertGreaterEqual(len(adv.attack_cases()), 15)
        self.assertGreaterEqual(len(adv.benign_cases()), 3)


class TestCheckerRejectsEveryAttack(unittest.TestCase):
    def test_every_attack_is_flagged_with_expected_codes(self):
        slipped = []
        for case in adv.attack_cases():
            report = check_cad_code(case.snippet(), kernel=case.kernel,
                                    required_def=None)
            if report.ok:
                slipped.append((case.name, case.snippet()))
                continue
            missing = set(case.expected_codes) - set(report.codes())
            self.assertFalse(
                missing,
                "attack %s fired but missed expected codes %s (got %s)"
                % (case.name, sorted(missing), sorted(set(report.codes()))))
        self.assertEqual(
            slipped, [],
            "CHECKER GAP: attack(s) slipped past check_cad_code: %s"
            % [name for name, _ in slipped])


class TestCheckerAcceptsEveryBenign(unittest.TestCase):
    def test_no_benign_case_is_over_refused(self):
        for case in adv.benign_cases():
            report = check_cad_code(case.snippet(), kernel=case.kernel,
                                    required_def=None)
            self.assertTrue(
                report.ok,
                "benign %s over-refused: %s"
                % (case.name, sorted(set(report.codes()))))


class TestDocumentedGapsAreStillOpen(unittest.TestCase):
    """A documented gap is a case check_cad_code returns ok=True on -- a KNOWN
    hole, kept visible until closed. The two originally here (subclasses-escape,
    breakpoint) have since been closed and moved to the attack set, so the gap
    set is empty and the goal state. If a future pass adds a gap, this test
    keeps it honestly uncaught until someone fixes it; the day the checker
    catches it, this fails and forces the case to move to _ATTACKS."""

    def test_gaps_remain_uncaught(self):
        for case in adv.gap_cases():
            report = check_cad_code(case.snippet(), kernel=case.kernel,
                                    required_def=None)
            self.assertTrue(
                report.ok,
                "documented gap %s is now CAUGHT (%s) -- update the corpus"
                % (case.name, sorted(set(report.codes()))))

    def test_the_two_original_gaps_are_now_closed(self):
        """Regression: the subclasses-escape and breakpoint holes stay shut."""
        by_name = {c.name: c for c in adv.attack_cases()}
        for name in ("gap_subclasses_escape", "gap_breakpoint"):
            self.assertIn(name, by_name, "%s must be an attack now" % name)
            report = check_cad_code(by_name[name].snippet(),
                                    kernel=by_name[name].kernel,
                                    required_def=None)
            self.assertFalse(report.ok, "%s regressed to uncaught" % name)


class TestSelfcheckPasses(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        self.assertEqual(adv.main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
