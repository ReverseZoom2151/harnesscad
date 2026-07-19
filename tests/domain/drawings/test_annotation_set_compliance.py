# Tests for harnesscad.domain.drawings.annotation_set_compliance.
#
# Known-bad vectors and the iff-shaped properties are ported from the
# CAD-Annotator reference repo (Apache-2.0):
#   artifacts/api-server/src/lib/compliance-engine.test.ts   (Properties 5-8)
#   artifacts/cad-annotator/src/lib/compliance-summary.test.ts (Property 10)
# Credit: CAD-Annotator contributors, Apache License 2.0.
#
# PROPERTY-TEST SUBSTITUTION: the upstream tests use fast-check. `hypothesis`
# is not installed in this harness and must not be added, so every fast-check
# property below is reproduced as (a) an EXHAUSTIVE table-driven sweep with
# itertools.product over the full small domain (all 14 geometric
# characteristics x datum counts 0-4, x all 4 material conditions, etc.) and
# (b) random sampling with random.Random(SEED) where the domain is too large
# to enumerate. SEED = 20260719 (fixed, so failures reproduce exactly).
# Both directions of every iff are asserted.

import itertools
import random
import unittest

from harnesscad.domain.drawings.annotation_schema import (
    BoundingBox,
    DatumAnnotation,
    DimensionAnnotation,
    FcfAnnotation,
    NoteAnnotation,
    SurfaceFinishAnnotation,
)
from harnesscad.domain.drawings.annotation_set_compliance import (
    DATUM_COUNT_RANGE,
    MMC_LMC_PERMITTED,
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ComplianceIssue,
    ComplianceSummaryCounts,
    check_datum_ref_exists,
    check_duplicate_datum_letters,
    check_fcf_datum_count,
    check_mmc_lmc_applicability,
    check_tolerance_positive,
    compute_compliance_summary,
    main,
    validate_compliance,
)

SEED = 20260719

ALL_CHARACTERISTICS = tuple(sorted(DATUM_COUNT_RANGE))
ALL_MATERIAL_CONDITIONS = ("MMC", "LMC", "RFS", None)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _bbox():
    return BoundingBox(x=1.0, y=2.0, width=5.0, height=5.0, color="green")


def _base(ann_id, label="L", value="V"):
    return {
        "id": ann_id,
        "label": label,
        "value": value,
        "view": "Front View",
        "bounding_box": _bbox(),
        "confidence": 0.9,
    }


def fcf(ann_id, characteristic="position", tolerance=1.0, mc=None, datums=()):
    return FcfAnnotation(
        geometric_characteristic=characteristic,
        tolerance_value=tolerance,
        material_condition=mc,
        datum_references=tuple(datums),
        **_base(ann_id)
    )


def datum(ann_id, letter):
    return DatumAnnotation(datum_letter=letter, **_base(ann_id))


def note(ann_id):
    return NoteAnnotation(**_base(ann_id))


def dimension(ann_id):
    return DimensionAnnotation(
        dimension_type="linear", nominal_value=10.0, **_base(ann_id)
    )


def surface(ann_id):
    return SurfaceFinishAnnotation(roughness_value=1.6, **_base(ann_id))


def rule_ids(issues, ann_id=None):
    return sorted(
        i.rule_id for i in issues if ann_id is None or i.annotation_id == ann_id
    )


# --------------------------------------------------------------------------- #
# Known-bad vectors (ported from compliance-engine.test.ts / the harness CLI
# selfcheck fixture). A refusal predicate with no known-bad inputs is untested.
# --------------------------------------------------------------------------- #


class KnownBadFcfDatumCountTests(unittest.TestCase):
    def test_flatness_with_a_datum_is_bad(self):
        # form control, range (0, 0)
        issues = check_fcf_datum_count([fcf("a1", "flatness", datums=("A",))])
        self.assertEqual(rule_ids(issues), ["FCF_DATUM_COUNT"])
        self.assertEqual(issues[0].severity, SEVERITY_ERROR)
        self.assertEqual(issues[0].annotation_id, "a1")
        self.assertIn("flatness requires 0 datum reference(s), but 1", issues[0].description)

    def test_position_with_one_datum_is_bad(self):
        issues = check_fcf_datum_count([fcf("a1", "position", datums=("A",))])
        self.assertEqual(rule_ids(issues), ["FCF_DATUM_COUNT"])
        # range is a span, so the message uses the "lo-hi" form
        self.assertIn("requires 2-3 datum reference(s)", issues[0].description)

    def test_position_with_four_datums_is_bad(self):
        issues = check_fcf_datum_count(
            [fcf("a1", "position", datums=("A", "B", "C", "D"))]
        )
        self.assertEqual(rule_ids(issues), ["FCF_DATUM_COUNT"])
        self.assertIn("but 4 provided", issues[0].description)

    def test_concentricity_needs_exactly_one(self):
        self.assertEqual(check_fcf_datum_count([fcf("a", "concentricity", datums=("A",))]), [])
        bad = check_fcf_datum_count([fcf("a", "concentricity", datums=("A", "B"))])
        self.assertEqual(rule_ids(bad), ["FCF_DATUM_COUNT"])

    def test_symmetry_needs_exactly_three(self):
        self.assertEqual(
            check_fcf_datum_count([fcf("a", "symmetry", datums=("A", "B", "C"))]), []
        )
        self.assertEqual(rule_ids(check_fcf_datum_count([fcf("a", "symmetry")])),
                         ["FCF_DATUM_COUNT"])

    def test_unknown_characteristic_is_skipped_not_flagged(self):
        # Harness behaviour: DATUM_COUNT_RANGE.get(...) is None -> rule abstains.
        # (annotation_schema would have dropped such an FCF at parse time.)
        self.assertEqual(check_fcf_datum_count([fcf("a", "bogusControl", datums=("A",))]), [])

    def test_non_fcf_annotations_are_ignored(self):
        anns = [datum("d", "A"), note("n"), dimension("dim"), surface("s")]
        self.assertEqual(check_fcf_datum_count(anns), [])


class KnownBadDatumRefTests(unittest.TestCase):
    def test_dangling_datum_reference(self):
        anns = [datum("d1", "A"), fcf("f1", "profileOfSurface", datums=("A", "D"))]
        issues = check_datum_ref_exists(anns)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rule_id, "DATUM_REF_EXISTS")
        self.assertEqual(issues[0].annotation_id, "f1")
        self.assertEqual(issues[0].severity, SEVERITY_ERROR)
        self.assertIn('"D"', issues[0].description)

    def test_no_datums_declared_at_all(self):
        issues = check_datum_ref_exists([fcf("f1", "profileOfSurface", datums=("A", "B"))])
        self.assertEqual(len(issues), 2)
        self.assertEqual({i.annotation_id for i in issues}, {"f1"})

    def test_all_declared_is_clean(self):
        anns = [datum("d1", "A"), datum("d2", "B"), fcf("f1", "position", datums=("A", "B"))]
        self.assertEqual(check_datum_ref_exists(anns), [])

    def test_repeated_dangling_ref_yields_one_issue_per_occurrence(self):
        issues = check_datum_ref_exists([fcf("f1", "profileOfSurface", datums=("Z", "Z"))])
        self.assertEqual(len(issues), 2)

    def test_declaration_order_does_not_matter(self):
        anns = [fcf("f1", "profileOfSurface", datums=("A",)), datum("d1", "A")]
        self.assertEqual(check_datum_ref_exists(anns), [])


class KnownBadMmcLmcTests(unittest.TestCase):
    def test_mmc_on_flatness_is_bad(self):
        issues = check_mmc_lmc_applicability([fcf("f1", "flatness", mc="MMC")])
        self.assertEqual(rule_ids(issues), ["MMC_LMC_APPLICABILITY"])
        self.assertEqual(issues[0].severity, SEVERITY_ERROR)
        self.assertIn("MMC is not permitted for flatness", issues[0].description)

    def test_lmc_on_perpendicularity_is_bad(self):
        issues = check_mmc_lmc_applicability([fcf("f1", "perpendicularity", mc="LMC")])
        self.assertEqual(rule_ids(issues), ["MMC_LMC_APPLICABILITY"])

    def test_rfs_and_none_never_fire(self):
        for characteristic in ALL_CHARACTERISTICS:
            for mc in ("RFS", None):
                self.assertEqual(
                    check_mmc_lmc_applicability([fcf("f", characteristic, mc=mc)]),
                    [],
                    "%s / %s" % (characteristic, mc),
                )

    def test_permitted_characteristics_accept_mmc(self):
        for characteristic in sorted(MMC_LMC_PERMITTED):
            self.assertEqual(
                check_mmc_lmc_applicability([fcf("f", characteristic, mc="MMC")]), []
            )

    def test_unknown_characteristic_with_mmc_is_flagged(self):
        # Divergence note: unlike FCF_DATUM_COUNT (which abstains on unknown
        # characteristics), this rule uses a membership test, so an unknown
        # characteristic is NOT in MMC_LMC_PERMITTED and does fire.
        issues = check_mmc_lmc_applicability([fcf("f", "bogusControl", mc="MMC")])
        self.assertEqual(rule_ids(issues), ["MMC_LMC_APPLICABILITY"])


class KnownBadToleranceTests(unittest.TestCase):
    def test_zero_tolerance_is_bad(self):
        issues = check_tolerance_positive([fcf("f1", tolerance=0.0)])
        self.assertEqual(rule_ids(issues), ["TOLERANCE_POSITIVE"])
        self.assertEqual(issues[0].severity, SEVERITY_ERROR)
        # _fmt_num renders integral floats JS-style, with no trailing ".0"
        self.assertIn("but got 0.", issues[0].description)

    def test_negative_tolerance_is_bad(self):
        issues = check_tolerance_positive([fcf("f1", tolerance=-0.1)])
        self.assertEqual(rule_ids(issues), ["TOLERANCE_POSITIVE"])
        self.assertIn("-0.1", issues[0].description)

    def test_integral_negative_tolerance_formats_without_decimal(self):
        issues = check_tolerance_positive([fcf("f1", tolerance=-3.0)])
        self.assertIn("but got -3.", issues[0].description)

    def test_positive_tolerance_is_clean(self):
        self.assertEqual(check_tolerance_positive([fcf("f1", tolerance=0.001)]), [])

    def test_non_fcf_ignored(self):
        self.assertEqual(check_tolerance_positive([datum("d", "A"), note("n")]), [])


class KnownBadDuplicateDatumLetterTests(unittest.TestCase):
    """DUPLICATE_DATUM_LETTER is a HARNESS-ONLY rule -- it has no counterpart in
    compliance-engine.ts, so these vectors are original, not ported."""

    def test_two_datums_same_letter(self):
        anns = [datum("d1", "A"), datum("d2", "A")]
        issues = check_duplicate_datum_letters(anns)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rule_id, "DUPLICATE_DATUM_LETTER")
        self.assertEqual(issues[0].severity, SEVERITY_ERROR)
        # The SECOND annotation is blamed; the message names the FIRST one.
        self.assertEqual(issues[0].annotation_id, "d2")
        self.assertIn('"A"', issues[0].description)
        self.assertIn('"d1"', issues[0].description)

    def test_triplicate_blames_every_repeat_against_the_first(self):
        anns = [datum("d1", "A"), datum("d2", "A"), datum("d3", "A")]
        issues = check_duplicate_datum_letters(anns)
        self.assertEqual([i.annotation_id for i in issues], ["d2", "d3"])
        for issue in issues:
            self.assertIn('"d1"', issue.description)

    def test_distinct_letters_are_clean(self):
        anns = [datum("d%d" % i, letter) for i, letter in enumerate("ABC")]
        self.assertEqual(check_duplicate_datum_letters(anns), [])

    def test_fcf_datum_references_do_not_count_as_declarations(self):
        anns = [datum("d1", "A"), fcf("f1", "position", datums=("A", "A", "B"))]
        self.assertEqual(check_duplicate_datum_letters(anns), [])

    def test_rule_is_wired_into_validate_compliance(self):
        # Guards the harness divergence: the extra rule must actually run in
        # the aggregate entry point, not just standalone.
        issues = validate_compliance([datum("d1", "A"), datum("d2", "A")])
        self.assertIn("DUPLICATE_DATUM_LETTER", rule_ids(issues))


class KnownBadCombinedSetTests(unittest.TestCase):
    def test_multi_violation_annotation_collects_every_rule(self):
        # flatness + a datum + MMC + negative tolerance + dangling ref
        anns = [fcf("f1", "flatness", tolerance=-0.02, mc="MMC", datums=("Q",))]
        self.assertEqual(
            rule_ids(validate_compliance(anns)),
            [
                "DATUM_REF_EXISTS",
                "FCF_DATUM_COUNT",
                "MMC_LMC_APPLICABILITY",
                "TOLERANCE_POSITIVE",
            ],
        )

    def test_empty_set_is_clean(self):
        self.assertEqual(validate_compliance([]), [])
        self.assertEqual(
            compute_compliance_summary([], []),
            ComplianceSummaryCounts(errors=0, warnings=0, passing=0),
        )

    def test_fully_valid_set_is_clean(self):
        anns = [
            datum("d1", "A"),
            datum("d2", "B"),
            datum("d3", "C"),
            fcf("f1", "position", tolerance=0.05, mc="MMC", datums=("A", "B", "C")),
            fcf("f2", "flatness", tolerance=0.02),
            note("n1"),
        ]
        self.assertEqual(validate_compliance(anns), [])
        summary = compute_compliance_summary(anns, [])
        self.assertEqual(summary.passing, len(anns))

    def test_rule_output_order_is_grouped_by_rule(self):
        anns = [fcf("f1", "flatness", tolerance=-1.0, datums=("Z",)), datum("d1", "Z")]
        self.assertEqual(
            [i.rule_id for i in validate_compliance(anns)],
            ["FCF_DATUM_COUNT", "TOLERANCE_POSITIVE"],
        )

    def test_issue_to_dict_round_trip(self):
        issue = ComplianceIssue("a1", "TOLERANCE_POSITIVE", SEVERITY_ERROR, "d")
        self.assertEqual(
            issue.to_dict(),
            {
                "annotation_id": "a1",
                "rule_id": "TOLERANCE_POSITIVE",
                "severity": SEVERITY_ERROR,
                "description": "d",
            },
        )

    def test_counts_to_dict(self):
        self.assertEqual(
            ComplianceSummaryCounts(1, 2, 3).to_dict(),
            {"errors": 1, "warnings": 2, "passing": 3},
        )


class SelfcheckCliTests(unittest.TestCase):
    def test_selfcheck_returns_zero(self):
        self.assertEqual(main(["--selfcheck"]), 0)

    def test_selfcheck_json_returns_zero(self):
        self.assertEqual(main(["--selfcheck", "--json"]), 0)

    def test_no_args_prints_help(self):
        self.assertEqual(main([]), 0)


# --------------------------------------------------------------------------- #
# IFF-properties (fast-check Properties 5-8 and 10, reproduced exhaustively).
# Each asserts BOTH directions: the rule fires when the predicate holds AND
# does not fire when it does not.
# --------------------------------------------------------------------------- #


class Property5DatumCountIffTests(unittest.TestCase):
    """Upstream Property 5: FCF_DATUM_COUNT issue iff count outside range.
    Exhaustive over 14 characteristics x datum counts 0..4 (70 cases)."""

    def test_iff_exhaustive(self):
        checked = 0
        for characteristic, count in itertools.product(
            ALL_CHARACTERISTICS, range(0, 5)
        ):
            ann = fcf("f1", characteristic, datums=tuple(LETTERS[:count]))
            issues = check_fcf_datum_count([ann])
            lo, hi = DATUM_COUNT_RANGE[characteristic]
            should_fire = count < lo or count > hi
            fired = any(
                i.rule_id == "FCF_DATUM_COUNT" and i.annotation_id == "f1"
                for i in issues
            )
            self.assertEqual(fired, should_fire, "%s/%d" % (characteristic, count))
            if fired:
                self.assertEqual(issues[0].severity, SEVERITY_ERROR)
                self.assertTrue(issues[0].description)
                self.assertIn(str(count), issues[0].description)
            checked += 1
        self.assertEqual(checked, len(ALL_CHARACTERISTICS) * 5)

    def test_iff_holds_when_batched_into_one_set(self):
        # Same property, but every FCF in a single call -- proves the rule does
        # not depend on the set containing exactly one annotation.
        anns = []
        expected = set()
        for characteristic, count in itertools.product(ALL_CHARACTERISTICS, range(0, 5)):
            ann_id = "%s_%d" % (characteristic, count)
            anns.append(fcf(ann_id, characteristic, datums=tuple(LETTERS[:count])))
            lo, hi = DATUM_COUNT_RANGE[characteristic]
            if count < lo or count > hi:
                expected.add(ann_id)
        fired = {i.annotation_id for i in check_fcf_datum_count(anns)}
        self.assertEqual(fired, expected)


class Property6DatumRefIffTests(unittest.TestCase):
    """Upstream Property 6: one DATUM_REF_EXISTS issue per referenced letter
    that is not declared, and none for declared letters. Randomised over
    declared/referenced letter subsets with random.Random(SEED)."""

    def test_iff_random_sampling(self):
        rng = random.Random(SEED)
        for _ in range(400):
            declared = rng.sample(LETTERS[:6], rng.randint(0, 5))
            referenced = rng.sample(LETTERS[:6], rng.randint(1, 3))
            anns = [datum("d%d" % i, letter) for i, letter in enumerate(declared)]
            anns.append(fcf("f1", "profileOfSurface", datums=tuple(referenced)))
            issues = check_datum_ref_exists(anns)

            for ref in referenced:
                fired = any('"%s"' % ref in i.description for i in issues)
                self.assertEqual(
                    fired,
                    ref not in set(declared),
                    "ref=%s declared=%s" % (ref, declared),
                )
            # no spurious issues: every issue is this rule, on the FCF, error
            for issue in issues:
                self.assertEqual(issue.rule_id, "DATUM_REF_EXISTS")
                self.assertEqual(issue.annotation_id, "f1")
                self.assertEqual(issue.severity, SEVERITY_ERROR)
            self.assertEqual(
                len(issues), len([r for r in referenced if r not in set(declared)])
            )

    def test_iff_exhaustive_single_letter(self):
        # Exhaustive over the whole alphabet with a single declared letter.
        for declared_letter, referenced_letter in itertools.product("ABC", LETTERS):
            anns = [
                datum("d1", declared_letter),
                fcf("f1", "profileOfSurface", datums=(referenced_letter,)),
            ]
            fired = bool(check_datum_ref_exists(anns))
            self.assertEqual(fired, declared_letter != referenced_letter)


class Property7MmcLmcIffTests(unittest.TestCase):
    """Upstream Property 7: MMC_LMC_APPLICABILITY iff the characteristic is not
    in MMC_LMC_PERMITTED AND the modifier is MMC or LMC. Exhaustive over
    14 characteristics x {MMC, LMC, RFS, None} (56 cases)."""

    def test_iff_exhaustive(self):
        for characteristic, mc in itertools.product(
            ALL_CHARACTERISTICS, ALL_MATERIAL_CONDITIONS
        ):
            issues = check_mmc_lmc_applicability([fcf("f1", characteristic, mc=mc)])
            should_fire = mc in ("MMC", "LMC") and characteristic not in MMC_LMC_PERMITTED
            fired = any(
                i.rule_id == "MMC_LMC_APPLICABILITY" and i.annotation_id == "f1"
                for i in issues
            )
            self.assertEqual(fired, should_fire, "%s/%s" % (characteristic, mc))
            if fired:
                self.assertEqual(issues[0].severity, SEVERITY_ERROR)
                self.assertIn(str(mc), issues[0].description)
                self.assertIn(characteristic, issues[0].description)


class Property8ToleranceIffTests(unittest.TestCase):
    """Upstream Property 8: TOLERANCE_POSITIVE iff tolerance_value <= 0.
    Boundary table plus random sampling over [-1000, 1000] with
    random.Random(SEED)."""

    BOUNDARIES = (-1000.0, -1.0, -1e-9, -0.0, 0.0, 1e-9, 0.001, 1.0, 1000.0)

    def test_iff_boundaries(self):
        for value in self.BOUNDARIES:
            issues = check_tolerance_positive([fcf("f1", tolerance=value)])
            fired = any(i.rule_id == "TOLERANCE_POSITIVE" for i in issues)
            self.assertEqual(fired, value <= 0, "tolerance=%r" % value)
            if fired:
                self.assertEqual(issues[0].severity, SEVERITY_ERROR)

    def test_iff_random_sampling(self):
        rng = random.Random(SEED)
        for _ in range(500):
            value = rng.uniform(-1000.0, 1000.0)
            fired = bool(check_tolerance_positive([fcf("f1", tolerance=value)]))
            self.assertEqual(fired, value <= 0, "tolerance=%r" % value)


class ValidateComplianceIffTests(unittest.TestCase):
    """Aggregate iff: validate_compliance returns an empty list iff no
    individual rule fires. Exhaustive over a small cross-product of set shapes
    (characteristic x datum count x material condition x tolerance sign x
    whether the referenced datums are declared)."""

    def test_compliant_iff_no_rule_fires(self):
        rules = (
            check_fcf_datum_count,
            check_datum_ref_exists,
            check_mmc_lmc_applicability,
            check_tolerance_positive,
            check_duplicate_datum_letters,
        )
        cases = 0
        for characteristic, count, mc, tol, declare, dup in itertools.product(
            ("position", "flatness", "perpendicularity", "symmetry", "profileOfSurface"),
            (0, 1, 2, 3),
            ("MMC", "RFS", None),
            (-1.0, 0.0, 0.5),
            (True, False),
            (True, False),
        ):
            refs = tuple(LETTERS[:count])
            anns = []
            if declare:
                anns.extend(datum("d%d" % i, letter) for i, letter in enumerate(refs))
            if dup:
                anns.append(datum("dup1", "Z"))
                anns.append(datum("dup2", "Z"))
            anns.append(fcf("f1", characteristic, tolerance=tol, mc=mc, datums=refs))

            aggregate = validate_compliance(anns)
            per_rule = [i for rule in rules for i in rule(anns)]

            # direction 1: aggregate is empty iff no rule fires
            self.assertEqual(bool(aggregate), bool(per_rule))
            # direction 2: aggregate is exactly the union of the rules
            self.assertEqual(sorted(map(id_key, aggregate)), sorted(map(id_key, per_rule)))
            cases += 1
        self.assertEqual(cases, 5 * 4 * 3 * 3 * 2 * 2)

    def test_dup_flag_alone_decides_duplicate_rule(self):
        clean = validate_compliance([datum("d1", "Z")])
        dirty = validate_compliance([datum("d1", "Z"), datum("d2", "Z")])
        self.assertEqual(clean, [])
        self.assertEqual(rule_ids(dirty), ["DUPLICATE_DATUM_LETTER"])


def id_key(issue):
    return (issue.annotation_id, issue.rule_id, issue.severity, issue.description)


class Property10SummaryTests(unittest.TestCase):
    """Upstream Property 10 (compliance-summary.test.ts): for all annotation
    sets and issue lists, errors + warnings + passing == len(annotations).
    Randomised with random.Random(SEED)."""

    RULES = (
        "FCF_DATUM_COUNT",
        "DATUM_REF_EXISTS",
        "MMC_LMC_APPLICABILITY",
        "TOLERANCE_POSITIVE",
        "DUPLICATE_DATUM_LETTER",
    )

    def test_counts_sum_to_total(self):
        rng = random.Random(SEED)
        for _ in range(300):
            n = rng.randint(0, 30)
            anns = [note("ann_%d" % i) for i in range(n)]
            ids = [a.id for a in anns]
            issues = []
            if ids:
                for _ in range(rng.randint(0, 20)):
                    issues.append(
                        ComplianceIssue(
                            annotation_id=rng.choice(ids),
                            rule_id=rng.choice(self.RULES),
                            severity=rng.choice([SEVERITY_ERROR, SEVERITY_WARNING]),
                            description="d",
                        )
                    )
            s = compute_compliance_summary(anns, issues)
            self.assertEqual(s.errors + s.warnings + s.passing, n)
            self.assertGreaterEqual(s.errors, 0)
            self.assertGreaterEqual(s.warnings, 0)
            self.assertGreaterEqual(s.passing, 0)

    def test_no_issues_means_all_passing(self):
        rng = random.Random(SEED)
        for _ in range(100):
            n = rng.randint(1, 20)
            anns = [note("ann_%d" % i) for i in range(n)]
            s = compute_compliance_summary(anns, [])
            self.assertEqual(s, ComplianceSummaryCounts(errors=0, warnings=0, passing=n))

    def test_any_error_severity_makes_the_annotation_an_error(self):
        rng = random.Random(SEED)
        for _ in range(100):
            k = rng.randint(1, 5)
            issues = [
                ComplianceIssue(
                    "ann_0",
                    rng.choice(self.RULES),
                    rng.choice([SEVERITY_ERROR, SEVERITY_WARNING]),
                    "d",
                )
                for _ in range(k)
            ]
            issues[0] = ComplianceIssue(
                issues[0].annotation_id, issues[0].rule_id, SEVERITY_ERROR, "d"
            )
            s = compute_compliance_summary([note("ann_0")], issues)
            self.assertEqual(s, ComplianceSummaryCounts(errors=1, warnings=0, passing=0))

    def test_only_warnings_makes_the_annotation_a_warning(self):
        rng = random.Random(SEED)
        for _ in range(100):
            k = rng.randint(1, 5)
            issues = [
                ComplianceIssue("ann_0", rng.choice(self.RULES), SEVERITY_WARNING, "d")
                for _ in range(k)
            ]
            s = compute_compliance_summary([note("ann_0")], issues)
            self.assertEqual(s, ComplianceSummaryCounts(errors=0, warnings=1, passing=0))

    def test_issues_for_unknown_annotation_ids_are_ignored(self):
        anns = [note("ann_0")]
        issues = [ComplianceIssue("ghost", "FCF_DATUM_COUNT", SEVERITY_ERROR, "d")]
        s = compute_compliance_summary(anns, issues)
        self.assertEqual(s, ComplianceSummaryCounts(errors=0, warnings=0, passing=1))

    def test_unknown_severity_is_bucketed_as_warning(self):
        # Harness behaviour (not a ported vector): the classifier is
        # "error in severities -> error, else warning", so an out-of-vocabulary
        # severity such as "info" lands in the warning bucket. Documented here
        # so a future change to the bucketing is caught.
        s = compute_compliance_summary(
            [note("ann_0")], [ComplianceIssue("ann_0", "R", "info", "d")]
        )
        self.assertEqual(s, ComplianceSummaryCounts(errors=0, warnings=1, passing=0))

    def test_end_to_end_summary_over_real_rule_output(self):
        anns = [
            datum("d1", "A"),
            datum("d2", "A"),  # DUPLICATE_DATUM_LETTER on d2
            fcf("f1", "position", tolerance=0.05, datums=("A", "B")),  # DATUM_REF_EXISTS B
            note("n1"),
        ]
        issues = validate_compliance(anns)
        s = compute_compliance_summary(anns, issues)
        self.assertEqual(s, ComplianceSummaryCounts(errors=2, warnings=0, passing=2))
        self.assertEqual(s.errors + s.warnings + s.passing, len(anns))


if __name__ == "__main__":
    unittest.main()
