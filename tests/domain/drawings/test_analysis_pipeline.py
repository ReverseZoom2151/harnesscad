"""Tests for harnesscad.domain.drawings.analysis_pipeline.

Known-bad vectors and orchestration cases are ported from the CAD-Annotator
reference repo (Apache-2.0), file
artifacts/api-server/src/lib/pipeline-orchestrator.test.ts. Where the harness
diverges from the TypeScript orchestrator the divergence is called out in a
comment and the HARNESS behaviour is asserted.

hypothesis is not installed in this repo, so the "property" style cases below
are table-driven exhaustive enumerations over small domains using stdlib
itertools plus random.Random(FIXED_SEED) (FIXED_SEED = 20260719). This
substitutes for the fast-check generators used upstream.
"""

import itertools
import json
import random
import unittest
from dataclasses import replace as dc_replace
from unittest import mock

from harnesscad.domain.drawings.analysis_pipeline import (
    STAGES,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    AnalysisSession,
    AnnotationEdit,
    StageError,
    apply_annotation_edit,
    main,
    run_analysis_pipeline,
    session_status,
)
from harnesscad.domain.drawings.annotation_schema import parse_annotation_response

FIXED_SEED = 20260719

MODULE = "harnesscad.domain.drawings.analysis_pipeline"


# --------------------------------------------------------------------------- #
# Detection-response builders (port of makeDatum/makeFcf/makeDimension +
# makeDetectionResponse from pipeline-orchestrator.test.ts). The harness takes
# raw response TEXT rather than an OpenAI choices envelope, so the envelope is
# collapsed into the JSON payload.
# --------------------------------------------------------------------------- #

_BOX = {"x": 10, "y": 20, "width": 15, "height": 8, "color": "green"}


def make_datum(letter, ann_id=None):
    return {
        "type": "datum",
        "id": ann_id or ("datum_%s" % letter),
        "label": "Datum %s" % letter,
        "value": letter,
        "view": "Front View",
        "boundingBox": dict(_BOX),
        "confidence": 0.95,
        "datumLetter": letter,
    }


def make_fcf(**overrides):
    raw = {
        "type": "fcf",
        "id": "fcf_1",
        "label": "Position 0.05 MMC A B C",
        "value": "0.05",
        "view": "Front View",
        "boundingBox": dict(_BOX),
        "confidence": 0.9,
        "geometricCharacteristic": "position",
        "toleranceValue": 0.05,
        "materialCondition": "MMC",
        "datumReferences": ["A", "B", "C"],
    }
    raw.update(overrides)
    return raw


def make_dimension(**overrides):
    raw = {
        "type": "dimension",
        "id": "dim_1",
        "label": "40.2 +/-0.1",
        "value": "40.2",
        "view": "Front View",
        "boundingBox": dict(_BOX),
        "confidence": 0.95,
        "dimensionType": "linear",
        "nominalValue": 40.2,
        "plusTolerance": 0.1,
        "minusTolerance": -0.1,
        "unit": "mm",
    }
    raw.update(overrides)
    return raw


_DEFAULT_VIEWS = ["Front View", "Side View"]
_UNSET = object()


def detection_text(annotations, views=_UNSET, description="Test drawing"):
    """Build a raw detection response. ``views`` is embedded VERBATIM (so
    malformed view payloads can be exercised); omit it for the default."""
    return json.dumps(
        {
            "annotations": list(annotations),
            "views": list(_DEFAULT_VIEWS) if views is _UNSET else views,
            "description": description,
        }
    )


def run_text(text, **kwargs):
    return run_analysis_pipeline(lambda: text, **kwargs)


def raiser(exc):
    def _raise(*_args, **_kwargs):
        raise exc

    return _raise


# --------------------------------------------------------------------------- #
# session_status -- persistSession status rule
# --------------------------------------------------------------------------- #


class SessionStatusTests(unittest.TestCase):
    def test_completed_when_no_errors(self):
        self.assertEqual(session_status([]), STATUS_COMPLETED)

    def test_failed_when_detection_error(self):
        errs = [StageError(stage="detection", message="Failed")]
        self.assertEqual(session_status(errs), STATUS_FAILED)

    def test_partial_when_non_detection_error(self):
        errs = [StageError(stage="compliance", message="Failed")]
        self.assertEqual(session_status(errs), STATUS_PARTIAL)

    def test_detection_error_dominates_regardless_of_position(self):
        errs = [
            StageError(stage="dfm", message="LLM timeout"),
            StageError(stage="detection", message="API down"),
        ]
        self.assertEqual(session_status(errs), STATUS_FAILED)

    def test_exhaustive_over_all_stage_subsets(self):
        # Substitutes for a fast-check property (hypothesis unavailable):
        # enumerate every subset of the 4 stages exhaustively -- 16 cases.
        for size in range(len(STAGES) + 1):
            for combo in itertools.combinations(STAGES, size):
                errs = [StageError(stage=s, message="boom") for s in combo]
                expected = (
                    STATUS_FAILED
                    if "detection" in combo
                    else (STATUS_PARTIAL if combo else STATUS_COMPLETED)
                )
                self.assertEqual(session_status(errs), expected, combo)

    def test_order_independent_over_shuffled_permutations(self):
        # Random.Random(FIXED_SEED) shuffles, seed 20260719: status must not
        # depend on the order errors were collected in.
        rng = random.Random(FIXED_SEED)
        errs = [StageError(stage=s, message="boom") for s in STAGES]
        for _ in range(20):
            rng.shuffle(errs)
            self.assertEqual(session_status(list(errs)), STATUS_FAILED)

    def test_unknown_stage_name_is_partial_not_failed(self):
        # Only the literal "detection" is fatal; anything else degrades.
        self.assertEqual(
            session_status([StageError(stage="Detection", message="x")]),
            STATUS_PARTIAL,
        )


# --------------------------------------------------------------------------- #
# Sequential stage execution (TS: describe "sequential stage execution")
# --------------------------------------------------------------------------- #


class HappyPathTests(unittest.TestCase):
    def test_runs_full_pipeline_and_returns_completed_session(self):
        anns = [
            make_datum("A"),
            make_datum("B"),
            make_datum("C"),
            make_fcf(),
            make_dimension(),
        ]
        s = run_text(
            detection_text(anns),
            session_id="sess-1",
            image_reference="synthetic://d.png",
        )
        self.assertEqual(s.status, STATUS_COMPLETED)
        self.assertEqual(len(s.annotations), 5)
        self.assertEqual(s.views, ("Front View", "Side View"))
        self.assertEqual(s.description, "Test drawing")
        self.assertEqual(s.stage_errors, ())
        self.assertEqual(s.session_id, "sess-1")
        self.assertEqual(s.image_reference, "synthetic://d.png")
        self.assertEqual(s.edits, ())

    def test_returns_compliance_issues_from_the_compliance_engine(self):
        # TS vector: position FCF with 0 datum references -> FCF_DATUM_COUNT.
        s = run_text(
            detection_text(
                [make_fcf(id="fcf_bad", datumReferences=[], materialCondition=None)]
            )
        )
        self.assertTrue(len(s.compliance_issues) > 0)
        issue = next(
            (i for i in s.compliance_issues if i.rule_id == "FCF_DATUM_COUNT"), None
        )
        self.assertIsNotNone(issue)
        self.assertEqual(issue.annotation_id, "fcf_bad")

    def test_returns_deterministic_dfm_datum_scheme_finding(self):
        # TS vector: only one datum present -> datum_scheme_completeness warning.
        s = run_text(detection_text([make_datum("A"), make_fcf(datumReferences=["A"])]))
        finding = next(
            (f for f in s.dfm_findings if f.category == "datum_scheme_completeness"),
            None,
        )
        self.assertIsNotNone(finding)
        self.assertEqual(finding.severity, "warning")

    def test_description_is_carried_through(self):
        # Divergence: the harness has no includeDescription flag -- the parsed
        # description is always carried onto the session.
        s = run_text(detection_text([make_dimension()], description="A bracket"))
        self.assertEqual(s.description, "A bracket")

    def test_deterministic_across_repeated_runs(self):
        text = detection_text([make_datum("A"), make_fcf(datumReferences=["A", "Z"])])
        a = run_text(text, session_id="s").to_dict()
        b = run_text(text, session_id="s").to_dict()
        self.assertEqual(a, b)


# --------------------------------------------------------------------------- #
# Orchestration invariants: ordering + data flow between stages
# --------------------------------------------------------------------------- #


class StageOrderingTests(unittest.TestCase):
    def test_stage_constant(self):
        self.assertEqual(STAGES, ("detection", "requery", "compliance", "dfm"))

    def test_stages_run_in_declared_order(self):
        calls = []
        text = detection_text([make_datum("A")])

        def rq(anns):
            calls.append("requery")
            return anns

        real_parse = parse_annotation_response

        def spy_parse(content):
            calls.append("detection")
            return real_parse(content)

        with mock.patch(MODULE + ".parse_annotation_response", spy_parse), mock.patch(
            MODULE + ".validate_compliance",
            side_effect=lambda a: calls.append("compliance") or [],
        ), mock.patch(
            MODULE + ".review_dfm",
            side_effect=lambda a, llm=None: calls.append("dfm") or [],
        ):
            run_text(text, requery=rq)

        self.assertEqual(calls, ["detection", "requery", "compliance", "dfm"])

    def test_requery_output_feeds_compliance_and_dfm(self):
        text = detection_text([make_datum("A"), make_datum("B")])
        replacement = []

        def rq(anns):
            # Drop everything: downstream stages must see the REQUERIED set,
            # not the detected set.
            return replacement

        with mock.patch(
            MODULE + ".validate_compliance", return_value=[]
        ) as vc, mock.patch(MODULE + ".review_dfm", return_value=[]) as rd:
            s = run_text(text, requery=rq)

        self.assertEqual(vc.call_args[0][0], [])
        self.assertEqual(rd.call_args[0][0], [])
        self.assertEqual(s.annotations, ())

    def test_detection_output_feeds_compliance_when_no_requery(self):
        text = detection_text([make_datum("A")])
        with mock.patch(MODULE + ".validate_compliance", return_value=[]) as vc:
            run_text(text)
        passed = vc.call_args[0][0]
        self.assertEqual([a.id for a in passed], ["datum_A"])

    def test_dfm_llm_is_forwarded_to_review_dfm(self):
        sentinel = lambda prompt: "{}"
        with mock.patch(MODULE + ".review_dfm", return_value=[]) as rd:
            run_text(detection_text([make_datum("A")]), dfm_llm=sentinel)
        self.assertIs(rd.call_args[1]["llm"], sentinel)

    def test_requery_receives_a_copy_not_the_live_list(self):
        # The harness passes list(annotations); mutating the argument must not
        # corrupt the session when requery then returns its own list.
        text = detection_text([make_datum("A"), make_datum("B")])

        def rq(anns):
            anns.clear()
            return [make_datum("A")] and []

        s = run_text(text, requery=rq)
        self.assertEqual(s.annotations, ())  # requery's return value wins

    def test_requery_none_is_skipped_entirely(self):
        s = run_text(detection_text([make_datum("A")]), requery=None)
        self.assertEqual(s.status, STATUS_COMPLETED)
        self.assertEqual(len(s.annotations), 1)


# --------------------------------------------------------------------------- #
# Known-bad vectors: stage failures (TS: describe "error handling")
# --------------------------------------------------------------------------- #


class DetectionFailureTests(unittest.TestCase):
    def test_detection_exception_yields_failed_empty_session(self):
        # TS vector: mockCreate.mockRejectedValueOnce(new Error("OpenAI API error"))
        s = run_analysis_pipeline(
            raiser(RuntimeError("OpenAI API error")), session_id="sess-x"
        )
        self.assertEqual(s.status, STATUS_FAILED)
        self.assertEqual(s.annotations, ())
        self.assertEqual(s.compliance_issues, ())
        self.assertEqual(s.dfm_findings, ())
        self.assertEqual(len(s.stage_errors), 1)
        self.assertEqual(s.stage_errors[0].stage, "detection")
        self.assertEqual(s.stage_errors[0].message, "OpenAI API error")
        self.assertEqual(s.session_id, "sess-x")

    def test_detection_failure_does_not_raise(self):
        # A fatal stage still RETURNS a session rather than propagating.
        try:
            run_analysis_pipeline(raiser(RuntimeError("model timeout")))
        except Exception as exc:  # pragma: no cover - guard
            self.fail("pipeline raised: %r" % (exc,))

    def test_detection_failure_leaves_default_views_and_description(self):
        s = run_analysis_pipeline(raiser(RuntimeError("boom")))
        self.assertEqual(s.views, ("View 1",))
        self.assertIsNone(s.description)

    def test_blank_message_falls_back_to_stage_default(self):
        # TS vector: "handles non-Error thrown objects in stage failures" ->
        # message "Detection stage failed". The harness equivalent of a
        # message-less throw is an exception whose str() is empty.
        s = run_analysis_pipeline(raiser(RuntimeError("")))
        self.assertEqual(s.stage_errors[0].message, "Detection stage failed")

    def test_downstream_stages_are_short_circuited_on_detection_failure(self):
        with mock.patch(MODULE + ".validate_compliance") as vc, mock.patch(
            MODULE + ".review_dfm"
        ) as rd:
            run_analysis_pipeline(raiser(RuntimeError("down")), requery=lambda a: a)
        vc.assert_not_called()
        rd.assert_not_called()

    def test_various_exception_types_all_recorded_as_detection_errors(self):
        # Small exhaustive table over exception classes (no hypothesis).
        for exc in (
            RuntimeError("model timeout"),
            ValueError("bad payload"),
            KeyError("choices"),
            TimeoutError("deadline"),
            ZeroDivisionError("division by zero"),
        ):
            with self.subTest(exc=type(exc).__name__):
                s = run_analysis_pipeline(raiser(exc))
                self.assertEqual(s.status, STATUS_FAILED)
                self.assertEqual(s.stage_errors[0].stage, "detection")
                self.assertEqual(s.stage_errors[0].message, str(exc))


class RequeryFailureTests(unittest.TestCase):
    def test_requery_failure_is_partial_and_keeps_detected_annotations(self):
        text = detection_text([make_datum("A"), make_datum("B"), make_datum("C")])
        s = run_text(text, requery=raiser(RuntimeError("vision service unreachable")))
        self.assertEqual(s.status, STATUS_PARTIAL)
        self.assertEqual(len(s.stage_errors), 1)
        self.assertEqual(s.stage_errors[0].stage, "requery")
        self.assertEqual(s.stage_errors[0].message, "vision service unreachable")
        # continued with the ORIGINAL annotations
        self.assertEqual(len(s.annotations), 3)

    def test_compliance_and_dfm_still_run_after_requery_failure(self):
        text = detection_text([make_datum("A"), make_fcf(datumReferences=["A"])])
        s = run_text(text, requery=raiser(RuntimeError("nope")))
        self.assertTrue(
            any(f.category == "datum_scheme_completeness" for f in s.dfm_findings)
        )

    def test_requery_blank_message_falls_back(self):
        s = run_text(detection_text([make_datum("A")]), requery=raiser(ValueError("")))
        self.assertEqual(s.stage_errors[0].message, "Re-query stage failed")

    def test_requery_returning_non_list_iterable_is_materialised(self):
        text = detection_text([make_datum("A"), make_datum("B")])
        s = run_text(text, requery=lambda anns: iter(anns[:1]))
        self.assertEqual(s.status, STATUS_COMPLETED)
        self.assertEqual(len(s.annotations), 1)

    def test_requery_returning_none_is_a_requery_stage_error(self):
        # list(None) raises TypeError inside the try -> recorded, not raised.
        s = run_text(detection_text([make_datum("A")]), requery=lambda anns: None)
        self.assertEqual(s.status, STATUS_PARTIAL)
        self.assertEqual(s.stage_errors[0].stage, "requery")
        # annotations keep their pre-requery value (assignment never happened)
        self.assertEqual(len(s.annotations), 1)


class ComplianceFailureTests(unittest.TestCase):
    def test_compliance_failure_is_partial_with_empty_issues(self):
        # The TS test could not force this (validateCompliance is imported
        # directly); with unittest.mock the harness path IS exercisable.
        text = detection_text([make_dimension()])
        with mock.patch(
            MODULE + ".validate_compliance",
            side_effect=RuntimeError("Rule engine crashed"),
        ):
            s = run_text(text)
        self.assertEqual(s.status, STATUS_PARTIAL)
        self.assertEqual(s.compliance_issues, ())
        self.assertEqual(s.stage_errors[0].stage, "compliance")
        self.assertEqual(s.stage_errors[0].message, "Rule engine crashed")
        # partial results preserved
        self.assertEqual(len(s.annotations), 1)

    def test_dfm_still_runs_after_compliance_failure(self):
        text = detection_text([make_datum("A")])
        with mock.patch(
            MODULE + ".validate_compliance", side_effect=RuntimeError("x")
        ), mock.patch(MODULE + ".review_dfm", return_value=[]) as rd:
            run_text(text)
        rd.assert_called_once()

    def test_compliance_blank_message_falls_back(self):
        with mock.patch(MODULE + ".validate_compliance", side_effect=ValueError("")):
            s = run_text(detection_text([make_dimension()]))
        self.assertEqual(s.stage_errors[0].message, "Compliance stage failed")


class DfmFailureTests(unittest.TestCase):
    def test_dfm_stage_failure_is_partial_with_empty_findings(self):
        text = detection_text([make_datum("A"), make_fcf()])
        with mock.patch(MODULE + ".review_dfm", side_effect=RuntimeError("DFM failed")):
            s = run_text(text)
        self.assertEqual(s.status, STATUS_PARTIAL)
        self.assertEqual(s.dfm_findings, ())
        self.assertEqual(s.stage_errors[0].stage, "dfm")
        # annotations + compliance survive (TS: "continues with partial results")
        self.assertEqual(len(s.annotations), 2)

    def test_dfm_llm_failure_is_swallowed_by_review_dfm_not_the_pipeline(self):
        # TS note: "the LLM error is caught inside reviewDfm". Harness confirms:
        # a raising dfm_llm produces NO stage error, and the deterministic
        # findings still come back.
        text = detection_text([make_datum("A"), make_fcf(datumReferences=["A"])])
        s = run_text(text, dfm_llm=raiser(RuntimeError("DFM API error")))
        self.assertEqual(s.status, STATUS_COMPLETED)
        self.assertEqual(s.stage_errors, ())
        self.assertTrue(
            any(f.category == "datum_scheme_completeness" for f in s.dfm_findings)
        )

    def test_dfm_llm_empty_and_malformed_responses_are_tolerated(self):
        # Known-bad LLM payloads: empty string, empty choices analogue (empty
        # JSON object), markdown-fenced JSON, and outright garbage.
        text = detection_text([make_datum("A"), make_fcf(datumReferences=["A"])])
        for payload in (
            "",
            "{}",
            '{"choices": []}',
            "not json at all",
            '```json\n{"findings": []}\n```',
        ):
            with self.subTest(payload=payload[:20]):
                s = run_text(text, dfm_llm=lambda _p, r=payload: r)
                self.assertEqual(s.status, STATUS_COMPLETED)
                self.assertEqual(s.stage_errors, ())

    def test_markdown_fenced_dfm_json_is_parsed_and_merged(self):
        text = detection_text([make_datum("A"), make_datum("B"), make_datum("C")])
        fenced = (
            "Here you go:\n```json\n"
            + json.dumps(
                {
                    "findings": [
                        {
                            "id": "dfm_1",
                            "category": "general",
                            "severity": "info",
                            "description": "Drawing looks good",
                            "recommendation": "No changes needed",
                        }
                    ]
                }
            )
            + "\n```"
        )
        s = run_text(text, dfm_llm=lambda _p: fenced)
        self.assertEqual(s.status, STATUS_COMPLETED)
        self.assertTrue(any(f.id == "dfm_1" for f in s.dfm_findings))

    def test_dfm_blank_message_falls_back(self):
        with mock.patch(MODULE + ".review_dfm", side_effect=KeyError()):
            s = run_text(detection_text([make_datum("A")]))
        # KeyError() str() is "" -> default message
        self.assertEqual(s.stage_errors[0].message, "DFM stage failed")


class MultipleStageErrorTests(unittest.TestCase):
    def test_collects_multiple_stage_errors_in_stage_order(self):
        text = detection_text([make_dimension()])
        with mock.patch(
            MODULE + ".validate_compliance", side_effect=RuntimeError("c-boom")
        ), mock.patch(MODULE + ".review_dfm", side_effect=RuntimeError("d-boom")):
            s = run_text(text, requery=raiser(RuntimeError("r-boom")))
        self.assertEqual([e.stage for e in s.stage_errors], ["requery", "compliance", "dfm"])
        self.assertEqual(s.status, STATUS_PARTIAL)
        self.assertEqual(len(s.annotations), 1)

    def test_exhaustive_non_detection_failure_subsets(self):
        # Exhaustive over the 8 subsets of the 3 recoverable stages
        # (hypothesis unavailable -> itertools enumeration).
        text = detection_text([make_dimension()])
        recoverable = ("requery", "compliance", "dfm")
        for size in range(len(recoverable) + 1):
            for combo in itertools.combinations(recoverable, size):
                with self.subTest(failing=combo):
                    rq = raiser(RuntimeError("r")) if "requery" in combo else None
                    ctx = []
                    if "compliance" in combo:
                        ctx.append(
                            mock.patch(
                                MODULE + ".validate_compliance",
                                side_effect=RuntimeError("c"),
                            )
                        )
                    if "dfm" in combo:
                        ctx.append(
                            mock.patch(
                                MODULE + ".review_dfm", side_effect=RuntimeError("d")
                            )
                        )
                    with contextlib_nested(ctx):
                        s = run_text(text, requery=rq)
                    self.assertEqual([e.stage for e in s.stage_errors], list(combo))
                    self.assertEqual(
                        s.status, STATUS_PARTIAL if combo else STATUS_COMPLETED
                    )
                    # partial results: annotations always survive
                    self.assertEqual(len(s.annotations), 1)


class contextlib_nested(object):
    """Tiny helper: enter a list of context managers together."""

    def __init__(self, managers):
        self._managers = list(managers)

    def __enter__(self):
        for m in self._managers:
            m.__enter__()
        return self

    def __exit__(self, *exc):
        for m in reversed(self._managers):
            m.__exit__(*exc)
        return False


# --------------------------------------------------------------------------- #
# Known-bad vectors: empty / malformed DETECTION payloads
# --------------------------------------------------------------------------- #


class MalformedDetectionInputTests(unittest.TestCase):
    def test_empty_and_malformed_payloads_yield_completed_empty_sessions(self):
        # DIVERGENCE from the TS orchestrator: the harness detection stage only
        # fails when detect() RAISES. A malformed / empty response text is
        # absorbed by the tolerant parse_annotation_response, so the session is
        # "completed" with zero annotations, NOT "failed".
        for payload in (
            "",  # empty content
            "no json here",  # no JSON blob
            "{ not json",  # unparseable
            "{}",  # empty object == empty choices analogue
            '{"annotations": "nope"}',  # annotations not a list
            '{"annotations": null}',
            '{"choices": []}',  # TS "empty choices" vector
        ):
            with self.subTest(payload=payload):
                s = run_text(payload)
                self.assertEqual(s.status, STATUS_COMPLETED)
                self.assertEqual(s.annotations, ())
                self.assertEqual(s.compliance_issues, ())
                self.assertEqual(s.stage_errors, ())
                self.assertEqual(s.views, ("View 1",))
                self.assertIsNone(s.description)

    def test_markdown_fenced_detection_json_is_parsed(self):
        payload = "Analysis:\n```json\n" + detection_text([make_datum("A")]) + "\n```"
        s = run_text(payload)
        self.assertEqual(s.status, STATUS_COMPLETED)
        self.assertEqual(len(s.annotations), 1)
        self.assertEqual(s.views, ("Front View", "Side View"))

    def test_individually_malformed_annotations_are_dropped_silently(self):
        anns = [
            make_datum("A"),
            make_datum("A", ann_id="bad_letter"),
            make_fcf(id="bad_gc", geometricCharacteristic="unicorn"),
            {"type": "bogus", "id": "bad_type"},
        ]
        anns[1]["datumLetter"] = "abc"  # not a single uppercase letter
        s = run_text(detection_text(anns))
        self.assertEqual(s.status, STATUS_COMPLETED)
        self.assertEqual([a.id for a in s.annotations], ["datum_A"])

    def test_persists_empty_arrays_when_no_annotations_detected(self):
        # TS vector: "persists empty arrays when no annotations are detected".
        s = run_text(detection_text([]))
        self.assertEqual(s.annotations, ())
        self.assertEqual(s.compliance_issues, ())
        self.assertEqual(s.status, STATUS_COMPLETED)

    def test_views_default_when_views_list_is_empty_or_wrong_type(self):
        for views in ([], "Front View", None, [1, 2]):
            with self.subTest(views=views):
                s = run_text(detection_text([make_datum("A")], views=views))
                self.assertEqual(s.views, ("View 1",))


# --------------------------------------------------------------------------- #
# to_dict / session record shape
# --------------------------------------------------------------------------- #


class SessionSerialisationTests(unittest.TestCase):
    def test_to_dict_is_json_serialisable_and_has_expected_keys(self):
        s = run_text(detection_text([make_datum("A"), make_fcf()]))
        d = s.to_dict()
        json.dumps(d)  # must not raise
        self.assertEqual(
            sorted(d),
            sorted(
                [
                    "session_id",
                    "image_reference",
                    "status",
                    "description",
                    "views",
                    "stage_errors",
                    "annotations",
                    "compliance_issues",
                    "compliance_summary",
                    "dfm_findings",
                    "edits",
                ]
            ),
        )

    def test_compliance_summary_counts_partition_the_annotations(self):
        s = run_text(
            detection_text(
                [make_datum("A"), make_fcf(datumReferences=["A", "Z"]), make_dimension()]
            )
        )
        summary = s.to_dict()["compliance_summary"]
        self.assertEqual(
            summary["errors"] + summary["warnings"] + summary["passing"],
            len(s.annotations),
        )

    def test_stage_error_to_dict(self):
        self.assertEqual(
            StageError(stage="dfm", message="LLM timeout").to_dict(),
            {"stage": "dfm", "message": "LLM timeout"},
        )

    def test_failed_session_to_dict(self):
        s = run_analysis_pipeline(raiser(RuntimeError("API down")))
        d = s.to_dict()
        self.assertEqual(d["status"], STATUS_FAILED)
        self.assertEqual(d["annotations"], [])
        self.assertEqual(d["stage_errors"], [{"stage": "detection", "message": "API down"}])


# --------------------------------------------------------------------------- #
# Edit audit trail + optimistic-update-with-revalidation
# --------------------------------------------------------------------------- #


def _completed_session():
    """Session whose FCF carries a dangling datum reference D."""
    anns = [
        make_datum("A"),
        make_datum("B"),
        make_fcf(id="fcf_1", datumReferences=["A", "D"], materialCondition=None),
    ]
    return run_text(detection_text(anns), session_id="sess-edit")


class AnnotationEditTests(unittest.TestCase):
    def setUp(self):
        self.session = _completed_session()
        self.fcf = next(a for a in self.session.annotations if a.id == "fcf_1")

    def test_baseline_has_dangling_datum_reference(self):
        rules = [i.rule_id for i in self.session.compliance_issues]
        self.assertIn("DATUM_REF_EXISTS", rules)

    def test_edit_revalidates_compliance(self):
        fixed = dc_replace(self.fcf, datum_references=("A", "B"))
        new_session, _edit = apply_annotation_edit(self.session, fixed)
        rules = [i.rule_id for i in new_session.compliance_issues]
        self.assertNotIn("DATUM_REF_EXISTS", rules)

    def test_edit_snapshots_previous_and_new_values(self):
        fixed = dc_replace(self.fcf, datum_references=("A", "B"))
        _new_session, edit = apply_annotation_edit(self.session, fixed)
        self.assertEqual(edit.session_id, "sess-edit")
        self.assertEqual(edit.annotation_id, "fcf_1")
        self.assertEqual(edit.previous_value["datum_references"], ["A", "D"])
        self.assertEqual(edit.new_value["datum_references"], ["A", "B"])
        self.assertEqual(edit.sequence, 1)
        json.dumps(edit.to_dict())

    def test_input_session_is_not_mutated(self):
        before = self.session.to_dict()
        fixed = dc_replace(self.fcf, datum_references=("A", "B"))
        apply_annotation_edit(self.session, fixed)
        self.assertEqual(self.session.edits, ())
        self.assertEqual(self.session.to_dict(), before)

    def test_sequence_numbers_are_deterministic_and_monotonic(self):
        session = self.session
        for expected in (1, 2, 3, 4, 5):
            fixed = dc_replace(self.fcf, tolerance_value=0.05 + expected)
            session, edit = apply_annotation_edit(session, fixed)
            self.assertEqual(edit.sequence, expected)
            self.assertEqual(len(session.edits), expected)
            self.assertEqual(session.edits[-1], edit)
        self.assertEqual([e.sequence for e in session.edits], [1, 2, 3, 4, 5])

    def test_unknown_annotation_id_raises_value_error(self):
        stranger = dc_replace(self.fcf, id="does_not_exist")
        with self.assertRaises(ValueError) as ctx:
            apply_annotation_edit(self.session, stranger)
        self.assertIn("does_not_exist", str(ctx.exception))
        self.assertIn("sess-edit", str(ctx.exception))

    def test_edit_on_empty_session_raises(self):
        empty = run_text(detection_text([]))
        with self.assertRaises(ValueError):
            apply_annotation_edit(empty, self.fcf)

    def test_edit_preserves_annotation_order_and_untouched_neighbours(self):
        fixed = dc_replace(self.fcf, datum_references=("A", "B"))
        new_session, _ = apply_annotation_edit(self.session, fixed)
        self.assertEqual(
            [a.id for a in new_session.annotations],
            [a.id for a in self.session.annotations],
        )
        self.assertIs(new_session.annotations[0], self.session.annotations[0])

    def test_edit_preserves_session_identity_fields(self):
        fixed = dc_replace(self.fcf, datum_references=("A", "B"))
        new_session, _ = apply_annotation_edit(self.session, fixed)
        for attr in ("session_id", "image_reference", "description", "views", "status"):
            self.assertEqual(getattr(new_session, attr), getattr(self.session, attr))

    def test_status_is_not_recomputed_by_an_edit(self):
        # DIVERGENCE / documented harness behaviour: apply_annotation_edit
        # revalidates COMPLIANCE but leaves ``status`` alone -- status reflects
        # stage errors from the pipeline run, not compliance cleanliness.
        partial = run_text(
            detection_text([make_datum("A"), make_datum("B"), make_fcf(id="fcf_1")]),
            requery=raiser(RuntimeError("boom")),
            session_id="sess-p",
        )
        self.assertEqual(partial.status, STATUS_PARTIAL)
        fcf = next(a for a in partial.annotations if a.id == "fcf_1")
        new_session, _ = apply_annotation_edit(
            partial, dc_replace(fcf, datum_references=("A", "B"))
        )
        self.assertEqual(new_session.status, STATUS_PARTIAL)
        self.assertEqual(new_session.stage_errors, partial.stage_errors)

    def test_no_op_edit_still_records_an_audit_entry(self):
        # Replacing an annotation with an identical value is still an edit.
        new_session, edit = apply_annotation_edit(self.session, self.fcf)
        self.assertEqual(len(new_session.edits), 1)
        self.assertEqual(edit.previous_value, edit.new_value)

    def test_duplicate_annotation_ids_replace_every_match(self):
        # Known-bad vector: an LLM may emit two annotations with the SAME id --
        # annotation_schema does not de-duplicate. The harness loop replaces
        # ALL matches and snapshots the LAST match as previous_value.
        # Documented here as observed behaviour (see report: audit-trail
        # fidelity issue, not a crash).
        other = dc_replace(self.fcf, datum_references=("A", "C"))
        dup = dc_replace(self.session, annotations=self.session.annotations + (other,))
        fixed = dc_replace(self.fcf, datum_references=("A", "B"))
        new_session, edit = apply_annotation_edit(dup, fixed)
        matches = [a for a in new_session.annotations if a.id == "fcf_1"]
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(a.datum_references == ("A", "B") for a in matches))
        # previous_value snapshots the LAST match, silently losing the first.
        self.assertEqual(edit.previous_value["datum_references"], ["A", "C"])

    def test_annotation_edit_to_dict(self):
        edit = AnnotationEdit(
            session_id="s",
            annotation_id="a",
            previous_value={"x": 1},
            new_value={"x": 2},
            sequence=7,
        )
        self.assertEqual(
            edit.to_dict(),
            {
                "session_id": "s",
                "annotation_id": "a",
                "previous_value": {"x": 1},
                "new_value": {"x": 2},
                "sequence": 7,
            },
        )

    def test_edits_appear_in_session_to_dict(self):
        fixed = dc_replace(self.fcf, datum_references=("A", "B"))
        new_session, edit = apply_annotation_edit(self.session, fixed)
        self.assertEqual(new_session.to_dict()["edits"], [edit.to_dict()])

    def test_random_edit_sequences_keep_trail_consistent(self):
        # Substitutes for a fast-check property: random.Random(20260719) picks
        # 30 random annotation ids to edit; sequence numbers must remain
        # 1..n and the trail length must equal the number of edits.
        rng = random.Random(FIXED_SEED)
        session = self.session
        ids = [a.id for a in session.annotations]
        for n in range(1, 31):
            target_id = rng.choice(ids)
            target = next(a for a in session.annotations if a.id == target_id)
            session, edit = apply_annotation_edit(
                session, dc_replace(target, label="edit-%d" % n)
            )
            self.assertEqual(edit.sequence, n)
        self.assertEqual([e.sequence for e in session.edits], list(range(1, 31)))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class CliTests(unittest.TestCase):
    def test_selfcheck_passes(self):
        with mock.patch("sys.stdout"):
            self.assertEqual(main(["--selfcheck"]), 0)

    def test_selfcheck_json_emits_valid_json(self):
        with mock.patch("builtins.print") as pr:
            self.assertEqual(main(["--selfcheck", "--json"]), 0)
        payload = pr.call_args_list[0][0][0]
        parsed = json.loads(payload)
        self.assertEqual(
            sorted(parsed), ["completed", "edited", "failed", "partial"]
        )
        self.assertEqual(parsed["completed"]["status"], STATUS_COMPLETED)
        self.assertEqual(parsed["partial"]["status"], STATUS_PARTIAL)
        self.assertEqual(parsed["failed"]["status"], STATUS_FAILED)

    def test_no_args_prints_help_and_returns_zero(self):
        with mock.patch("sys.stdout"):
            self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()
