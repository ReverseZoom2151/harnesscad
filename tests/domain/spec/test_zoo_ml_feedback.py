"""Tests for the Zoo ML text-to-CAD response + feedback model."""

import unittest

from harnesscad.domain.spec import zoo_ml_feedback as zmf


def _resp(rid, status, feedback=None, created="2026-01-01"):
    return zmf.TextToCadResponse(
        id=rid, prompt="a bracket", status=status, feedback=feedback, created_at=created,
        output_keys=("source.step",),
    )


class StatusTest(unittest.TestCase):
    def test_terminal_and_success(self):
        self.assertTrue(zmf.is_terminal(zmf.ApiCallStatus.COMPLETED))
        self.assertTrue(zmf.is_terminal(zmf.ApiCallStatus.FAILED))
        self.assertFalse(zmf.is_terminal(zmf.ApiCallStatus.IN_PROGRESS))
        self.assertTrue(zmf.is_success(zmf.ApiCallStatus.COMPLETED))
        self.assertFalse(zmf.is_success(zmf.ApiCallStatus.FAILED))


class ResponseTest(unittest.TestCase):
    def test_accepted_flag(self):
        r = _resp("a", zmf.ApiCallStatus.COMPLETED, zmf.MlFeedback.THUMBS_UP)
        self.assertTrue(r.succeeded)
        self.assertTrue(r.rated)
        self.assertTrue(r.accepted)

    def test_rejected_not_accepted(self):
        r = _resp("b", zmf.ApiCallStatus.COMPLETED, zmf.MlFeedback.THUMBS_DOWN)
        self.assertTrue(r.rated)
        self.assertFalse(r.accepted)

    def test_unrated(self):
        r = _resp("c", zmf.ApiCallStatus.COMPLETED, None)
        self.assertFalse(r.rated)
        self.assertFalse(r.accepted)


class AcceptanceStatsTest(unittest.TestCase):
    def test_rate_over_rated_only(self):
        responses = [
            _resp("a", zmf.ApiCallStatus.COMPLETED, zmf.MlFeedback.THUMBS_UP),
            _resp("b", zmf.ApiCallStatus.COMPLETED, zmf.MlFeedback.THUMBS_DOWN),
            _resp("c", zmf.ApiCallStatus.COMPLETED, None),      # unrated - excluded
            _resp("d", zmf.ApiCallStatus.FAILED, zmf.MlFeedback.THUMBS_UP),  # failed - excluded
        ]
        stats = zmf.acceptance_stats(responses)
        self.assertEqual(stats.total, 4)
        self.assertEqual(stats.completed, 3)
        self.assertEqual(stats.rated, 2)
        self.assertEqual(stats.accepted, 1)
        self.assertEqual(stats.rejected, 1)
        self.assertAlmostEqual(stats.acceptance_rate, 0.5)
        self.assertAlmostEqual(stats.completion_rate, 0.75)

    def test_empty_no_silent_pass(self):
        stats = zmf.acceptance_stats([])
        self.assertEqual(stats.acceptance_rate, 0.0)
        self.assertEqual(stats.completion_rate, 0.0)


class SortAndPaginateTest(unittest.TestCase):
    def test_sort_desc_default(self):
        responses = [_resp("a", zmf.ApiCallStatus.COMPLETED, created="2026-01-01"),
                     _resp("b", zmf.ApiCallStatus.COMPLETED, created="2026-03-01")]
        ordered = zmf.sort_by_created_at(responses)
        self.assertEqual(ordered[0].id, "b")

    def test_sort_asc(self):
        responses = [_resp("a", zmf.ApiCallStatus.COMPLETED, created="2026-03-01"),
                     _resp("b", zmf.ApiCallStatus.COMPLETED, created="2026-01-01")]
        ordered = zmf.sort_by_created_at(responses, zmf.CreatedAtSortMode.ASC)
        self.assertEqual(ordered[0].id, "b")

    def test_paginate(self):
        responses = [_resp(str(i), zmf.ApiCallStatus.COMPLETED) for i in range(5)]
        page = zmf.paginate(responses, page_size=2, page=1)
        self.assertEqual([r.id for r in page], ["2", "3"])

    def test_paginate_bad_size(self):
        with self.assertRaises(ValueError):
            zmf.paginate([], page_size=0)


if __name__ == "__main__":
    unittest.main()
