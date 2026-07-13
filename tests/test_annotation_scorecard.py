import unittest

from harnesscad.data.dataengine.annotation.scorecard import (
    AnnotationKind,
    Candidate,
    MultiViewAnnotationJob,
    QualityPolicy,
    score_candidates,
)


class ScorecardTests(unittest.TestCase):
    def test_consensus_accepts_cross_view_candidate(self):
        cards = score_candidates(
            AnnotationKind.TAG,
            [
                Candidate("mounting bracket", 0.9, "front", ("visible flange",)),
                Candidate("Mounting  Bracket", 0.8, "side", ("right angle",)),
                Candidate("plate", 0.95, "top", ("flat face",)),
            ],
            QualityPolicy(minimum_views=2, required_evidence=True),
        )
        bracket = next(card for card in cards if card.value == "mounting bracket")
        self.assertTrue(bracket.accepted)
        self.assertAlmostEqual(bracket.confidence, 0.85)
        plate = next(card for card in cards if card.value == "plate")
        self.assertFalse(plate.accepted)

    def test_type_specific_filters_are_explained(self):
        cards = score_candidates(
            AnnotationKind.CAPTION,
            [
                Candidate("Unknown component", 0.4, "front"),
                Candidate("unknown component", 0.5, "side"),
            ],
            QualityPolicy(
                minimum_confidence=0.8,
                forbidden_terms=("unknown",),
                required_evidence=True,
            ),
        )
        self.assertEqual(
            set(cards[0].reasons),
            {"confidence_below_threshold", "forbidden_term", "missing_evidence"},
        )

    def test_empty_values_and_invalid_confidence_are_rejected(self):
        with self.assertRaises(ValueError):
            Candidate("", 0.5, "front")
        with self.assertRaises(ValueError):
            Candidate("tag", 1.1, "front")


class MultiViewJobTests(unittest.TestCase):
    def test_job_uses_injected_annotator_and_tracks_rejections(self):
        def tagger(view, payload):
            common = Candidate("bracket", 0.9, view, (str(payload),))
            if view == "front":
                return [common, Candidate("uncertain", 0.2, view)]
            return [common]

        job = MultiViewAnnotationJob(
            {AnnotationKind.TAG: tagger},
            {
                AnnotationKind.TAG: QualityPolicy(
                    minimum_confidence=0.7,
                    minimum_views=2,
                    required_evidence=True,
                )
            },
        )
        result = job.run(
            "part-1",
            {"front": "front.png", "side": "side.png"},
            metadata={"source": "fixture"},
        )
        self.assertEqual([card.value for card in result.accepted], ["bracket"])
        self.assertEqual(len(result.rejected_candidates), 1)
        self.assertEqual(result.metadata["source"], "fixture")

    def test_mismatched_candidate_view_is_rejected(self):
        job = MultiViewAnnotationJob(
            {
                AnnotationKind.TAG: lambda view, payload: [
                    Candidate("plate", 0.9, "wrong-view")
                ]
            },
            {},
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            job.run("part", {"front": object()})

    def test_job_requires_views(self):
        job = MultiViewAnnotationJob({}, {})
        with self.assertRaises(ValueError):
            job.run("part", {})


if __name__ == "__main__":
    unittest.main()
