"""Truth-boundary tests for the OpenCAD discipline-record repair."""

import unittest
from unittest.mock import patch

from harnesscad.eval.corpus import consensus
from harnesscad.eval.corpus import discipline_examples as examples
from harnesscad.eval.corpus.spec import Source, Split


class TestTruthBoundary(unittest.TestCase):
    def test_only_three_records_are_trusted(self):
        trusted = examples.trusted_examples()
        self.assertEqual(
            [record.example_id for record in trusted],
            [
                "opencad-software-hmi-panel",
                "opencad-firmware-programmer-fixture",
                "opencad-device-cable-grommet",
            ],
        )

    def test_retired_records_cannot_green_light(self):
        retired = [record for record in examples.all_examples()
                   if record.truth_status == examples.RETIRED]
        self.assertEqual(len(retired), 2)
        for record in retired:
            ok, _volume, detail = examples.verify_example(record)
            self.assertFalse(ok)
            self.assertTrue(detail.startswith("retired:"), detail)
            self.assertIsNone(record.expected_volume_mm3)

    def test_panel_closed_form_adds_back_its_overlapping_hole_lens(self):
        panel = examples.example_by_id("opencad-software-hmi-panel")
        ok, measured, detail = examples.verify_example(panel)
        self.assertTrue(ok, detail)
        # This is 40.3936 mm3 larger than the old double-subtracted formula.
        self.assertAlmostEqual(panel.expected_volume_mm3, 22610.880553, places=6)
        self.assertAlmostEqual(measured, panel.expected_volume_mm3, places=6)


class TestCorroborationRoute(unittest.TestCase):
    def test_briefs_are_analytic_dev_inputs_with_complete_geometry(self):
        briefs = examples.corroboration_briefs()
        self.assertEqual(len(briefs), 3)
        for brief in briefs:
            self.assertEqual(brief.source, Source.ANALYTIC)
            self.assertEqual(brief.split, Split.DEV)
            self.assertTrue(brief.reference)
            self.assertTrue(brief.citation)
            self.assertGreater(brief.volume, 0.0)
            self.assertTrue(all(axis > 0.0 for axis in brief.bbox))

    def test_consensus_route_receives_only_the_trusted_trio(self):
        sentinel = object()
        with patch.object(consensus, "corroborate_all", return_value=sentinel) as call:
            self.assertIs(consensus.corroborate_discipline_examples(), sentinel)
        briefs = call.call_args.args[0]
        self.assertEqual([brief.id for brief in briefs], [
            "discipline_opencad-software-hmi-panel",
            "discipline_opencad-firmware-programmer-fixture",
            "discipline_opencad-device-cable-grommet",
        ])
