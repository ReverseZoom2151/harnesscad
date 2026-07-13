"""Tests for the submission contract: candidates, meta, manifest."""
import tempfile
import unittest
from pathlib import Path

from harnesscad.data.dataengine.schemas.cgb_submission_package import (
    MAX_NOTES_CHARS,
    REQUIRED_META_KEYS,
    SampleEntry,
    build_manifest,
    default_meta,
    discover_samples,
    pick_candidate,
    scan_directory,
    validate_meta,
    validate_submission,
)


def _good_meta():
    meta = default_meta("Jane", "my agent v1")
    meta["agree_to_publish"] = True
    return meta


class TestCandidateDiscovery(unittest.TestCase):
    def test_step_preferred_over_stp(self):
        self.assertEqual(pick_candidate(["output.stp", "output.step"]), "output.step")

    def test_stp_accepted(self):
        self.assertEqual(pick_candidate(["output.stp"]), "output.stp")

    def test_other_files_ignored(self):
        self.assertIsNone(pick_candidate(["model.py", "render.png", "out.step"]))

    def test_empty_file_is_not_a_candidate(self):
        self.assertIsNone(pick_candidate(["output.step"], {"output.step": 0}))

    def test_missing_sample_folder_is_kept(self):
        entries = discover_samples({"101": ["output.step"], "102": ["agent.log"]})
        self.assertEqual([e.name for e in entries], ["101", "102"])
        self.assertEqual(entries[1].status, "missing")
        self.assertIsNone(entries[1].candidate)


class TestMeta(unittest.TestCase):
    def test_required_keys(self):
        self.assertEqual(len(REQUIRED_META_KEYS), 5)
        self.assertEqual(validate_meta(_good_meta()), [])

    def test_missing_key(self):
        meta = _good_meta()
        del meta["agent_url"]
        errors = validate_meta(meta)
        self.assertTrue(any("agent_url" in e for e in errors))

    def test_consent_defaults_false_and_blocks_acceptance(self):
        meta = default_meta("Jane", "v1")
        self.assertFalse(meta["agree_to_publish"])
        errors = validate_meta(meta)
        self.assertTrue(any("agree_to_publish" in e for e in errors))

    def test_empty_submitter(self):
        meta = _good_meta()
        meta["submitter_name"] = "   "
        self.assertTrue(any("submitter_name" in e for e in validate_meta(meta)))

    def test_long_notes_rejected(self):
        meta = _good_meta()
        meta["notes"] = "x" * (MAX_NOTES_CHARS + 1)
        self.assertTrue(any("notes" in e for e in validate_meta(meta)))


class TestManifestAndValidation(unittest.TestCase):
    def test_manifest_counts(self):
        entries = discover_samples({"101": ["output.step"], "102": []})
        manifest = build_manifest(entries, _good_meta())
        self.assertEqual(manifest["n_samples"], 2)
        self.assertEqual(manifest["n_with_candidate"], 1)
        self.assertEqual(manifest["n_missing"], 1)

    def test_accepted_submission(self):
        entries = discover_samples({"101": ["output.step"]})
        report = validate_submission(entries, _good_meta())
        self.assertTrue(report.accepted)
        self.assertEqual(report.errors, [])

    def test_empty_submission_rejected(self):
        report = validate_submission([], _good_meta())
        self.assertFalse(report.accepted)

    def test_unknown_sample_folder_rejected(self):
        entries = [SampleEntry("999", "output.step")]
        report = validate_submission(entries, _good_meta(), expected_samples=["101"])
        self.assertFalse(report.accepted)
        self.assertTrue(any("unknown sample folder" in e for e in report.errors))

    def test_not_submitted_sample_is_allowed_and_counted(self):
        entries = discover_samples({"101": ["output.step"]})
        report = validate_submission(entries, _good_meta(), expected_samples=["101", "102"])
        self.assertTrue(report.accepted)
        self.assertEqual(report.manifest["n_not_submitted"], 1)


class TestScanDirectory(unittest.TestCase):
    def test_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "101").mkdir()
            (root / "101" / "output.step").write_text("ISO-10303-21;")
            (root / "102").mkdir()
            (root / "102" / "agent.log").write_text("failed")
            (root / "103").mkdir()
            (root / "103" / "output.step").write_text("")  # empty: not a candidate
            entries = scan_directory(root)
        self.assertEqual([e.name for e in entries], ["101", "102", "103"])
        self.assertEqual(entries[0].candidate, "output.step")
        self.assertEqual(entries[1].status, "missing")
        self.assertEqual(entries[2].status, "missing")

    def test_missing_dir_raises(self):
        with self.assertRaises(NotADirectoryError):
            scan_directory(Path(tempfile.gettempdir()) / "cgb_does_not_exist_12345")


if __name__ == "__main__":
    unittest.main()
