import tempfile
import unittest
from pathlib import Path

from audit.closure import validate_register


class AuditClosureTests(unittest.TestCase):
    def test_closed_register_requires_coverage_code_and_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            corpus = root / "corpus"
            repo.mkdir()
            corpus.mkdir()
            (repo / "feature.py").write_text("x = 1\n", encoding="utf-8")
            (repo / "test_feature.py").write_text("pass\n", encoding="utf-8")
            (corpus / "source.txt").write_text("idea\n", encoding="utf-8")
            register = {
                "coverage": [{
                    "path": "source.txt", "status": "reviewed", "method": "full text"
                }],
                "ideas": [{
                    "id": "I-001",
                    "statement": "Build the feature.",
                    "disposition": "implemented",
                    "sources": [{"path": "source.txt", "locator": "line 1"}],
                    "code": ["feature.py"],
                    "tests": ["test_feature.py"],
                }],
            }
            report = validate_register(register, repo_root=repo, corpus_root=corpus)
            self.assertTrue(report.closed, report.to_dict())

    def test_open_partial_and_uncovered_file_fail_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            corpus = root / "corpus"
            repo.mkdir()
            corpus.mkdir()
            (corpus / "source.txt").write_text("idea\n", encoding="utf-8")
            register = {
                "coverage": [],
                "ideas": [{
                    "id": "I-001",
                    "statement": "Build part of it.",
                    "disposition": "partial",
                    "sources": [{"path": "source.txt", "locator": "line 1"}],
                }],
            }
            report = validate_register(register, repo_root=repo, corpus_root=corpus)
            self.assertFalse(report.closed)
            self.assertIn("open-disposition", {issue.code for issue in report.issues})
            self.assertIn("uncovered-file", {issue.code for issue in report.issues})

    def test_external_requires_rationale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            corpus = root / "corpus"
            repo.mkdir()
            corpus.mkdir()
            (corpus / "source.txt").write_text("idea\n", encoding="utf-8")
            register = {
                "coverage": [{
                    "path": "source.txt", "status": "reviewed", "method": "full text"
                }],
                "ideas": [{
                    "id": "I-001",
                    "statement": "Connect proprietary software.",
                    "disposition": "external",
                    "sources": [{"path": "source.txt", "locator": "line 1"}],
                }],
            }
            report = validate_register(register, repo_root=repo, corpus_root=corpus)
            self.assertIn("missing-rationale", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
