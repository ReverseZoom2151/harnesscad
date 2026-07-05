import tempfile
import unittest
from pathlib import Path

from security import DataPolicy, SecureIngestGate, redact_metadata


class SecurityPolicyTests(unittest.TestCase):
    def test_on_prem_rejects_network_egress(self):
        with self.assertRaises(ValueError):
            DataPolicy(execution_mode="on_prem", allow_network=True)

    def test_redacts_secrets_and_pii_without_mutation(self):
        source = {
            "author_email": "person@example.com",
            "api_key": "secret",
            "material": "steel",
            "nested": {"customer": "ACME"},
        }
        result = redact_metadata(source, DataPolicy())
        self.assertEqual(result["material"], "steel")
        self.assertTrue(result["api_key"].startswith("[REDACTED:"))
        self.assertTrue(result["author_email"].startswith("[REDACTED:"))
        self.assertTrue(result["nested"]["customer"].startswith("[REDACTED:"))
        self.assertEqual(source["api_key"], "secret")

    def test_allowed_file_is_hashed_and_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "part.step")
            path.write_bytes(b"ISO-10303-21")
            gate = SecureIngestGate()
            decision = gate.inspect(path, {"owner": "Alice"}, root=tmp)
            self.assertTrue(decision.allowed)
            self.assertEqual(len(decision.content_sha256), 64)
            self.assertTrue(decision.redacted_metadata["owner"].startswith("[REDACTED:"))
            self.assertEqual(gate.audit_log()[0]["sequence"], 1)

    def test_extension_size_and_traversal_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "root")
            root.mkdir()
            denied = Path(root, "payload.exe")
            denied.write_bytes(b"x")
            gate = SecureIngestGate(DataPolicy(max_bytes=2))
            self.assertEqual(gate.inspect(denied, root=root).code, "extension-denied")

            large = Path(root, "large.step")
            large.write_bytes(b"123")
            self.assertEqual(gate.inspect(large, root=root).code, "file-too-large")

            outside = Path(tmp, "outside.step")
            outside.write_bytes(b"x")
            self.assertEqual(gate.inspect(outside, root=root).code, "path-outside-root")


if __name__ == "__main__":
    unittest.main()
