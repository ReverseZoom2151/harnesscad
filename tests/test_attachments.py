import hashlib
import tempfile
import unittest
from pathlib import Path

from harnesscad.agents.agent.attachments import (
    Attachment,
    AttachmentKind,
    AttachmentProvenance,
    DeterministicEncoder,
    condition_attachment,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"payload"
JPEG = b"\xff\xd8\xff\xe0" + b"payload"


class TestAttachments(unittest.TestCase):
    def setUp(self):
        self.provenance = AttachmentProvenance(
            "user-upload", author="engineer", source_id="brief-7"
        )
        self.encoder = DeterministicEncoder()

    def test_bytes_are_validated_hashed_and_encoded(self):
        result = condition_attachment(
            Attachment(AttachmentKind.IMAGE, self.provenance, data=PNG),
            self.encoder,
        )
        digest = hashlib.sha256(PNG).hexdigest()
        self.assertEqual(result.mime, "image/png")
        self.assertEqual(result.sha256, digest)
        self.assertEqual(result.encoded["sha256"], digest)
        self.assertIs(result.provenance, self.provenance)

    def test_expected_hash_is_enforced(self):
        digest = hashlib.sha256(PNG).hexdigest()
        result = condition_attachment(
            Attachment(
                AttachmentKind.SKETCH, self.provenance, data=PNG,
                expected_sha256=digest.upper(),
            ),
            self.encoder,
        )
        self.assertEqual(result.kind, AttachmentKind.SKETCH)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            condition_attachment(
                Attachment(
                    AttachmentKind.IMAGE, self.provenance, data=PNG,
                    expected_sha256="0" * 64,
                ),
                self.encoder,
            )

    def test_declared_mime_must_match_magic(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            condition_attachment(
                Attachment(
                    AttachmentKind.IMAGE, self.provenance, data=JPEG,
                    declared_mime="image/png",
                ),
                self.encoder,
            )

    def test_size_limit_applies_to_bytes(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            condition_attachment(
                Attachment(AttachmentKind.IMAGE, self.provenance, data=PNG),
                self.encoder,
                max_bytes=4,
            )

    def test_path_requires_allowed_root_and_checks_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(PNG)
            attachment = Attachment(AttachmentKind.IMAGE, self.provenance, path=path)
            with self.assertRaisesRegex(ValueError, "allowed root"):
                condition_attachment(attachment, self.encoder)
            result = condition_attachment(
                attachment, self.encoder, allowed_roots=[Path(tmp)]
            )
            self.assertEqual(result.mime, "image/png")

            misleading = Path(tmp) / "image.jpg"
            misleading.write_bytes(PNG)
            with self.assertRaisesRegex(ValueError, "extension"):
                condition_attachment(
                    Attachment(
                        AttachmentKind.IMAGE, self.provenance, path=misleading
                    ),
                    self.encoder,
                    allowed_roots=[Path(tmp)],
                )

    def test_path_outside_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            path = Path(other) / "x.png"
            path.write_bytes(PNG)
            with self.assertRaises(PermissionError):
                condition_attachment(
                    Attachment(AttachmentKind.IMAGE, self.provenance, path=path),
                    self.encoder,
                    allowed_roots=[Path(root)],
                )

    def test_safe_svg_is_accepted_and_active_svg_rejected(self):
        safe = b'<svg xmlns="urn:x"><path d="M0 0"/></svg>'
        result = condition_attachment(
            Attachment(AttachmentKind.SKETCH, self.provenance, data=safe),
            self.encoder,
        )
        self.assertEqual(result.mime, "image/svg+xml")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            condition_attachment(
                Attachment(
                    AttachmentKind.SKETCH, self.provenance,
                    data=b"<svg><script>alert(1)</script></svg>",
                ),
                self.encoder,
            )

    def test_unknown_content_and_invalid_construction_rejected(self):
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            condition_attachment(
                Attachment(AttachmentKind.IMAGE, self.provenance, data=b"text"),
                self.encoder,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            Attachment(AttachmentKind.IMAGE, self.provenance)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            Attachment(
                AttachmentKind.IMAGE, self.provenance, data=PNG, path=Path("x")
            )
        with self.assertRaises(ValueError):
            AttachmentProvenance("")

    def test_encoder_receives_validated_bytes_only(self):
        calls = []

        class Encoder:
            def encode(self, data, *, mime, kind):
                calls.append((data, mime, kind))
                return "provider-token"

        result = condition_attachment(
            Attachment(AttachmentKind.IMAGE, self.provenance, data=PNG),
            Encoder(),
        )
        self.assertEqual(result.encoded, "provider-token")
        self.assertEqual(calls, [(PNG, "image/png", AttachmentKind.IMAGE)])


if __name__ == "__main__":
    unittest.main()
