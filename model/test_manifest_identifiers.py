"""Privacy and backward-compatibility tests for corpus manifest identifiers."""
import hashlib
import unittest

from prepare_corpus import IDENTIFIER_FORMAT, identifier_fingerprint
from sft import manifest_source_groups


class ManifestIdentifierTests(unittest.TestCase):
    def test_fingerprints_are_domain_separated(self):
        value = "customer@example.org"
        self.assertNotEqual(
            identifier_fingerprint(value, "record"),
            identifier_fingerprint(value, "source-group")
        )
        self.assertNotEqual(identifier_fingerprint(value, "record"), value)

    def test_new_manifest_uses_source_group_fingerprints(self):
        raw_group = "private-source-group-42"
        fingerprint = identifier_fingerprint(raw_group, "source-group")
        manifest = {
            "identifier_format": IDENTIFIER_FORMAT,
            "splits": {"test": {"source_group_ids": [fingerprint]}}
        }
        self.assertEqual(manifest_source_groups(manifest), {fingerprint})
        self.assertNotIn(raw_group, str(manifest))

    def test_legacy_manifest_raw_groups_remain_supported(self):
        raw_group = "legacy-source-group-42"
        manifest = {
            "splits": {"test": {"source_group_ids": [raw_group]}}
        }
        self.assertEqual(
            manifest_source_groups(manifest),
            {identifier_fingerprint(raw_group, "source-group")}
        )

    def test_unknown_identifier_format_is_rejected(self):
        manifest = {
            "identifier_format": "unknown-v9",
            "splits": {"test": {"source_group_ids": ["a" * 64]}}
        }
        with self.assertRaisesRegex(ValueError, "unsupported"):
            manifest_source_groups(manifest)

    def test_malformed_fingerprints_are_rejected(self):
        manifest = {
            "identifier_format": IDENTIFIER_FORMAT,
            "splits": {"test": {"source_group_ids": ["not-a-sha256"]}}
        }
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            manifest_source_groups(manifest)


if __name__ == "__main__":
    unittest.main()
