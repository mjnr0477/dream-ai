import unittest

from verify_dream_ai_sft_audits import verify_audit_reports


def manifest():
    return {
        "schema": "dream-sft-splits-v1",
        "dataset_sha256": "d" * 64,
        "splits": {
            "train": {"records": 8, "sha256": "a" * 64},
            "validation": {"records": 2, "sha256": "b" * 64},
        },
    }


def report(count, digest):
    return {
        "schema": "dream-corpus-audit-v1",
        "input_sha256": digest,
        "records": count,
        "error_count": 0,
        "unreviewed_records": 0,
        "possible_pii_findings": 0,
        "unreviewed_records": 0,
        "error_count": 0,
        "parse_errors": 0,
        "finding_counts": {},
        "errors": [],
    }


class DreamSFTAuditGateTests(unittest.TestCase):
    def test_accepts_clean_reports_bound_to_exact_manifest_hashes(self):
        result = verify_audit_reports(manifest(), report(8, "a" * 64), report(2, "b" * 64))
        self.assertTrue(result["passed"])
        self.assertEqual(result["splits"]["train"]["records"], 8)
        self.assertEqual(result["splits"]["validation"]["records"], 2)

    def test_rejects_possible_pii_findings(self):
        train = report(8, "a" * 64)
        train["possible_pii_findings"] = 1
        with self.assertRaisesRegex(ValueError, "possible_pii_findings"):
            verify_audit_reports(manifest(), train, report(2, "b" * 64))

    def test_rejects_any_unresolved_audit_finding(self):
        validation = report(2, "b" * 64)
        validation["finding_counts"] = {"possible_email": 1}
        with self.assertRaisesRegex(ValueError, "audit_findings"):
            verify_audit_reports(manifest(), report(8, "a" * 64), validation)

    def test_rejects_stale_audit_report_for_different_file_content(self):
        train = report(8, "e" * 64)
        with self.assertRaisesRegex(ValueError, "stale_train_audit_hash"):
            verify_audit_reports(manifest(), train, report(2, "b" * 64))

    def test_rejects_unreviewed_examples(self):
        train = report(8, "a" * 64)
        train["unreviewed_records"] = 1
        with self.assertRaisesRegex(ValueError, "unreviewed_records"):
            verify_audit_reports(manifest(), train, report(2, "b" * 64))

    def test_rejects_missing_report_sections(self):
        train = report(8, "a" * 64)
        del train["errors"]
        with self.assertRaisesRegex(ValueError, "audit_parse_errors"):
            verify_audit_reports(manifest(), train, report(2, "b" * 64))


if __name__ == "__main__":
    unittest.main()
