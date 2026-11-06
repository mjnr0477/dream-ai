import copy
import unittest
import hashlib
import tempfile
from pathlib import Path

from dream_ai_optin_pipeline import export_optin_rows, prepare_sft_splits


SESSION_HASH = "a" * 64


def optin_row(record_id="11111111-1111-4111-8111-111111111111", **overrides):
    row = {
        "id": record_id,
        "mode": "general",
        "input_text": "What is photosynthesis and why is sunlight important?",
        "target_text": "Photosynthesis is the process plants use to convert light energy into chemical energy.",
        "prompt": "What is photosynthesis and why is sunlight important?",
        "response": "Photosynthesis is the process plants use to convert light energy into chemical energy.",
        "consented": True,
        "consent_version": "dream-learning-opt-in-v2",
        "consented_at": "2026-11-06T10:00:00Z",
        "review_status": "pending",
        "status": "pending",
        "created_at": "2026-11-06T10:00:00Z",
        "metadata": {"source_group_sha256": SESSION_HASH, "language": "en"},
    }
    row.update(overrides)
    return row


def approved_sft_record(index, group=None):
    return {
        "id": "dream-optin-" + f"{index + 1:064x}",
        "source": "DREAM AI opt-in exchange",
        "source_group": group or f"dream-ai-session-{index:064x}",
        "license": "DREAM AI Contributor Permission v2",
        "license_url": "https://dream-ai-live.onrender.com/learning-data-terms.html",
        "source_url": "https://dream-ai-live.onrender.com",
        "rights_basis": "explicit-permission",
        "consent_version": "dream-learning-opt-in-v2",
        "source_date": "2026-11-06",
        "language": "en",
        "rights_confirmed": True,
        "privacy_reviewed": True,
        "quality_reviewed": True,
        "approval_granted": True,
        "messages": [
            {"role": "user", "content": f"Explain concept number {index} in simple terms."},
            {"role": "assistant", "content": f"Concept {index} is explained with a clear and independently reviewed answer."},
        ],
    }


class DreamAIOptinPipelineTests(unittest.TestCase):
    def test_malformed_mode_is_safely_downgraded_to_general(self):
        row = optin_row(mode={"unexpected": "object"})
        candidates, report = export_optin_rows([row])
        self.assertEqual(report["exported"], 1)
        self.assertEqual(candidates[0]["dream_mode"], "general")

    def test_exports_only_v2_consented_pending_rows_as_unapproved_sft_candidates(self):
        candidates, report = export_optin_rows([optin_row()])
        self.assertEqual(report["exported"], 1)
        self.assertEqual(len(candidates), 1)
        row = candidates[0]
        self.assertEqual(row["source_group"], "dream-ai-session-" + SESSION_HASH)
        self.assertEqual(row["source_date"], "2026-11-06")
        self.assertTrue(row["id"].startswith("dream-optin-"))
        self.assertNotEqual(row["id"], "11111111-1111-4111-8111-111111111111")
        self.assertNotIn("11111111-1111-4111-8111-111111111111", str(row))
        self.assertEqual(row["messages"][0]["role"], "user")
        self.assertEqual(row["messages"][1]["role"], "assistant")
        for flag in ("rights_confirmed", "privacy_reviewed", "quality_reviewed", "approval_granted"):
            self.assertIs(row[flag], False)

    def test_excludes_v1_unconsented_and_already_reviewed_rows(self):
        old = optin_row("22222222-2222-4222-8222-222222222222", consent_version="dream-learning-opt-in-v1")
        not_consented = optin_row("33333333-3333-4333-8333-333333333333", consented=False)
        reviewed = optin_row("44444444-4444-4444-8444-444444444444", status="approved", review_status="approved")
        candidates, report = export_optin_rows([old, not_consented, reviewed])
        self.assertEqual(candidates, [])
        self.assertEqual(report["exported"], 0)
        self.assertEqual(sum(report["skipped_reasons"].values()), 3)

    def test_rejects_rows_without_a_valid_source_group_hash(self):
        row = optin_row(metadata={"language": "en", "source_group_sha256": "not-a-hash"})
        candidates, report = export_optin_rows([row])
        self.assertEqual(candidates, [])
        self.assertEqual(report["skipped_reasons"].get("invalid_source_group_hash"), 1)

    def test_export_deduplicates_dialogue_and_never_puts_text_in_report(self):
        first = optin_row()
        second = optin_row("55555555-5555-4555-8555-555555555555", mode="coding")
        candidates, report = export_optin_rows([first, second])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(report["skipped_reasons"].get("duplicate_dialogue"), 1)
        report_text = str(report)
        self.assertNotIn(first["prompt"], report_text)
        self.assertNotIn(first["response"], report_text)
        self.assertNotIn(SESSION_HASH, report_text)

    def test_splits_approved_records_by_source_group_without_leakage(self):
        records = [approved_sft_record(i) for i in range(8)]
        train, validation, manifest = prepare_sft_splits(records)
        train_groups = {item["source_group"] for item in train}
        validation_groups = {item["source_group"] for item in validation}
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertFalse(train_groups & validation_groups)
        self.assertEqual(len(train) + len(validation), len(records))
        self.assertEqual(manifest["records"], len(records))
        self.assertNotIn("dream-ai-session-", str(manifest))

    def test_splits_are_deterministic_when_input_order_changes(self):
        records = [approved_sft_record(i) for i in range(10)]
        train_a, val_a, manifest_a = prepare_sft_splits(records)
        train_b, val_b, manifest_b = prepare_sft_splits(list(reversed(records)))
        self.assertEqual([row["id"] for row in train_a], [row["id"] for row in train_b])
        self.assertEqual([row["id"] for row in val_a], [row["id"] for row in val_b])
        self.assertEqual(manifest_a, manifest_b)

    def test_manifest_split_hash_matches_exact_jsonl_bytes_written(self):
        records = [approved_sft_record(i) for i in range(8)]
        train, validation, manifest = prepare_sft_splits(records)
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.jsonl"
            validation_path = Path(directory) / "validation.jsonl"
            from dream_ai_optin_pipeline import write_jsonl
            write_jsonl(train_path, train)
            write_jsonl(validation_path, validation)
            self.assertEqual(
                hashlib.sha256(train_path.read_bytes()).hexdigest(),
                manifest["splits"]["train"]["sha256"]
            )
            self.assertEqual(
                hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                manifest["splits"]["validation"]["sha256"]
            )

    def test_refuses_fewer_than_four_independent_source_groups(self):
        records = [approved_sft_record(i) for i in range(3)]
        with self.assertRaisesRegex(ValueError, "at least 4 independent source groups"):
            prepare_sft_splits(records)

    def test_refuses_any_record_missing_a_human_review_flag(self):
        records = [approved_sft_record(i) for i in range(4)]
        records[0]["quality_reviewed"] = False
        with self.assertRaisesRegex(ValueError, "quality_reviewed"):
            prepare_sft_splits(records)

    def test_refuses_duplicate_dialogue_content_even_across_different_ids(self):
        records = [approved_sft_record(i) for i in range(4)]
        records[1]["messages"] = copy.deepcopy(records[0]["messages"])
        with self.assertRaisesRegex(ValueError, "duplicate dialogue content"):
            prepare_sft_splits(records)

    def test_refuses_legacy_v1_even_when_review_flags_are_true(self):
        records = [approved_sft_record(i) for i in range(4)]
        records[0]["consent_version"] = "dream-learning-opt-in-v1"
        with self.assertRaisesRegex(ValueError, "dream-learning-opt-in-v2"):
            prepare_sft_splits(records)

    def test_refuses_unrecognized_source_or_permission_metadata(self):
        records = [approved_sft_record(i) for i in range(4)]
        records[0]["license_url"] = "https://example.org/other-license"
        with self.assertRaisesRegex(ValueError, "DREAM AI contributor terms"):
            prepare_sft_splits(records)

    def test_refuses_raw_source_group_identifiers(self):
        records = [approved_sft_record(i) for i in range(4)]
        records[0]["source_group"] = "raw-browser-session-uuid"
        with self.assertRaisesRegex(ValueError, "privacy-preserving session fingerprint"):
            prepare_sft_splits(records)

    def test_refuses_unhashed_record_identifiers(self):
        records = [approved_sft_record(i) for i in range(4)]
        records[0]["id"] = "raw-database-row-uuid"
        with self.assertRaisesRegex(ValueError, "hashed DREAM opt-in candidate id"):
            prepare_sft_splits(records)

    def test_refuses_non_https_provenance(self):
        records = [approved_sft_record(i) for i in range(4)]
        records[0]["source_url"] = "http://example.com/source"
        with self.assertRaisesRegex(ValueError, "https"):
            prepare_sft_splits(records)


if __name__ == "__main__":
    unittest.main()
