"""Tests for the explicit human approval gate for DREAM corpus candidates."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.review_candidates import candidate_problem, review_candidate, review_file


def candidate(record_id="candidate-1"):
    return {
        "id": record_id,
        "source": "OpenAssistant OASST1",
        "source_group": "oasst-tree-1",
        "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "rights_basis": "open-license",
        "source_date": "2023-04-12",
        "rights_confirmed": False,
        "privacy_reviewed": False,
        "quality_reviewed": False,
        "approval_granted": False,
        "language": "en",
        "text": "This is a sufficiently long example record for DREAM corpus review and training."
    }


class ReviewCandidatesTests(unittest.TestCase):
    def test_valid_candidate_passes_metadata_preflight_only(self):
        item = candidate()
        self.assertIsNone(candidate_problem(item))
        self.assertIs(item["approval_granted"], False)

    def test_all_four_explicit_yes_answers_are_required_for_approval(self):
        answers = iter(["y", "yes", "y", "yes"])
        reviewed, stop, missing = review_candidate(
            candidate(), input_fn=lambda _prompt: next(answers), output_fn=lambda *_: None
        )
        self.assertFalse(stop)
        self.assertEqual(missing, [])
        self.assertIsNotNone(reviewed)
        for field in ("rights_confirmed", "privacy_reviewed", "quality_reviewed", "approval_granted"):
            self.assertIs(reviewed[field], True)
        self.assertIn("human_reviewed_at", reviewed)

    def test_one_negative_answer_prevents_approval(self):
        answers = iter(["y", "n", "y", "y"])
        reviewed, stop, missing = review_candidate(
            candidate(), input_fn=lambda _prompt: next(answers), output_fn=lambda *_: None
        )
        self.assertIsNone(reviewed)
        self.assertFalse(stop)
        self.assertEqual(missing, ["privacy_reviewed"])

    def test_quit_does_not_create_an_approval(self):
        reviewed, stop, missing = review_candidate(
            candidate(), input_fn=lambda _prompt: "q", output_fn=lambda *_: None
        )
        self.assertIsNone(reviewed)
        self.assertTrue(stop)
        self.assertEqual(missing, [])

    def test_invalid_provenance_is_rejected_without_prompting(self):
        item = candidate()
        item["source_url"] = "http://example.com/dataset"
        self.assertEqual(candidate_problem(item), "invalid_source_url")
        reviewed, stop, missing = review_candidate(
            item,
            input_fn=lambda _prompt: self.fail("Invalid candidates must not be approved interactively"),
            output_fn=lambda *_: None
        )
        self.assertIsNone(reviewed)
        self.assertFalse(stop)
        self.assertEqual(missing, ["invalid_source_url"])

    def test_review_file_persists_decisions_and_skips_them_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidates.jsonl"
            output = root / "approved.jsonl"
            decisions = root / "decisions.jsonl"
            source.write_text(
                json.dumps(candidate("accepted-1")) + "\n" +
                json.dumps(candidate("rejected-1")) + "\n",
                encoding="utf-8"
            )
            answers = iter(["y", "y", "y", "y", "y", "n", "y", "y"])
            summary = review_file(
                source, output, decisions, limit=2,
                input_fn=lambda _prompt: next(answers), output_fn=lambda *_: None
            )
            self.assertEqual(summary["presented"], 2)
            self.assertEqual(summary["approved"], 1)
            self.assertEqual(summary["rejected"], 1)

            approved_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            decision_rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(approved_rows), 1)
            self.assertEqual(approved_rows[0]["id"], "accepted-1")
            self.assertEqual(len(decision_rows), 2)
            self.assertNotIn("text", decision_rows[0])
            self.assertNotIn("text", decision_rows[1])
            self.assertIn("id_hash", decision_rows[0])
            self.assertIn("id_hash", decision_rows[1])
            self.assertNotIn("id", decision_rows[0])
            self.assertNotIn("id", decision_rows[1])

            resumed = review_file(
                source, output, decisions, limit=2,
                input_fn=lambda _prompt: self.fail("Previously reviewed records must not be prompted again"),
                output_fn=lambda *_: None
            )
            self.assertEqual(resumed["skipped"], 2)
            self.assertEqual(resumed["presented"], 0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)

    def test_candidate_input_cannot_be_reused_as_output_or_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "candidates.jsonl"
            source.write_text(json.dumps(candidate()) + "\\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be separate files"):
                review_file(source, source, root / "decisions.jsonl")

    def test_rejects_identifying_ids_and_malformed_message_records(self):
        item = candidate("reviewer@example.com")
        self.assertEqual(candidate_problem(item), "invalid_id")
        self.assertEqual(candidate_problem(candidate("5551234567")), "invalid_id")

        item = candidate("conversation-1")
        item.pop("text")
        item["messages"] = [
            {"role": "user", "content": "A valid user message that is sufficiently long."},
            {"role": "tool", "content": "Unexpected role content that should not be accepted."}
        ]
        self.assertEqual(candidate_problem(item), "invalid_messages")

    def test_valid_user_assistant_conversation_can_be_reviewed(self):
        item = candidate("conversation-2")
        item.pop("text")
        item["messages"] = [
            {"role": "user", "content": "Explain how rain forms in the atmosphere."},
            {"role": "assistant", "content": "Water evaporates, cools and condenses into droplets that form clouds and precipitation."}
        ]
        self.assertIsNone(candidate_problem(item))


if __name__ == "__main__":
    unittest.main()
