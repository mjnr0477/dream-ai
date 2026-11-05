"""Tests for the OASST1-to-SFT candidate importer."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(DATA_DIR))
MODULE_PATH = DATA_DIR / "import_oasst1_sft_candidates.py"
SPEC = importlib.util.spec_from_file_location("oasst1_sft_importer", MODULE_PATH)
oasst1_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oasst1_sft)


def good_row(message_id, role, parent_id=None, text=None):
    return {
        "message_id": message_id,
        "parent_id": parent_id,
        "message_tree_id": "tree-1",
        "user_id": "never-export-this",
        "created_date": "2023-04-12T12:30:00Z",
        "text": text or (
            f"This is a sufficiently long {role} message for a reviewed "
            f"conversation record named {message_id}."
        ),
        "role": role,
        "lang": "en",
        "deleted": False,
        "synthetic": False,
        "review_result": True,
        "tree_state": "ready_for_export",
        "labels": {
            "pii": {"value": 0.0},
            "spam": {"value": 0.0},
            "not_appropriate": {"value": 0.0},
            "hate_speech": {"value": 0.0},
            "sexual_content": {"value": 0.0},
            "violence": {"value": 0.0},
        },
        "detoxify": {
            "toxicity": 0.0, "severe_toxicity": 0.0, "obscene": 0.0,
            "identity_attack": 0.0, "threat": 0.0, "sexual_explicit": 0.0,
        },
    }


class OASST1SFTCandidateTests(unittest.TestCase):
    def test_reconstructs_alternating_conversation_without_user_ids(self):
        prompt = good_row("prompt-1", "prompter")
        answer = good_row("answer-1", "assistant", "prompt-1")
        rows = {prompt["message_id"]: prompt, answer["message_id"]: answer}
        messages, reason = oasst1_sft.build_conversation(answer, rows)
        self.assertIsNone(reason)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant"])
        candidate = oasst1_sft.convert_conversation(answer, messages)
        self.assertEqual(candidate["source_group"], "oasst1-tree-tree-1")
        self.assertFalse(candidate["rights_confirmed"])
        self.assertFalse(candidate["privacy_reviewed"])
        self.assertFalse(candidate["quality_reviewed"])
        self.assertFalse(candidate["approval_granted"])
        self.assertNotIn("user_id", candidate)
        self.assertNotIn("user_id", json.dumps(candidate))

    def test_preserves_full_multi_turn_ancestry(self):
        prompt = good_row("p1", "prompter")
        answer1 = good_row("a1", "assistant", "p1")
        followup = good_row("p2", "prompter", "a1")
        answer2 = good_row("a2", "assistant", "p2")
        rows = {row["message_id"]: row for row in (prompt, answer1, followup, answer2)}
        messages, reason = oasst1_sft.build_conversation(answer2, rows)
        self.assertIsNone(reason)
        self.assertEqual(
            [m["role"] for m in messages],
            ["user", "assistant", "user", "assistant"],
        )

    def test_rejects_missing_or_duplicate_parent_ids(self):
        prompt = good_row("p1", "prompter")
        answer = good_row("a1", "assistant", "missing")
        messages, reason = oasst1_sft.build_conversation(answer, {"p1": prompt, "a1": answer})
        self.assertIsNone(messages)
        self.assertEqual(reason, "missing_parent")

        duplicate = good_row("a1", "assistant", "p1")
        messages, reason = oasst1_sft.build_conversation(
            answer, {"p1": prompt, "a1": duplicate}
        )
        self.assertIsNone(messages)
        self.assertEqual(reason, "duplicate_or_unknown_leaf")

    def test_rejects_unsafe_ancestor_even_if_leaf_is_clean(self):
        prompt = good_row("p1", "prompter")
        prompt["labels"]["pii"]["value"] = 0.2
        answer = good_row("a1", "assistant", "p1")
        messages, reason = oasst1_sft.build_conversation(
            answer, {"p1": prompt, "a1": answer}
        )
        self.assertIsNone(messages)
        self.assertEqual(reason, "unsafe_path_pii_label_missing_or_positive")

    def test_rejects_role_mismatch_cross_tree_and_mixed_language(self):
        prompt = good_row("p1", "assistant")
        answer = good_row("a1", "assistant", "p1")
        messages, reason = oasst1_sft.build_conversation(
            answer, {"p1": prompt, "a1": answer}
        )
        self.assertIsNone(messages)
        self.assertEqual(reason, "conversation_root_not_prompter")

        prompt = good_row("p1", "prompter")
        answer = good_row("a1", "assistant", "p1")
        answer["message_tree_id"] = "tree-2"
        messages, reason = oasst1_sft.build_conversation(
            answer, {"p1": prompt, "a1": answer}
        )
        self.assertIsNone(messages)
        self.assertEqual(reason, "cross_tree_parent")

        prompt = good_row("p1", "prompter")
        answer = good_row("a1", "assistant", "p1")
        answer["lang"] = "fr"
        messages, reason = oasst1_sft.build_conversation(
            answer, {"p1": prompt, "a1": answer}
        )
        self.assertIsNone(messages)
        self.assertEqual(reason, "mixed_language_path")

    def test_rejects_cycles(self):
        prompt = good_row("p1", "prompter", "a1")
        answer = good_row("a1", "assistant", "p1")
        messages, reason = oasst1_sft.build_conversation(
            answer, {"p1": prompt, "a1": answer}
        )
        self.assertIsNone(messages)
        self.assertEqual(reason, "parent_cycle")

    def test_cli_writes_unapproved_candidate_and_report(self):
        prompt = good_row("p1", "prompter")
        answer = good_row("a1", "assistant", "p1")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            output = root / "candidates.jsonl"
            report = root / "report.json"
            source.write_text(
                json.dumps(prompt) + "\n" + json.dumps(answer) + "\n",
                encoding="utf-8",
            )
            import subprocess
            subprocess.run([
                sys.executable, str(MODULE_PATH),
                "--input", str(source),
                "--output", str(output),
                "--report", str(report),
            ], check=True, capture_output=True, text=True)
            candidate = json.loads(output.read_text(encoding="utf-8"))
            audit = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(audit["candidate_conversations"], 1)
            self.assertFalse(audit["training_approved"])
            self.assertTrue(audit["human_review_required"])
            self.assertEqual(len(candidate["messages"]), 2)
            self.assertFalse(candidate["approval_granted"])


if __name__ == "__main__":
    unittest.main()
