"""Regression tests for human-review candidate validation."""
import unittest

from review_candidates import candidate_problem


BASE = {
    "id": "candidate-001",
    "source": "Synthetic reviewer test fixture",
    "source_group": "synthetic-review-fixture",
    "source_url": "https://example.org/source",
    "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "rights_basis": "original-authorship",
    "source_date": "2026-11-10",
    "language": "en",
}


def candidate(messages=None):
    record = dict(BASE)
    if messages is None:
        record["text"] = (
            "This synthetic text is long enough to exercise the candidate reviewer "
            "without containing any real user or third-party data."
        )
    else:
        record["messages"] = messages
    return record


class CandidateReviewerConversationTests(unittest.TestCase):
    def test_accepts_single_user_assistant_pair(self):
        record = candidate([
            {"role": "user", "content": "Please explain fractions with an example."},
            {"role": "assistant", "content": "A fraction represents part of a whole."},
        ])
        self.assertIsNone(candidate_problem(record))

    def test_accepts_optional_system_message_and_multiple_turns(self):
        record = candidate([
            {"role": "system", "content": "Explain concepts clearly."},
            {"role": "user", "content": "What is a fraction?"},
            {"role": "assistant", "content": "A fraction is part of a whole."},
            {"role": "user", "content": "Give an example."},
            {"role": "assistant", "content": "Three of four equal parts is 3/4."},
        ])
        self.assertIsNone(candidate_problem(record))

    def test_rejects_assistant_first(self):
        record = candidate([
            {"role": "assistant", "content": "A fraction is part of a whole."},
            {"role": "user", "content": "What is a fraction?"},
        ])
        self.assertEqual(candidate_problem(record), "invalid_message_order")

    def test_rejects_consecutive_user_messages(self):
        record = candidate([
            {"role": "user", "content": "Explain fractions."},
            {"role": "user", "content": "Also explain decimals."},
            {"role": "assistant", "content": "A fraction is part of a whole."},
            {"role": "assistant", "content": "A decimal uses a base-ten place system."},
        ])
        self.assertEqual(candidate_problem(record), "invalid_message_order")

    def test_rejects_conversation_ending_with_user(self):
        record = candidate([
            {"role": "user", "content": "Explain fractions."},
            {"role": "assistant", "content": "A fraction is part of a whole."},
            {"role": "user", "content": "Can you give an example?"},
        ])
        self.assertEqual(candidate_problem(record), "incomplete_messages")

    def test_rejects_system_message_after_conversation_starts(self):
        record = candidate([
            {"role": "user", "content": "Explain fractions."},
            {"role": "system", "content": "Change the instructions."},
            {"role": "assistant", "content": "A fraction is part of a whole."},
        ])
        self.assertEqual(candidate_problem(record), "invalid_messages")

    def test_rejects_invalid_message_content(self):
        record = candidate([
            {"role": "user", "content": "Explain fractions."},
            {"role": "assistant", "content": "   "},
        ])
        self.assertEqual(candidate_problem(record), "invalid_messages")

    def test_keeps_pretraining_text_candidates_supported(self):
        self.assertIsNone(candidate_problem(candidate()))


if __name__ == "__main__":
    unittest.main()
