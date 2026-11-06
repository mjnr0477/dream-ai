import unittest

from license_policy import (
    POLICY_VERSION,
    classify_record_license,
    require_commercial_training_license,
)


def record(license_name, license_url, **overrides):
    value = {
        "license": license_name,
        "license_url": license_url,
        "rights_basis": "open-license",
        "rights_confirmed": True,
    }
    value.update(overrides)
    return value


class CommercialTrainingLicensePolicyTests(unittest.TestCase):
    def test_policy_is_versioned(self):
        self.assertEqual(POLICY_VERSION, "dream-commercial-training-license-policy-v1")

    def test_allows_cc0_1_0(self):
        result = classify_record_license(record(
            "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"
        ))
        self.assertEqual(result["status"], "allowed")
        self.assertFalse(result["attribution_required"])

    def test_allows_cc_by_4_with_attribution(self):
        result = classify_record_license(record(
            "CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/"
        ))
        self.assertEqual(result["status"], "allowed")
        self.assertTrue(result["attribution_required"])

    def test_allows_apache_2_for_oasst1_style_dataset_terms(self):
        result = classify_record_license(record(
            "Apache-2.0", "https://www.apache.org/licenses/LICENSE-2.0"
        ))
        self.assertEqual(result["status"], "allowed")
        self.assertEqual(result["code"], "apache_2_0")

    def test_blocks_noncommercial_even_when_human_rights_flag_is_true(self):
        result = classify_record_license(record(
            "CC-BY-NC-SA-4.0",
            "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        ))
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["code"], "noncommercial_restriction")

    def test_restrictive_url_wins_over_incorrect_permissive_display_label(self):
        result = classify_record_license(record(
            "CC-BY-4.0",
            "https://creativecommons.org/licenses/by-nc/4.0/",
        ))
        self.assertEqual(result["status"], "blocked")

    def test_holds_sharealike_for_separate_review(self):
        result = classify_record_license(record(
            "CC-BY-SA-4.0",
            "https://creativecommons.org/licenses/by-sa/4.0/",
        ))
        self.assertEqual(result["status"], "review_required")
        self.assertIn("share_alike", result["code"])

    def test_holds_no_derivatives_for_separate_review(self):
        result = classify_record_license(record(
            "CC-BY-ND-4.0",
            "https://creativecommons.org/licenses/by-nd/4.0/",
        ))
        self.assertEqual(result["status"], "review_required")
        self.assertIn("no_derivatives", result["code"])

    def test_holds_dataset_collection_license_without_item_level_rights(self):
        result = classify_record_license(record(
            "ODC-By-1.0-claimed",
            "https://opendatacommons.org/licenses/by/1-0/",
        ))
        self.assertEqual(result["status"], "review_required")
        self.assertEqual(result["code"], "dataset_collection_license_does_not_clear_item_rights")

    def test_unknown_license_is_held_for_review(self):
        result = classify_record_license(record(
            "Some-Unknown-License", "https://example.org/license"
        ))
        self.assertEqual(result["status"], "review_required")

    def test_separate_written_commercial_training_permission_overrides_restriction(self):
        result = classify_record_license(record(
            "CC-BY-NC-SA-4.0",
            "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            rights_basis="explicit-permission",
            permission_scope="Worldwide permission for commercial AI model training.",
            permission_evidence_url="https://example.org/permission/123",
        ))
        self.assertEqual(result["status"], "allowed")
        self.assertEqual(result["code"], "separate_commercial_training_permission")

    def test_restricted_license_override_requires_evidence_and_scope(self):
        result = classify_record_license(record(
            "CC-BY-NC-4.0",
            "https://creativecommons.org/licenses/by-nc/4.0/",
            rights_basis="explicit-permission",
            permission_scope="Training use only.",
        ))
        self.assertEqual(result["status"], "blocked")

    def test_dream_v2_user_permission_is_recognized_only_with_consent_and_review(self):
        base = {
            "license": "DREAM AI Contributor Permission v2",
            "license_url": "https://dream-ai-live.onrender.com/learning-data-terms.html",
            "rights_basis": "explicit-permission",
            "consent_version": "dream-learning-opt-in-v2",
            "consented": True,
            "consented_at": "2026-11-06T10:00:00+00:00",
            "human_reviewed_at": "2026-11-06T11:00:00+00:00",
            "rights_confirmed": True,
        }
        self.assertEqual(classify_record_license(base)["status"], "allowed")
        base["consented"] = False
        self.assertEqual(classify_record_license(base)["status"], "review_required")

    def test_prepare_function_fails_closed_for_unresolved_terms(self):
        with self.assertRaisesRegex(ValueError, "noncommercial_restriction"):
            require_commercial_training_license(record(
                "CC-BY-NC-4.0", "https://creativecommons.org/licenses/by-nc/4.0/"
            ), line_number=7)


if __name__ == "__main__":
    unittest.main()
