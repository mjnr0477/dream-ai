import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

import import_wikidata_candidates as importer


def binding(country="Q114", capital="Q19689", country_label="Kenya", capital_label="Nairobi"):
    return {
        "country": {"type": "uri", "value": f"http://www.wikidata.org/entity/{country}"},
        "countryLabel": {"type": "literal", "value": country_label},
        "capital": {"type": "uri", "value": f"http://www.wikidata.org/entity/{capital}"},
        "capitalLabel": {"type": "literal", "value": capital_label},
    }


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, limit=-1):
        return self.payload[:limit]


class WikidataCandidateImporterTests(unittest.TestCase):
    def test_query_is_bounded_and_scoped_to_structured_capital_statements(self):
        query = importer.build_query(50, 100)
        self.assertIn("wdt:P31 wd:Q3624078", query)
        self.assertIn("wdt:P36 ?capital", query)
        self.assertIn('LANG(?countryLabel) = "en"', query)
        self.assertIn('LANG(?capitalLabel) = "en"', query)
        self.assertTrue(query.endswith("LIMIT 50\nOFFSET 100"))
        with self.assertRaises(ValueError):
            importer.build_query(101, 0)
        with self.assertRaises(ValueError):
            importer.build_query(10, -1)

    def test_candidate_is_provenance_rich_and_all_review_flags_remain_false(self):
        candidate = importer.build_candidate(
            binding(), "2026-11-23", "a" * 64
        )
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["source_group"], "wikidata-country-Q114")
        self.assertEqual(candidate["source_url"], "https://www.wikidata.org/wiki/Q114")
        self.assertEqual(candidate["license"], "CC0-1.0")
        self.assertEqual(candidate["license_url"], importer.LICENSE_URL)
        self.assertEqual(candidate["metadata"]["wikidata_property_id"], "P36")
        self.assertEqual(candidate["metadata"]["wikidata_object_qid"], "Q19689")
        for flag in importer.REVIEW_FLAGS:
            self.assertIs(candidate[flag], False)

    def test_all_statements_for_one_country_share_a_source_group(self):
        first = importer.build_candidate(binding(capital="Q19689"), "2026-11-23", "a" * 64)
        second = importer.build_candidate(
            binding(capital="Q1748", capital_label="Mombasa"),
            "2026-11-23",
            "a" * 64,
        )
        other = importer.build_candidate(
            binding(country="Q1033", capital="Q1861", country_label="Nigeria", capital_label="Abuja"),
            "2026-11-23",
            "a" * 64,
        )
        self.assertEqual(first["source_group"], second["source_group"])
        self.assertNotEqual(first["source_group"], other["source_group"])
        self.assertNotEqual(first["id"], second["id"])

    def test_rejects_non_wikidata_entity_uris_and_invalid_labels(self):
        invalid_uri = binding()
        invalid_uri["country"]["value"] = "https://example.com/entity/Q114"
        self.assertIsNone(
            importer.build_candidate(invalid_uri, "2026-11-23", "a" * 64)
        )
        invalid_label = binding()
        invalid_label["capitalLabel"]["value"] = " " * 3
        self.assertIsNone(
            importer.build_candidate(invalid_label, "2026-11-23", "a" * 64)
        )

    def test_fetch_page_uses_current_v2_endpoint_and_identifying_user_agent(self):
        payload = {"results": {"bindings": [binding()]}}
        response = FakeResponse(json.dumps(payload).encode("utf-8"))
        with patch.object(importer, "urlopen", return_value=response) as opener:
            rows = importer.fetch_page(10, 0, timeout=20)
        self.assertEqual(len(rows), 1)
        request = opener.call_args.args[0]
        self.assertTrue(request.full_url.startswith(importer.ENDPOINT + "?"))
        self.assertEqual(request.get_header("Accept"), "application/sparql-results+json")
        self.assertIn("DREAM-AI", request.get_header("User-agent"))
        self.assertEqual(opener.call_args.kwargs["timeout"], 20)

    def test_rejects_malformed_sparql_json_schema(self):
        response = FakeResponse(b'{"results":{"bindings":{}}}')
        with patch.object(importer, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "SPARQL JSON results schema"):
                importer.fetch_page(10, 0)

    def test_rate_limit_fails_closed_without_retrying_another_endpoint(self):
        error = HTTPError(importer.ENDPOINT, 429, "Too Many Requests", {}, None)
        with patch.object(importer, "urlopen", side_effect=error) as opener:
            with self.assertRaisesRegex(RuntimeError, "rate-limited"):
                importer.fetch_page(10, 0)
        self.assertEqual(opener.call_count, 1)

    def test_pagination_deduplicates_and_preserves_country_group_isolation(self):
        pages = [
            [binding(), binding(country="Q1033", capital="Q1861", country_label="Nigeria", capital_label="Abuja")],
            [binding(), binding(capital="Q1748", capital_label="Mombasa")],
        ]
        calls = []

        def fetcher(limit, offset, timeout):
            calls.append((limit, offset, timeout))
            return pages.pop(0)

        records, summary = importer.collect_candidates(
            max_rows=4,
            page_size=2,
            timeout=20,
            collection_date="2026-11-23",
            page_fetcher=fetcher,
        )
        self.assertEqual(calls, [(2, 0, 20), (2, 2, 20)])
        self.assertEqual(summary["pages"], 2)
        self.assertEqual(summary["scanned_rows"], 4)
        self.assertEqual(summary["duplicate_statements"], 1)
        self.assertEqual(len(records), 3)
        kenya_groups = {
            row["source_group"] for row in records
            if row["metadata"]["wikidata_subject_qid"] == "Q114"
        }
        nigeria_groups = {
            row["source_group"] for row in records
            if row["metadata"]["wikidata_subject_qid"] == "Q1033"
        }
        self.assertEqual(len(kenya_groups), 1)
        self.assertEqual(len(nigeria_groups), 1)
        self.assertNotEqual(kenya_groups, nigeria_groups)

    def test_export_writes_hash_bound_report_without_record_text(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "wikidata-candidates.jsonl"
            report_path = Path(temp) / "wikidata-report.json"
            report = importer.export_candidates(
                output,
                report_path,
                max_rows=2,
                page_size=2,
                collection_date="2026-11-23",
                page_fetcher=lambda limit, offset, timeout: [binding()],
            )
            raw = output.read_bytes()
            record = json.loads(raw.decode("utf-8").splitlines()[0])
            self.assertEqual(report["output_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertIs(report["training_approved"], False)
            self.assertIs(report["human_review_required"], True)
            for flag in importer.REVIEW_FLAGS:
                self.assertIs(record[flag], False)
            report_text = report_path.read_text(encoding="utf-8")
            self.assertNotIn("Kenya", report_text)
            self.assertNotIn("Nairobi", report_text)

    def test_failed_fetch_does_not_replace_existing_candidate_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "wikidata-candidates.jsonl"
            report_path = Path(temp) / "wikidata-report.json"
            output.write_text("previous candidate file\n", encoding="utf-8")

            def fail_fetch(limit, offset, timeout):
                raise RuntimeError("simulated endpoint failure")

            with self.assertRaisesRegex(RuntimeError, "simulated endpoint failure"):
                importer.export_candidates(
                    output,
                    report_path,
                    max_rows=2,
                    page_size=2,
                    collection_date="2026-11-23",
                    page_fetcher=fail_fetch,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "previous candidate file\n")
            self.assertFalse(report_path.exists())

    def test_invalid_limits_and_collection_dates_are_rejected_before_fetch(self):
        with self.assertRaises(ValueError):
            importer.collect_candidates(max_rows=0)
        with self.assertRaises(ValueError):
            importer.collect_candidates(max_rows=10, page_size=11)
        with self.assertRaises(ValueError):
            importer.collect_candidates(collection_date="2026-11-23T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
