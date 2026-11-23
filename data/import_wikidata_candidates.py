#!/usr/bin/env python3
"""Export a bounded Wikidata CC0 sample as UNAPPROVED DREAM review candidates.

The importer reads only structured sovereign-state capital statements and English
entity labels. It never approves records or writes to an approved training corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

ENDPOINT = "https://query-next.wikidata.org/sparql"
LICENSE = "CC0-1.0"
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"
SOURCE_NAME = "Wikidata structured data (sovereign-state capitals)"
USER_AGENT = (
    "DREAM-AI-Wikidata-Candidate-Importer/0.1 "
    "(https://github.com/mjnr0477/dream-ai)"
)
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
REVIEW_FLAGS = (
    "rights_confirmed",
    "privacy_reviewed",
    "quality_reviewed",
    "approval_granted",
)
QUERY_TEMPLATE = """SELECT DISTINCT ?country ?countryLabel ?capital ?capitalLabel WHERE {
  ?country wdt:P31 wd:Q3624078 ;
           wdt:P36 ?capital .
  ?country rdfs:label ?countryLabel .
  FILTER(LANG(?countryLabel) = "en")
  ?capital rdfs:label ?capitalLabel .
  FILTER(LANG(?capitalLabel) = "en")
}
ORDER BY ?country ?capital"""


def query_template_sha256() -> str:
    return hashlib.sha256(QUERY_TEMPLATE.encode("utf-8")).hexdigest()


def build_query(limit: int, offset: int) -> str:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("page limit must be an integer from 1 to 100")
    if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 1000000:
        raise ValueError("page offset must be an integer from 0 to 1000000")
    return f"{QUERY_TEMPLATE}\nLIMIT {limit}\nOFFSET {offset}"


def fetch_page(limit: int, offset: int, timeout: int = 30) -> list[dict]:
    """Fetch one bounded page from the current Wikidata Query Service v2 endpoint."""
    query = build_query(limit, offset)
    params = urlencode({"query": query, "timeout": f"{timeout}s"})
    request = Request(
        f"{ENDPOINT}?{params}",
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 429:
            raise RuntimeError(
                "Wikidata Query Service rate-limited the request (HTTP 429); retry later."
            ) from exc
        raise RuntimeError(
            f"Wikidata Query Service returned HTTP {exc.code}; no candidate files were written."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            "Wikidata Query Service network/endpoint request failed; no candidate files were written."
        ) from exc

    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Wikidata response exceeded the 5 MiB safety limit")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Wikidata returned invalid UTF-8 SPARQL JSON") from exc
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("results"), dict)
        or not isinstance(payload["results"].get("bindings"), list)
    ):
        raise ValueError("Wikidata response did not match the SPARQL JSON results schema")
    bindings = payload["results"]["bindings"]
    if any(not isinstance(item, dict) for item in bindings):
        raise ValueError("Wikidata response contains a non-object binding")
    if len(bindings) > limit:
        raise ValueError("Wikidata returned more rows than the requested page limit")
    return bindings


def binding_value(binding: dict, key: str) -> str | None:
    item = binding.get(key)
    if not isinstance(item, dict):
        return None
    value = item.get("value")
    return value if isinstance(value, str) else None


def qid_from_uri(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or parsed.netloc != "www.wikidata.org":
        return None
    match = re.fullmatch(r"/entity/(Q[1-9][0-9]*)", parsed.path)
    return match.group(1) if match else None


def clean_label(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > 160:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in normalized):
        return None
    return normalized


def build_candidate(binding: dict, collection_date: str, query_sha256: str) -> dict | None:
    """Convert one statement to a provenance-rich record with all review flags false."""
    country_qid = qid_from_uri(binding_value(binding, "country"))
    capital_qid = qid_from_uri(binding_value(binding, "capital"))
    country_label = clean_label(binding_value(binding, "countryLabel"))
    capital_label = clean_label(binding_value(binding, "capitalLabel"))
    if not all((country_qid, capital_qid, country_label, capital_label)):
        return None
    if country_qid == capital_qid:
        return None

    return {
        "id": f"wikidata-{country_qid}-P36-{capital_qid}",
        "source": SOURCE_NAME,
        "source_group": f"wikidata-country-{country_qid}",
        "source_url": f"https://www.wikidata.org/wiki/{country_qid}",
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "rights_basis": "open-license",
        "source_date": collection_date,
        "language": "en",
        "text": (
            f"Question: What is one listed capital of {country_label}? "
            f"Answer: {capital_label}."
        ),
        "rights_confirmed": False,
        "privacy_reviewed": False,
        "quality_reviewed": False,
        "approval_granted": False,
        "metadata": {
            "source_endpoint": ENDPOINT,
            "query_template_sha256": query_sha256,
            "wikidata_subject_qid": country_qid,
            "wikidata_property_id": "P36",
            "wikidata_object_qid": capital_qid,
            "wikidata_object_url": f"https://www.wikidata.org/wiki/{capital_qid}",
            "retrieval_language": "en",
        },
    }


def validate_collection_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("collection date must use strict YYYY-MM-DD format") from exc
    if parsed.isoformat() != value:
        raise ValueError("collection date must use strict YYYY-MM-DD format")
    return value


def collect_candidates(
    max_rows: int = 200,
    page_size: int = 50,
    timeout: int = 30,
    collection_date: str | None = None,
    page_fetcher=None,
) -> tuple[list[dict], dict]:
    """Fetch a bounded live snapshot; any request failure aborts before files are written."""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 1000:
        raise ValueError("max_rows must be an integer from 1 to 1000")
    if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100:
        raise ValueError("page_size must be an integer from 1 to 100")
    if page_size > max_rows:
        raise ValueError("page_size cannot exceed max_rows")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 5 <= timeout <= 55:
        raise ValueError("timeout must be an integer from 5 to 55 seconds")

    collected = collection_date or datetime.now(timezone.utc).date().isoformat()
    validate_collection_date(collected)
    query_sha256 = query_template_sha256()
    fetcher = page_fetcher or fetch_page
    candidates: list[dict] = []
    seen_ids: set[str] = set()
    scanned_rows = 0
    pages = 0
    rejected_invalid_binding = 0
    duplicate_statements = 0

    while scanned_rows < max_rows:
        limit = min(page_size, max_rows - scanned_rows)
        bindings = fetcher(limit, scanned_rows, timeout)
        if not isinstance(bindings, list) or any(not isinstance(item, dict) for item in bindings):
            raise ValueError("page fetcher returned an invalid bindings collection")
        if len(bindings) > limit:
            raise ValueError("page fetcher returned more rows than requested")
        pages += 1
        scanned_rows += len(bindings)

        for binding in bindings:
            candidate = build_candidate(binding, collected, query_sha256)
            if candidate is None:
                rejected_invalid_binding += 1
                continue
            if candidate["id"] in seen_ids:
                duplicate_statements += 1
                continue
            seen_ids.add(candidate["id"])
            candidates.append(candidate)

        if len(bindings) < limit:
            break

    candidates.sort(key=lambda item: item["id"])
    if not candidates:
        raise RuntimeError("No valid candidate statements were returned; no files were written.")

    summary = {
        "collection_date": collected,
        "query_template_sha256": query_sha256,
        "pages": pages,
        "scanned_rows": scanned_rows,
        "candidate_records": len(candidates),
        "rejected_invalid_binding": rejected_invalid_binding,
        "duplicate_statements": duplicate_statements,
        "max_rows": max_rows,
        "page_size": page_size,
    }
    return candidates, summary


def atomic_write(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlink output: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def export_candidates(
    output_path: str | Path,
    report_path: str | Path | None = None,
    max_rows: int = 200,
    page_size: int = 50,
    timeout: int = 30,
    collection_date: str | None = None,
    page_fetcher=None,
) -> dict:
    output = Path(output_path)
    report = Path(report_path) if report_path is not None else output.with_name(output.name + ".report.json")
    if output.absolute() == report.absolute():
        raise ValueError("candidate output and report must use different paths")

    candidates, summary = collect_candidates(
        max_rows=max_rows,
        page_size=page_size,
        timeout=timeout,
        collection_date=collection_date,
        page_fetcher=page_fetcher,
    )
    jsonl_bytes = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
        for item in candidates
    ).encode("utf-8")
    report_payload = {
        "schema": "dream-wikidata-candidate-export-v1",
        "dataset": "Wikidata sovereign-state capital statements",
        "source_endpoint": ENDPOINT,
        "source_license": LICENSE,
        "license_url": LICENSE_URL,
        "scope": "English labels for Wikidata sovereign-state (Q3624078) P36 capital statements",
        **summary,
        "output_sha256": hashlib.sha256(jsonl_bytes).hexdigest(),
        "training_approved": False,
        "human_review_required": True,
        "review_flags_written_as_false": list(REVIEW_FLAGS),
        "source_group_policy": (
            "One source group per country QID; all capital statements for the same "
            "country remain in the same group during later splitting."
        ),
        "warning": (
            "Candidate-only export. CC0 status does not establish factual accuracy, "
            "privacy suitability, quality or DREAM training approval. A human must "
            "review every record before it can enter an approved corpus."
        ),
    }
    report_bytes = (
        json.dumps(report_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")

    atomic_write(output, jsonl_bytes)
    atomic_write(report, report_bytes)
    return report_payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=200, help="Maximum source rows to scan (1-1000)")
    parser.add_argument("--page-size", type=int, default=50, help="Rows per sequential request (1-100)")
    parser.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds (5-55)")
    parser.add_argument("--output", required=True, help="Candidate JSONL output path; never an approved corpus")
    parser.add_argument("--report", help="Optional text-free report path (defaults beside output)")
    args = parser.parse_args(argv)
    try:
        report = export_candidates(
            output_path=args.output,
            report_path=args.report,
            max_rows=args.max_rows,
            page_size=args.page_size,
            timeout=args.timeout,
        )
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
