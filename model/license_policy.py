"""Conservative license screening for DREAM's potentially commercial model corpus.

This is machine-readable triage, not legal advice. Humans must still verify the
source, rights holder, attribution, privacy and the applicable terms.
"""
from __future__ import annotations

import re
from datetime import datetime
from urllib.parse import urlsplit


POLICY_VERSION = "dream-commercial-training-license-policy-v1"
COMMERCIAL_USE_INTENDED = True
ALLOWED_STANDARD_LICENSES = {
    "cc0-1.0", "cc-by-4.0", "apache-2.0", "mit",
}
RESTRICTED_LICENSE_CODE = "noncommercial_restriction"


def _normalized(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _is_https_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.username and not parsed.password
    except ValueError:
        return False


def _timestamp_present(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _cc_path(license_url: object) -> str:
    if not isinstance(license_url, str):
        return ""
    try:
        parsed = urlsplit(license_url)
    except ValueError:
        return ""
    if (parsed.hostname or "").lower() not in {"creativecommons.org", "www.creativecommons.org"}:
        return ""
    return parsed.path.lower().strip("/")


def _has_separate_commercial_permission(record: dict) -> bool:
    if record.get("rights_basis") != "explicit-permission":
        return False
    scope = record.get("permission_scope")
    if not isinstance(scope, str):
        return False
    normalized_scope = " ".join(scope.lower().split())
    covers_training = "training" in normalized_scope and (
        "model" in normalized_scope or "ai" in normalized_scope
    )
    covers_commercial = any(
        term in normalized_scope
        for term in ("commercial", "revenue", "monetized", "monetised", "paid service")
    )
    return (
        covers_training
        and covers_commercial
        and _is_https_url(record.get("permission_evidence_url"))
        and record.get("rights_confirmed") is True
    )


def _is_dream_v2_contribution(record: dict) -> bool:
    return (
        record.get("license") == "DREAM AI Contributor Permission v2"
        and record.get("license_url") == "https://dream-ai-live.onrender.com/learning-data-terms.html"
        and record.get("rights_basis") == "explicit-permission"
        and record.get("consent_version") == "dream-learning-opt-in-v2"
        and record.get("consented") is True
        and _timestamp_present(record.get("consented_at"))
        and _timestamp_present(record.get("human_reviewed_at"))
        and record.get("rights_confirmed") is True
    )


def classify_record_license(record: dict) -> dict:
    """Classify declared terms for a commercial-ready AI corpus."""
    if not isinstance(record, dict):
        return {"status": "review_required", "code": "invalid_record", "attribution_required": False}

    if _is_dream_v2_contribution(record):
        return {
            "status": "allowed",
            "code": "dream_explicit_contributor_permission_v2",
            "attribution_required": True,
        }

    if _has_separate_commercial_permission(record):
        return {
            "status": "allowed",
            "code": "separate_commercial_training_permission",
            "attribution_required": True,
        }

    name = _normalized(record.get("license"))
    path = _cc_path(record.get("license_url"))
    license_url = record.get("license_url", "")
    url_text = license_url.lower() if isinstance(license_url, str) else ""
    combined = name + " " + path.replace("/", "-") + " " + url_text

    has_noncommercial = (
        re.search(r"(?:^|[-/])nc(?:[-/]|$)", combined) is not None
        or "noncommercial" in combined
        or "non-commercial" in combined
    )
    if has_noncommercial:
        return {
            "status": "blocked",
            "code": RESTRICTED_LICENSE_CODE,
            "attribution_required": True,
        }

    if re.search(r"(?:^|[-/])nd(?:[-/]|$)", combined) or "no-derivatives" in combined:
        return {
            "status": "review_required",
            "code": "no_derivatives_terms_require_separate_permission",
            "attribution_required": True,
        }

    if re.search(r"(?:^|[-/])sa(?:[-/]|$)", combined) or "sharealike" in combined or "share-alike" in combined:
        return {
            "status": "review_required",
            "code": "share_alike_obligations_require_review",
            "attribution_required": True,
        }

    if "odc-by" in name or "open-data-commons-attribution" in combined:
        return {
            "status": "review_required",
            "code": "dataset_collection_license_does_not_clear_item_rights",
            "attribution_required": True,
        }

    if name == "cc0-1-0" and path == "publicdomain/zero/1-0":
        return {"status": "allowed", "code": "cc0_1_0", "attribution_required": False}

    if name == "cc-by-4-0" and path == "licenses/by/4-0":
        return {"status": "allowed", "code": "cc_by_4_0", "attribution_required": True}

    if name == "apache-2-0" and (
        "apache.org/licenses/license-2.0" in url_text
        or "spdx.org/licenses/apache-2.0" in url_text
    ):
        return {"status": "allowed", "code": "apache_2_0", "attribution_required": True}

    if name == "mit" and (
        "opensource.org/license/mit" in url_text
        or "spdx.org/licenses/mit.html" in url_text
        or "mit-license.org" in url_text
    ):
        return {"status": "allowed", "code": "mit", "attribution_required": True}

    if name in {"public-domain", "public-domain-mark-1-0"} and (
        record.get("rights_basis") == "public-domain"
        and "creativecommons.org/publicdomain/mark/1.0" in url_text
    ):
        return {"status": "allowed", "code": "human_verified_public_domain", "attribution_required": False}

    return {
        "status": "review_required",
        "code": "unknown_or_unverified_license_terms",
        "attribution_required": True,
    }


def require_commercial_training_license(record: dict, line_number: int | None = None) -> dict:
    """Raise ValueError unless the declared license passes this policy."""
    result = classify_record_license(record)
    if result["status"] != "allowed":
        prefix = f"record {line_number}: " if line_number is not None else ""
        raise ValueError(prefix + f"license policy {result['status']}: {result['code']}")
    return result
