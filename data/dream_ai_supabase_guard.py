"""Fail-closed boundary checks for the dedicated DREAM AI Supabase project."""
from __future__ import annotations

import re
from urllib.parse import urlsplit

# These project refs belong to other products and must never be used by DREAM AI.
NON_DREAM_PROJECT_REFS = frozenset({
    "ldpcawixzxcolbmakrfc",  # Dream Education
    "lzmrvalonsxtapigtjcx",  # Mariona
})


def validate_dream_ai_supabase_url(value: str) -> str:
    """Return a normalized HTTPS project URL or reject unsafe/cross-app targets."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("DREAM_AI_SUPABASE_URL must identify a dedicated DREAM AI project.")

    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname.lower() if parsed.hostname else ""
        port = parsed.port
    except ValueError as exc:
        raise ValueError("DREAM_AI_SUPABASE_URL is malformed.") from exc

    if (
        parsed.scheme.lower() != "https"
        or not host.endswith(".supabase.co")
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("DREAM_AI_SUPABASE_URL must be a plain HTTPS Supabase project URL.")

    project_ref = host[:-len(".supabase.co")]
    if not project_ref or "." in project_ref or not re.fullmatch(r"[a-z0-9-]+", project_ref):
        raise ValueError("DREAM_AI_SUPABASE_URL must contain exactly one Supabase project reference.")

    if project_ref in NON_DREAM_PROJECT_REFS:
        raise ValueError(
            "Refusing a known non-DREAM Supabase project. Use a dedicated DREAM AI project only."
        )

    return "https://" + host
