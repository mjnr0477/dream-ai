"""Safely connect a verified Colab DREAM runtime to the existing Render service."""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

_MODEL_ENV_KEYS = ("DREAM_MODEL_URL", "DREAM_MODEL_NAME", "DREAM_MODEL_TOKEN")
_SERVICE_ID_PATTERN = re.compile(r"^srv-[a-z0-9]+$")
_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _request_json(url, method, api_key, payload, opener):
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method=method,
    )
    try:
        with opener(request, timeout=20) as response:
            raw = response.read()
    except HTTPError as error:
        raise RuntimeError(f"Render API request failed with HTTP {error.code}.") from None
    except URLError:
        raise RuntimeError("Render API could not be reached. Check the connection and retry.") from None

    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Render API returned an invalid response.") from None


def update_render_model_env(
    *,
    service_id,
    api_key,
    model_url,
    model_name,
    model_token,
    confirm=False,
    opener=None,
):
    """Update only DREAM's three model variables, then request one deploy.

    The caller must obtain explicit confirmation before calling this function.
    Individual env-var endpoints are used so unrelated service variables survive.
    """
    if confirm is not True:
        raise PermissionError("Explicit confirmation is required before changing Render.")
    if not isinstance(service_id, str) or not _SERVICE_ID_PATTERN.fullmatch(service_id):
        raise ValueError("A valid Render web-service ID is required.")
    if not isinstance(api_key, str) or len(api_key.strip()) < 10 or any(c.isspace() for c in api_key):
        raise ValueError("A valid Render API key is required.")
    if not isinstance(model_url, str):
        raise ValueError("DREAM_MODEL_URL must be an HTTPS URL.")
    parsed = urlparse(model_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ValueError("DREAM_MODEL_URL must be a credential-free HTTPS /v1 URL.")
    if not isinstance(model_name, str) or not _MODEL_NAME_PATTERN.fullmatch(model_name):
        raise ValueError("DREAM_MODEL_NAME contains unsupported characters.")
    if not isinstance(model_token, str) or not 16 <= len(model_token) <= 512 or any(c.isspace() for c in model_token):
        raise ValueError("DREAM_MODEL_TOKEN must be a non-empty generated secret.")
    send = opener or urlopen
    base = f"https://api.render.com/v1/services/{service_id}"
    values = (
        ("DREAM_MODEL_URL", model_url),
        ("DREAM_MODEL_NAME", model_name),
        ("DREAM_MODEL_TOKEN", model_token),
    )
    updated_keys = []
    for key, value in values:
        try:
            _request_json(
                f"{base}/env-vars/{key}",
                "PUT",
                api_key,
                {"value": value},
                send,
            )
        except Exception:
            possibly_changed = ", ".join([*updated_keys, key])
            raise RuntimeError(
                f"Render update failed while setting {key}; these variables may now be saved: "
                f"{possibly_changed}. No deploy was requested; review all three DREAM variables "
                "before any future deploy."
            ) from None
        updated_keys.append(key)

    try:
        deploy = _request_json(
            f"{base}/deploys",
            "POST",
            api_key,
            {"clearCache": "do_not_clear"},
            send,
        )
    except Exception:
        raise RuntimeError(
            "All three DREAM variables were saved, but Render did not confirm a deploy. "
            "Check the service deploy status before retrying."
        ) from None
    deploy_id = deploy.get("id") if isinstance(deploy, dict) else None
    if not isinstance(deploy_id, str) or not deploy_id:
        raise RuntimeError("Render accepted the environment updates but did not return a deploy ID.")

    return {
        "deploy_id": deploy_id,
        "status": deploy.get("status", "created"),
        "updated_keys": list(_MODEL_ENV_KEYS),
    }
