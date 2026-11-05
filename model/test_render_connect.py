import json
import unittest
from urllib.error import HTTPError

from render_connect import update_render_model_env


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class RenderConnectTests(unittest.TestCase):
    def setUp(self):
        self.api_key = "render-secret-api-key"
        self.model_token = "dream-runtime-token-0123456789"
        self.values = {
            "service_id": "srv-dbkoa2ui0phs73arqvk0",
            "api_key": self.api_key,
            "model_url": "https://dream-runtime.trycloudflare.com/v1",
            "model_name": "dream-native-v1",
            "model_token": self.model_token,
            "confirm": True,
        }

    def test_updates_only_dream_variables_then_triggers_one_deploy(self):
        requests = []

        def fake_open(request, timeout):
            requests.append((request, timeout))
            if request.full_url.endswith("/deploys"):
                return FakeResponse({"id": "dep-test123", "status": "queued"})
            return FakeResponse({"key": request.full_url.rsplit("/", 1)[-1]})

        result = update_render_model_env(**self.values, opener=fake_open)

        self.assertEqual(result["deploy_id"], "dep-test123")
        self.assertEqual(result["status"], "queued")
        self.assertEqual(
            result["updated_keys"],
            ["DREAM_MODEL_URL", "DREAM_MODEL_NAME", "DREAM_MODEL_TOKEN"],
        )
        self.assertEqual(len(requests), 4)
        expected_keys = ["DREAM_MODEL_URL", "DREAM_MODEL_NAME", "DREAM_MODEL_TOKEN"]
        for (request, timeout), key in zip(requests[:3], expected_keys):
            self.assertEqual(request.get_method(), "PUT")
            self.assertTrue(request.full_url.endswith(f"/env-vars/{key}"))
            self.assertEqual(json.loads(request.data.decode("utf-8")), {"value": self.values[
                {"DREAM_MODEL_URL": "model_url", "DREAM_MODEL_NAME": "model_name", "DREAM_MODEL_TOKEN": "model_token"}[key]
            ]})
            self.assertEqual(request.get_header("Authorization"), f"Bearer {self.api_key}")
            self.assertEqual(timeout, 20)

        deploy_request, _ = requests[-1]
        self.assertEqual(deploy_request.get_method(), "POST")
        self.assertTrue(deploy_request.full_url.endswith("/deploys"))
        self.assertEqual(
            json.loads(deploy_request.data.decode("utf-8")),
            {"clearCache": "do_not_clear"},
        )
        self.assertNotIn(self.api_key, repr(result))
        self.assertNotIn(self.model_token, repr(result))

    def test_requires_explicit_confirmation_before_any_network_call(self):
        calls = []
        values = {**self.values, "confirm": False}
        with self.assertRaises(PermissionError):
            update_render_model_env(**values, opener=lambda *_args, **_kwargs: calls.append(True))
        self.assertEqual(calls, [])

    def test_rejects_non_https_or_non_v1_runtime_url(self):
        for url in (
            "http://dream-runtime.trycloudflare.com/v1",
            "https://user:pass@dream-runtime.trycloudflare.com/v1",
            "https://dream-runtime.trycloudflare.com/",
            "https://dream-runtime.trycloudflare.com/v1?token=leak",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    update_render_model_env(
                        **{**self.values, "model_url": url},
                        opener=lambda *_args, **_kwargs: self.fail("network must not be called"),
                    )

    def test_rejects_invalid_service_or_missing_credentials(self):
        for override in (
            {"service_id": "not-a-render-service"},
            {"api_key": ""},
            {"model_token": "short"},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    update_render_model_env(
                        **{**self.values, **override},
                        opener=lambda *_args, **_kwargs: self.fail("network must not be called"),
                    )

    def test_partial_environment_update_is_reported_and_never_deploys(self):
        requests = []

        def fail_second(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/env-vars/DREAM_MODEL_NAME"):
                raise HTTPError(request.full_url, 403, "Forbidden", None, None)
            return FakeResponse({})

        with self.assertRaises(RuntimeError) as raised:
            update_render_model_env(**self.values, opener=fail_second)

        self.assertIn("DREAM_MODEL_URL", str(raised.exception))
        self.assertIn("No deploy was requested", str(raised.exception))
        self.assertNotIn(self.api_key, str(raised.exception))
        self.assertNotIn(self.model_token, str(raised.exception))
        self.assertEqual(len(requests), 2)
        self.assertFalse(any(request.full_url.endswith("/deploys") for request in requests))

    def test_deploy_failure_reports_that_all_variables_were_saved(self):
        requests = []

        def fail_deploy(request, timeout):
            requests.append(request)
            if request.full_url.endswith("/deploys"):
                raise HTTPError(request.full_url, 503, "Unavailable", None, None)
            return FakeResponse({})

        with self.assertRaisesRegex(RuntimeError, "All three DREAM variables were saved"):
            update_render_model_env(**self.values, opener=fail_deploy)
        self.assertEqual(len(requests), 4)
        self.assertTrue(requests[-1].full_url.endswith("/deploys"))

    def test_render_http_errors_do_not_echo_credentials(self):
        def denied(request, timeout):
            raise HTTPError(request.full_url, 403, "Forbidden", None, None)

        with self.assertRaisesRegex(RuntimeError, "No deploy was requested") as raised:
            update_render_model_env(**self.values, opener=denied)
        self.assertNotIn(self.api_key, str(raised.exception))
        self.assertNotIn(self.model_token, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
