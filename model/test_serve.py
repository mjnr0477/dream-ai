"""Local HTTP integration tests for the native DREAM model server."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import torch

from dream_transformer import DreamConfig, DreamTransformer

SERVER = Path(__file__).with_name("serve.py")


class DreamServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.checkpoint = cls.root / "dream-test.pt"
        config = DreamConfig(
            block_size=64, d_model=16, n_heads=4, n_layers=1, dropout=0.0
        )
        model = DreamTransformer(config)
        torch.save({
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "tokenizer": "utf8-byte-v1",
            "training": {"initialization": "random-smoke-test"}
        }, cls.checkpoint)

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        env = os.environ.copy()
        env.update({
            "DREAM_CHECKPOINT": str(cls.checkpoint),
            "DREAM_HOST": "127.0.0.1",
            "DREAM_PORT": str(cls.port),
            "DREAM_SERVER_TOKEN": "test-secret"
        })
        cls.process = subprocess.Popen(
            [sys.executable, str(SERVER)],
            cwd=str(SERVER.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError("DREAM model server exited during startup")
            try:
                cls.request("GET", "/health")
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("DREAM model server did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()
            cls.process.wait(timeout=5)
        cls.temp.cleanup()

    @classmethod
    def request(cls, method, path, payload=None, token=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(
            cls.base + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health_identifies_native_checkpoint(self):
        status, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["engine"], "native-dream-transformer")
        self.assertEqual(body["tokenizer"], "utf8-byte-v1")

    def test_requires_bearer_token_for_chat(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1}
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_returns_native_model_completion(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hello DREAM"}], "max_tokens": 1},
            token="test-secret"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "dream-native")
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")

    def test_rejects_invalid_temperature(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                "temperature": float("nan")
            },
            token="test-secret"
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "temperature_out_of_range")


if __name__ == "__main__":
    unittest.main()
