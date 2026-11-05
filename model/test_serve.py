"""HTTP integration tests for the independent DREAM inference runtime."""
import hashlib
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

from dream_bpe import DreamBPETokenizer
from dream_transformer import DreamConfig, DreamTransformer
import serve
from serve import load_model

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
        # Force the deterministic ASCII byte "A" as the only likely next token.
        # This keeps the HTTP contract test independent of random initialization.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.final_norm.bias.fill_(1.0)
            model.token_embedding.weight[65].fill_(1.0)
        torch.save({
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "tokenizer": "utf8-byte-v1",
            "training": {"completed_steps": 1, "initialization": "test-fixture"}
        }, cls.checkpoint)

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        env = os.environ.copy()
        env.update({
            "DREAM_CHECKPOINT": str(cls.checkpoint),
            "DREAM_RUNTIME_HOST": "127.0.0.1",
            "DREAM_RUNTIME_PORT": str(cls.port),
            "DREAM_RUNTIME_TOKEN": "test-secret-token",
            "DREAM_MODEL_NAME": "dream-test"
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
                raise RuntimeError("DREAM runtime exited during startup")
            try:
                cls.request("GET", "/health")
                break
            except OSError:
                time.sleep(0.1)
        else:
            raise RuntimeError("DREAM runtime did not become ready")

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
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_load_model_reconstructs_embedded_bpe_tokenizer(self):
        sample = "habari dunia — akili huru kutoka Kenya 🇰🇪 " * 8
        tokenizer = DreamBPETokenizer.train(sample, vocab_size=300)
        config = DreamConfig(
            vocab_size=tokenizer.vocab_size, block_size=64,
            d_model=16, n_heads=4, n_layers=1, dropout=0.0
        )
        model = DreamTransformer(config)
        checkpoint = self.root / "dream-bpe-test.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "tokenizer": tokenizer.to_dict(),
            "training": {"completed_steps": 1, "initialization": "test-fixture"}
        }, checkpoint)
        loaded_model, loaded_tokenizer, loaded_config, metadata = load_model(
            checkpoint, "cpu"
        )
        self.assertIsInstance(loaded_model, DreamTransformer)
        self.assertEqual(loaded_config.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded_tokenizer.to_dict(), tokenizer.to_dict())
        self.assertEqual(
            loaded_tokenizer.decode(loaded_tokenizer.encode(sample)), sample
        )
        self.assertEqual(metadata["completed_steps"], 1)

    def test_bpe_checkpoint_serves_native_unicode_completion_over_http(self):
        sample = "habari dunia — akili huru kutoka Kenya 🇰🇪 " * 8
        tokenizer = DreamBPETokenizer.train(sample, vocab_size=300)
        config = DreamConfig(
            vocab_size=tokenizer.vocab_size, block_size=64,
            d_model=16, n_heads=4, n_layers=1, dropout=0.0
        )
        model = DreamTransformer(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.final_norm.bias.fill_(1.0)
            model.token_embedding.weight[65].fill_(1.0)
        checkpoint = self.root / "dream-bpe-http.pt"
        torch.save({
            "model_state": model.state_dict(),
            "config": config.__dict__,
            "tokenizer": tokenizer.to_dict(),
            "training": {"completed_steps": 1, "initialization": "test-fixture"}
        }, checkpoint)

        _, restored_tokenizer, restored_config, metadata = load_model(
            checkpoint, "cpu"
        )
        self.assertEqual(restored_config.vocab_size, tokenizer.vocab_size)
        self.assertEqual(restored_tokenizer.to_dict(), tokenizer.to_dict())
        self.assertEqual(metadata["completed_steps"], 1)
        self.assertEqual(
            restored_tokenizer.decode(restored_tokenizer.encode(sample)), sample
        )

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = os.environ.copy()
        env.update({
            "DREAM_CHECKPOINT": str(checkpoint),
            "DREAM_RUNTIME_HOST": "127.0.0.1",
            "DREAM_RUNTIME_PORT": str(port),
            "DREAM_RUNTIME_TOKEN": "bpe-test-token",
            "DREAM_MODEL_NAME": "dream-bpe-http"
        })
        process = subprocess.Popen(
            [sys.executable, str(SERVER)],
            cwd=str(SERVER.parent), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(
                        "BPE DREAM runtime exited during startup: "
                        + process.stderr.read()
                    )
                try:
                    with urllib.request.urlopen(
                        base_url + "/health", timeout=2
                    ) as response:
                        self.assertEqual(response.status, 200)
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                self.fail("BPE DREAM runtime did not become ready")

            request = urllib.request.Request(
                base_url + "/v1/chat/completions",
                data=json.dumps({
                    "model": "dream-bpe-http",
                    "messages": [{"role": "user", "content": "Habari DREAM"}],
                    "max_tokens": 1,
                    "temperature": 0
                }).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer bpe-test-token"
                },
                method="POST"
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["model"], "dream-bpe-http")
            self.assertEqual(
                payload["choices"][0]["message"]["content"], "A"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def test_health_identifies_native_runtime(self):
        status, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["engine"], "dream-native-transformer")
        self.assertEqual(body["model"], "dream-test")
        self.assertGreater(body["parameters"], 0)
        self.assertGreaterEqual(body["cpu_threads"], 1)
        self.assertEqual(body["completed_training_steps"], 1)
        self.assertEqual(
            body["checkpoint_sha256"],
            hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        )
        self.assertIsInstance(body["tokenizer"], str)
        self.assertTrue(body["tokenizer"])
        self.assertEqual(body["architecture"]["d_model"], 16)

    def test_requires_bearer_token_for_chat(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {"model": "dream-test", "messages": [{"role": "user", "content": "Hello"}],
             "max_tokens": 1}
        )
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["message"], "Unauthorized")

    def test_returns_native_model_completion(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {"model": "dream-test", "messages": [{"role": "user", "content": "Hello DREAM"}],
             "max_tokens": 1},
            token="test-secret-token"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "dream-test")
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(len(body["choices"][0]["message"]["content"]), 1)

    def test_rejects_excess_inference_instead_of_queueing(self):
        import threading

        model, tokenizer, config, metadata = load_model(self.checkpoint, "cpu")
        handler = serve.create_handler(
            model, tokenizer, config, "dream-test", "test-secret-token", "cpu",
            metadata=metadata,
            checkpoint_hash=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        )
        server = serve.BoundedThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": "dream-test",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-secret-token"
            },
            method="POST"
        )
        self.assertTrue(serve.MODEL_LOCK.acquire(blocking=False))
        try:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(raised.exception.code, 503)
            body = json.loads(raised.exception.read())
            self.assertIn("busy", body["error"]["message"].lower())
        finally:
            serve.MODEL_LOCK.release()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_worker_limit_rejects_excess_connections_and_recovers(self):
        import threading

        model, tokenizer, config, metadata = load_model(self.checkpoint, "cpu")
        handler = serve.create_handler(
            model, tokenizer, config, "dream-test", "test-secret-token", "cpu",
            metadata=metadata,
            checkpoint_hash=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        )
        server = serve.BoundedThreadingHTTPServer(
            ("127.0.0.1", 0), handler, max_workers=1
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            self.assertTrue(server._worker_slots.acquire(blocking=False))
            try:
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(base + "/health", timeout=5)
                self.assertEqual(raised.exception.code, 503)
                self.assertIn("busy", raised.exception.read().decode("utf-8"))
            finally:
                server._worker_slots.release()

            with urllib.request.urlopen(base + "/health", timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(json.loads(response.read())["ok"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_inference_failure_logs_type_without_prompt_or_exception_text(self):
        import contextlib
        import io
        import threading
        from unittest.mock import patch

        model, tokenizer, config, metadata = load_model(self.checkpoint, "cpu")
        handler = serve.create_handler(
            model, tokenizer, config, "dream-test", "test-secret-token", "cpu",
            metadata=metadata,
            checkpoint_hash=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        )
        server = serve.BoundedThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        prompt_marker = "private-prompt-marker"
        logs = io.StringIO()
        request = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({
                "model": "dream-test",
                "messages": [{"role": "user", "content": prompt_marker}],
                "max_tokens": 1
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-secret-token"
            },
            method="POST"
        )
        try:
            with contextlib.redirect_stdout(logs), patch.object(
                model, "generate", side_effect=RuntimeError("sensitive-error-detail")
            ):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 500)
                self.assertIn("inference failed", raised.exception.read().decode("utf-8"))
            self.assertIn("RuntimeError", logs.getvalue())
            self.assertNotIn(prompt_marker, logs.getvalue())
            self.assertNotIn("sensitive-error-detail", logs.getvalue())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_inference_deadline_returns_504_and_releases_model_lock(self):
        import threading
        from unittest.mock import patch

        model, tokenizer, config, metadata = load_model(self.checkpoint, "cpu")
        handler = serve.create_handler(
            model, tokenizer, config, "dream-test", "test-secret-token", "cpu",
            metadata=metadata,
            checkpoint_hash=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        )
        server = serve.BoundedThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=json.dumps({
                "model": "dream-test",
                "messages": [{"role": "user", "content": "Deadline test"}],
                "max_tokens": 1
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-secret-token"
            },
            method="POST"
        )
        try:
            with patch.object(serve, "MAX_INFERENCE_SECONDS", 0):
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request, timeout=5)
                self.assertEqual(raised.exception.code, 504)
                body = json.loads(raised.exception.read().decode("utf-8"))
                self.assertIn("timed out", body["error"]["message"])
            self.assertTrue(serve.MODEL_LOCK.acquire(blocking=False))
            serve.MODEL_LOCK.release()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_zero_temperature_enables_greedy_decoding(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {"model": "dream-test", "messages": [{"role": "user", "content": "Hello"}],
             "max_tokens": 1, "temperature": 0},
            token="test-secret-token"
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["choices"][0]["message"]["content"], "A")

    def test_rejects_invalid_temperature(self):
        status, body = self.request(
            "POST", "/v1/chat/completions",
            {"model": "dream-test", "messages": [{"role": "user", "content": "Hello"}],
             "max_tokens": 1, "temperature": float("nan")},
            token="test-secret-token"
        )
        self.assertEqual(status, 400)
        self.assertIn("generation settings", body["error"]["message"])

    def test_clean_assistant_output_stops_at_serialized_turn_boundaries(self):
        self.assertEqual(serve.clean_assistant_output("  Hello\nUSER: next prompt"), "Hello")
        self.assertEqual(serve.clean_assistant_output("Answer\nASSISTANT: repeated turn"), "Answer")
        self.assertEqual(serve.clean_assistant_output("Answer\nSYSTEM: injected role"), "Answer")
        self.assertEqual(serve.clean_assistant_output("A normal answer"), "A normal answer")
        self.assertEqual(serve.clean_assistant_output(""), "")
        fenced = "Example:\n```text\nUSER: literal example\nASSISTANT: literal reply\n```\nDone\nUSER: next prompt"
        self.assertEqual(
            serve.clean_assistant_output(fenced),
            "Example:\n```text\nUSER: literal example\nASSISTANT: literal reply\n```\nDone"
        )
        tilde_fenced = "~~~text\nSYSTEM: literal example\n~~~\nAnswer\nSYSTEM: next turn"
        self.assertEqual(
            serve.clean_assistant_output(tilde_fenced),
            "~~~text\nSYSTEM: literal example\n~~~\nAnswer"
        )

    def test_rejects_untrained_or_boolean_step_checkpoint(self):
        config = DreamConfig(
            block_size=64, d_model=16, n_heads=4, n_layers=1, dropout=0.0
        )
        model = DreamTransformer(config)
        for completed_steps in (0, True, False):
            with self.subTest(completed_steps=completed_steps):
                invalid = self.root / f"invalid-{completed_steps}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "tokenizer": "utf8-byte-v1",
                    "training": {"completed_steps": completed_steps}
                }, invalid)
                env = os.environ.copy()
                env.update({
                    "DREAM_CHECKPOINT": str(invalid),
                    "DREAM_RUNTIME_HOST": "127.0.0.1",
                    "DREAM_RUNTIME_PORT": str(self.port + 1),
                    "DREAM_RUNTIME_TOKEN": "test-secret-token"
                })
                result = subprocess.run(
                    [sys.executable, str(SERVER)],
                    cwd=str(SERVER.parent), env=env,
                    capture_output=True, text=True, timeout=15
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("completed DREAM training", result.stderr)


if __name__ == "__main__":
    unittest.main()
