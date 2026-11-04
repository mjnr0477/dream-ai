"""Serve DREAM's own trained checkpoint through an authenticated HTTP API.

The OpenAI-compatible response shape is only a transport format. All inference
runs locally in this process using the native DREAM transformer weights.
"""
import argparse
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch

from checkpoint_tokenizer import load_tokenizer
from dream_transformer import DreamConfig, DreamTransformer


MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGES = 12
MAX_MESSAGE_CHARS = 8000
MAX_OUTPUT_TOKENS = 256
MODEL_LOCK = threading.Lock()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("sft_training") or checkpoint.get("training") or {}
    completed_steps = metadata.get("completed_steps", 0)
    if (not isinstance(completed_steps, int) or
            isinstance(completed_steps, bool) or completed_steps < 1):
        raise ValueError("Refusing to serve a checkpoint without completed DREAM training.")
    config = DreamConfig(**checkpoint["config"])
    tokenizer = load_tokenizer(checkpoint, config)
    model = DreamTransformer(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, tokenizer, config, metadata


def format_prompt(messages):
    return "".join(
        f"{item['role'].upper()}: {item['content'].strip()}\n"
        for item in messages
    ) + "ASSISTANT:"


def create_handler(model, tokenizer, config, model_name, token, device):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DREAMNativeRuntime/0.2"

        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            # Never log request bodies or user prompts.
            print(f"DREAM runtime: {self.address_string()} {fmt % args}")

        def do_GET(self):
            if urlparse(self.path).path != "/health":
                return self.send_json(404, {"error": {"message": "Not found"}})
            return self.send_json(200, {
                "ok": True,
                "engine": "dream-native-transformer",
                "model": model_name,
                "device": device,
                "parameters": sum(parameter.numel() for parameter in model.parameters())
            })

        def do_POST(self):
            if urlparse(self.path).path not in ("/v1/chat/completions", "/chat/completions"):
                return self.send_json(404, {"error": {"message": "Not found"}})
            if token:
                supplied = self.headers.get("Authorization", "")
                if not hmac.compare_digest(supplied, f"Bearer {token}"):
                    return self.send_json(401, {"error": {"message": "Unauthorized"}})
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                return self.send_json(415, {"error": {"message": "application/json required"}})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.send_json(400, {"error": {"message": "Invalid content length"}})
            if length < 1 or length > MAX_BODY_BYTES:
                return self.send_json(413, {"error": {"message": "Invalid request size"}})
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self.send_json(400, {"error": {"message": "Body must be valid UTF-8 JSON"}})
            if not isinstance(payload, dict):
                return self.send_json(400, {"error": {"message": "Body must be an object"}})
            if payload.get("model") not in (None, model_name):
                return self.send_json(400, {"error": {"message": "Unknown DREAM model name"}})

            messages = payload.get("messages")
            if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_MESSAGES:
                return self.send_json(400, {"error": {"message": "messages must contain 1 to 12 items"}})
            for item in messages:
                if (
                    not isinstance(item, dict)
                    or item.get("role") not in ("system", "user", "assistant")
                    or not isinstance(item.get("content"), str)
                    or not item["content"].strip()
                    or len(item["content"]) > MAX_MESSAGE_CHARS
                ):
                    return self.send_json(400, {"error": {"message": "Invalid message role or content"}})
            if messages[-1]["role"] != "user":
                return self.send_json(400, {"error": {"message": "Last message must be from user"}})

            try:
                max_tokens = payload.get("max_tokens", 160)
                temperature = payload.get("temperature", 0.7)
                top_k = payload.get("top_k", 40)
                if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
                    raise ValueError
                if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
                    raise ValueError
                if isinstance(top_k, bool) or not isinstance(top_k, int):
                    raise ValueError
                if not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
                    raise ValueError
                if not 0 <= temperature <= 2.0:
                    raise ValueError
                if not 0 <= top_k <= config.vocab_size:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                return self.send_json(400, {"error": {"message": "Invalid generation settings"}})

            prompt_ids = tokenizer.encode(format_prompt(messages))
            if not prompt_ids:
                return self.send_json(400, {"error": {"message": "Prompt is empty"}})
            prompt_ids = prompt_ids[-config.block_size:]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            try:
                with MODEL_LOCK, torch.inference_mode():
                    if float(temperature) == 0:
                        generated = model.generate(
                            input_ids, max_new_tokens=max_tokens, temperature=0,
                            constrain_utf8=tokenizer.vocab_size == 256
                        )[0].tolist()
                    else:
                        torch.manual_seed(secrets.randbits(63))
                        generated = model.generate(
                            input_ids,
                            max_new_tokens=max_tokens,
                            temperature=float(temperature),
                            top_k=top_k,
                            constrain_utf8=tokenizer.vocab_size == 256
                        )[0].tolist()
                answer = tokenizer.decode(generated[len(prompt_ids):]).strip()
            except Exception:
                return self.send_json(500, {"error": {"message": "DREAM inference failed"}})
            if not answer:
                return self.send_json(502, {"error": {"message": "DREAM generated an empty answer"}})

            answer_tokens = len(tokenizer.encode(answer))
            return self.send_json(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "length"
                }],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": answer_tokens,
                    "total_tokens": len(prompt_ids) + answer_tokens
                }
            })

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get(
        "DREAM_CHECKPOINT", "checkpoints/dream-instruct.pt"
    ))
    parser.add_argument("--host", default=os.environ.get("DREAM_RUNTIME_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DREAM_RUNTIME_PORT", "8081")))
    parser.add_argument("--model-name", default=os.environ.get("DREAM_MODEL_NAME", "dream-native"))
    parser.add_argument("--token", default=os.environ.get("DREAM_RUNTIME_TOKEN", ""))
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        raise SystemExit("Port must be between 1 and 65535.")
    if not args.model_name.strip():
        raise SystemExit("Model name must not be empty.")
    if args.host not in ("127.0.0.1", "localhost", "::1") and len(args.token) < 24:
        raise SystemExit("A runtime token of at least 24 characters is required for non-loopback binding.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, tokenizer, config, metadata = load_model(args.checkpoint, device)
    except Exception as exc:
        raise SystemExit(f"Cannot load DREAM checkpoint: {exc}") from exc

    handler = create_handler(model, tokenizer, config, args.model_name, args.token, device)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(json.dumps({
        "engine": "DREAM native inference",
        "model": args.model_name,
        "checkpoint": args.checkpoint,
        "completed_training_steps": metadata.get("completed_steps"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": device,
        "host": args.host,
        "port": args.port,
        "authenticated": bool(args.token)
    }))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
