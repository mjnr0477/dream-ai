"""Serve the native DREAM checkpoint over a local OpenAI-compatible HTTP API.

The protocol is only an HTTP message format. All generated tokens come from
DREAM's own checkpoint loaded by this process.
"""
import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer


CHECKPOINT = os.environ.get("DREAM_CHECKPOINT", "checkpoints/dream-small.pt")
HOST = os.environ.get("DREAM_HOST", "127.0.0.1")
PORT = int(os.environ.get("DREAM_PORT", "8090"))
TOKEN = os.environ.get("DREAM_SERVER_TOKEN", "")
if HOST not in ("127.0.0.1", "localhost", "::1") and not TOKEN:
    raise RuntimeError("DREAM_SERVER_TOKEN is required when binding beyond localhost")
MAX_BODY = 64 * 1024
MAX_OUTPUT_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    if checkpoint.get("tokenizer") != "utf8-byte-v1":
        raise ValueError("Unsupported DREAM tokenizer in checkpoint")
    config = DreamConfig(**checkpoint["config"])
    model = DreamTransformer(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, DreamByteTokenizer(), config


MODEL, TOKENIZER, CONFIG = load_model()


class Handler(BaseHTTPRequestHandler):
    server_version = "DREAM-Model/0.1"

    def _json(self, status, value):
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {
                "ok": True,
                "engine": "native-dream-transformer",
                "checkpoint": os.path.basename(CHECKPOINT),
                "device": DEVICE,
                "parameters": sum(p.numel() for p in MODEL.parameters()),
                "tokenizer": "utf8-byte-v1"
            })
        return self._json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not_found"})
        if TOKEN and not secrets.compare_digest(
            self.headers.get("Authorization", ""),
            "Bearer " + TOKEN
        ):
            return self._json(401, {"error": "unauthorized"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY:
                return self._json(413, {"error": "invalid_body_size"})
            body = json.loads(self.rfile.read(length))
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._json(400, {"error": "messages_required"})
            parts = []
            for item in messages[-8:]:
                if not isinstance(item, dict):
                    return self._json(400, {"error": "invalid_message"})
                role = item.get("role")
                content = item.get("content")
                if role not in ("system", "user", "assistant") or not isinstance(content, str):
                    return self._json(400, {"error": "invalid_message"})
                if len(content) > 8000:
                    return self._json(400, {"error": "message_too_long"})
                parts.append(role.upper() + ": " + content.strip())
            prompt = "\n".join(parts) + "\nASSISTANT:"
            prompt_ids = TOKENIZER.encode(prompt)
            if not prompt_ids:
                return self._json(400, {"error": "empty_prompt"})
            max_tokens = body.get("max_tokens", 128)
            if not isinstance(max_tokens, int) or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
                return self._json(400, {"error": "max_tokens_out_of_range"})

            ids = torch.tensor(
                [prompt_ids[-CONFIG.block_size:]], dtype=torch.long, device=DEVICE
            )
            with torch.inference_mode():
                generated = MODEL.generate(
                    ids, max_new_tokens=max_tokens,
                    temperature=max(0.1, min(float(body.get("temperature", 0.8)), 1.5))
                )[0].tolist()
            answer = TOKENIZER.decode(generated[len(ids[0]):]).strip()
            if not answer:
                answer = TOKENIZER.decode(generated).strip()
            return self._json(200, {
                "id": "dream-native-completion",
                "object": "chat.completion",
                "model": "dream-native",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "length"
                }]
            })
        except (ValueError, TypeError, json.JSONDecodeError):
            return self._json(400, {"error": "invalid_request"})
        except Exception:
            return self._json(500, {"error": "dream_inference_failed"})

    def log_message(self, fmt, *args):
        print("DREAM model server:", fmt % args)


if __name__ == "__main__":
    print("Starting native DREAM model on", HOST, PORT, "device", DEVICE)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
