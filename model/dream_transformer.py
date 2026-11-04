"""DREAM decoder-only transformer, initialized from random weights.

This is DREAM's own experimental architecture. It does not load weights from
another model. The byte tokenizer is deterministic and has no external vocab.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DreamConfig:
    vocab_size: int = 256
    block_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    def __post_init__(self):
        dimensions = (
            self.vocab_size, self.block_size, self.d_model,
            self.n_heads, self.n_layers
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in dimensions
        ):
            raise ValueError("Model dimensions must be positive integers")
        if self.vocab_size != 256:
            raise ValueError("The current DREAM byte tokenizer requires vocab_size=256")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be a finite number in the range [0, 1)")


class DreamByteTokenizer:
    """UTF-8 byte tokenizer: IDs 0..255 map directly to bytes."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        return list(text.encode("utf-8"))

    def decode(self, token_ids) -> str:
        """Decode byte IDs without silently dropping or coercing invalid values."""
        values = []
        for token in token_ids:
            if not isinstance(token, int) or isinstance(token, bool):
                raise ValueError("Token IDs must be integers")
            if token < 0 or token >= self.vocab_size:
                raise ValueError("Token ID is outside the UTF-8 byte vocabulary")
            values.append(token)
        return bytes(values).decode("utf-8", errors="replace")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DreamConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, length, channels
        )
        return self.proj(attended)

    def forward_cached(self, x, past_key_value=None):
        """Decode one token against cached keys/values without recomputing history."""
        batch, length, channels = x.shape
        if past_key_value is not None and length != 1:
            raise ValueError("Cached decoding accepts exactly one new token")
        q, k, v = self.qkv(x).split(channels, dim=-1)
        shape = (batch, length, self.n_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        # With one appended token every key is in its causal past, so no
        # triangular mask is needed (SDPA's is_causal mask is upper-left aligned).
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0,
            is_causal=past_key_value is None
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch, length, channels
        )
        return self.proj(attended), (k, v)


class FeedForward(nn.Module):
    def __init__(self, config: DreamConfig):
        super().__init__()
        hidden = 4 * config.d_model
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class DreamBlock(nn.Module):
    def __init__(self, config: DreamConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

    def forward_cached(self, x, past_key_value=None):
        attended, present = self.attention.forward_cached(
            self.norm1(x), past_key_value
        )
        x = x + attended
        x = x + self.feed_forward(self.norm2(x))
        return x, present


class DreamTransformer(nn.Module):
    def __init__(self, config: DreamConfig | None = None):
        super().__init__()
        self.config = config or DreamConfig()
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.d_model
        )
        self.position_embedding = nn.Embedding(
            self.config.block_size, self.config.d_model
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            DreamBlock(self.config) for _ in range(self.config.n_layers)
        )
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.language_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.language_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        batch, length = token_ids.shape
        if length < 1 or length > self.config.block_size:
            raise ValueError("Input sequence length is outside configured block_size")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")
        if torch.any((token_ids < 0) | (token_ids >= self.config.vocab_size)):
            raise ValueError("token_ids contain IDs outside the configured vocabulary")
        positions = torch.arange(length, device=token_ids.device)
        x = self.dropout(
            self.token_embedding(token_ids) +
            self.position_embedding(positions)[None, :, :]
        )
        for block in self.blocks:
            x = block(x)
        logits = self.language_head(self.final_norm(x))
        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape or targets.dtype != torch.long:
                raise ValueError("targets must be torch.long and match token_ids shape")
            invalid_targets = (targets != -100) & (
                (targets < 0) | (targets >= self.config.vocab_size)
            )
            if torch.any(invalid_targets):
                raise ValueError(
                    "targets contain IDs outside the configured vocabulary "
                    "or the supported ignore_index=-100"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss

    def forward_cached(self, token_ids, past_key_values=None):
        """Forward a prompt or one next token while returning per-layer KV caches.

        This inference-only path preserves the model's learned absolute positions.
        A cached call after prefill must contain exactly one new token.
        """
        if self.training:
            raise RuntimeError("Cached inference requires model.eval()")
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")
        if torch.any((token_ids < 0) | (token_ids >= self.config.vocab_size)):
            raise ValueError("token_ids contain IDs outside the configured vocabulary")

        batch, length = token_ids.shape
        if past_key_values is None:
            past_length = 0
            past_key_values = (None,) * self.config.n_layers
        else:
            if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) != self.config.n_layers:
                raise ValueError("past_key_values must contain one key/value pair per layer")
            if length != 1:
                raise ValueError("Cached decoding accepts exactly one new token")
            lengths = set()
            for pair in past_key_values:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("Each cache entry must contain key and value tensors")
                key, value = pair
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                    raise ValueError("Cache keys and values must be tensors")
                if key.shape != value.shape or key.ndim != 4:
                    raise ValueError("Cache key/value tensor shapes are invalid")
                if key.shape[:2] != (batch, self.config.n_heads) or key.shape[3] != self.config.d_model // self.config.n_heads:
                    raise ValueError("Cache dimensions do not match this model")
                if key.device != token_ids.device or value.device != token_ids.device:
                    raise ValueError("Cache and token tensors must use the same device")
                lengths.add(key.shape[2])
            if len(lengths) != 1:
                raise ValueError("All layer caches must have the same sequence length")
            past_length = lengths.pop()

        if past_length + length > self.config.block_size:
            raise ValueError("Cached context exceeds configured block_size")
        positions = torch.arange(
            past_length, past_length + length, device=token_ids.device
        )
        x = self.dropout(
            self.token_embedding(token_ids) +
            self.position_embedding(positions)[None, :, :]
        )
        presents = []
        for block, past in zip(self.blocks, past_key_values):
            x, present = block.forward_cached(x, past)
            presents.append(present)
        logits = self.language_head(self.final_norm(x))
        return logits, tuple(presents)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100, temperature=0.8,
                 top_k=40):
        if (not isinstance(max_new_tokens, int) or
                isinstance(max_new_tokens, bool) or max_new_tokens < 0):
            raise ValueError("max_new_tokens must be a non-negative integer")
        if (isinstance(temperature, bool) or
                not isinstance(temperature, (int, float)) or
                not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and non-negative")
        if (not isinstance(top_k, int) or isinstance(top_k, bool) or
                top_k < 0):
            raise ValueError("top_k must be a non-negative integer")
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("generation prompt must be a torch.Tensor")
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("generation prompt must have shape [batch, sequence]")
        if token_ids.dtype != torch.long:
            raise TypeError("generation prompt must use torch.long")
        if torch.any((token_ids < 0) | (token_ids >= self.config.vocab_size)):
            raise ValueError("generation prompt contains IDs outside the configured vocabulary")
        self.eval()
        if max_new_tokens == 0:
            return token_ids
        # Cache is exact while the complete prompt + generation fits the learned
        # absolute-position window. Once it would overflow, use the original
        # sliding-window path so positions are re-based exactly as before.
        if token_ids.shape[1] + max_new_tokens <= self.config.block_size:
            logits, past_key_values = self.forward_cached(token_ids)
            for step in range(max_new_tokens):
                logits = logits[:, -1, :]
                if temperature == 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    sampled_logits = logits / temperature
                    if top_k:
                        k = min(top_k, sampled_logits.size(-1))
                        threshold = torch.topk(sampled_logits, k).values[:, [-1]]
                        sampled_logits = sampled_logits.masked_fill(
                            sampled_logits < threshold, float("-inf")
                        )
                    probabilities = F.softmax(sampled_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                token_ids = torch.cat((token_ids, next_token), dim=1)
                if step + 1 < max_new_tokens:
                    logits, past_key_values = self.forward_cached(
                        next_token, past_key_values
                    )
            return token_ids

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
