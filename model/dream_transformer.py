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
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("Model dimensions must be positive integers")
        if self.vocab_size < 256 or self.vocab_size > 65536:
            raise ValueError("vocab_size must be between 256 and 65536")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (isinstance(self.dropout, bool) or
                not isinstance(self.dropout, (int, float)) or
                not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in the range [0, 1)")


class DreamByteTokenizer:
    """UTF-8 byte tokenizer: IDs 0..255 map directly to bytes."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        return list(text.encode("utf-8"))

    def decode(self, token_ids) -> str:
        raw = bytes(int(token) for token in token_ids if 0 <= int(token) <= 255)
        return raw.decode("utf-8", errors="replace")


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
        if batch < 1 or length < 1 or length > self.config.block_size:
            raise ValueError("Input batch and sequence must be non-empty and sequence within block_size")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.config.vocab_size):
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
            # -100 is the standard ignore_index used for prompt/padding
            # positions in assistant-only supervised fine-tuning.
            supervised_targets = targets != -100
            if not torch.any(supervised_targets):
                raise ValueError("targets must contain at least one supervised token")
            if torch.any(supervised_targets & (
                (targets < 0) | (targets >= self.config.vocab_size)
            )):
                raise ValueError("targets contain IDs outside the configured vocabulary")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100, temperature=0.8,
                 top_k=40):
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if (isinstance(temperature, bool) or
                not isinstance(temperature, (int, float)) or
                not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and non-negative")
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if token_ids.ndim != 2 or token_ids.shape[1] < 1:
            raise ValueError("generation prompt must have shape [batch, sequence]")
        self.eval()
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
