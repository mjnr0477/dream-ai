"""DREAM decoder-only transformer, initialized from random weights.

This is DREAM's own experimental architecture. It does not load weights from
another model. The legacy byte tokenizer and optional byte-fallback BPE tokenizer
are deterministic and use no borrowed vocabulary.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


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
        if self.vocab_size < 256:
            raise ValueError(
                "vocab_size must be at least 256 to preserve DREAM's UTF-8 byte fallback"
            )
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

    def token_byte_length(self, token_id: int) -> int:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or not 0 <= token_id < self.vocab_size:
            raise ValueError("Token ID is outside the UTF-8 byte vocabulary")
        return 1

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
        self.gradient_checkpointing = False
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
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = activation_checkpoint(block, x, use_reentrant=False)
            else:
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
                if key.shape != value.shape or key.ndim != 4 or key.shape[2] < 1:
                    raise ValueError("Cache key/value tensor shapes are invalid")
                if key.shape[:2] != (batch, self.config.n_heads) or key.shape[3] != self.config.d_model // self.config.n_heads:
                    raise ValueError("Cache dimensions do not match this model")
                if key.device != token_ids.device or value.device != token_ids.device:
                    raise ValueError("Cache and token tensors must use the same device")
                if (
                    key.dtype != value.dtype
                    or key.dtype != self.token_embedding.weight.dtype
                    or not key.is_floating_point()
                    or key.layout != torch.strided
                    or value.layout != torch.strided
                ):
                    raise ValueError("Cache key/value tensor dtypes or layouts are invalid")
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

    @staticmethod
    def _utf8_next_token_mask(pending_bytes, next_min, next_max, remaining_after):
        """Return valid next-byte choices that can finish within the token budget."""
        batch_size = pending_bytes.shape[0]
        token_ids = torch.arange(256, device=pending_bytes.device)
        no_pending = pending_bytes == 0
        allowed = no_pending[:, None] & (token_ids[None, :] <= 0x7F)

        if remaining_after >= 1:
            two_byte_lead = (token_ids >= 0xC2) & (token_ids <= 0xDF)
            allowed |= no_pending[:, None] & two_byte_lead[None, :]
        if remaining_after >= 2:
            three_byte_lead = (token_ids >= 0xE0) & (token_ids <= 0xEF)
            allowed |= no_pending[:, None] & three_byte_lead[None, :]
        if remaining_after >= 3:
            four_byte_lead = (token_ids >= 0xF0) & (token_ids <= 0xF4)
            allowed |= no_pending[:, None] & four_byte_lead[None, :]

        continuation = (
            (pending_bytes > 0)[:, None]
            & ((pending_bytes - 1) <= remaining_after)[:, None]
            & (token_ids[None, :] >= next_min[:, None])
            & (token_ids[None, :] <= next_max[:, None])
        )
        allowed |= continuation
        return allowed

    @staticmethod
    def _advance_utf8_state(pending_bytes, next_min, next_max, next_token):
        """Advance a per-sequence UTF-8 decoder state after selecting one byte."""
        was_pending = pending_bytes > 0
        continued = torch.clamp(pending_bytes - 1, min=0)
        two_byte = (next_token >= 0xC2) & (next_token <= 0xDF)
        three_byte = (next_token >= 0xE0) & (next_token <= 0xEF)
        four_byte = (next_token >= 0xF0) & (next_token <= 0xF4)

        new_pending = (
            two_byte.to(torch.long)
            + (2 * three_byte.to(torch.long))
            + (3 * four_byte.to(torch.long))
        )
        pending_bytes = torch.where(was_pending, continued, new_pending)

        minimum = torch.full_like(next_min, 0x80)
        maximum = torch.full_like(next_max, 0xBF)
        minimum = torch.where(was_pending, minimum, next_min)
        maximum = torch.where(was_pending, maximum, next_max)
        minimum = torch.where(next_token == 0xE0, torch.full_like(minimum, 0xA0), minimum)
        maximum = torch.where(next_token == 0xED, torch.full_like(maximum, 0x9F), maximum)
        minimum = torch.where(next_token == 0xF0, torch.full_like(minimum, 0x90), minimum)
        maximum = torch.where(next_token == 0xF4, torch.full_like(maximum, 0x8F), maximum)
        # Once a constrained continuation is consumed, subsequent continuations
        # use the ordinary 0x80..0xBF range.
        minimum = torch.where(was_pending & (continued > 0), torch.full_like(minimum, 0x80), minimum)
        maximum = torch.where(was_pending & (continued > 0), torch.full_like(maximum, 0xBF), maximum)
        return pending_bytes, minimum, maximum

    def _sample_next_byte(self, logits, temperature, top_k, utf8_state, remaining_after):
        pending_bytes, next_min, next_max = utf8_state
        allowed = self._utf8_next_token_mask(
            pending_bytes, next_min, next_max, remaining_after
        )
        logits = logits.masked_fill(~allowed, float("-inf"))
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
        next_token_values = next_token.squeeze(-1)
        next_state = self._advance_utf8_state(
            pending_bytes, next_min, next_max, next_token_values
        )
        return next_token, next_state


    @staticmethod
    def _advance_utf8_piece(state, piece):
        """Advance a UTF-8 decoder state across every byte in one BPE token."""
        pending, minimum, maximum = state
        for value in piece:
            if pending:
                if not minimum <= value <= maximum:
                    return None
                pending -= 1
                minimum, maximum = 0x80, 0xBF
                continue
            if value <= 0x7F:
                continue
            if 0xC2 <= value <= 0xDF:
                pending = 1
            elif value == 0xE0:
                pending, minimum = 2, 0xA0
            elif 0xE1 <= value <= 0xEC or 0xEE <= value <= 0xEF:
                pending = 2
            elif value == 0xED:
                pending, maximum = 2, 0x9F
            elif value == 0xF0:
                pending, minimum = 3, 0x90
            elif 0xF1 <= value <= 0xF3:
                pending = 3
            elif value == 0xF4:
                pending, maximum = 3, 0x8F
            else:
                return None
        return pending, minimum, maximum

    def _sample_next_bpe_token(
        self, logits, temperature, top_k, utf8_states, token_byte_pieces,
        remaining_after
    ):
        """Sample BPE tokens that preserve valid UTF-8 and can finish in budget."""
        rows = []
        for state in utf8_states:
            row = []
            for piece in token_byte_pieces:
                next_state = self._advance_utf8_piece(state, piece)
                row.append(next_state is not None and next_state[0] <= remaining_after)
            rows.append(row)
        allowed = torch.tensor(rows, dtype=torch.bool, device=logits.device)
        if not allowed.any(dim=1).all().item():
            raise RuntimeError("No UTF-8-valid BPE token can complete within the token budget")
        masked_logits = logits.masked_fill(~allowed, float("-inf"))
        next_token = self._sample_next_token(masked_logits, temperature, top_k)
        selected = next_token.squeeze(-1).tolist()
        next_states = [
            self._advance_utf8_piece(state, token_byte_pieces[token_id])
            for state, token_id in zip(utf8_states, selected)
        ]
        return next_token, next_states

    def _sample_next_token(self, logits, temperature, top_k):
        """Sample from the full vocabulary without byte-level UTF-8 constraints."""
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k).values[:, [-1]]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens=100, temperature=0.8,
                 top_k=40, constrain_utf8=True, token_byte_pieces=None):
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
        if token_byte_pieces is not None:
            if (not isinstance(token_byte_pieces, (tuple, list))
                    or len(token_byte_pieces) != self.config.vocab_size
                    or any(not isinstance(piece, bytes) or not piece
                           for piece in token_byte_pieces)):
                raise ValueError("token_byte_pieces must contain one non-empty bytes value per vocabulary ID")
        self.eval()
        if max_new_tokens == 0:
            return token_ids
        bpe_utf8_states = (
            [(0, 0x80, 0xBF)] * token_ids.shape[0]
            if token_byte_pieces is not None else None
        )
        # Cache is exact while the complete prompt + generation fits the learned
        # absolute-position window. Once it would overflow, use the original
        # sliding-window path so positions are re-based exactly as before.
        if token_ids.shape[1] + max_new_tokens <= self.config.block_size:
            logits, past_key_values = self.forward_cached(token_ids)
            utf8_state = (
                torch.zeros(token_ids.shape[0], dtype=torch.long, device=token_ids.device),
                torch.full((token_ids.shape[0],), 0x80, dtype=torch.long, device=token_ids.device),
                torch.full((token_ids.shape[0],), 0xBF, dtype=torch.long, device=token_ids.device),
            )
            for step in range(max_new_tokens):
                logits = logits[:, -1, :]
                if constrain_utf8 and token_byte_pieces is not None:
                    next_token, bpe_utf8_states = self._sample_next_bpe_token(
                        logits, temperature, top_k, bpe_utf8_states,
                        token_byte_pieces, max_new_tokens - step - 1
                    )
                elif constrain_utf8:
                    next_token, utf8_state = self._sample_next_byte(
                        logits, temperature, top_k, utf8_state,
                        max_new_tokens - step - 1
                    )
                else:
                    next_token = self._sample_next_token(logits, temperature, top_k)
                token_ids = torch.cat((token_ids, next_token), dim=1)
                if step + 1 < max_new_tokens:
                    logits, past_key_values = self.forward_cached(
                        next_token, past_key_values
                    )
            return token_ids

        utf8_state = (
            torch.zeros(token_ids.shape[0], dtype=torch.long, device=token_ids.device),
            torch.full((token_ids.shape[0],), 0x80, dtype=torch.long, device=token_ids.device),
            torch.full((token_ids.shape[0],), 0xBF, dtype=torch.long, device=token_ids.device),
        )
        for step in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            if constrain_utf8 and token_byte_pieces is not None:
                next_token, bpe_utf8_states = self._sample_next_bpe_token(
                    logits, temperature, top_k, bpe_utf8_states,
                    token_byte_pieces, max_new_tokens - step - 1
                )
            elif constrain_utf8:
                next_token, utf8_state = self._sample_next_byte(
                    logits, temperature, top_k, utf8_state,
                    max_new_tokens - step - 1
                )
            else:
                next_token = self._sample_next_token(logits, temperature, top_k)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
