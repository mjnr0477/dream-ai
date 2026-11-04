"""Deterministic, byte-fallback BPE tokenizer for experimental DREAM checkpoints.

This module is deliberately standalone: existing utf8-byte-v1 checkpoints and
runtime behavior remain unchanged until an end-to-end migration is tested.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re


class DreamBPETokenizer:
    """A small dependency-free byte-level BPE tokenizer with lossless UTF-8 decode."""

    format_version = "dream-bpe-v1"
    base_vocab_size = 256

    def __init__(self, merges=None):
        self.merges = [tuple(map(int, pair)) for pair in (merges or [])]
        self._validate_merges()
        self._pieces = [bytes([i]) for i in range(self.base_vocab_size)]
        for left, right in self.merges:
            self._pieces.append(self._pieces[left] + self._pieces[right])

    @property
    def vocab_size(self):
        return self.base_vocab_size + len(self.merges)

    def _validate_merges(self):
        for index, pair in enumerate(self.merges):
            if len(pair) != 2:
                raise ValueError("Each BPE merge must contain exactly two token IDs")
            left, right = pair
            limit = self.base_vocab_size + index
            if left < 0 or right < 0 or left >= limit or right >= limit:
                raise ValueError("BPE merges must reference only previously defined tokens")

    @classmethod
    def train(cls, text, vocab_size=1024, min_frequency=2):
        if not isinstance(text, str):
            raise TypeError("BPE training input must be text")
        if not isinstance(vocab_size, int) or not cls.base_vocab_size <= vocab_size <= 65536:
            raise ValueError("vocab_size must be between 256 and 65536")
        if not isinstance(min_frequency, int) or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer")

        # Keep whitespace runs and non-whitespace runs separate so merges do
        # not accidentally combine across lexical boundaries.
        frequencies = Counter(re.findall(r"\s+|[^\s]+", text))
        sequences = {}
        for piece, count in frequencies.items():
            sequences[tuple(piece.encode("utf-8"))] = count

        merges = []
        while cls.base_vocab_size + len(merges) < vocab_size:
            pair_counts = Counter()
            for sequence, frequency in sequences.items():
                for pair in zip(sequence, sequence[1:]):
                    pair_counts[pair] += frequency
            if not pair_counts:
                break
            best_count = max(pair_counts.values())
            if best_count < min_frequency:
                break
            # Lexicographic tie-breaking makes training reproducible.
            best_pair = min(pair for pair, count in pair_counts.items()
                            if count == best_count)
            new_id = cls.base_vocab_size + len(merges)
            merges.append(best_pair)
            updated = {}
            for sequence, frequency in sequences.items():
                result = []
                index = 0
                while index < len(sequence):
                    if (index + 1 < len(sequence)
                            and (sequence[index], sequence[index + 1]) == best_pair):
                        result.append(new_id)
                        index += 2
                    else:
                        result.append(sequence[index])
                        index += 1
                key = tuple(result)
                updated[key] = updated.get(key, 0) + frequency
            sequences = updated
        return cls(merges)

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        output = []
        for piece in re.findall(r"\s+|[^\s]+", text):
            tokens = list(piece.encode("utf-8"))
            # Most learned merges are irrelevant to any given piece. Avoid
            # rescanning its token sequence unless both merge inputs are present.
            present = set(tokens)
            for merge_index, (left, right) in enumerate(self.merges):
                if left not in present or right not in present:
                    continue
                new_id = self.base_vocab_size + merge_index
                merged = []
                index = 0
                changed = False
                while index < len(tokens):
                    if (index + 1 < len(tokens)
                            and tokens[index] == left and tokens[index + 1] == right):
                        merged.append(new_id)
                        index += 2
                        changed = True
                    else:
                        merged.append(tokens[index])
                        index += 1
                if changed:
                    tokens = merged
                    present = set(tokens)
            output.extend(tokens)
        return output

    def token_byte_length(self, token_id):
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError("Token ID must be an integer")
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError("Token ID is outside this tokenizer vocabulary")
        return len(self._pieces[token_id])

    def decode(self, token_ids):
        pieces = []
        for token in token_ids:
            if not isinstance(token, int) or isinstance(token, bool):
                raise ValueError("Token IDs must be integers")
            if token < 0 or token >= self.vocab_size:
                raise ValueError("Token ID is outside this tokenizer vocabulary")
            pieces.append(self._pieces[token])
        raw = b"".join(pieces)
        return raw.decode("utf-8", errors="replace")

    def to_dict(self):
        return {"format": self.format_version, "vocab_size": self.vocab_size,
                "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("format") != cls.format_version:
            raise ValueError("Unsupported DREAM BPE tokenizer format")
        tokenizer = cls(payload.get("merges", []))
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match its merges")
        return tokenizer

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
