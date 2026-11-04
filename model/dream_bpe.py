"""Deterministic byte-fallback BPE tokenizer for native DREAM checkpoints.

Legacy utf8-byte-v1 checkpoints remain supported; BPE vocabulary and byte pieces
are reconstructed from the checkpoint metadata for training and inference.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import heapq
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

    @property
    def token_byte_pieces(self):
        return tuple(self._pieces)

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
        if (not isinstance(vocab_size, int) or isinstance(vocab_size, bool)
                or not cls.base_vocab_size <= vocab_size <= 65536):
            raise ValueError("vocab_size must be between 256 and 65536")
        if (not isinstance(min_frequency, int) or isinstance(min_frequency, bool)
                or min_frequency < 1):
            raise ValueError("min_frequency must be a positive integer")

        # Keep whitespace runs and non-whitespace runs separate so merges do
        # not accidentally combine across lexical boundaries. Each distinct
        # piece is represented once and weighted by its corpus frequency.
        frequencies = Counter(re.findall(r"\s+|[^\s]+", text))
        sequences = [
            (list(piece.encode("utf-8")), frequency)
            for piece, frequency in frequencies.items()
        ]

        # Maintain pair counts and an inverted index. A merge only changes
        # sequences containing that pair, rather than rescanning the entire
        # vocabulary of unique words for every merge.
        pair_counts = Counter()
        pair_sequences = defaultdict(set)
        for sequence_id, (sequence, frequency) in enumerate(sequences):
            for pair, occurrences in Counter(zip(sequence, sequence[1:])).items():
                pair_counts[pair] += occurrences * frequency
                pair_sequences[pair].add(sequence_id)

        versions = defaultdict(int)
        heap = []
        for pair, count in pair_counts.items():
            heapq.heappush(heap, (-count, pair, versions[pair]))

        merges = []
        while cls.base_vocab_size + len(merges) < vocab_size:
            best_pair = None
            best_count = 0
            while heap:
                negative_count, pair, version = heapq.heappop(heap)
                count = -negative_count
                if version == versions[pair] and count == pair_counts[pair]:
                    best_pair, best_count = pair, count
                    break
            if best_pair is None or best_count < min_frequency:
                break

            new_id = cls.base_vocab_size + len(merges)
            merges.append(best_pair)
            affected_ids = sorted(pair_sequences.get(best_pair, ()))
            deltas = Counter()

            for sequence_id in affected_ids:
                sequence, frequency = sequences[sequence_id]
                old_counts = Counter(zip(sequence, sequence[1:]))
                merged = []
                index = 0
                while index < len(sequence):
                    if (index + 1 < len(sequence)
                            and (sequence[index], sequence[index + 1]) == best_pair):
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(sequence[index])
                        index += 1
                new_counts = Counter(zip(merged, merged[1:]))

                for pair in old_counts.keys() | new_counts.keys():
                    old_occurrences = old_counts.get(pair, 0)
                    new_occurrences = new_counts.get(pair, 0)
                    if old_occurrences and not new_occurrences:
                        pair_sequences[pair].discard(sequence_id)
                        if not pair_sequences[pair]:
                            del pair_sequences[pair]
                    elif new_occurrences and not old_occurrences:
                        pair_sequences[pair].add(sequence_id)
                    difference = new_occurrences - old_occurrences
                    if difference:
                        deltas[pair] += difference * frequency
                sequences[sequence_id] = (merged, frequency)

            for pair, delta in deltas.items():
                pair_counts[pair] += delta
                versions[pair] += 1
                if pair_counts[pair] > 0:
                    heapq.heappush(
                        heap, (-pair_counts[pair], pair, versions[pair])
                    )

        return cls(merges)

    def encode(self, text):
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be text")
        output = []
        for piece in re.findall(r"\s+|[^\s]+", text):
            tokens = list(piece.encode("utf-8"))
            length = len(tokens)
            if length < 2 or not self.merges:
                output.extend(tokens)
                continue

            # A linked list over original byte positions supports O(1) merges.
            # The heap processes the lowest-ranked available pair first, and
            # original position breaks ties to preserve left-to-right behavior.
            previous = [index - 1 for index in range(length)]
            following = [index + 1 for index in range(length)]
            following[-1] = -1
            alive = [True] * length
            candidates = []
            for index in range(length - 1):
                rank = self._merge_ranks.get((tokens[index], tokens[index + 1]))
                if rank is not None:
                    heapq.heappush(candidates, (rank, index))

            while candidates:
                rank, left = heapq.heappop(candidates)
                if not alive[left]:
                    continue
                right = following[left]
                if right < 0 or not alive[right]:
                    continue
                if self._merge_ranks.get((tokens[left], tokens[right])) != rank:
                    continue

                tokens[left] = self.base_vocab_size + rank
                alive[right] = False
                after = following[right]
                following[left] = after
                if after >= 0:
                    previous[after] = left

                before = previous[left]
                if before >= 0:
                    new_rank = self._merge_ranks.get((tokens[before], tokens[left]))
                    if new_rank is not None:
                        heapq.heappush(candidates, (new_rank, before))
                if after >= 0:
                    new_rank = self._merge_ranks.get((tokens[left], tokens[after]))
                    if new_rank is not None:
                        heapq.heappush(candidates, (new_rank, left))

            index = 0
            while index >= 0:
                output.append(tokens[index])
                index = following[index]
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
