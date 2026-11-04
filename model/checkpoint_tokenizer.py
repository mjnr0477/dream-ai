"""Strict tokenizer reconstruction for DREAM native checkpoints."""
from dream_bpe import DreamBPETokenizer
from dream_transformer import DreamByteTokenizer


def tokenizer_name(metadata):
    if metadata == "utf8-byte-v1":
        return "utf8-byte-v1"
    if isinstance(metadata, dict) and metadata.get("format") == DreamBPETokenizer.format_version:
        return DreamBPETokenizer.format_version
    return "unknown"


def load_tokenizer(checkpoint, config):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be an object")
    metadata = checkpoint.get("tokenizer")
    if metadata == "utf8-byte-v1":
        tokenizer = DreamByteTokenizer()
    elif isinstance(metadata, dict):
        tokenizer = DreamBPETokenizer.from_dict(metadata)
    else:
        raise ValueError("Missing or unsupported tokenizer metadata")
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match the model configuration")
    return tokenizer
