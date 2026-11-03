"""Generate text from a trained DREAM checkpoint."""
import argparse
import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer
from dream_bpe import DreamBPETokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dream-small.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    tokenizer_name = checkpoint.get("tokenizer", "utf8-byte-v1")
    if tokenizer_name == "utf8-byte-v1":
        tokenizer = DreamByteTokenizer()
    elif tokenizer_name == DreamBPETokenizer.format_version:
        tokenizer = DreamBPETokenizer.from_dict(checkpoint["tokenizer_data"])
    else:
        raise SystemExit("Unsupported DREAM checkpoint tokenizer.")
    config = DreamConfig(**checkpoint["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise SystemExit("Checkpoint model vocabulary does not match its tokenizer.")
    model = DreamTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt = tokenizer.encode(args.prompt)
    if not prompt:
        raise SystemExit("Prompt must not be empty.")
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated = model.generate(
        ids, max_new_tokens=args.tokens, temperature=args.temperature
    )[0].tolist()
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
