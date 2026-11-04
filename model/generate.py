"""Generate text from a trained DREAM checkpoint."""
import argparse
import torch

from checkpoint_tokenizer import load_tokenizer
from dream_transformer import DreamConfig, DreamTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/dream-small.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = DreamConfig(**checkpoint["config"])
    try:
        tokenizer = load_tokenizer(checkpoint, config)
    except ValueError as exc:
        raise SystemExit(f"Unsupported checkpoint tokenizer: {exc}") from exc
    model = DreamTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt = tokenizer.encode(args.prompt)
    if not prompt:
        raise SystemExit("Prompt must not be empty.")
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    generated = model.generate(
        ids, max_new_tokens=args.tokens, temperature=args.temperature,
        constrain_utf8=True,
        token_byte_pieces=getattr(tokenizer, "token_byte_pieces", None)
    )[0].tolist()
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
