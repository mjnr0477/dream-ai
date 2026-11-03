"""Generate text from a trained DREAM checkpoint."""
import argparse
import torch

from dream_transformer import DreamByteTokenizer, DreamConfig, DreamTransformer


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
    model = DreamTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokenizer = DreamByteTokenizer()
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
