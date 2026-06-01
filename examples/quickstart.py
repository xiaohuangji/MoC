"""Minimal MoC forward/backward example."""
from __future__ import annotations

import torch

from moc import PRESETS, build_model


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This quickstart expects a CUDA GPU.")

    config = PRESETS["130m"]
    model = build_model(config, ffn_type="moc").cuda().bfloat16()
    tokens = torch.randint(0, config.vocab_size, (2, 256), device="cuda")
    logits, loss = model(tokens, labels=tokens)
    loss.backward()
    print({"logits_shape": tuple(logits.shape), "loss": float(loss.detach().cpu())})


if __name__ == "__main__":
    main()
