#!/usr/bin/env python3
"""
Interactive CLI chat loop for an OpenMythos checkpoint.

These are base (non-instruction-tuned) language models, so this is plain
continued-text generation, not a chat-templated conversation — each turn's
generated text is simply appended to the running context as the next prompt.

    # Released 100M GQA checkpoint (mythos_100m config)
    python examples/chat_cli.py --variant 100m \
        --checkpoint /root/openmythos/models/100m/mythos_100m_mixed_80k.pt

    # A mythos_100m_v2 smoke-training checkpoint
    python examples/chat_cli.py --variant 100m_v2 \
        --checkpoint /root/openmythos/models/100m/mythos_100m_v2_smoke_600.pt
"""

from __future__ import annotations

import argparse
import os

import torch

from open_mythos import MythosTokenizer, OpenMythos, load_mythos_100m
from open_mythos.variants import mythos_100m_v2

# CPU decode here is dominated by per-token dispatch cost of many small
# linear/MoE ops (~100ms/token profiled), not by available core count —
# pin thread count explicitly so behavior doesn't depend on the caller's
# environment.
torch.set_num_threads(os.cpu_count() or 1)


def load_model(variant: str, checkpoint: str, device: str) -> OpenMythos:
    if variant == "100m":
        return load_mythos_100m(checkpoint, device=device)
    if variant == "100m_v2":
        cfg = mythos_100m_v2()
        model = OpenMythos(cfg)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()
        return model
    raise ValueError(f"unknown variant: {variant!r} (expected '100m' or '100m_v2')")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--variant", choices=["100m", "100m_v2"], default="100m")
    p.add_argument(
        "--checkpoint", default="/root/openmythos/models/100m/mythos_100m_mixed_80k.pt"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument(
        "--n-loops",
        type=int,
        default=1,
        help="recurrent loop depth. Profiled at ~37%% faster decode at 1 vs. "
        "the trained depth of 4, with byte-identical output on the current "
        "smoke-training checkpoints — they're undertrained enough that "
        "recurrence depth isn't yet doing meaningful work. Revisit this "
        "default (raise back toward 4) once the model is trained further "
        "and loop depth starts to actually change the output.",
    )
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="trim running context to this many tokens before each turn "
        "(defaults to the model's max_seq_len, minus room for one reply)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[load] variant={args.variant} checkpoint={args.checkpoint}")
    model = load_model(args.variant, args.checkpoint, args.device)
    tok = MythosTokenizer()
    max_seq_len = model.cfg.max_seq_len
    max_context = args.max_context_tokens or (max_seq_len - args.max_new_tokens)

    print(
        f"[ready] params={sum(p.numel() for p in model.parameters()):,}  "
        f"max_seq_len={max_seq_len}  n_loops={args.n_loops}\n"
        f"Type a prompt and press enter. Commands: /reset (clear context), /quit.\n"
    )

    context_ids: list[int] = []
    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in ("/quit", "/exit"):
            break
        if user_text == "/reset":
            context_ids = []
            print("[context cleared]")
            continue

        context_ids.extend(tok.encode(user_text))
        if len(context_ids) > max_context:
            context_ids = context_ids[-max_context:]

        input_ids = torch.tensor([context_ids], device=args.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                n_loops=args.n_loops,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        gen_ids = out[0, len(context_ids):].tolist()
        reply = tok.decode(gen_ids)
        print(reply)
        context_ids.extend(gen_ids)


if __name__ == "__main__":
    main()
