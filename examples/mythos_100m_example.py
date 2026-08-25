"""
Load the released mythos_100m_mixed_80k.pt checkpoint (legacy GQA attention),
verify it, and benchmark generation.

    python examples/mythos_100m_example.py --checkpoint /path/to/mythos_100m_mixed_80k.pt
"""

from __future__ import annotations

import argparse
import time

import torch

from open_mythos import MythosTokenizer, load_mythos_100m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--checkpoint", default="/root/openmythos/models/100m/mythos_100m_mixed_80k.pt"
    )
    p.add_argument("--prompt", default="The history of artificial intelligence began")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--n-loops", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"[load] checkpoint={args.checkpoint}")
    model = load_mythos_100m(args.checkpoint, device=args.device)
    total = sum(x.numel() for x in model.parameters())
    print(f"[load] params={total:,}  attn_type={model.cfg.attn_type}")

    tok = MythosTokenizer()
    ids = tok.encode(args.prompt)
    input_ids = torch.tensor([ids], device=args.device)
    print(f"[prompt] {len(ids)} tokens: {args.prompt!r}")

    # ---- smoke test: single forward pass, no KV cache ----
    with torch.no_grad():
        logits = model(input_ids, n_loops=args.n_loops)
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"
    print(
        f"[smoke] logits shape={tuple(logits.shape)}  "
        f"abs-max={logits.abs().max().item():.3f}  no NaN/Inf: OK"
    )

    # ---- benchmark: prefill + KV-cache decode throughput ----
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        n_loops=args.n_loops,
        temperature=0.8,
        top_k=50,
    )
    dt = time.perf_counter() - t0
    gen_ids = out[0, len(ids):].tolist()
    assert all(t == t for t in gen_ids), "NaN token id in generation output"

    print(
        f"[bench] generated {len(gen_ids)} tokens in {dt:.3f}s "
        f"-> {len(gen_ids) / dt:.2f} tok/s (n_loops={args.n_loops}, device={args.device})"
    )
    print(f"[output] {tok.decode(gen_ids)!r}")


if __name__ == "__main__":
    main()
