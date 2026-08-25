#!/usr/bin/env python3
"""
Smoke-training run for mythos_100m_v2() on FineWeb-Edu (streamed), CPU-only.

This is NOT a production pretraining run — it exists to verify the config
actually trains (loss decreases, no NaN/crash) on a real dataset before
committing to a long, expensive pretraining job. Defaults are tuned for a
2-CPU / ~4GB-available VPS with other production services running
alongside it: small batch, short seq len, a few hundred steps.

    python training/100m_v2_smoke.py
    python training/100m_v2_smoke.py --steps 500 --batch-size 4
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from open_mythos import OpenMythos, MythosTokenizer
from open_mythos.variants import mythos_100m_v2


class PackedLMDataset(Dataset):
    """Flatten a streamed HF text dataset into fixed-length next-token pairs."""

    def __init__(self, hf_ds, tokenizer: MythosTokenizer, seq_len: int, max_tokens: int):
        buf: list[int] = []
        for sample in hf_ds:
            text = sample.get("text")
            if not text or not text.strip():
                continue
            buf.extend(tokenizer.encode(text))
            if len(buf) >= max_tokens:
                break
        self.seq_len = seq_len
        n_pairs = max(1, (len(buf) - 1) // seq_len)
        buf = buf[: n_pairs * seq_len + 1]
        self.data = torch.tensor(buf, dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int):
        s = idx * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--train-tokens", type=int, default=400_000)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gen-prompt",
        default="In the field of artificial intelligence research, recurrent-depth transformers",
    )
    p.add_argument("--gen-max-new-tokens", type=int, default=60)
    p.add_argument("--save-checkpoint", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    cfg = mythos_100m_v2()
    print(f"[setup] mythos_100m_v2  seq_len={cfg.max_seq_len}  vocab={cfg.vocab_size}")

    tokenizer = MythosTokenizer()  # openai/gpt-oss-20b, vocab_size == cfg.vocab_size

    print(f"[setup] dataset={args.dataset} config={args.dataset_config} (streaming)")
    raw = load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)
    train_ds = PackedLMDataset(raw, tokenizer, cfg.max_seq_len, args.train_tokens)
    print(f"[setup] packed tokens={train_ds.data.numel():,}  pairs={len(train_ds)}")

    torch.manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = OpenMythos(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    non_embed = total - model.embed.weight.numel()
    print(f"[setup] params total={total:,}  non_embed={non_embed:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    losses: list[float] = []
    data_iter = iter(loader)
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        model.train()
        opt.zero_grad()
        logits = model(x, n_loops=cfg.max_loop_iters)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        assert not torch.isnan(loss), f"NaN loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        if step == 1 or step % args.log_every == 0:
            dt = time.perf_counter() - t0
            tok_s = step * x.numel() / dt
            print(f"  step {step:>4}/{args.steps}  loss={loss.item():.4f}  {tok_s:,.0f} tok/s")

    total_wall = time.perf_counter() - t0
    initial = sum(losses[:10]) / min(10, len(losses))
    final = sum(losses[-10:]) / min(10, len(losses))
    print(f"\n[done] wall={total_wall:.1f}s  initial_loss(avg first 10)={initial:.4f}  "
          f"final_loss(avg last 10)={final:.4f}  delta={final - initial:+.4f}")
    print(f"[done] no NaN/crash across {args.steps} steps: OK")

    if args.save_checkpoint:
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"[done] saved checkpoint -> {args.save_checkpoint}")

    # ---- post-training generation check ----
    model.eval()
    ids = tokenizer.encode(args.gen_prompt)
    input_ids = torch.tensor([ids])
    torch.manual_seed(args.seed)
    out = model.generate(
        input_ids,
        max_new_tokens=args.gen_max_new_tokens,
        n_loops=cfg.max_loop_iters,
        temperature=0.8,
        top_k=50,
    )
    gen_ids = out[0, len(ids):].tolist()
    print(f"\n[gen] prompt={args.gen_prompt!r}")
    print(f"[gen] continuation={tokenizer.decode(gen_ids)!r}")
    print(f"[gen] any NaN token id: {any(t != t for t in gen_ids)}")


if __name__ == "__main__":
    main()
