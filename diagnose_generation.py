#!/usr/bin/env python3
"""
Phase 2/3 diagnostic: is OpenMythos's incoherent output a sampling problem or
a model/training/weight-quality problem?

Runs a fixed prompt set through every available checkpoint at multiple
temperatures (including greedy, temperature=0), and records per-step token
probability, entropy, repetition statistics, NaN/Inf, and tokenizer
round-trip correctness. Does not modify any checkpoint or model code path
beyond instrumenting generation locally in this script.

    python diagnose_generation.py
    python diagnose_generation.py --out diagnostics_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from open_mythos import MythosTokenizer, OpenMythos, load_mythos_100m
from open_mythos.variants import mythos_100m_v2

CHECKPOINTS = [
    ("100m_released_80k", "100m", "/root/openmythos/models/100m/mythos_100m_mixed_80k.pt"),
    ("100m_v2_smoke_1000", "100m_v2", "/root/openmythos/models/100m/mythos_100m_v2_smoke_1000.pt"),
]

PROMPTS = [
    ("empty", ""),
    ("hello", "Hello"),
    ("who_are_you", "Who are you?"),
    ("what_is_python", "What is Python?"),
    ("write_add_fn", "Write a Python function to add two numbers."),
    ("find_bug", "Find the bug in this Python code:\ndef add(a, b):\n    return a - b"),
    ("known_continuation", "The quick brown fox jumps over the lazy"),
]

TEMPERATURES = [0.0, 0.2, 0.5, 0.7, 1.0]

TOKENIZER_ROUNDTRIP_SAMPLES = [
    "Hello, world!",
    "def add(a, b):\n    return a + b\n",
    "    if x == None:\n\tprint('bad indent mix')",
    "print(\"emoji test 🚀 and unicode café\")",
    "a_very_long_identifier_name_1234567890" * 3,
]


@dataclass
class GenResult:
    prompt_name: str
    checkpoint: str
    temperature: float
    n_loops: int
    max_new_tokens: int
    prompt_tokens: int
    generated_tokens: int
    token_ids: list[int]
    decoded_text: str
    mean_token_prob: float
    mean_entropy: float
    nan_seen: bool
    inf_seen: bool
    distinct_1_ratio: float
    distinct_2_ratio: float
    max_immediate_repeat_run: int
    eos_token_seen: bool


def load_model(variant: str, checkpoint: str) -> Optional[OpenMythos]:
    if not Path(checkpoint).exists():
        return None
    if variant == "100m":
        return load_mythos_100m(checkpoint, device="cpu")
    cfg = mythos_100m_v2()
    model = OpenMythos(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def generate_with_stats(
    model: OpenMythos,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    n_loops: int,
    temperature: float,
    top_k: int = 50,
) -> tuple[torch.Tensor, list[float], list[float], bool, bool]:
    """Mirrors OpenMythos.generate()'s decode loop, instrumented with
    per-step chosen-token probability, full-distribution entropy, and
    NaN/Inf flags — without modifying the library's generate() further."""
    kv_cache: dict = {}
    prompt_len = input_ids.shape[1]
    token_probs: list[float] = []
    entropies: list[float] = []
    nan_seen = False
    inf_seen = False

    for step in range(max_new_tokens):
        cur_ids = input_ids if step == 0 else input_ids[:, -1:]
        start_pos = 0 if step == 0 else prompt_len + step - 1
        logits = model.forward(cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
        logits = logits[:, -1, :]

        if torch.isnan(logits).any():
            nan_seen = True
        if torch.isinf(logits).any():
            inf_seen = True

        full_probs = F.softmax(logits.float(), dim=-1)
        entropy = -(full_probs * (full_probs.clamp_min(1e-12)).log()).sum(-1).item()
        entropies.append(entropy / math.log(2))  # bits

        if temperature <= 0.0:
            next_tok = logits.argmax(dim=-1, keepdim=True)
            chosen_prob = full_probs.gather(-1, next_tok).item()
        else:
            scaled = logits / temperature
            if top_k > 0:
                v, _ = scaled.topk(top_k)
                scaled = scaled.clone()
                scaled[scaled < v[:, -1:]] = float("-inf")
            probs = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            chosen_prob = probs.gather(-1, next_tok).item()

        token_probs.append(chosen_prob)
        input_ids = torch.cat([input_ids, next_tok], dim=1)

    return input_ids, token_probs, entropies, nan_seen, inf_seen


def repetition_stats(token_ids: list[int]) -> tuple[float, float, int]:
    if not token_ids:
        return 1.0, 1.0, 0
    unigrams = token_ids
    distinct_1 = len(set(unigrams)) / len(unigrams)
    bigrams = list(zip(token_ids, token_ids[1:])) or [(0, 0)]
    distinct_2 = len(set(bigrams)) / len(bigrams)
    max_run = 1
    cur_run = 1
    for a, b in zip(token_ids, token_ids[1:]):
        if a == b:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    return distinct_1, distinct_2, max_run


def run_tokenizer_roundtrip(tok: MythosTokenizer) -> list[dict]:
    results = []
    for text in TOKENIZER_ROUNDTRIP_SAMPLES:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        results.append(
            {
                "original": text,
                "decoded": decoded,
                "exact_match": decoded == text,
                "n_tokens": len(ids),
            }
        )
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out", default="diagnostics_report.json")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--n-loops", type=int, default=4)
    args = p.parse_args()

    tok = MythosTokenizer()
    print(f"[tokenizer] {tok.tokenizer.name_or_path}  vocab_size={tok.vocab_size}")
    print(
        f"[tokenizer] bos_token={tok.tokenizer.bos_token!r} "
        f"eos_token={tok.tokenizer.eos_token!r} "
        f"pad_token={tok.tokenizer.pad_token!r}"
    )

    roundtrip = run_tokenizer_roundtrip(tok)
    n_ok = sum(r["exact_match"] for r in roundtrip)
    print(f"[tokenizer] round-trip: {n_ok}/{len(roundtrip)} exact matches")
    for r in roundtrip:
        if not r["exact_match"]:
            print(f"  MISMATCH: {r['original']!r} -> {r['decoded']!r}")

    eos_id = tok.tokenizer.eos_token_id

    all_results: list[GenResult] = []

    for ckpt_name, variant, ckpt_path in CHECKPOINTS:
        model = load_model(variant, ckpt_path)
        if model is None:
            print(f"\n[skip] {ckpt_name}: checkpoint not found at {ckpt_path}")
            continue

        total = sum(x.numel() for x in model.parameters())
        print(
            f"\n[checkpoint] {ckpt_name}  variant={variant}  params={total:,}  "
            f"attn_type={model.cfg.attn_type}  max_seq_len={model.cfg.max_seq_len}"
        )

        for prompt_name, prompt_text in PROMPTS:
            ids = tok.encode(prompt_text)
            if not ids:
                # Empty prompt: tok.encode("") -> []. bos_token_id (199998) is
                # OUT OF RANGE for this model's vocab_size=199998 embedding
                # table (valid ids 0..199997) -- see BOS/EOS/PAD finding in
                # the readiness report. Use token id 0 as an in-vocab,
                # arbitrary seed instead of crashing on an OOB special token.
                ids = [0]
            input_ids = torch.tensor([ids])
            room = model.cfg.max_seq_len - len(ids) - 1
            max_new = max(1, min(args.max_new_tokens, room))

            for temp in TEMPERATURES:
                torch.manual_seed(0)
                out_ids, probs, entropies, nan_seen, inf_seen = generate_with_stats(
                    model, input_ids, max_new, args.n_loops, temp
                )
                gen_ids = out_ids[0, len(ids):].tolist()
                decoded = tok.decode(gen_ids)
                d1, d2, max_run = repetition_stats(gen_ids)

                result = GenResult(
                    prompt_name=prompt_name,
                    checkpoint=ckpt_name,
                    temperature=temp,
                    n_loops=args.n_loops,
                    max_new_tokens=max_new,
                    prompt_tokens=len(ids),
                    generated_tokens=len(gen_ids),
                    token_ids=gen_ids,
                    decoded_text=decoded,
                    mean_token_prob=sum(probs) / max(1, len(probs)),
                    mean_entropy=sum(entropies) / max(1, len(entropies)),
                    nan_seen=nan_seen,
                    inf_seen=inf_seen,
                    distinct_1_ratio=d1,
                    distinct_2_ratio=d2,
                    max_immediate_repeat_run=max_run,
                    eos_token_seen=(eos_id is not None and eos_id in gen_ids),
                )
                all_results.append(result)

                print(
                    f"  [{prompt_name:<20} T={temp:>3}] "
                    f"mean_p={result.mean_token_prob:.3f} "
                    f"entropy={result.mean_entropy:.2f}bits "
                    f"distinct1={d1:.2f} maxrun={max_run} "
                    f"nan={nan_seen} inf={inf_seen}  "
                    f"-> {decoded[:70]!r}"
                )

        del model  # free memory between checkpoints (CPU, ~700MB-1.4GB each)

    report = {
        "tokenizer_roundtrip": roundtrip,
        "generations": [asdict(r) for r in all_results],
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[done] wrote {len(all_results)} generation records to {args.out}")


if __name__ == "__main__":
    main()
