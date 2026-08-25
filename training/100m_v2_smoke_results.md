# mythos_100m_v2 smoke-training run

Verifies `mythos_100m_v2()` (added to fix the embed/core param imbalance in
`mythos_100m()`) actually trains on real data before committing to a full,
expensive pretraining job. Not a production run — 90k streamed tokens from
FineWeb-Edu, CPU-only (2 vCPU, no GPU), 150 steps.

    python training/100m_v2_smoke.py --steps 150 --batch-size 2 --train-tokens 90000 --log-every 10

## Config
- `mythos_100m_v2()`: dim=512, n_heads=8, n_kv_heads=4, attn_type=gqa, max_seq_len=256,
  max_loop_iters=4, prelude/coda=2, n_experts=8, expert_dim=4096
- params: total=173,455,906, non_embedding=71,056,930
- dataset: `HuggingFaceFW/fineweb-edu` (config `sample-10BT`, streamed), 90,113 tokens
  packed into 352 (x, y) pairs of seq_len=256
- batch_size=2, lr=3e-4, AdamW(betas=(0.9, 0.95), weight_decay=0.1), grad clip 1.0

## Result

```
  step    1/150  loss=12.2967   55 tok/s
  step   10/150  loss=10.7064   56 tok/s
  step   20/150  loss=9.6698    60 tok/s
  step   30/150  loss=8.9799    61 tok/s
  step   40/150  loss=8.2633    65 tok/s
  step   50/150  loss=7.8722    68 tok/s
  step   60/150  loss=8.2601    71 tok/s
  step   70/150  loss=7.6367    70 tok/s
  step   80/150  loss=8.2974    70 tok/s
  step   90/150  loss=7.9454    71 tok/s
  step  100/150  loss=8.0815    72 tok/s
  step  110/150  loss=7.7469    72 tok/s
  step  120/150  loss=7.6044    72 tok/s
  step  130/150  loss=7.6135    73 tok/s
  step  140/150  loss=7.7184    73 tok/s
  step  150/150  loss=8.2243    72 tok/s

wall=1072.6s  initial_loss(avg first 10)=11.3890  final_loss(avg last 10)=7.7675  delta=-3.6215
no NaN/crash across 150 steps: OK
```

Loss drops steadily from 12.30 to a ~7.6-8.2 range and stays there — expected
plateau/noise given the run only sees ~1 epoch over a 90k-token buffer (352
pairs at batch_size=2), not a sign of a training problem. Confirms the config,
GQA attention, MoE routing, LTI injection, and ACT halting all backprop
cleanly through `mythos_100m_v2()` with no NaN/Inf/crash. A real pretraining
run needs a GPU and orders of magnitude more data/steps — this VPS has
neither, so this was intentionally scoped as a correctness smoke test only.

## Follow-up run: 400 steps, 200k tokens, + generation check + checkpoint save

Same CPU-only host, more data/steps, plus a post-training generation sample
and `--save-checkpoint` (script changes below). Checkpoint itself
(`mythos_100m_v2_smoke_400.pt`, ~694MB) is left local on the VPS — too large
for this repo (GitHub's 100MB file limit) and still just a smoke-test
artifact, not a release checkpoint.

    python training/100m_v2_smoke.py --steps 400 --batch-size 2 --train-tokens 200000 \
      --log-every 20 --save-checkpoint /root/openmythos/models/100m/mythos_100m_v2_smoke_400.pt

```
  step    1/400  loss=12.3411  45 tok/s
  step   20/400  loss=9.8123   57 tok/s
  step   40/400  loss=8.5083   64 tok/s
  step   60/400  loss=8.0489   69 tok/s
  step   80/400  loss=8.0561   69 tok/s
  step  100/400  loss=8.2639   69 tok/s
  step  120/400  loss=7.7046   67 tok/s
  step  140/400  loss=8.2229   68 tok/s
  step  160/400  loss=8.0125   66 tok/s
  step  180/400  loss=7.9475   67 tok/s
  step  200/400  loss=7.8922   66 tok/s
  step  220/400  loss=8.3461   67 tok/s
  step  240/400  loss=7.8329   66 tok/s
  step  260/400  loss=7.3605   67 tok/s
  step  280/400  loss=7.0476   66 tok/s
  step  300/400  loss=7.1648   67 tok/s
  step  320/400  loss=6.8919   66 tok/s
  step  340/400  loss=7.7785   66 tok/s
  step  360/400  loss=7.8027   66 tok/s
  step  380/400  loss=7.0486   65 tok/s
  step  400/400  loss=6.4351   65 tok/s

wall=3156.2s  initial_loss(avg first 10)=11.4601  final_loss(avg last 10)=6.8952  delta=-4.5649
no NaN/crash across 400 steps: OK
```

Post-training generation check (same model instance, `n_loops=4`, `temperature=0.8`, `top_k=50`):

- prompt: `"In the field of artificial intelligence research, recurrent-depth transformers"`
- continuation: `", and the end of the other new- The 100. A- This is the new- The same new.\nBs on 2009. In the new- An for the of the one of the environment on the new, and the 2. In high pressure, there is one"`
- no NaN token ids

Loss improved further than the 150-step run (delta -4.56 vs -3.62, final avg 6.90
vs 7.77). More notably, the degenerate repetition loop seen when generating
from the *released* `mythos_100m_mixed_80k.pt` checkpoint (drifting into a
repeated Chinese-legal or JSON-template sentence verbatim, even at
`n_loops=8`/`16`) is **gone** here — output is now varied, fragmented English
rather than a stuck loop. Still far from coherent (only ~200k tokens / 1
epoch seen) — a real pretraining run needs a GPU and orders of magnitude more
data/steps than this VPS can provide, so this remains a correctness/behavior
smoke test, not a usable checkpoint.
