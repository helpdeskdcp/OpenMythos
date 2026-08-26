# OpenMythos Coding-Readiness Report

Phase 1-4 diagnosis only, per the explicit instruction not to start retraining
or build the agent layer until diagnosis is complete. Evidence backing every
claim below is in `diagnostics/diagnostics_report.json` (70 generation
records, 2 checkpoints × 7 prompts × 5 temperatures) and reproducible with
`python diagnose_generation.py`.

## TL;DR

**The gibberish is not a sampling problem.** Greedy (temperature=0) decoding
is exactly as incoherent as sampled decoding on both checkpoints tested. It
is also not a numerical-instability or tokenizer-corruption problem (no
NaN/Inf anywhere, tokenizer round-trips 5/5 exactly). It is a genuine
**model/training-quality problem**, with three independent, compounding
causes:

1. **Severe undertraining relative to a 200k-token vocabulary.** The released
   checkpoint spends 90.7% of its 112.9M params on the embedding/output
   table and only 9.3% (10.5M) on the actual transformer core, trained for
   just 80k steps on an undocumented dataset. My own from-scratch
   experiments this session (up to 1000 steps, better 41%-core ratio)
   show the *same* qualitative incoherence, confirming the shared bottleneck
   is tokens-seen, not just this one checkpoint's specific run.
2. **No instruction-tuning stage exists at all.** This is a 100% base
   (pretrain-only) language model. "do you know python coding" isn't being
   misunderstood — the model has never seen a single (instruction,
   response) pair, so it just continues the text as a base LM would.
3. **A real bug: BOS/EOS/PAD token IDs are outside the model's vocabulary.**
   `bos_token_id=199998`, `eos_token_id=200002`, `pad_token_id=199999`, but
   the model's `vocab_size=199998` (valid embedding/output indices
   `0..199997`). The model can never be fed these tokens (confirmed
   `IndexError`) and can never generate EOS as output (the output layer has
   no logit position for it). This explains why `generate()` has no
   EOS-stopping logic — it's structurally impossible today.

None of these are fixed by better sampling, more agent scaffolding, or a
bigger prompt. They require training changes (Phase 6/7, not yet done).

## Evidence

### A. Sampling vs. model quality (Phase 3)

Ran greedy (T=0.0) and T={0.2, 0.5, 0.7, 1.0} on 7 prompts × 2 checkpoints.
Sample of the released 100M checkpoint (`mythos_100m_mixed_80k.pt`):

| prompt | T | mean token prob | entropy (bits) | distinct-1 | max repeat run | output (truncated) |
|---|---|---|---|---|---|---|
| hello | 0.0 | 0.221 | 7.88 | 0.53 | 2 | `的，因而非同意而非同意。2000年，英國國家大會議會議員會議在美國國會議會議員會議` |
| find_bug | 0.0 | 0.850 | 2.06 | 0.45 | 1 | `` \n    return '\n```{'contexts': , 'labels': , 'meshes': , 'reasoning_req `` |
| find_bug | 0.5 | 0.945 | 2.06 | 0.45 | 1 | *(byte-identical continuation to T=0.0 and T=0.2)* |

And the experimental `mythos_100m_v2` checkpoint (1000 training steps):

| prompt | T | mean token prob | entropy (bits) | distinct-1 | max repeat run | output (truncated) |
|---|---|---|---|---|---|---|
| hello | 0.0 | 0.111 | 9.62 | **0.07** | **6** | `. 5. 5. 5. 5. 5. 5. 5. 5. 5.    5.      ` |
| empty | 0.0 | 0.064 | 11.36 | 0.15 | 1 | `, and the time of the time of the time of the time of the time of the ` |

**Reading this**: per the diagnostic's own classification rule (spec Phase
3), greedy generation being incoherent classifies this as a model/training
problem, not a sampling problem — confirmed on both checkpoints. The
`find_bug` prompt is a striking case: **mean token probability of 0.85-0.97
at every temperature from 0.0 to 0.7**, with entropy collapsed to ~2 bits —
the model isn't uncertain, it's *confidently* emitting the same memorized
`{'contexts': ..., 'labels': ..., 'meshes': ...}` QA-dataset-schema template
regardless of what was asked. That's a classic sign of an undertrained model
that has memorized one high-frequency template from its training data as a
strong attractor, not random noise from sampling. Full data:
`diagnostics/diagnostics_report.json`.

No NaN/Inf was observed in logits across all 70 runs (both checkpoints) —
the weights themselves are numerically healthy; this rules out a
gradient-explosion/corrupted-checkpoint explanation.

### B. Tokenizer (Phase 1/2)

- `openai/gpt-oss-20b` AutoTokenizer, `tok.vocab_size` (base BPE) = 199,998,
  `len(tokenizer)` (incl. added/special tokens) = 200,019.
- Round-trip test (`text -> encode -> decode`) on 5 samples including a
  Python function and mixed-indentation/unicode text: **5/5 exact matches**.
  The tokenizer itself is not corrupted.
- `bos_token_id=199998`, `eos_token_id=200002`, `pad_token_id=199999` — see
  the bug above. These are the tokenizer's *added* special tokens, all
  landing in the `199998..200018` range that the model's `vocab_size=199998`
  excludes entirely.

### C. Architecture (Phase 1, from source inspection, not guessed)

- `open_mythos/main.py`: Recurrent-Depth Transformer — `prelude` (N dense
  transformer layers) → `RecurrentBlock` (1 shared block looped
  `n_loops` times, with `LTIInjection` for stable state update and
  `ACTHalting` for adaptive per-position compute) → `coda` (N dense layers).
  Attention is swappable per config: `GQAttention` (`wq/wk/wv/wo`, tested
  this session against the released checkpoint with 0 missing/unexpected
  keys) or `MLAttention` (DeepSeek-V2-style latent KV compression, used by
  every *other* variant in `variants.py` — 1b/3b/10b/50b/100b/500b/1t — none
  of which have a real trained checkpoint). FFN is dense SwiGLU in
  prelude/coda, DeepSeek-style MoE (shared + routed experts) in the
  recurrent block. RoPE for position, RMSNorm throughout, weight-tied
  embed/head.
- Released checkpoint: 112,898,594 params (GQA, `dim=512, n_heads=8,
  n_kv_heads=4, max_seq_len=256, max_loop_iters=4, prelude=2, coda=2,
  n_experts=4, expert_dim=256`).
- Inference (`OpenMythos.generate()`): KV-cached autoregressive decode,
  `top_k` only (no `top_p`, no `repetition_penalty` anywhere in the
  codebase), **no EOS-based stopping** (structurally blocked by the vocab
  bug above), and **had a division-by-zero crash at `temperature=0.0`**
  (see Minimal Fixes below).

### D. Training (Phase 4, from source inspection — do not conflate the two pipelines)

- The one production pretraining script in the repo,
  `training/3b_fine_web_edu.py`, is **GPU/FSDP-only** and targets
  `mythos_3b` (a different, MLA-attention config). It did not and could not
  have produced the released 100M **GQA** checkpoint — that checkpoint's
  actual training code/log is **not present anywhere in this repository or
  on this VPS**. Its dataset ("mixed"), step count context, LR schedule,
  and hyperparameters are undocumented; the "80k" in its filename is the
  only provenance available.
- That script does have warmup+cosine LR, gradient clipping (`max_norm=1.0`),
  and CUDA-only bf16/fp16 autocast — reasonable choices, but irrelevant to
  what actually trained the released checkpoint.
- **No validation split or held-out perplexity tracking exists anywhere in
  the production path.** Only the benchmarking demo script
  (`tests/small_benchmark.py`) has a held-out eval concept, and it has never
  been used to select a checkpoint.
- My own experimental continued-training this session (documented in
  `training/100m_v2_smoke_results.md`) used FineWeb-Edu (general English web
  text) — **zero Python/code content** — for up to 1000 cumulative steps
  (~800k tokens). Far below any reasonable token budget even for the
  smaller 10.5M-71M active core (Chinchilla-style ~20 tokens/param
  suggests ≥200M tokens as a floor), and confirms the qualitative failure
  mode (incoherent under greedy decoding) is not specific to one dataset —
  it reproduces under a completely different, English-only corpus too.

### E. Coding data (Phase 6, inspection only — no training done)

Neither training corpus available to us contains meaningful code:
FineWeb-Edu (my experiments) is general web text; the released checkpoint's
"mixed" dataset is undocumented but its generation behavior (drifting into
Chinese legal/political text and a QA-dataset JSON template, never into
code) gives no evidence of a code-heavy mix either. **There is currently no
curated Python/code training corpus anywhere in this project.** This
directly explains why `write_add_fn` and `find_bug` never produce a single
valid Python statement across all 10 generations tested for each (2
checkpoints × 5 temperatures) — the model has essentially never seen code.

## Minimal fixes made this session (transparently reported, as instructed)

Only one production code change, to inference (Layer F), needed just to run
the Phase-3 greedy-decoding diagnostic the spec requires:

- **`open_mythos/main.py`, `OpenMythos.generate()`**: `temperature=0.0`
  divided logits by zero, producing NaN/Inf softmax probabilities and
  crashing `torch.multinomial` (confirmed `RuntimeError`). Added a genuine
  argmax branch for `temperature <= 0.0`; sampling behavior for
  `temperature > 0` is byte-for-byte unchanged. Verified: deterministic
  across repeated calls, all 123 existing tests still pass, no other
  behavior touched.

**Nothing else was changed**: no architecture, tokenizer, threshold,
inference-sampling default, or production checkpoint was modified. The
`diagnose_generation.py` script uses its own local instrumented copy of the
decode loop (for per-step probability/entropy stats) rather than modifying
the library's `generate()` further.

## Recommended training changes (NOT implemented — Phase 6/7, deferred)

In priority order:

1. **Fix `vocab_size` to include the tokenizer's special tokens** (e.g.
   `len(tokenizer)=200019`, rounded up) before any further pretraining.
   Without this, EOS-based stopping is impossible by construction, and any
   future SFT stage has no valid end-of-turn token to train on.
2. **Insert real BOS/EOS at document boundaries** in the training data
   pipeline once (1) is fixed — currently absent (couldn't have been done
   correctly with the out-of-range ids anyway), so document boundaries are
   currently invisible to the model, likely contributing to context
   bleeding between unrelated training documents.
3. **Scale training tokens by 2-3 orders of magnitude** before expecting
   basic fluency. Document and version the data mixture explicitly instead
   of an opaque "mixed" label — include a real held-out validation split
   and track perplexity, selecting checkpoints by it instead of just saving
   periodically.
4. **Add a curated Python/code component** to the data mixture once base
   fluency is established (Gate 1-2) — prioritize quality (real, working,
   documented code + comments + a modest amount of error-message/traceback
   text) over raw volume, per the spec's own guidance.
5. **Only after (1)-(4) show healthy greedy continuations**, build an actual
   instruction-tuning/SFT stage with real (instruction, response) pairs —
   none exist today.

## Coding-agent architecture (Phase 8-9, designed — not built)

Proposed, matching the spec's layering, to be implemented only once Gates
1-4 below pass:

```
Chat CLI -> Context Manager -> Planner -> Model Inference -> Tool Router
                                                                 |
                            read_file / list_files / search_code /
                            run_python / run_pytest / apply_patch /
                            git_diff / git_status
```

- Tools return real stdout/stderr/exit codes; the model never fabricates a
  tool result (this needs the model to reliably emit a structured
  tool-call format, which in turn needs SFT — another reason this is
  gated behind training, not built first).
- Command safety: `SAFE` (read-only: `pwd`, `ls`, `grep`, `git status`,
  `git diff`, `python -m py_compile`, `pytest`) always runs; `REQUIRES
  APPROVAL` (`rm`, `git reset`, destructive `git checkout`, package
  install, service restart, network/DB ops) is shown to the user before
  execution, never silently run.
- Context management: system prompt + rolling conversation history +
  targeted file/tool-result context, with truncation and retrieval instead
  of dumping the whole repo into every prompt — bounded by the current
  model's tiny `max_seq_len=256` regardless (another reason to fix training
  before building this: a 256-token context can't hold a file, a diff, and
  a conversation at once).

## Quality gates (Phase 13) — measured, not assumed

| Gate | Status | Evidence |
|---|---|---|
| 1. Coherent basic conversation | **FAIL** | Greedy "hello" → repetitive/incoherent on both checkpoints (distinct-1 as low as 0.07) |
| 2. Correct Python syntax generation | **FAIL** | 0/10 `write_add_fn` generations (2 checkpoints × 5 temps) contain a valid Python function |
| 3. Basic Python debugging | **FAIL** | `find_bug` output is either the same memorized JSON-schema loop or unrelated prose; never engages the actual bug |
| 4. Understand tracebacks | **NOT YET TESTABLE** | No SFT/instruction signal exists for this format; not meaningfully assessable until Gate 1-3 pass |
| 5-8 (file edit, run tests, multi-step debug, multi-turn context) | **NOT ATTEMPTED** | Building these against an incapable base model would hide the real problem — the spec's own stated principle |

## What was deliberately NOT done this turn

Per the spec's explicit "do NOT immediately start retraining" and "only
after diagnosis is complete should implementation begin": `coding_eval.py`,
`training_quality_report.py`, `mythos_cli.py`, `agent_tools.py`,
`context_manager.py`, `tool_router.py`, `safety_policy.py`, and their tests
were **not created**. Building an agent/tool layer on top of a model that
fails Gate 1-3 would be exactly the "hide model-quality problems behind an
agent wrapper" anti-pattern the spec warns against. Recommend: fix the
vocab_size/EOS bug and re-run `diagnose_generation.py` after a real,
larger, validated pretraining run; only build the agent layer once Gates
1-4 pass with measured evidence, same as this report.
