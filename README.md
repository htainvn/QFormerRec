# CoLLM-QFormer (`QFormerRec`)

A candidate-aware **Q-Former CIE** for [CoLLM](https://github.com/zyang1580/CoLLM). It
replaces CoLLM's CIE module (MF embedding → single MLP → one soft token per user/item)
with a Q-Former that reads a per-user **collaborative memory bank** and emits `L`
**preference tokens**, letting the prompt drop the history-title list entirely.

The collaborative model stays **MF**. Everything CoLLM already does well — MF, the dataset
loaders, prompt splicing, LoRA wiring, the Yes/No answer read-out, AUC/UAUC — is reused.

**→ [`RUNSHEET_COLAB.md`](RUNSHEET_COLAB.md) is the operating manual.** Start there.

## Design

```
 PER ROW, point-in-time (the sample's own `his`, = CoLLM's <ItemTitleList>)
                    ┌─ 50 recent liked items ───┐
 MF (frozen or not) ├─ user emb ────────────────┤
                    ├─ 8 cosine-sim neighbours ─┤──► MemoryEncoder ──► mem (B,65,d_q)
 trainable protos   ├─ 3 genre prototypes ──────┤    + type & recency emb   + mask
                    └─ 3 user-cluster protos ───┘    + per-slot prior    + type_ids (B,65)
 TRAIN-SPLIT-ONLY fitted objects (KNN graph, KMeans centroids, genre means)

 candidate item ──► per-token FiLM ──► L queries ──┐
                    (zero-init)                    │
                                    2 × [self-attn → cross-attn(+type_bias+log prior) → FFN]
                                                   │
                                    Z (B,L,d_q) ──►├─ LLMProjector (RMS-matched) ─► <PrefTokens>
                                                   └─ CFAuxHead ─► s_cf  (deep supervision)
```

Prompt (stage 2/3 and all evaluation) — no history titles:

```
#Question: A user's viewing preferences are encoded in the features <PrefTokens>, and the
user's identity is encoded in the feature <UserID>. Using all available information, make
a prediction about whether the user would enjoy the movie titled <TargetItemTitle> with
the feature <TargetItemID>? Answer with "Yes" or "No". \n#Answer:
```

86.5 tokens vs CoLLM-MF's 205.2 — measured, see the runsheet §6.4.

History slots are **per-row and point-in-time**: the sample's own `his` list,
most-recent-first, which is exactly what CoLLM renders into `<ItemTitleList>`.
Nothing about them is precomputed. CoLLM caps that list at 10 only because it is
rendered as *text*; memory slots cost zero prompt tokens, so `k_hist` is a config
value and defaults to 50 (ML-1M mean history: 38 train / 75 test). A history item
whose MF row was never trained (0.8 % ML-1M, 10 % Amazon-Book) gets a single
learned `unk_item` vector and the distinct slot type `SLOT_HIST_UNK`, so
`type_bias` can learn to discount it.

The other slot types stay **train-split-only**: the KNN neighbour graph, the
KMeans centroids and the genre prototype means are *fitted objects*, and fitting
them across valid/test would use other users' futures — that would be real
leakage. `history_source: train_only` reverts the history slots to the old
per-user train-split lookup; that is the `-pit-history` ablation row.

Losses: `L_bce` (CoLLM's, unchanged) + `0.5·L_rank` (within-user pairwise — the direct
lever on UAUC) + `0.2·L_cf` + `0.1·L_div` + `0.05·L_attn` + `0.05·L_var`
(the three anti-collapse terms) + optional `L_align`.

## Layout

```
qformerrec/
  models/qformer_cie.py            all new modules + losses (no CoLLM imports)
  models/minigpt4rec_qformer.py    MiniGPT4RecQFormer(MiniGPT4Rec_v2), splicing, forward, diagnostics
  datasets/rec_datasets_qformer.py dataset + builders (fixes the ML-1M warm/cold split)
  datasets/samplers.py             UserGroupedBatchSampler (L_rank needs same-user pairs)
  tasks/rec_qformer_task.py        eval + UAUC-safe metric + grad clipping + timing
  runners/runner_qformer.py        per-group lr_scale, grouped sampler wiring, diagnostics
scripts/
  pretrain_mf.py                   stage 0 (parameterised port of CoLLM's MF baseline)
  build_memory.py                  stage 0b: the train-only fitted objects + item_in_train
  check_pit_history.py             proves the per-row history is point-in-time, not leakage
  smoke_test.py                    50 CPU checks, no LLM needed
  integration_test.py              end to end on a tiny random LLaMA, incl. 18 ablations
train_configs/                     stage1/2/3 × {ml1m, amazon}
prompts/                           short / terse / +titles / tallrec, per dataset
train_qformer.py                   entry point for all stages
```

`QFormerRec` is an **overlay**: it imports `minigpt4.*` from `$COLLM_ROOT` and registers
its model / task / runner / builders on import. **No file in the CoLLM checkout is
modified**, so the baseline remains reproducible from the same tree.

## Quick start

```bash
export COLLM_ROOT=/path/to/CoLLM
python scripts/pretrain_mf.py   --data_dir data/ml-1m/ --out ckpt/mf_ml1m_d256.pth
python scripts/build_memory.py  --data_dir data/ml-1m/ --mf_ckpt ckpt/mf_ml1m_d256.pth \
                                --out ckpt/memory_index_ml1m.pkl --dataset ml1m \
                                --genre_source metadata --movies_dat raw/ml-1m/movies.dat
python scripts/check_pit_history.py --data_dir data/ml-1m/ --k_hist 10 20 50
python scripts/smoke_test.py --memory_index ckpt/memory_index_ml1m.pkl
python train_qformer.py --cfg-path train_configs/stage1_lora_ml1m.yaml   # cache once
python train_qformer.py --cfg-path train_configs/stage2_qformer_ml1m.yaml
python train_qformer.py --cfg-path train_configs/stage3_qformer_ml1m.yaml
```

Targets: UAUC > 0.6956 (ML-1M) and > 0.6319 (Amazon-Book), AUC not below CoLLM-MF T2.
**ML-1M UAUC is the hard target** — budget the tuning there.

## Fixes to inherited CoLLM behaviour

Found while building this; all are load-bearing.

1. **UAUC returns `NaN` on scikit-learn ≥ ~1.3.** `uAUC_me` skips single-class users via a
   bare `except` around `roc_auc_score`, which now *returns* `nan` instead of raising. On
   the released ML-1M valid split that is 44 of 283 evaluable users, so the primary metric
   — the one checkpoints are selected on — silently becomes NaN.
2. **Single-process evaluation crashes.** `RecBaseTask.evaluation`'s non-distributed branch
   calls `results_logits_.cput()` and assigns a 3-tuple to a scalar, so it fails whenever
   `world_size == 1` — exactly the single-GPU setup.
3. **The ML-1M warm/cold split cannot run.** Its builder points at a
   `test_warm_cold_ood2.pkl` that is absent from the released data and filters on a `warm`
   column that is also absent. Derived from `not_cold` here (4153 warm + 3178 cold = 7331).
4. **Per-group learning rates are silently destroyed** — CoLLM's LR schedulers assign
   `lr` to every param group. Hence `lr_scale` + `linear_warmup_cosine_lr_scaled`.
5. **No gradient clipping** in the training loop; added at 1.0, with `scaler.unscale_`
   first so it is correct under fp16 AMP.
6. `RunnerBase.model` returns `None` when the model already sits on the target device, and
   `PrefetchLoader` hard-requires CUDA — both blocked CPU debugging.

And one bug of our own worth recording: `L_attn`'s Bhattacharyya coefficient needs an
`eps` inside the `sqrt`. Masked memory slots get exactly zero attention and
`d/dx √x → ∞` at 0, so the plain formula NaNs every gradient on essentially every real
batch. Same for `L_var`'s `std()` at exactly zero variance — the collapse case it exists
to punish. Both are covered by regression checks in `smoke_test.py`.
