# CoLLM-QFormer (`QFormerRec`)

<https://github.com/htainvn/QFormerRec>

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
  metrics.py                       UAUC that does not depend on the sklearn version
  compat.py                        import shims, environment check, live output + file log
  tasks/rec_qformer_task.py        eval + grad clipping + timing
  runners/runner_qformer.py        per-group lr_scale, grouped sampler wiring, diagnostics
scripts/
  pretrain_mf.py                   stage 0 (parameterised port of CoLLM's MF baseline)
  build_memory.py                  stage 0b: the train-only fitted objects + item_in_train
  check_pit_history.py             proves the per-row history is point-in-time, not leakage
  s3_sync.sh                       push/pull checkpoints to S3-compatible storage
  doctor.py                        one-command setup diagnostic
  smoke_test.py                    75 CPU checks, no LLM needed
  integration_test.py              end to end on a tiny random LLaMA, incl. 21 ablations
train_configs/                     stage1/2/3 × {ml1m, amazon}
prompts/                           short / terse / +titles / tallrec, per dataset
train_qformer.py                   entry point for all stages
```

`QFormerRec` is an **overlay**: it imports `minigpt4.*` from `$COLLM_ROOT` and registers
its model / task / runner / builders on import. **No file in the CoLLM checkout is
modified**, so the baseline remains reproducible from the same tree.

## Quick start

Clone both repos side by side; nothing needs installing, `train_qformer.py` puts both
trees on `sys.path`.

```bash
git clone https://github.com/zyang1580/CoLLM.git        # the baseline
git clone https://github.com/htainvn/QFormerRec.git     # this work
cd QFormerRec
pip install -r requirements.txt   # transformers 4.36.2 + peft 0.9.0; see the file's comments

export COLLM_ROOT=../CoLLM
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
4. **"LoRA-only" checkpoints contain the entire frozen backbone.**
   `_save_checkpoint` filters frozen weights by *key name*, but `named_parameters()`
   deduplicates by tensor identity while `state_dict()` does not — and peft leaves the
   backbone reachable under both `llama_model.*` and `llama_model_lora.base_model.*`. The
   second copy is never in the name→requires_grad map, so it survives the filter: an
   observed stage-1 checkpoint was **13.5 GB with 0.2 % of it LoRA**. Filtering on
   `id(tensor)` gives 3.2 MB for the same content, and reloads bit-exactly.
5. **Early stopping is hard-coded and silent** — patience 20 *validations*
   (`20 x iters_per_epoch` steps) with no report of how close the stop is, so a run that
   peaked early is indistinguishable from one still improving. Now `run.patience`, with the
   gap logged after every validation.
6. **Per-group learning rates are silently destroyed** — CoLLM's LR schedulers assign
   `lr` to every param group. Hence `lr_scale` + `linear_warmup_cosine_lr_scaled`.
7. **No gradient clipping** in the training loop; added at 1.0, with `scaler.unscale_`
   first so it is correct under fp16 AMP.
8. `RunnerBase.model` returns `None` when the model already sits on the target device, and
   `PrefetchLoader` hard-requires CUDA — both blocked CPU debugging.
9. **CoLLM's `requirements.txt` cannot be installed on a current Python.** Colab is on
   3.12; `transformers==4.28.0` requires `tokenizers<0.14`, which has no cp312 wheel, so
   pip attempts a Rust source build and fails. `scikit-learn==1.2.2` compiles from source,
   and `decord` has no wheel for ≥3.11 at all. This project pins `transformers==4.36.2` +
   `peft==0.9.0` (the widest window where CoLLM's vendored LLaMA and
   `prepare_model_for_int8_training` both still work), leaves scikit-learn/numpy unpinned,
   and stubs `decord` in [`qformerrec/compat.py`](qformerrec/compat.py).
   `check_environment()` names each of these failure modes instead of letting them surface
   as ImportErrors inside CoLLM.
10. **transformers 5.x silently NaNs CoLLM's vendored LLaMA.** Its loader reports
   `self_attn.rotary_emb.inv_freq | MISSING` and leaves the vendored
   `LlamaRotaryEmbedding`'s `cos_cached` / `sin_cached` *buffers* as uninitialised memory;
   4.x re-runs the module init instead. Every **parameter** stays finite, so the only
   symptom is `loss=nan` after a full model build. Colab preinstalls 5.13.1, so this is the
   default outcome if the pins are not applied. `check_vendored_llama()` reproduces it in
   under a second — via the realistic path (a checkpoint written by *transformers'* class,
   loaded by the *vendored* one) and by inspecting buffers, since neither a version test nor
   a parameter scan catches it — and the entry points refuse to start.

11. **peft ≥ 0.5 makes LoRA fp16, which `GradScaler` cannot train.** Vicuna is loaded
   `torch_dtype=float16`, and modern peft casts new adapters to the base layer's dtype, so
   in stages 1 and 3 every trainable tensor is fp16 — and `GradScaler` rejects fp16
   gradients in `step()` just as much as in `unscale_()`, so `amp: True` cannot start.
   CoLLM only escaped this by pinning peft 0.4.0, which left adapters in fp32. The runner
   casts trainable params back to fp32 (`run.fp32_trainable`, default on), which restores
   the precision CoLLM trained with and is the standard mixed-precision LoRA arrangement.

12. **CoLLM's warmup does not warm up when `warmup_steps > iters_per_epoch`.** It gates on
   the global step but ramps on the within-epoch step, so the lr sawtooths for the warmup
   period and then jumps to the full `init_lr` in one step — the shipped stage-1 setting
   (300 vs 100) hits this. The visible symptom is a run that improves for exactly
   `warmup_steps / iters_per_epoch` epochs, peaks, then decays to chance.
   `linear_warmup_cosine_lr_scaled` ramps on the global step instead.

And one bug of our own worth recording: `L_attn`'s Bhattacharyya coefficient needs an
`eps` inside the `sqrt`. Masked memory slots get exactly zero attention and
`d/dx √x → ∞` at 0, so the plain formula NaNs every gradient on essentially every real
batch. Same for `L_var`'s `std()` at exactly zero variance — the collapse case it exists
to punish. Both are covered by regression checks in `smoke_test.py`.
