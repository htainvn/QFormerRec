# CoLLM-QFormer — Colab runsheet

Copy-pasteable cells, in order. Every command here was exercised locally on the real
ML-1M split (with a tiny stand-in LLaMA in place of Vicuna); the numbers quoted as
"measured" come from those runs.

**Read §0 first — the GPU you get decides which stages are feasible.**

---

## 0. Pick the runtime, and know the constraint

The design spec assumes **1× A100 80 GB**. Colab does not offer that; Colab Pro/Pro+ A100s
are **40 GB**. What fits:

| Colab GPU | VRAM | Stage 1 (long prompt) | Stage 2/3 (short prompt) | Verdict |
|---|---|---|---|---|
| A100 | 40 GB | `batch_size_train: 16` | `U=8 × m=6` (B=48) | **use this** |
| L4 | 22.5 GB | `batch_size_train: 8` + grad ckpt | B=32 (`U=8 × m=4`) + grad ckpt | slow but possible |
| T4 | 16 GB | no | no | fp16 7B training will not fit |

Budget arithmetic: bf16/fp16 Vicuna-7B weights ≈ 13.5 GB; gradients must flow from the
answer logit back to the input soft tokens, so activations for all 32 layers are retained
— that is the dominant term (~3.7 MB per token per sample). Stage 2 at B=48 × ~90 tokens
≈ 12 GB activations → ~26 GB total, comfortable on 40 GB. Stage 1 at B=32 × ~205 tokens
≈ 24 GB → ~38 GB, too tight on a 40 GB card, hence `batch_size_train: 16` there.

```python
!nvidia-smi
import torch; print(torch.__version__, torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory/1e9, "GB")
```

**Colab sessions die.** Put `output_dir` on Drive and use `resume_ckpt_path` (§7).

---

## 1. Environment

The version pins matter. Two of them are load-bearing:

* `transformers==4.28.0` — CoLLM vendors its own `modeling_llama.py` against this API.
* `peft==0.4.0` — CoLLM imports `prepare_model_for_int8_training`, which was renamed in
  peft ≥ 0.10, and later peft versions also change the LoRA state-dict key layout.

`scikit-learn` is pinned for a subtler reason: see §5.3.

```python
from google.colab import drive; drive.mount('/content/drive')

!pip -q install "transformers==4.28.0" "peft==0.4.0" "omegaconf==2.3.0" \
    "webdataset==0.2.48" "timm==0.6.13" "scikit-learn==1.2.2" \
    sentencepiece decord iopath opencv-python-headless
# torch / torchvision / pandas / numpy: keep Colab's preinstalled versions.
# If numpy 2.x causes import errors in this stack: !pip -q install "numpy<2"
```

```python
%cd /content
!git clone https://github.com/zyang1580/CoLLM.git          # or copy your own checkout
# put the QFormerRec package next to it
!cp -r /content/drive/MyDrive/QFormerRec /content/QFormerRec

import os
os.environ["COLLM_ROOT"] = "/content/CoLLM"                # read by train_qformer.py
%cd /content/QFormerRec
!mkdir -p /content/data /content/ckpt /content/logs
```

`QFormerRec` is a **standalone overlay package**: it imports `minigpt4.*` from
`COLLM_ROOT` and registers its own model / task / runner / builders on import. **No file
in the CoLLM tree is modified**, so the baseline stays reproducible from the same
checkout.

---

## 2. Data

```python
%cd /content/data
!unzip -o -q /content/CoLLM/collm-datasets/ml-1m.zip        # -> ml-1m/{train,valid,valid_small,test}_ood2.pkl
!unzip -o -q /content/CoLLM/collm-datasets/amazon_book.zip  # -> book/...
!ls -la ml-1m book
```

Verified schema of the released pickles (identical for both datasets):
`uid, iid, label, timestamp, his, his_title, title, flag, not_cold`.

| split | ML-1M rows | users | items | pos rate |
|---|---|---|---|---|
| train | 33 891 | 740 | 3 087 | 0.537 |
| valid | 10 401 | 356 | 2 506 | 0.528 |
| valid_small | 5 200 | 331 | 1 985 | 0.531 |
| test | 7 331 | 320 | 2 203 | 0.545 |

`user_num = 839`, `item_num = 3256` (index 0 is padding for both). Splits are
timestamp-ordered: `train.ts.max() < valid.ts.min() < test.ts.min()` — checked, and
asserted again by `build_memory.py`.

**Amazon-Book RAM warning.** `book/train_ood2.pkl` is 500 MB on disk and expands to
several GB as Python objects. Use a **high-RAM** runtime for the Amazon memory build.

### 2.1 ML-1M genres (optional but recommended)

`genre_source: metadata` needs the original `movies.dat`. The item titles in the CoLLM
pickles are byte-identical to ML-1M's, so the join is on the title string:

```python
%cd /content/data
!wget -q https://files.grouplens.org/datasets/movielens/ml-1m.zip -O ml1m_raw.zip
!unzip -o -q ml1m_raw.zip 'ml-1m/movies.dat' -d raw && ls raw/ml-1m/movies.dat
```

`build_memory.py` asserts that >80 % of item titles match, so a bad file fails loudly
rather than silently producing empty genres. If you cannot get `movies.dat`, use
`--genre_source item_kmeans` (both paths are implemented and tested).

Amazon-Book **has no category field** in the released pickles (columns confirmed above),
so Amazon must use `--genre_source item_kmeans` — KMeans(32) over the MF item embeddings
as pseudo-genres. State this in the writeup.

---

## 3. Stage 0 — pretrain MF

CoLLM's `baseline_train_mf_ood.py` has its paths and hyper-parameters hard-coded inside a
commented-out `__main__`; `scripts/pretrain_mf.py` is the same model/loss/optimiser/
early-stopping rule with arguments. Defaults reproduce CoLLM's tuned setting
(d=256, Adam lr 1e-3, wd 1e-4, batch 2048, early stop on valid AUC, patience 100).

```python
%cd /content/QFormerRec
!python scripts/pretrain_mf.py --data_dir /content/data/ml-1m/ \
    --out /content/ckpt/mf_ml1m_d256.pth        # ~5 min on any GPU

!python scripts/pretrain_mf.py --data_dir /content/data/book/ \
    --out /content/ckpt/mf_amazon_d256.pth      # ~20 min
```

**Sanity gate:** the printed `best` must land near CoLLM's published MF row —
ML-1M `test_auc ≈ 0.6482`, `test_uauc ≈ 0.6361`; Amazon `0.7134 / 0.5565`. If it does
not, stop: everything downstream inherits this checkpoint.

---

## 4. Stage 0b — build the collaborative memory index

```python
!python scripts/build_memory.py \
    --data_dir /content/data/ml-1m/ --mf_ckpt /content/ckpt/mf_ml1m_d256.pth \
    --out /content/ckpt/memory_index_ml1m.pkl --dataset ml1m \
    --genre_source metadata --movies_dat /content/data/raw/ml-1m/movies.dat

!python scripts/build_memory.py \
    --data_dir /content/data/book/ --mf_ckpt /content/ckpt/mf_amazon_d256.pth \
    --out /content/ckpt/memory_index_amazon.pkl --dataset amazon_book \
    --genre_source item_kmeans --n_user_clusters 256
```

This artifact holds only the **train-split-only fitted objects** — the KNN neighbour graph,
the KMeans centroids, the genre prototype means — plus `item_in_train` (which item ids MF
actually saw). It does **not** hold the history slots: those are per-row and point-in-time,
read from each sample's own `his` column by the dataset, with no precomputation (§4.1).
The one exception is a small train-only `hist_items` table kept solely for the
`-pit-history` ablation.

Measured on ML-1M: 718/839 users have ≥1 neighbour (mean top-1 cosine 0.33), 3087/3256
items (94.8 %) were seen in training. Runs in well under a minute.

Slot budget is set by the **model** config, not this artifact:
`1 user + k_hist + 8 neighbours + 3 genres + 3 clusters`. With the default `k_hist: 50`
that is **65 slots**.

### 4.1 History slots are point-in-time, and why that is not leakage

History slots come from **each row's own `his` column** — the same list CoLLM renders into
`<ItemTitleList>`. So the Q-Former reads exactly the user behaviour the baseline reads, and
the comparison is apples-to-apples on that axis. Nothing is precomputed for them.

`his` is a point-in-time history, verified against the released ML-1M pickles
(2 426 445 history entries across all splits):

| Property | Measured |
|---|---|
| entries that are positives (`label == 1`) | **100 %** |
| entries dated strictly before their own row | 99.28 % |
| entries with the same timestamp as their row (ties) | 0.72 % |
| entries dated **after** their own row | **0** |
| index 0 at a non-leading position | **0 rows** (it is pure left padding) |

Zero future items. The 0.72 % remainder are same-second ties, not leakage. Because the
splits are time-ordered, a *test* row's history does contain valid- and test-period items —
that is what "point-in-time" means, and it is what the baseline consumes too.

**What stays train-split-only, and must:** the KNN neighbour graph, the KMeans centroids
and the genre prototype means. Those are *fitted* objects; fitting them over valid/test
would mix in **other users' futures**, which is real leakage and a different thing entirely
from a user's own past. `build_memory.py` still asserts
`train.ts.max() <= valid.ts.min() <= test.ts.min()` for them. Item titles (for the genre
map) are read from all splits — item metadata, not interactions.

**Untrained MF rows.** 99.2 % of ML-1M history items and 90.0 % of Amazon-Book ones appear
somewhere in the train split, so their MF embedding is trained. The rest would otherwise
contribute a randomly-initialised row, i.e. pure noise. Those slots instead get a single
learned `unk_item` vector and the distinct slot type `SLOT_HIST_UNK`, so `type_bias` can
learn to discount them. The rate is logged every epoch (`hist_unk_rate`) and per eval
split — measured on ML-1M at `k_hist=50`: 0.96 % on the cold test split, 0.57 % on warm.

### 4.2 Verify it before retraining — mandatory gate

```python
!python scripts/check_pit_history.py --data_dir /content/data/ml-1m/ --k_hist 10 20 50
!python scripts/check_pit_history.py --data_dir /content/data/book/  --k_hist 10 50
```

Asserts (fatal) the split ordering, that `his` is positives-only, per-row causality with
ties allowed, and that index 0 is leading-padding only. Then it prints the **provenance of
the `k_hist` items actually used** — the check that the per-row path took effect at all:

```
k_hist split    train   valid    test  unk_item        filled
    10 test      9.3%   14.2%   76.5%    0.83%    9.37/10
    10 valid    18.9%   81.1%    0.0%    0.72%    9.20/10
    50 test     22.1%   19.8%   58.2%    0.80%   34.64/50
```

**If ML-1M test comes back mostly `train`, the change did not take effect.** Expected
roughly 9 % train / 14 % valid / 77 % test at `k_hist=10`; Amazon-Book should be near
70 / 16 / 14. The integration test asserts this independently.

## 5. The four stages

### 5.1 Stage 1 — LoRA warmup (train once per dataset, then cache forever)

Unchanged CoLLM stage 1: legacy `mini_gpt4rec_v2` arch, TALLRec prompt **with** history
titles, LoRA trainable, everything else frozen. It does not depend on any Q-Former
hyper-parameter, so every later run and every ablation reuses this one checkpoint. This
is the single biggest time saving available.

First point the configs at your Vicuna weights:

```python
!sed -i 's#/content/vicuna-7b-v0#/content/drive/MyDrive/vicuna-7b-v0#' train_configs/*.yaml
```

> **Vicuna version.** CoLLM's published numbers are on **Vicuna-7B v0**. Use a v0 copy if
> you have one. v1.1/v1.5 will shift absolute numbers, so if you switch you must re-run
> the CoLLM-MF and TALLRec baselines yourself — the table in the spec would no longer
> apply. You can also skip this stage entirely by dropping in CoLLM's published stage-1
> checkpoint (linked from their README) as `ckpt_lora`.

```python
!sed -i 's#batch_size_train: 32#batch_size_train: 16#' train_configs/stage1_lora_ml1m.yaml  # 40GB A100
!python train_qformer.py --cfg-path train_configs/stage1_lora_ml1m.yaml 2>&1 | tail -40
```

Then copy the winner somewhere permanent and pin it:

```python
import glob, shutil
best = sorted(glob.glob('/content/logs/stage1_ml1m/*/checkpoint_best.pth'))[-1]
shutil.copy(best, '/content/ckpt/stage1_lora_ml1m.pth')
!sed -i 's#ckpt_lora: .*#ckpt_lora: /content/ckpt/stage1_lora_ml1m.pth#' \
    train_configs/stage2_qformer_ml1m.yaml train_configs/stage3_qformer_ml1m.yaml
```

### 5.2 Stage 2 — train the Q-Former CIE on the short prompt

```python
!python train_qformer.py --cfg-path train_configs/stage2_qformer_ml1m.yaml
```

Trains: Q-Former, candidate-aware query generator, LLM projector, genre/cluster
prototypes, CoLLM's `<UserID>`/`<TargetItemID>` MLP. Frozen: Vicuna, LoRA, and (with
`freeze_rec: True`) MF.

**Keep `freeze_rec: True` for the whole search.** CoLLM notes that tuning the
collaborative model needs ≥5× the training effort on ML-1M. Unfreeze only for the final
2–3 runs.

### 5.3 What to watch in the log — this is the part that decides success

The runner prints a `[qformer-diag]` block every `diag_log_freq` steps and once per epoch,
and appends it to `<output_dir>/<job_id>/qformer_diagnostics.jsonl`:

```
[qformer-diag] epoch0
  token_cosine_offdiag=0.0400  pref_token_norm=0.1602  llm_emb_norm=0.1595
  z_std_mean=0.2075  rank_pairs_per_batch=15.5  proj_scale=0.0201
  hist_slots_filled=6.94  hist_unk_rate=0.0000  history_source=pit  k_hist=50
  loss_bce=0.6904 loss_rank=0.6929 loss_cf=0.6871 loss_div=0.0490 ...
  token0: attn_by_type=[user:0.091 hist:0.213 neighbor:0.193 genre:0.222 cluster:0.281] ...
```

Note on `hist_slots_filled`: on **train** batches it reads ~7, not the 37.7 per-row mean,
and that is correct — the user-grouped sampler draws *users* uniformly (mean history 10.4),
and within a user it favours earlier rows, which have shorter histories. On the eval splits
it reads ~33–36 out of 50. `hist_unk_rate` is ~0 on train batches by construction (every
train history item is a train item) and nonzero on valid/test.

| Watch | Healthy | If not |
|---|---|---|
| `pref_token_norm` vs `llm_emb_norm` | within 2× | the RMS matching is off — soft prompt will do nothing |
| `token_cosine_offdiag` | **< 0.4** | > 0.6 → raise `lambda_div` to 0.3 before touching anything else |
| `z_std_mean` | not trending to 0 | tokens are collapsing to a constant; check `lambda_var` |
| per-token `top3` attn types | **differ across tokens** | tokens are redundant; check `type_bias` is learning (lr too low?) |
| `rank_pairs_per_batch` | ≫ 0 (≈15 at U=4×m=4; ~72 at U=8×m=6) | the user-grouped sampler is off → `L_rank` has no signal |
| `hist_slots_filled` | ≫ 0 | 0 → `PitHistItems` is not reaching the model |
| `hist_unk_rate` | ~1 % (ML-1M), ~10 % (Amazon) | much higher → wrong `item_in_train`, i.e. a stale memory index |
| `history_source` | `pit` | `train_only` means you are running the ablation by accident |
| `valid_uauc` in `log.txt` | a real number | **NaN → see below** |

**The NaN-UAUC trap.** CoLLM's `uAUC_me` skips single-class users via a bare `except`
around `roc_auc_score`. scikit-learn ≥ ~1.3 *returns `nan`* for single-class input instead
of raising, so those users are no longer skipped and the mean comes out `NaN`. On the
released ML-1M valid split that is **44 of 283 evaluable users** — i.e. UAUC, the metric
this project is selected and reported on, silently becomes NaN, checkpoint selection never
fires, and every reported UAUC is garbage. Two independent defences are in place: the pin
`scikit-learn==1.2.2` in §1, and `uauc_score()` in the task, which filters those users
explicitly and is numerically identical to `uAUC_me` under sklearn ≤ 1.2. The eval log
also reports `uauc_users_scored` / `uauc_users_skipped` so you can see it happening.

### 5.4 Stage 3 — joint tuning with LoRA at lr/10

Mirrors CoLLM's T1/T2 tuning, which is where its best numbers came from. **Do not skip
it.** Stage 3 needs *both* checkpoints: the stage-2 best (Q-Former etc.) and the stage-1
LoRA, because `_save_checkpoint` only stores parameters that were trainable in that stage.

```python
best2 = sorted(glob.glob('/content/logs/stage2_ml1m/*/checkpoint_best.pth'))[-1]
!sed -i "s#ckpt: /content/logs/stage2_ml1m/checkpoint_best.pth#ckpt: $best2#" \
    train_configs/stage3_qformer_ml1m.yaml
!python train_qformer.py --cfg-path train_configs/stage3_qformer_ml1m.yaml
```

Check the `Load ckpt_lora` / `Load ckpt` lines report `n_unexpected=0`. A nonzero count
means a key-name mismatch and the weights did **not** load.

Set `freeze_rec: False` in stage 3 for the T2-style variant.

### 5.5 Evaluation — overall, warm and cold

```python
best3 = sorted(glob.glob('/content/logs/stage3_ml1m/*/checkpoint_best.pth'))[-1]
!python train_qformer.py --cfg-path train_configs/stage3_qformer_ml1m.yaml \
    --options run.evaluate=True model.ckpt=$best3 \
              "run.test_splits=[test,test_warm,test_cold,valid]" \
              run.output_dir=/content/logs/eval_ml1m
```

Warm/cold use CoLLM's own `not_cold` flag on the test file (warm = user **and** item each
appear ≥3× in train). CoLLM's ML-1M builder pointed at a `test_warm_cold_ood2.pkl` that
does not exist in the released data and filtered on a `warm` column that is also absent,
so that split raised `FileNotFoundError`; the builder here derives it from `not_cold`, as
CoLLM's own Amazon path does. Measured partition on ML-1M test: **4153 warm + 3178 cold =
7331**, exact.

Each split logs `auc`, `uauc`, `mean_prompt_tokens`, `eval_seconds`, `ms_per_sample` and
`token_cosine_offdiag` — that is the efficiency table, no extra instrumentation needed.

---

## 6. Search order, ablations, and the honest budget

### 6.1 Order (stop as soon as the target is met)

Do **all** search on ML-1M only (`freeze_rec: True`, 1 seed). Nothing is lost
scientifically: CoLLM-MF T2 already clears the Amazon-Book UAUC bar, so **ML-1M UAUC is
the hard target**.

| # | Knob | Values | How |
|---|---|---|---|
| 1 | `L` | 2, 4, 8 | `--options model.qformer.n_query=8` |
| 2 | `λ_rank` | 0, 0.25, 0.5, 1.0 | `--options model.loss.lambda_rank=0.25` |
| 3 | `k_hist`, `k_neighbor` | **10/20/50**, 0/4/8 | `--options model.qformer.memory.k_hist=20` |
| 4 | `λ_div` | 0.05, 0.1, 0.3 | `--options model.loss.lambda_div=0.3` |
| 5 | Q-Former layers | 1, 2, 3 | `--options model.qformer.n_layers=3` |

`k_hist` needs **no** index rebuild — history is per-row now. It is capped only by the
dataset's `pit_hist_width` (default 50); going beyond that raises a clear error rather than
truncating silently. `k_neighbor`/`k_genre`/`k_cluster` can only shrink below what the
artifact holds.

Then **3 seeds for the final ML-1M row only** (`--options run.seed=43`). ML-1M has 839
users and swings ±0.004 UAUC across seeds — a single run beating BinLLM by 0.002 is not a
result.

Amazon-Book gets the **winning config only**: 1 seed for the main table, plus 1 confirming
seed if time allows. Say so in the writeup.

### 6.2 Acceptance criteria

UAUC > **0.6956** on ML-1M and > **0.6319** on Amazon-Book, with AUC not below CoLLM-MF T2
(0.7418 / 0.8288).

### 6.3 Every ablation row, as a command

All of these were verified to build, train a step and evaluate:

| Row | Override |
|---|---|
| `-memory` | `model.qformer.memory.k_hist=0 model.qformer.memory.k_neighbor=0 model.qformer.memory.k_genre=0 model.qformer.memory.k_cluster=0` |
| `-hist` / `-neighbor` / `-genre` / `-cluster` | the corresponding single `k_*=0` |
| **`-pit-history`** | `model.qformer.memory.history_source=train_only model.qformer.memory.k_hist=10` — quantifies what the point-in-time history is worth versus a per-user train-only lookup |
| `k_hist` sweep | `model.qformer.memory.k_hist=10` / `=20` / `=50` |
| `-candidate` | `model.qformer.use_candidate=False` |
| `L=1` | `model.qformer.n_query=1` |
| `-anticollapse` | `model.loss.lambda_div=0 model.loss.lambda_attn=0 model.loss.lambda_var=0` |
| `-rank` | `model.loss.lambda_rank=0` and `run.user_grouped_sampler=False` |
| `-norm-match` | `model.qformer.match_llm_norm=False` |
| `-slot-prior` | `model.qformer.use_slot_prior=False` |
| `+titles` | `model.n_titles_kept=3 model.prompt_path=prompts/qformer_movie_titles.txt` |
| `+align` | `model.loss.lambda_align=0.01` |
| stage-3 off | just report the stage-2 numbers |

The `-anticollapse` row is the one reviewers will look for: report both the metric drop
**and** `token_cosine_offdiag` from the diagnostics (expect it to jump toward 0.9).
`-rank` must also flip the sampler off, or you are ablating the loss while keeping the
batch construction it motivated.

### 6.4 Prompt length — the measured numbers

Measured with the real LLaMA tokenizer over 2000 ML-1M test rows (mean tokens, `<s>`
included):

| Prompt | Tokens | Note |
|---|---|---|
| TALLRec (history titles) | **178.2** | stage-1 prompt |
| CoLLM-MF (history titles + 2 ID tokens) | **205.2** | the baseline to beat |
| **ours, short** (`qformer_movie.txt`) | **86.5** | CoLLM's exact instruction wording — **2.4× shorter** |
| ours, short, `+titles` (3 kept) | 130.8 | cold-start fallback |
| ours, terse (`qformer_movie_terse.txt`) | **60.5** | reworded instruction |
| ours, short, `soft_token_glue: "."` | 89.5 | CoLLM's `.`-joined placeholder spacing |

Two honest notes. (a) The spec's "~55–70 tokens" is only reachable by **rewording the
instruction**; keeping CoLLM's wording — which is what makes the comparison clean — costs
86.5. Both prompts ship; report whichever you use, and if you use the terse one, say that
the instruction text changed. (b) CoLLM glues repeated placeholders with `"."`; consecutive
`<unk>` already tokenise to separate ids (verified), so the default here uses no glue and
saves `L-1` tokens. `soft_token_glue: "."` restores CoLLM's exact spacing.

### 6.5 Time budget

CoLLM reports 36 min (ML-1M) and 418 min (Amazon-Book) for full training on **2** GPUs;
on one, expect roughly double. Realistic plan: ~2 days of GPU time for the ML-1M search
plus ablations, ~1 day for the two Amazon-Book runs. That is why stage 1 is cached and why
the search never touches Amazon.

Note the configs here are *more* eval-efficient than CoLLM's: `iters_per_epoch: 200`
(ML-1M) instead of 50, so validation runs 4× less often for the same number of updates.
Stage 2 as shipped = 40 × 200 × 48 ≈ 11 real epochs over the 33 891 train rows.

---

## 7. Surviving a Colab disconnect

```yaml
run:
  output_dir: /content/drive/MyDrive/collm_logs/stage2_ml1m   # absolute path is honoured
  resume_ckpt_path: /content/drive/MyDrive/collm_logs/stage2_ml1m/<job_id>/checkpoint_best.pth
```

`resume_ckpt_path` restores model + optimizer + epoch. Note each launch creates a new
`<job_id>` subdirectory, so pass the *old* job's path when resuming.

---

## 8. Failure modes, in the order you will hit them

| Symptom | Cause | Fix |
|---|---|---|
| `valid_uauc: NaN` | sklearn ≥1.3 vs `uAUC_me` | pin `scikit-learn==1.2.2` (§5.3); already handled by `uauc_score` |
| metrics ≈ TALLRec, soft tokens ignored | norm mismatch | check `pref_token_norm` vs `llm_emb_norm` in the diag block |
| all `L` tokens identical | `λ_div` too low, or `type_bias` not learning | check `token_cosine_offdiag`; raise `lambda_div` to 0.3 |
| stage-2 loss flat | LoRA not actually frozen, or FiLM not zero-init | the log prints every trainable param with its group — read it |
| `n_unexpected` > 0 on ckpt load | wrong `ckpt`/`ckpt_lora` or a peft version change | check the printed key names |
| train UAUC ≫ val UAUC on ML-1M | 839 users, 33k rows — it overfits fast | fewer epochs, more dropout, `k_neighbor: 4` |
| cold UAUC collapses | the title-list removal | `model.n_titles_kept=3` + the `_titles` prompt (costs ~44 tokens) |
| UAUC > 0.75 on ML-1M | too good — suspect leakage | re-read §4.1; check `meta.train_only` in the index |
| `history_source='pit' needs the per-row PitHistItems field` | the dataset is not the Q-Former one | use builder `movie_ood_qf` / `amazon_ood_qf`, not CoLLM's `movie_ood` |
| `k_hist=80 but the dataset only emits 50` | `k_hist` > `pit_hist_width` | raise `pit_hist_width` in the dataset config block |
| `memory index predates the point-in-time history change` | stale artifact | rerun `build_memory.py` so it carries `item_in_train` |
| provenance report is ~100 % train on the test split | the per-row path is not active | check `history_source: pit` and that the model logs `hist_slots_filled` > 0 |
| `AssertionError: expected N <unk> slots` | prompt truncated by `max_txt_len`, or a title tokenised to `<unk>` | raise `max_txt_len`; the assert names the count mismatch |
| OOM | in order | SDPA/flash attn → stage-1 batch 24 → 16 → gradient checkpointing → 8-bit base. Do **not** drop stages 2–3 below a real batch of 32 without switching the regularisers to per-micro-batch (`strict_batch: False` acknowledges this) |
| `RuntimeError: dummy base class Stream` | running on CPU | expected; the CPU path is for `scripts/*_test.py` only |

---

## 9. Self-checks before burning GPU hours

Both run on CPU in a couple of minutes and need no Vicuna weights.

```python
!python scripts/check_pit_history.py --data_dir /content/data/ml-1m/ --k_hist 10 20 50
```
The point-in-time gate from §4.2 — run it whenever the data or `k_hist` changes.

```python
!python scripts/smoke_test.py --memory_index /content/ckpt/memory_index_ml1m.pkl
```
63 checks: slot budget and masking, attention finiteness when a user's only valid slot is
itself, FiLM zero-init, RMS matching, the three anti-collapse losses (including the
`sqrt(0)` NaN-gradient case), `L_rank` pair counting, gradient flow to MF / prototypes /
`type_bias`, the splicing index math, per-row `type_ids` with `SLOT_HIST_UNK` routing, the
empty-history row, and the real memory index's bounds under both history sources.

```python
!python scripts/integration_test.py --data_dir /content/data/ml-1m/ \
    --mf_ckpt /content/ckpt/mf_ml1m_d256.pth \
    --memory_index /content/ckpt/memory_index_ml1m.pkl --work_dir /tmp/itest
```
Builds a **tiny random LLaMA** and runs the real thing through it: registry wiring, the
real tokenizer's splicing, stage-2 vs stage-3 freeze patterns, forward/backward with every
loss on, the optimizer's per-group `lr_scale`, a real training epoch, the single-process
evaluation, the diagnostics dump, **all 21 ablation configs** (including `-pit-history`),
and an independent check that the test split's history really is point-in-time.

---

## 10. Deliverables checklist

1. `train_qformer.py` reproducing all four stages from one config each. ✔ shipped
2. `scripts/build_memory.py` with the leakage assertions. ✔ shipped
3. Results table: AUC/UAUC overall + warm + cold, both datasets, 3 seeds (ML-1M).
4. Ablation table from §6.3, with `token_cosine_offdiag` on the `-anticollapse` row.
5. Efficiency table: prompt tokens (§6.4), train time, `ms_per_sample` vs CoLLM-MF and
   TALLRec — run those two baselines through the same eval command for a fair clock.
6. The `type_bias.softmax(-1)` heatmap (L × T): every row of
   `<output_dir>/<job_id>/qformer_diagnostics.jsonl` carries `type_bias_softmax` and
   `attn_by_type`; plot the last one.

```python
import json, numpy as np, matplotlib.pyplot as plt
rows = [json.loads(l) for l in open('/content/logs/stage3_ml1m/<job_id>/qformer_diagnostics.jsonl')]
d = rows[-1]
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for ax, key, title in zip(axes, ["type_bias_softmax", "attn_by_type"],
                          ["learned type_bias (softmax)", "actual attention mass"]):
    m = np.array(d[key]); im = ax.imshow(m, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(d["slot_type_names"]))); ax.set_xticklabels(d["slot_type_names"])
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels([f"token{i}" for i in range(m.shape[0])])
    ax.set_title(title); plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig('/content/type_bias_heatmap.png', dpi=150)
```

---

## 11. Deviations from the spec, and why

Judgment calls worth knowing about before you read the results.

| Spec | Implemented | Why |
|---|---|---|
| history from a per-user train-only lookup | **per-row point-in-time** (`history_source: pit`), from the row's own `his` | matches what CoLLM feeds `<ItemTitleList>`; verified point-in-time (§4.1). `train_only` kept for the `-pit-history` ablation |
| "delete the train-only history builder" | kept, demoted to ablation-only | deleting it would make the `-pit-history` ablation row impossible to run |
| `k_i` capped at 10 | config `k_hist`, default 50 | CoLLM's cap exists because history is *rendered as text*; memory slots cost zero prompt tokens |
| fitted objects (KNN, KMeans, genre means) | unchanged, still train-split-only | fitting across valid/test would use *other users'* futures — real leakage |
| the rest of the memory index | held in the model as non-persistent buffers, keyed by `UserID` | tiny; no collate work, no dataset changes, covers every split automatically |
| bf16 throughout | CoLLM's fp16 + GradScaler for the LLM; the whole CIE block in **fp32** | keeps the frozen backbone byte-identical to CoLLM's setup; the Q-Former is tiny, so fp32 attention/softmax/losses cost nothing and are much easier to trust |
| `s = logit(Yes) − logit(No)` everywhere | `L_rank` uses that; `L_bce` **and eval** use `logit(Yes)` exactly as CoLLM | eval must stay numerically comparable. `rank_score: yes` switches `L_rank` too |
| new files under `minigpt4/` | standalone `qformerrec/` overlay package | zero edits to the CoLLM tree, so the baseline stays reproducible and the diff is reviewable |
| prompt ~55–70 tokens | 86.5 with CoLLM's wording; 60.5 with the terse prompt (both ship) | see §6.4 |
| `T` new files incl. `samplers.py` under `minigpt4/datasets` | `qformerrec/datasets/samplers.py` | same reason as above |
| — | added: gradient clipping (1.0), per-group `lr_scale` scheduler, UAUC-safe metric, warm/cold split fix, per-step `empty_cache` made optional | CoLLM's loop has no clipping, its schedulers overwrite per-group lrs, its UAUC NaNs on modern sklearn, and its ML-1M warm/cold split does not run at all |

Also worth stating in the paper: Amazon-Book genres are KMeans pseudo-genres, not real
categories (§2.1), and our memory is train-only while CoLLM's prompt history is not
(§4.1).
