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

Colab's runtime is **Python 3.12**. CoLLM's own `requirements.txt` is from the 3.9/3.10
era and **cannot be installed there** — do not use it:

* `peft==0.4.0` created LoRA adapters in fp32; **peft ≥ 0.5 casts them to the base fp16
  dtype**, which `GradScaler` cannot unscale. We pin 0.9.0 (0.4.0 predates the
  `prepare_model_for_int8_training` removal but is otherwise ageing) and cast trainable
  params back to fp32 in the runner — see `run.fp32_trainable`.
* `transformers==4.28.0` requires `tokenizers<0.14`, and tokenizers has **no cp312 wheel
  below 0.14**. pip falls back to building it from Rust source and dies with
  `Failed building wheel for tokenizers`.
* `scikit-learn==1.2.2` has no cp312 wheel either — it compiles from source (slow, and
  unnecessary).
* `decord` has no wheel for Python ≥ 3.11 at all (last release 0.6.0, 2021).

So the pins here have both a lower and an upper bound, and each bound has a reason:

| Package | Pin | Lower bound because | Upper bound because |
|---|---|---|---|
| `transformers` | **4.36.2** | < 4.34 needs `tokenizers<0.14` → no cp312 wheel | **5.x silently produces `loss=nan`** — see below. Verified working on 4.28.0 and 4.49.0 |
| `peft` | **0.9.0** | — | 0.10.0 removed `prepare_model_for_int8_training`, which CoLLM imports **by name** |
| `scikit-learn` | *unpinned* | — | — (see §5.3: the NaN-UAUC problem is fixed in code, not by a pin) |
| `decord` | **not installed** | — | stubbed by `qformerrec/compat.py`; CoLLM only imports it for video datasets |

### 1.1 Mount Drive and clone both repos

```python
from google.colab import drive; drive.mount('/content/drive')

%cd /content
!git clone -q https://github.com/zyang1580/CoLLM.git         # the baseline
!git clone -q https://github.com/htainvn/QFormerRec.git      # this work

import os
os.environ["COLLM_ROOT"] = "/content/CoLLM"                  # read by train_qformer.py
%cd /content/QFormerRec
!mkdir -p /content/data /content/ckpt /content/logs
!git log --oneline -1                                        # record the commit you ran
```

`QFormerRec` is a **standalone overlay package**: it imports `minigpt4.*` from
`COLLM_ROOT` and registers its own model / task / runner / builders on import. **No file
in the CoLLM tree is modified**, so the baseline stays reproducible from the same checkout.
The package itself needs no install — `train_qformer.py` puts both trees on `sys.path`.

Pin a revision when you want a run reproducible later:

```python
# !cd /content/QFormerRec && git checkout -q <sha>
```

Iterating on the code? **Pull, do not re-clone** — a re-clone wipes the config edits you
make in §6/§7:

```python
%cd /content/QFormerRec
!git pull -q && git log --oneline -1
```

**If the repo is private**, the anonymous HTTPS clone fails with a 404 (Colab has no SSH
key, so the `git@github.com:` remote you push with will not work there either). Use a
fine-grained PAT with read access and keep it out of the saved notebook:

```python
from getpass import getpass
tok = getpass("GitHub token: ")
!git clone -q https://{tok}@github.com/htainvn/QFormerRec.git
del tok
```

The Drive mount is only for the Vicuna weights (§6) and for keeping logs and checkpoints
across session restarts (§7) — the code comes from git.

### 1.2 Install the pinned stack

```python
!pip -q install "transformers==4.36.2" "peft==0.9.0" \
    omegaconf webdataset timm iopath sentencepiece opencv-python-headless \
    scikit-learn pandas scipy
# torch / torchvision / numpy: keep Colab's preinstalled versions.
# Do NOT install decord, and do NOT pin scikit-learn or numpy.
```

Or equivalently `!pip -q install -r requirements.txt`, which carries the same pins and the
reasoning as comments.

Two things to expect, both harmless:

1. **Dependency-conflict warnings** naming `gradio` and the preinstalled `transformers`
   5.x / `huggingface_hub` 1.x. Colab ships transformers 5.x; pinning 4.36.2 downgrades
   `huggingface_hub` below 1.0, which the preinstalled gradio dislikes. Nothing here uses
   gradio.
2. Colab pre-imports some of these packages, so **restart the runtime after installing**
   (`Runtime → Restart session`), then resume from §1.3. Without a restart you can end up
   running against the *old* transformers still in memory.

> **The transformers 5.x trap — the single most expensive way to lose an afternoon.**
> Colab preinstalls transformers 5.13.1. On 5.x, CoLLM's vendored `modeling_llama.py`
> imports fine, loads fine, and reports **every parameter finite** — then the forward
> returns NaN and you get `loss=nan`. The mechanism: 5.x's loader reports
> `self_attn.rotary_emb.inv_freq | MISSING` and leaves the vendored
> `LlamaRotaryEmbedding`'s `cos_cached` / `sin_cached` **buffers** as uninitialised memory
> (values like `1e+24`, `6e-38`). transformers 4.x re-runs the module init and fills them
> correctly. Because those are *buffers*, not parameters, nothing looks wrong until the loss
> is NaN. `check_environment()` now reproduces this in under a second and **refuses to
> start** — if you must override, `QFORMERREC_ALLOW_BAD_ENV=1`, but the run will be
> garbage.

### 1.3 Verify the environment before going further

```python
%cd /content/QFormerRec
!python -c "from qformerrec.compat import check_environment as c; import sys; sys.exit(1 if c() else 0)" \
    && echo "environment OK"
```

`check_environment()` also runs automatically at the top of every entry point and prints a
`[compat]` line. It names the two failure modes above explicitly — a `peft` without
`prepare_model_for_int8_training`, and a `transformers` whose `utils` no longer exports the
docstring decorators CoLLM's vendored LLaMA imports — rather than letting them surface as
confusing ImportErrors deep inside CoLLM.

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

Measured, ML-1M:

```
k_hist split    train   valid    test  unk_item        filled
    10 test      9.3%   14.2%   76.5%    0.83%    9.37/10
    10 valid    18.9%   81.1%    0.0%    0.72%    9.20/10
    50 test     22.1%   19.8%   58.2%    0.80%   34.64/50
```

Measured, Amazon-Book (727 463 train rows; 49 s, 1.9 GB peak RSS):

```
k_hist split    train   valid    test  unk_item        filled
    10 test     70.1%   16.3%   13.6%   10.00%    8.90/10
    10 valid    86.5%   13.5%    0.0%    3.99%    8.97/10
    50 test     88.2%    7.0%    4.9%    4.13%   26.48/50
```

**If ML-1M test comes back mostly `train`, the change did not take effect.** The
integration test asserts this independently.

Two things the Amazon run prints that are expected, not faults:

* `[note] 100 (user, item) pairs appear with BOTH labels` — the released Amazon data has
  100 duplicate pairs with conflicting labels. The check therefore asks whether a pair has
  *a* positive occurrence rather than trusting the first label seen; 18 089 750 / 18 089 750
  history entries pass. (ML-1M has no such pairs.)
* the unk-item rate is **10 %** on Amazon test at `k_hist=10` (vs 0.8 % on ML-1M) — that is
  the 90 % train-coverage figure, and it is exactly what `unk_item` + `SLOT_HIST_UNK` exist
  to absorb. It falls to 4 % at `k_hist=50`, because reaching further back lands in
  train-era items.

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

### 5.0 Getting Vicuna-7B v0 — read this before `sed`-ing a path

`OSError: Incorrect path_or_model_id: '/content/drive/MyDrive/vicuna-7b-v0'` just means the
placeholder path in the configs does not exist yet. There is no one-line download, because
**`lmsys/vicuna-7b-v0` does not exist as merged weights** — checked against the Hub. Only
the *delta* is published, and it cannot be loaded directly.

Three options, in decreasing fidelity to CoLLM's published table:

**(a) Apply the official delta — matches the paper.** ~27 GB of downloads and a merge step.
Do it on local disk, not Drive.

```python
from huggingface_hub import snapshot_download
snapshot_download("huggyllama/llama-7b",      local_dir="/content/llama-7b")
snapshot_download("lmsys/vicuna-7b-delta-v0", local_dir="/content/vicuna-7b-delta-v0")
```
```python
!pip -q install "git+https://github.com/lm-sys/FastChat.git@v0.1.10"
!python -m fastchat.model.apply_delta \
    --base /content/llama-7b \
    --delta /content/vicuna-7b-delta-v0 \
    --target /content/vicuna-7b-v0
```

`huggyllama/llama-7b` is the usual ungated LLaMA-1 mirror (Meta does not redistribute the
originals; this is why the delta exists at all). FastChat is pinned to `v0.1.10` because
that is the release whose `apply_delta` matches v0 — CoLLM's `PrepareVicuna.md` says the
same. Note this pin will pull an old `transformers`, so **do the merge in a separate
runtime, or re-run §1.2 afterwards** to restore `transformers==4.36.2`.

**(b) A community merged-v0 mirror — fastest way to unblock.** Several exist, e.g.
`ZzZZCHS/vicuna-7b-v0` (13.5 GB, a genuine `LlamaForCausalLM` config). Provenance is
unverified — it is someone's re-upload of the merged result — so if the numbers matter,
prefer (a).

```python
from huggingface_hub import snapshot_download
snapshot_download("ZzZZCHS/vicuna-7b-v0", local_dir="/content/vicuna-7b-v0")
```

**(c) A newer Vicuna — cleanest licensing, but changes the baseline.** `lmsys/vicuna-7b-v1.5`
(Llama-2 based) and `lmsys/vicuna-7b-v1.3` are published as full merged weights, no delta
step:

```python
from huggingface_hub import snapshot_download
snapshot_download("lmsys/vicuna-7b-v1.5", local_dir="/content/vicuna-7b-v1.5")
```

If you go this way you **must re-run the CoLLM-MF and TALLRec baselines yourself** — the
reference table in §6.2 is v0-only, and comparing your v1.5 numbers against their v0 numbers
would not be a valid comparison. Say which Vicuna you used in the writeup either way.

Then point the configs at whatever you produced:

```python
VICUNA = "/content/vicuna-7b-v0"        # or /content/vicuna-7b-v1.5
!sed -i "s#/content/vicuna-7b-v0#$VICUNA#" train_configs/*.yaml
!ls $VICUNA && grep -h llama_model train_configs/stage2_qformer_ml1m.yaml
```

> **Why the Python API and not `huggingface-cli`?** That command was renamed to `hf` in
> `huggingface_hub` 0.34.0 and **removed outright in 1.0.0** (later 1.x re-added it as a
> deprecation shim). Which one you get depends on *when* you run the cell: the §1.2 pins put
> you on hub 0.36.2 (both `hf` and `huggingface-cli` work, the latter with a warning), while
> Colab preinstalls 1.23.0 (`huggingface-cli` is a shim). `snapshot_download` has the same
> `(repo_id, local_dir=...)` signature on both, so it works either way. If you want a shell
> command, `!hf download <repo> --local-dir <dir>` is the current spelling. Do not pass
> `local_dir_use_symlinks` — removed in hub 1.0.

Practical notes:

* **Do not load the weights from Drive.** A Drive-mounted 13.5 GB load is minutes of FUSE
  overhead every session; download to `/content` (fast local disk) and re-download after a
  restart, or `cp` from Drive to `/content` first.
* v0 configs carry `bos_token_id: 0, eos_token_id: 1, vocab_size: 32001`, which differs from
  v1.5's `1 / 2 / 32000`. That is the known MiniGPT-4-era v0 quirk and is harmless here —
  the code reads special-token ids from the *tokenizer*, not the config, so the `<unk>`
  soft-token splicing is unaffected. If the ids ever did collide, the splicing assert in
  §8 fires loudly rather than corrupting the prompt silently.

```python
# 40 GB A100: halve the batch -- AND double max_epoch, or you halve the sample budget too
!sed -i 's#batch_size_train: 32#batch_size_train: 16#' train_configs/stage1_lora_ml1m.yaml
!sed -i 's#max_epoch: 60#max_epoch: 120#'              train_configs/stage1_lora_ml1m.yaml
!python train_qformer.py --cfg-path train_configs/stage1_lora_ml1m.yaml
```

**"epoch" here does not mean a pass over the data.** CoLLM's runner validates every
`iters_per_epoch` steps, and stage 1 sets that to 100. So one "epoch" is
`100 x batch_size` samples:

| batch | samples per "epoch" | share of the 33 891 train rows | 60 "epochs" |
|---|---|---|---|
| 32 | 3 200 | 9.4 % | 5.7 real passes |
| 16 | 1 600 | **4.7 %** | **2.8 real passes** |

Two consequences worth internalising before you judge a run:

* **The first `[valid]` line is after ~1 600 samples.** Expect it to be ~0.50 AUC. It is not
  a failure signal. LoRA's `lora_B` is zero-initialised, so at step 0 the adapter contributes
  exactly nothing and you are reading base Vicuna zero-shot on the TALLRec prompt — which is
  about chance on ML-1M. Judge the *trend* and `best_epoch`, never epoch 0.
* **Reducing the batch silently reduces total training**, because the step count is fixed by
  `max_epoch x iters_per_epoch`. Halving the batch to fit 40 GB halves the samples seen, so
  double `max_epoch` to keep the budget.

Read the trend rather than the last line:

```python
import json, glob
log = sorted(glob.glob('/content/logs/stage1_ml1m/*/log.txt'))[-1]
rows = [json.loads(l) for l in open(log) if l.startswith('{') and 'valid_auc' in l]
for k, r in enumerate(rows):
    print(f"epoch {k:3d}  auc={r['valid_auc']:.4f}  uauc={r['valid_uauc']:.4f}  "
          f"best_epoch={r['valid_best_epoch']}")
print(f"\nbest auc={max(r['valid_auc'] for r in rows):.4f}  "
      f"best uauc={max(r['valid_uauc'] for r in rows):.4f}  over {len(rows)} validations")
```

Also check the loss is actually moving, which is the real "is it learning" signal:

```python
!grep "train epoch" /content/logs/stage1_ml1m/*/train.log | head -5
!grep "train epoch" /content/logs/stage1_ml1m/*/train.log | tail -5
```

**If AUC rises, peaks around epoch 3, then decays back to chance, the lr is too high.**
That exact curve was observed:

```
epoch 0  auc=0.5650   epoch 3  auc=0.6504  <- peak     epoch 7   auc=0.4916
epoch 1  auc=0.5863   epoch 4  auc=0.6345              epoch 9   auc=0.4918
epoch 2  auc=0.5510   epoch 5  auc=0.5349              epoch 11  auc=0.5170
```

The peak lands at epoch 3 for a reason, and it was two compounding faults, both now fixed:

1. **The inherited warmup did not warm up.** CoLLM's schedule gates on the *global* step
   (`total < warmup_steps`) but ramps on the *within-epoch* step. With the shipped
   `warmup_steps: 300` and `iters_per_epoch: 100`, the ramp restarted every epoch and then
   jumped straight to full lr:

   | epoch | lr, old (broken) | lr, fixed |
   |---|---|---|
   | 0 | 1e-05 → 3.4e-04 | 1.0e-06 → 3.4e-05 |
   | 1 | 1e-05 → 3.4e-04 *(restart)* | 3.4e-05 → 6.7e-05 |
   | 2 | 1e-05 → 3.4e-04 *(restart)* | 6.7e-05 → 1.0e-04 |
   | 3 | **1.0e-03** *(jump)* | 1.0e-04 |

   So the model improved for three epochs only because the lr was accidentally small, and
   collapsed the moment the real lr arrived. `linear_warmup_cosine_lr_scaled` now ramps on
   the global step.
2. **`init_lr: 1e-3` is a CIE-stage lr, not a LoRA one.** It is what CoLLM uses for its
   mapping module; for LoRA on a 7B it is roughly 10x too high. Stage 1 now ships
   `init_lr: 1e-4`, `min_lr: 1e-5`, `warmup_lr: 1e-6`.

Your peak checkpoint is not lost: selection is on validation UAUC, so
`checkpoint_best.pth` holds the best epoch (0.6348 in the run above), not the last one.
Confirm with `valid_best_epoch` in the trend print. A collapsed tail wastes time but not the
result — and it is likely how a 1e-3 run can still report a decent number.

If AUC is still ~0.50 after 20+ validations **and** the loss is flat, it is undertraining:
raise `max_epoch`, and check `warmup_steps` is small relative to
`max_epoch x iters_per_epoch`.

`prompt_tokens=nan` in the stage-1 `[valid]` line is expected, not a fault: stage 1 runs
CoLLM's legacy `mini_gpt4rec_v2`, which does not report prompt length, so there is nothing to
average. Stage 2/3 report a real number.

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

**Where the output goes.** Four places, and only two are files:

| Sink | Contents |
|---|---|
| cell output | everything, live (stdout is made line-buffered at startup) |
| `<output_dir>/<job_id>/train.log` | every `logging` record: param groups, the loss/lr curve, the diagnostics, eval metrics |
| `<output_dir>/<job_id>/log.txt` | JSON lines: the config, then one row per validation |
| `<output_dir>/<job_id>/qformer_diagnostics.jsonl` | the `L x T` heatmap data |

**Never pipe a training run into `tail` or `head`.** `tail -30` cannot emit anything until
its stdin closes, so it shows *nothing at all* until the run finishes — measured: 0 bytes
for the entire run, then 3 943 bytes at exit. This is the usual reason a training cell looks
dead. Just run the command bare, and Colab streams it:

```python
!python train_qformer.py --cfg-path train_configs/stage2_qformer_ml1m.yaml
```

If you want a saved copy too, use `tee`, which passes data through as it arrives (measured
live from t=5 s):

```python
!python train_qformer.py --cfg-path train_configs/stage2_qformer_ml1m.yaml 2>&1 \
    | tee /content/logs/stage2_run.log
```

`grep` in a pipe has the same problem unless you give it `--line-buffered`.

A second, independent cause of a silent cell is Python block-buffering stdout (8 KB) because
`!python` hands it a pipe rather than a TTY — CoLLM reports progress with `print()`, so it
accumulates. Measured before the fix: stalled at 12 511 bytes for 20 s, then dumped 5 KB at
exit. The entry point now calls `enable_live_output()` at startup, so this is handled
(`python -u` does the same for scripts of your own).

Either way `train.log` is written continuously, so you can watch from a second cell while
the first one runs:

```python
!tail -f /content/logs/stage2_ml1m/*/train.log     # or tail -20 for a snapshot
```

| Watch | Healthy | If not |
|---|---|---|
| `pref_token_norm` vs `llm_emb_norm` | within 2× | the RMS matching is off — soft prompt will do nothing |
| `token_cosine_offdiag` | **< 0.4** | > 0.6 → raise `lambda_div` to 0.3 before touching anything else |
| `z_std_mean` | not trending to 0 | tokens are collapsing to a constant; check `lambda_var` |
| per-token `top3` attn types | **differ across tokens** | tokens are redundant; check `type_bias` is learning (lr too low?) |
| `rank_pairs_per_batch` | ≫ 0 (≈15 at U=4×m=4; ~72 at U=8×m=6) | the user-grouped sampler is off → `L_rank` has no signal |
| `hist_slots_filled` | ≫ 0 | 0 → `PitHistItems` is not reaching the model |
| `hist_unk_rate` | ~1 % (ML-1M), ~10 % (Amazon) | much higher → wrong `item_in_train`, i.e. a stale memory index |
| `loss_bce` | a real number | `nan` → almost certainly transformers 5.x (§1.2) |
| `[runner] cast N trainable tensors ... to fp32` | expected in stages 1 and 3 | absent in stage 2 is normal — the Q-Former is built in fp32 already |
| `history_source` | `pit` | `train_only` means you are running the ablation by accident |
| `valid_uauc` in `log.txt` | a real number | **NaN → see below** |

**The NaN-UAUC trap.** CoLLM's `uAUC_me` skips single-class users via a bare `except`
around `roc_auc_score`. scikit-learn ≥ ~1.3 *returns `nan`* for single-class input instead
of raising, so those users are no longer skipped and the mean comes out `NaN`. On the
released ML-1M valid split that is **44 of 283 evaluable users** — i.e. UAUC, the metric
this project is selected and reported on, silently becomes NaN, checkpoint selection never
fires, and every reported UAUC is garbage. The fix is in code, not in a pin: `qformerrec/metrics.py::uauc_score()` filters those
users explicitly and is numerically identical to `uAUC_me` under sklearn ≤ 1.2 (verified —
both stacks return UAUC 0.487485 on the same inputs). That is why `scikit-learn` is
unpinned here even though CoLLM pins 1.2.2, which has no cp312 wheel. The eval log reports
`uauc_users_scored` / `uauc_users_skipped` so you can see the filtering happen, and
`scripts/pretrain_mf.py` uses the same function so the stage-0 gate cannot silently NaN
either.

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

Everything worth keeping is small — a few hundred MB per run — so push it off the runtime
as soon as each stage finishes. Two options; use either or both.

### 7.1 Checkpoints to S3-compatible storage

What each stage produces, and which command it needs:

| Stage | Path | Kind | Size (real Vicuna) |
|---|---|---|---|
| 0 — MF | `/content/ckpt/mf_{ml1m,amazon}_d256.pth` | loose **file** | 4 MB / 58 MB |
| 0b — memory index | `/content/ckpt/memory_index_*.pkl` | loose **file** | < 1 MB / few MB |
| 1 — LoRA | `/content/logs/stage1_*/<job_id>/` | **dir** | ~50 MB (LoRA + optim state) |
| 2 — Q-Former | `/content/logs/stage2_*/<job_id>/` | **dir** | ~170 MB |
| 3 — joint | `/content/logs/stage3_*/<job_id>/` | **dir** | ~220 MB |
| eval | `/content/logs/eval_*/<job_id>/` | **dir** | < 1 MB (metrics only) |

Each run dir holds `checkpoint_best.pth`, `train.log` (the full run log), `log.txt`
(per-validation metrics) and `qformer_diagnostics.jsonl` — you want all four, not just the
weights: the last three are deliverables 3–6, and together they are a few hundred KB.

```python
import os
os.environ["ENDPOINT_URL"] = "https://<your-endpoint>"     # same var as your other projects
RUN = "ml1m-pit-k50"        # tag this run; mirrors the `sella-aug` slot in your commands
os.environ["RUN"] = RUN
BUCKET = "s3://qformerrec"
os.environ["BUCKET"] = BUCKET
```

```python
# --- stage 0 + 0b. These are FILES, so `cp`, not `sync` (`s3 sync` requires a
# --- directory source and errors on a file).
!aws --endpoint-url="$ENDPOINT_URL" s3 cp /content/ckpt/mf_ml1m_d256.pth       $BUCKET/mf/$RUN/
!aws --endpoint-url="$ENDPOINT_URL" s3 cp /content/ckpt/memory_index_ml1m.pkl  $BUCKET/memory_index/$RUN/

# --- or push the whole ckpt dir in one shot, which is simpler and idempotent
!aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/ckpt $BUCKET/ckpt/$RUN/ --exclude "*.tmp"
```

```python
# --- stages 1-3 and eval. These ARE directories, so `sync` works as in your other
# --- projects. Syncing the parent picks up the <job_id> subdir automatically.
!aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/logs/stage1_ml1m $BUCKET/qformer_stage1/$RUN/
!aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/logs/stage2_ml1m $BUCKET/qformer_stage2/$RUN/
!aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/logs/stage3_ml1m $BUCKET/qformer_stage3/$RUN/
!aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/logs/eval_ml1m   $BUCKET/qformer_eval/$RUN/
```

Or everything in one call, which is what I would actually run after each stage:

```python
!aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/ckpt $BUCKET/ckpt/$RUN/
!aws --endpoint-url="$ENDPOINT_URL" s3 sync /content/logs $BUCKET/logs/$RUN/ \
    --exclude "*/result/*"
```

### 7.2 Restoring after a restart

This is the direction that actually saves you time — Colab keeps nothing under `/content`.

```python
!mkdir -p /content/ckpt /content/logs
!aws --endpoint-url="$ENDPOINT_URL" s3 sync $BUCKET/ckpt/$RUN/ /content/ckpt
!aws --endpoint-url="$ENDPOINT_URL" s3 sync $BUCKET/logs/$RUN/ /content/logs
!ls -la /content/ckpt && find /content/logs -name "checkpoint_best.pth"
```

Then re-point the configs, since the `<job_id>` will differ from the one in the yaml:

```python
import glob
best1 = sorted(glob.glob('/content/logs/stage1_ml1m/*/checkpoint_best.pth'))[-1]
best2 = sorted(glob.glob('/content/logs/stage2_ml1m/*/checkpoint_best.pth'))[-1]
!sed -i "s#ckpt_lora: .*#ckpt_lora: $best1#" train_configs/stage[23]_qformer_ml1m.yaml
!sed -i "s#^  ckpt: .*#  ckpt: $best2#"      train_configs/stage3_qformer_ml1m.yaml
!grep -E "^  ckpt" train_configs/stage3_qformer_ml1m.yaml
```

To resume an *interrupted* stage rather than start the next one, set `resume_ckpt_path` to
the restored checkpoint — it restores model + optimizer + epoch, which is why the optimizer
state is worth the extra MB:

```yaml
run:
  resume_ckpt_path: /content/logs/stage2_ml1m/<restored_job_id>/checkpoint_best.pth
```

Practical notes:

* **Never sync the Vicuna weights** (13.5 GB, and re-downloadable from the Hub in minutes).
  If you sync `/content` wholesale, exclude them.
* Add `--dryrun` first when you are unsure what a command will move.
* Prefer syncing **between** stages, not mid-training: `checkpoint_best.pth` is rewritten in
  place whenever validation UAUC improves, so a mid-write copy can be truncated.
* `sync` compares size and mtime, so re-running it is cheap and idempotent.
* If `aws` is missing from the runtime: `!pip -q install awscli`.

### 7.3 Or just use Drive

```yaml
run:
  output_dir: /content/drive/MyDrive/collm_logs/stage2_ml1m   # absolute path is honoured
  resume_ckpt_path: /content/drive/MyDrive/collm_logs/stage2_ml1m/<job_id>/checkpoint_best.pth
```

Writing checkpoints straight to Drive is fine (they are small and written rarely). Do not
put the *dataset* or the *Vicuna weights* on Drive — those are read constantly, and the FUSE
overhead is what makes Drive-backed runs slow.

## 8. Failure modes, in the order you will hit them

| Symptom | Cause | Fix |
|---|---|---|
| `huggingface-cli: command not found`, or a deprecation warning | renamed to `hf` in hub 0.34, removed in hub 1.0 | use `snapshot_download(repo_id, local_dir=...)` — stable across 0.x and 1.x — or `hf download` |
| `OSError: Incorrect path_or_model_id: '.../vicuna-7b-v0'` | the config's Vicuna path is a placeholder, and no merged `vicuna-7b-v0` exists on the Hub | see §5.0: apply the delta, use a merged mirror, or switch to v1.5 and re-run the baselines |
| `ValueError: Attempting to unscale FP16 gradients` | Vicuna loads in fp16 and peft ≥0.5 casts LoRA adapters to the base dtype, so the only trainable tensors in stage 1/3 are fp16 — `GradScaler` rejects those in `step()` as well as `unscale_()` | keep `run.fp32_trainable: True` (the default): the runner casts trainable params to fp32 and leaves the backbone fp16. CoLLM avoided this only by pinning peft 0.4.0, which created fp32 adapters |
| `loss=nan`, all params finite, log mentions `LOAD REPORT` or `v4.50` | transformers 5.x corrupting the vendored LLaMA's rotary buffers | `pip install "transformers==4.36.2" "peft==0.9.0"` **and restart the runtime**. `LOAD REPORT` appears only in 5.x, so it is a reliable tell |
| stage-1 AUC ~0.50 at the first validation | expected — that is after `iters_per_epoch x batch` = ~1 600 samples, and LoRA's `lora_B` starts at zero so the adapter contributes nothing yet | judge the trend and `best_epoch`, not epoch 0 |
| stage-1 AUC never leaves ~0.50 and the loss is flat | too little training, or `init_lr` too high for LoRA | you halved the batch without raising `max_epoch` (see §5.1); then try `init_lr: 3e-4` |
| `prompt_tokens=nan` in a stage-1 `[valid]` line | the legacy arch does not report prompt length | expected; stage 2/3 report it |
| training cell shows **nothing at all** until it finishes | the command is piped into `tail`/`head`, which cannot emit until stdin closes | drop the pipe, or use `\| tee run.log`; watch `train.log` from another cell |
| cell shows no output for minutes, then a burst | Python block-buffers stdout on a pipe | handled by `enable_live_output()`; for your own scripts use `python -u`. `train.log` is written live regardless |
| `RuntimeError: incompatible environment` at startup | the fail-fast check working as intended | read the `[compat] PROBLEM:` lines directly above it |
| `Failed building wheel for tokenizers` | `transformers==4.28.0` wants `tokenizers<0.14`, which has no cp312 wheel | use the §1 pins (`transformers==4.36.2`); do **not** use CoLLM's `requirements.txt` |
| `ModuleNotFoundError: No module named 'decord'` | no decord wheel for Python ≥3.11 | do not install it; entry points call `install_import_shims()` — if you wrote your own script, call it before importing `minigpt4` |
| `ImportError: cannot import name 'prepare_model_for_int8_training'` | peft ≥ 0.10 | `pip install peft==0.9.0` |
| scikit-learn compiles from source for minutes | the `==1.2.2` pin | drop the pin; UAUC no longer depends on sklearn's version |
| changes to installed versions seem ignored | Colab pre-imported the old ones | `Runtime → Restart session`, then re-run from the `%cd` cell |
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

All three run on CPU in a couple of minutes and need no Vicuna weights. Both test scripts
begin by printing and validating the environment, so they double as the §1 verification.

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
| CoLLM's `requirements.txt` (transformers 4.28 / peft 0.4 / sklearn 1.2.2, Python 3.9-era) | transformers 4.36.2 / peft 0.9.0, sklearn+numpy unpinned, decord stubbed | that stack cannot be installed on Colab's Python 3.12 at all (§1). Verified: all 75 smoke + 106 integration checks pass on transformers 4.36.2, peft 0.9.0, tokenizers 0.15.2, sklearn 1.9, numpy 2.4, torch 2.13 |
| — | added: gradient clipping (1.0), per-group `lr_scale` scheduler, UAUC-safe metric, warm/cold split fix, per-step `empty_cache` made optional | CoLLM's loop has no clipping, its schedulers overwrite per-group lrs, its UAUC NaNs on modern sklearn, and its ML-1M warm/cold split does not run at all |

Also worth stating in the paper: Amazon-Book genres are KMeans pseudo-genres, not real
categories (§2.1), and our memory is train-only while CoLLM's prompt history is not
(§4.1).
